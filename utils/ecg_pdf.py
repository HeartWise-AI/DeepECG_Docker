"""Extract the 12-lead waveforms drawn in a vector ECG PDF.

ECG management systems export the tracing as vector paths: a pink/red millimetre
grid, one 1 mV calibration pulse per row, and each row of leads as a polyline (often
split into pieces). This module reads those paths with pypdf and returns every lead
in seconds and millivolts. Nothing is guessed:

* scale  -- the printed "<n> mm/s" and "<n> mm/mV" text, checked against the grid
            spacing (5 mm major lines) and the height of the calibration pulses;
* layout -- the printed lead labels (required: nothing in the signals can tell the
            chest leads apart), checked against the signals themselves (Einthoven /
            Goldberger relations, and the rhythm strip against lead II).

Only the standard 3x4 layout with a 10 s lead-II rhythm strip is accepted: it is the
layout of the training renders (utils/ecg_page_renderer.py), so it is the only one that
can be redrawn faithfully. Anything else raises EcgPdfError.

These pages carry patient identifiers. The only text this module keeps is lead
labels and the scale line; error messages never quote page text.

Ported from DeepECG-PNG (branch issue_215/ecg-pdf-pipeline, commit a8d0376,
utils/ecg_pdf.py). Added here:
* limits (MAX_PDF_*): file size and page count before any page is parsed (check_pdf), then the
  decoded content, drawing operations and text of the whole file while its pages are walked,
  before pypdf extracts the ECG page's text;
* a stricter multi-page rule (read_ecg_pdf), and refusing XObjects other than forms and images;
* extract_png_from_pdf, which redraws the leads as a training page and returns it as PNG bytes
  in memory: nothing is written to disk.
"""
from __future__ import annotations

import io
import math
import os
import re
import stat
from dataclasses import dataclass, field

import numpy as np
from pypdf import PdfReader, apply_configuration
from pypdf.errors import LimitReachedError
from pypdf.generic import ContentStream, IndirectObject, TextStringObject
from scipy.signal import resample_poly

from utils.ecg_page_renderer import PAGE_WIDTH, render_training_page, save_training_page

# Limits. An exported ECG PDF (vendor sample) is 0.16 MB with one page, 0.5 MB of decoded
# drawing content, ~41k drawing operators with ~81k operands, and 60 text operations (521 bytes
# of text).
# File size and page count are refused before any page is parsed:
MAX_PDF_BYTES = 20_000_000
MAX_PDF_PAGES = 20
# For the whole file, every Form XObject counted each time it is drawn (a form can be drawn
# many times, so per-stream caps alone do not bound the work), checked while the pages are
# walked, before the text of the ECG page is extracted:
MAX_PDF_OPERATIONS = 2_000_000          # drawing operators + their operands
MAX_STREAM_PARSING_WORK = 1_000_000     # pypdf's parsing work for one content stream
MAX_PDF_CONTENT_BYTES = 20_000_000      # decoded content-stream bytes (also pypdf's cap per stream)
MAX_PDF_TEXT_OPERATIONS = 100_000       # Tj / TJ / ' / "
MAX_PDF_TEXT_BYTES = 2_000_000          # bytes of text they show
MAX_FONT_MAP_BYTES = 1_000_000          # a font's decoded ToUnicode map (also charged as content)
MAX_FONT_MAP_ENTRIES = 65_536
PDF_HEADER_WINDOW = 1024      # "%PDF-" must start within the first 1024 bytes
GRID_MIN_LINES = 60           # red 2-point lines for a page to have an ECG grid (pages: ~400)

# The redrawn page (render_training_page): the size of the training PNGs
PAGE_SIZE = (PAGE_WIDTH, PAGE_WIDTH // 2)
# mV -> the raw MUSE counts the training renderer reads (4.88 uV per count)
MV_PER_COUNT = 0.00488

# Lead order of the MHI training renders (ecg_plotter.py, dataset="MHI").
LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
STANDARD_ROWS = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))
RHYTHM_LEAD = "II"
COLUMN_SECONDS = 2.5
RECORD_SECONDS = 10.0

# Lead relations must hold to this RMS (mV). Exactly derived leads leave only the
# drawing quantisation (~0.005 mV here); a swapped or mislabelled lead leaves
# tenths of a millivolt.
RELATION_TOL_MV = 0.05

_LEAD_LABEL = re.compile(r"^(I|II|III|aVR|aVL|aVF|V[1-6])$")
# Bounded runs, and a number only where one starts: linear in the text (unbounded \d+ backtracks
# quadratically on a long run of digits)
_SPEED = re.compile(r"(?<![\d.,])(\d{1,4}(?:[.,]\d{1,3})?)\s{0,3}mm\s{0,3}/\s{0,3}s(?![a-z])", re.I)
_GAIN = re.compile(r"(?<![\d.,])(\d{1,4}(?:[.,]\d{1,3})?)\s{0,3}mm\s{0,3}/\s{0,3}mV", re.I)
_UNI_GLYPH = re.compile(r"/uni([0-9A-Fa-f]{4})")


class EcgPdfError(ValueError):
    """The PDF is not a vector ECG this module can read without guessing."""


class _Budget:
    """What reading one file may still cost (the MAX_PDF_* limits of the page walk)."""

    def __init__(self):
        self.left = {"operations": MAX_PDF_OPERATIONS, "content_bytes": MAX_PDF_CONTENT_BYTES,
                     "text_operations": MAX_PDF_TEXT_OPERATIONS, "text_bytes": MAX_PDF_TEXT_BYTES}
        self.fonts: dict = {}                  # font object -> _Font, built once per file

    def charge(self, kind: str, amount: int) -> None:
        self.left[kind] -= amount
        if self.left[kind] < 0:
            limit, what = {
                "operations": (MAX_PDF_OPERATIONS, "drawing operations and operands"),
                "content_bytes": (MAX_PDF_CONTENT_BYTES, "bytes of drawing content"),
                "text_operations": (MAX_PDF_TEXT_OPERATIONS, "text operations"),
                "text_bytes": (MAX_PDF_TEXT_BYTES, "bytes of text"),
            }[kind]
            raise EcgPdfError(f"PDF has more than {limit} {what}")


def _pypdf_limits():
    """pypdf's caps per stream: decompression lowered to MAX_PDF_CONTENT_BYTES, and the parsing
    work of one content stream (operators, operands, array elements) to MAX_STREAM_PARSING_WORK,
    so that one parse of a single stream cannot use the whole file's MAX_PDF_OPERATIONS."""
    cap = MAX_PDF_CONTENT_BYTES
    return apply_configuration(
        maximum_declared_stream_length=cap, array_based_stream_maximum_output_length=cap,
        jbig2_maximum_output_length=cap, lzw_maximum_output_length=cap,
        run_length_maximum_output_length=cap, zlib_maximum_output_length=cap,
        brotli_maximum_output_length=cap, image_maximum_buffer_size=cap,
        stream_parsing_work_maximum_count=MAX_STREAM_PARSING_WORK)


@dataclass
class Stroke:
    """A stroked polyline in display coordinates (PDF points): x grows with time
    (left to right as the page is shown), y grows upwards."""
    points: np.ndarray
    rgb: tuple
    width: float


@dataclass
class EcgPdf:
    leads: dict              # lead -> (t_seconds, mV) over its printed 2.5 s window
    rhythm: tuple            # (t_seconds, mV) over the 10 s rhythm strip
    rhythm_lead: str
    speed_mm_s: float
    gain_mm_mv: float
    pt_per_mm: float
    qc: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reading the page
# ---------------------------------------------------------------------------
def _mat_mul(a, b):
    """PDF matrix product a x b (row-vector convention: p' = p . M)."""
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def _display_transform(page):
    """User space -> the page as displayed (honours /Rotate)."""
    box = page.cropbox
    x0, y0, x1, y1 = (float(v) for v in (box.left, box.bottom, box.right, box.top))
    try:
        angle = float(page.rotation or 0)
    except (TypeError, ValueError):
        raise EcgPdfError("unsupported page rotation") from None
    if not math.isfinite(angle) or angle != int(angle) or int(angle) % 90:
        raise EcgPdfError("unsupported page rotation (not a multiple of 90 degrees)")
    rot = int(angle) % 360
    if rot == 0:
        return lambda x, y: (x - x0, y - y0)
    if rot == 90:
        return lambda x, y: (y - y0, x1 - x)
    if rot == 180:
        return lambda x, y: (x1 - x, y1 - y)
    if rot == 270:
        return lambda x, y: (y1 - y, x - x0)
    raise EcgPdfError(f"unsupported page rotation {rot}")


def _cmyk_to_rgb(c, m, y, k):
    return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))


def _color(vals):
    vals = [float(v) for v in vals]
    if len(vals) == 1:
        return (vals[0],) * 3
    if len(vals) == 3:
        return tuple(vals)
    if len(vals) == 4:
        return _cmyk_to_rgb(*vals)
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Text: read by the page walk itself (not pypdf's extract_text, whose font handling and
# text assembly are outside these limits). Only enough decoding for lead labels and the
# scale line: ToUnicode maps, /Differences glyph names, or the bytes of a simple encoding.
# ---------------------------------------------------------------------------
_GLYPH_NAMES = {"space": " ", "period": ".", "comma": ",", "slash": "/", "hyphen": "-", "minus": "-",
                "colon": ":", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_GLYPH_HEX = re.compile(r"(?:uni([0-9A-Fa-f]{4})|u([0-9A-Fa-f]{4,6}))")
_CMAP_TOKEN = re.compile(rb"<([0-9A-Fa-f\s]*)>|(\[)|(\])|(beginbfchar|endbfchar|beginbfrange|endbfrange)")


def _glyph_char(name: str) -> str:
    """The character of a glyph name; '?' for one that is not needed here (never longer)."""
    name = name.lstrip("/").split(".")[0]
    if len(name) == 1 and name.isascii() and name.isalnum():
        return name
    if name in _GLYPH_NAMES:
        return _GLYPH_NAMES[name]
    m = _GLYPH_HEX.fullmatch(name)
    if m:
        code = int(m.group(1) or m.group(2), 16)
        if code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF:
            return chr(code)
    return "?"


def _utf16(hexdigits) -> str:
    if not isinstance(hexdigits, bytes) or len(hexdigits) % 2:
        return "?"
    text = bytes.fromhex(hexdigits.decode("ascii")).decode("utf-16-be", errors="replace")
    return text if len(text) <= 4 else "?"          # no code may expand into a long string


def _shifted(text: str, k: int) -> str:
    """bfrange: the k-th code after the first maps to the first text with its last char + k."""
    if not text or text == "?" or ord(text[-1]) + k > 0x10FFFF:
        return "?"
    return text[:-1] + chr(ord(text[-1]) + k)


def _parse_to_unicode(data: bytes) -> dict:
    """code -> text of a ToUnicode CMap (bfchar / bfrange); EcgPdfError past MAX_FONT_MAP_ENTRIES."""
    out: dict = {}
    written = 0                                # entries written, overwrites included
    mode, args, array = None, [], None
    for m in _CMAP_TOKEN.finditer(data):
        hexdigits, open_, close, keyword = m.groups()
        if keyword is not None:
            mode = {b"beginbfchar": "char", b"beginbfrange": "range"}.get(keyword)
            args, array = [], None
            continue
        if mode is None:
            continue
        if open_ is not None:
            array = []
            continue
        if close is not None:
            if array is not None:
                args.append(array)
            array = None
        elif array is not None:
            array.append(re.sub(rb"\s", b"", hexdigits))
            continue
        else:
            args.append(re.sub(rb"\s", b"", hexdigits))
        if mode == "char" and len(args) == 2:
            src, dest = args
            if isinstance(src, bytes):
                out[int(src or b"0", 16)] = _utf16(dest)
                written += 1
            args = []
        elif mode == "range" and len(args) == 3:
            lo, hi, dest = args
            if isinstance(lo, bytes) and isinstance(hi, bytes):
                lo, hi = int(lo or b"0", 16), int(hi or b"0", 16)
                n = max(0, min(hi - lo + 1, MAX_FONT_MAP_ENTRIES + 1 - written))
                written += n
                if isinstance(dest, list):
                    for k, d in enumerate(dest[:n]):
                        out[lo + k] = _utf16(d)
                else:
                    first = _utf16(dest)
                    for k in range(n):
                        out[lo + k] = _shifted(first, k)
            args = []
        if written > MAX_FONT_MAP_ENTRIES:
            raise EcgPdfError("a font's ToUnicode map is too large")
    return out


class _Font:
    """Decodes the strings shown in one font: its ToUnicode map, else its /Differences glyph
    names, else the bytes as cp1252 (the simple encodings agree on the characters needed)."""

    def __init__(self, font, budget: "_Budget"):
        self.two_byte = font.get("/Subtype") == "/Type0"
        self.map: dict = {}
        to_unicode = font.get("/ToUnicode")
        to_unicode = to_unicode.get_object() if to_unicode is not None else None
        if to_unicode is not None and hasattr(to_unicode, "get_data"):
            data = to_unicode.get_data()
            if len(data) > MAX_FONT_MAP_BYTES:
                raise EcgPdfError("a font's ToUnicode map is too large")
            budget.charge("content_bytes", len(data))
            self.map = _parse_to_unicode(data)
        elif not self.two_byte:
            encoding = font.get("/Encoding")
            encoding = encoding.get_object() if encoding is not None else None
            differences = encoding.get("/Differences") if isinstance(encoding, dict) else None
            differences = differences.get_object() if differences is not None else []
            if len(differences) > 512:
                raise EcgPdfError("a font's /Differences array is too large")
            code = 0
            for item in differences:
                item = item.get_object() if hasattr(item, "get_object") else item
                if isinstance(item, (int, float)):
                    code = int(item)
                else:
                    if 0 <= code <= 255:
                        self.map[code] = _glyph_char(str(item))
                    code += 1

    def decode(self, data: bytes) -> str:
        if self.two_byte:
            return "".join(self.map.get(int.from_bytes(data[i:i + 2], "big"), "")
                           for i in range(0, len(data) - 1, 2))
        return "".join(self.map.get(b) or bytes([b]).decode("cp1252", errors="replace") for b in data)


def _operand_size(value) -> int:
    """1 + every element of the arrays and values of the dictionaries in it, at any depth."""
    size, stack = 0, [value]
    while stack:
        v = stack.pop()
        size += 1
        if isinstance(v, list):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(v.values())
    return size


def _numbers(operands, n):
    """The first n operands as floats, or None when they are not n numbers (the operator is then
    ignored, as pypdf's text extraction did)."""
    try:
        values = [float(v) for v in operands[:n]]
    except (TypeError, ValueError):
        return None
    return values if len(values) == n and all(math.isfinite(v) for v in values) else None


def _string_bytes(value) -> bytes:
    if isinstance(value, TextStringObject):
        return value.original_bytes
    return bytes(value) if isinstance(value, bytes) else str(value).encode("latin-1", errors="replace")


def read_strokes(reader: PdfReader, page, budget: _Budget | None = None) -> list:
    """Every stroked subpath on the page (Form XObjects included), in display space."""
    return _walk_page(reader, page, budget)[0]


def _walk_page(reader: PdfReader, page, budget: _Budget | None = None) -> tuple:
    """(strokes, texts) of the page, Form XObjects included: every stroked subpath, and every
    piece of text shown as [text, x, y, text object number] (display space, at the position
    where it starts).

    budget: what the file may still cost, shared by its pages and charged here for every
    content stream walked (each time a form is drawn): its decoded bytes, its operators and
    operands, the text it shows and the fonts it decodes; EcgPdfError once a limit is passed
    (None: the limits for this page). XObjects other than forms and images are refused.
    """
    to_display = _display_transform(page)
    strokes: list = []
    texts: list = []
    text_objects = iter(range(1 << 62))       # numbers each text object (BT ... ET) of the page
    budget = _Budget() if budget is None else budget

    def font_of(resources, name):
        fonts = resources.get("/Font") if resources is not None else None
        fonts = fonts.get_object() if fonts is not None else None
        ref = fonts.get(name) if hasattr(fonts, "get") else None
        if ref is None:
            return None
        key = (ref.idnum, ref.generation) if isinstance(ref, IndirectObject) else id(ref)
        if key not in budget.fonts:
            budget.fonts[key] = _Font(ref.get_object(), budget)
        return budget.fonts[key]

    def run(content, resources, ctm, depth, stroke_rgb=(0.0, 0.0, 0.0), width=1.0):
        if depth > 8:
            raise EcgPdfError("Form XObjects nested too deeply")
        if content is None:                    # a page without content (e.g. blank)
            return
        stack = []
        subpaths, cur = [], []
        # text state: text matrix, line matrix, leading, font; a new piece of text starts
        # wherever the text position is set (BT, Tm, Td, TD, T*, ', ")
        tm = tlm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        leading, font, positioned = 0.0, None, True
        text_object = next(text_objects)

        def user(x, y):
            return (ctm[0] * x + ctm[2] * y + ctm[4], ctm[1] * x + ctm[3] * y + ctm[5])

        def show(values):
            nonlocal positioned
            parts = []
            for v in values:
                if isinstance(v, (str, bytes)):
                    data = _string_bytes(v)
                    budget.charge("text_bytes", len(data))
                    parts.append(font.decode(data) if font is not None else "")
                elif isinstance(v, (int, float)) and float(v) <= -250:
                    parts.append(" ")              # a TJ gap of a quarter em or more: a space
            if positioned or not texts:
                texts.append([parts, *user(tm[4], tm[5]), text_object])
            else:
                texts[-1][0].extend(parts)         # a list: no quadratic string growth
            positioned = False

        def move_line(tx, ty):
            nonlocal tm, tlm, positioned
            tlm = _mat_mul([1.0, 0.0, 0.0, 1.0, tx, ty], tlm)
            tm, positioned = tlm, True

        budget.charge("content_bytes", len(content.get_data()))   # decoded (pypdf caps each stream)
        operations = ContentStream(content, reader).operations
        budget.charge("operations", len(operations) + sum(
            _operand_size(v) for operands, _ in operations for v in operands))
        for operands, op in operations:
            op = op.decode() if isinstance(op, bytes) else op
            if op in ("Tj", "TJ", "'", '"'):
                budget.charge("text_operations", 1)
                if not operands:
                    continue
                if op in ("'", '"'):
                    move_line(0.0, -leading)
                show(operands[0] if op == "TJ" and isinstance(operands[0], list) else [operands[-1]])
            elif op == "BT":
                tm = tlm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                positioned = True
                text_object = next(text_objects)
            elif op == "Tm":
                values = _numbers(operands, 6)
                if values is not None:
                    tm = tlm = values
                    positioned = True
            elif op in ("Td", "TD"):
                values = _numbers(operands, 2)
                if values is not None:
                    if op == "TD":
                        leading = -values[1]
                    move_line(*values)
            elif op == "T*":
                move_line(0.0, -leading)
            elif op == "TL":
                values = _numbers(operands, 1)
                if values is not None:
                    leading = values[0]
            elif op == "Tf":
                if operands:
                    font = font_of(resources, operands[0])
            elif op == "q":                    # the text font and leading are graphics state too
                stack.append((ctm[:], stroke_rgb, width, font, leading))
            elif op == "Q":
                if stack:
                    ctm, stroke_rgb, width, font, leading = stack.pop()
            elif op == "cm":
                ctm = _mat_mul([float(v) for v in operands], ctm)
            elif op in ("RG", "G", "K"):
                stroke_rgb = _color(operands)
            elif op in ("SC", "SCN"):
                nums = [v for v in operands if isinstance(v, (int, float)) or hasattr(v, "as_numeric")]
                if nums:
                    stroke_rgb = _color(nums)
            elif op == "w":
                width = float(operands[0])
            elif op == "m":
                if len(cur) > 1:
                    subpaths.append(cur)
                cur = [user(float(operands[0]), float(operands[1]))]
            elif op == "l":
                if cur:
                    cur.append(user(float(operands[0]), float(operands[1])))
            elif op in ("c", "v", "y"):
                if cur:
                    cur.append(user(float(operands[-2]), float(operands[-1])))
            elif op == "h":
                if len(cur) > 1:
                    cur.append(cur[0])
            elif op == "re":
                x, y, w, h = (float(v) for v in operands)
                if len(cur) > 1:
                    subpaths.append(cur)
                subpaths.append([user(x, y), user(x + w, y), user(x + w, y + h), user(x, y + h), user(x, y)])
                cur = []
            elif op in ("S", "s", "B", "B*", "b", "b*"):
                if op in ("s", "b", "b*") and len(cur) > 1:
                    cur.append(cur[0])
                if len(cur) > 1:
                    subpaths.append(cur)
                scale = float(np.sqrt(abs(ctm[0] * ctm[3] - ctm[1] * ctm[2])))
                for sp in subpaths:
                    pts = np.array([to_display(px, py) for px, py in sp], dtype=np.float64)
                    strokes.append(Stroke(pts, tuple(round(c, 3) for c in stroke_rgb), width * scale))
                subpaths, cur = [], []
            elif op in ("f", "F", "f*", "n"):
                subpaths, cur = [], []
            elif op == "Do" and resources is not None:
                xobjects = resources.get("/XObject")
                xobj = xobjects.get_object().get(operands[0]) if xobjects else None
                xobj = xobj.get_object() if xobj is not None else None
                subtype = xobj.get("/Subtype") if xobj is not None else None
                subtype = subtype.get_object() if subtype is not None else None
                if xobj is not None and subtype not in ("/Form", "/Image"):
                    raise EcgPdfError("unsupported XObject type (only forms and images)")
                if xobj is not None and subtype == "/Form":
                    matrix = [float(v) for v in xobj.get("/Matrix", [1, 0, 0, 1, 0, 0])]
                    sub_res = xobj.get("/Resources")
                    run(xobj, sub_res.get_object() if sub_res is not None else resources,
                        _mat_mul(matrix, ctm), depth + 1, stroke_rgb, width)

    res = page.get("/Resources")
    run(page.get_contents(), res.get_object() if res is not None else None, [1, 0, 0, 1, 0, 0], 0)
    return strokes, [["".join(parts), *to_display(x, y), block] for parts, x, y, block in texts]


def _decode_text(text: str) -> str:
    """pypdf leaves glyphs of some subset fonts as '/uniXXXX' names."""
    return _UNI_GLYPH.sub(lambda m: chr(int(m.group(1), 16)), text).strip()


def read_labels_and_scale(texts) -> tuple:
    """(lead labels as [(label, x, y)], speeds, gains) from the page's texts ([text, x, y, text
    object] in display space, from _walk_page). A lead label is a piece of text on its own; the
    scale is read from each text object's pieces joined by spaces (a value and its unit may be
    placed separately). Only lead labels and the scale line are kept -- no other page text is
    retained."""
    labels, speeds, gains, objects = [], set(), set(), {}
    for text, x, y, block in texts:
        t = _decode_text(text)
        if not t:
            continue
        if _LEAD_LABEL.match(t):
            labels.append((t, x, y))
            continue
        objects.setdefault(block, []).append(t)
    for pieces in objects.values():
        t = " ".join(pieces)
        speeds.update(float(m.group(1).replace(",", ".")) for m in _SPEED.finditer(t))
        gains.update(float(m.group(1).replace(",", ".")) for m in _GAIN.finditer(t))
    return labels, speeds, gains


# ---------------------------------------------------------------------------
# Scale, rows, columns
# ---------------------------------------------------------------------------
def _is_grid(s: Stroke) -> bool:
    r, g, b = s.rgb
    return len(s.points) == 2 and r >= 0.6 and r - max(g, b) >= 0.15


def _is_dark(s: Stroke) -> bool:
    return max(s.rgb) <= 0.35


def _grid_pt_per_mm(strokes) -> tuple:
    """Points per millimetre from the 5 mm major grid. Grid lines are grouped by
    (colour, width); the group with the widest regular spacing is the major grid
    and must be 5x the finer group when one exists."""
    groups: dict = {}
    for s in strokes:
        if not _is_grid(s):
            continue
        (x0, y0), (x1, y1) = s.points
        if abs(x0 - x1) < 1e-6:                     # vertical line: a time position
            groups.setdefault((s.rgb, round(s.width, 4)), []).append(x0)
    lines = [np.unique(np.round(xs, 4)) for xs in groups.values()]
    lines = [xs for xs in lines if len(xs) >= 5]
    if not lines:
        raise EcgPdfError("no millimetre grid found (not a vector ECG?)")

    def fit(xs, step):
        # Least-squares slope of position vs line index: one rounding error per line
        # instead of one per gap, so the scale is exact to ~1e-5 over the page. Lines
        # off the lattice (a 0.5 mm or 2.5 mm grid indexed at 1 mm) leave residuals.
        k = np.round((xs - xs[0]) / step)
        slope, icpt = np.polyfit(k, xs, 1)
        if np.abs(xs - (slope * k + icpt)).max() > 0.2 * step:
            raise EcgPdfError("grid lines are not on a regular 1 mm / 5 mm lattice")
        return float(slope)

    # The major (5 mm) group has the widest, gap-free spacing. Finer groups are indexed
    # on major/5: their median gap is biased by coordinate rounding and by the missing
    # lines where they coincide with major lines.
    medians = [float(np.median(np.diff(xs))) for xs in lines]
    major_idx = int(np.argmax(medians))
    major = fit(lines[major_idx], medians[major_idx])
    spacings = [major]
    for i, xs in enumerate(lines):
        if i != major_idx:
            spacings.append(fit(xs, major / 5.0))
    spacings.sort()
    if len(spacings) >= 2 and not 4.9 <= major / spacings[0] <= 5.1:
        raise EcgPdfError(f"grid spacings {np.round(spacings, 4).tolist()} are not a 1 mm / 5 mm pair")
    return major / 5.0, spacings


def _calibration_pulses(strokes, pt_per_mm):
    """Short dark polylines shaped like a 1 mV step: rise of 2-30 mm, 1-12 mm wide."""
    pulses = []
    for s in strokes:
        if not _is_dark(s) or not 4 <= len(s.points) <= 16:
            continue
        w = np.ptp(s.points[:, 0]) / pt_per_mm
        h = np.ptp(s.points[:, 1]) / pt_per_mm
        if 1.0 <= w <= 12.0 and 2.0 <= h <= 30.0:
            pulses.append(s)
    return pulses


def _series(point_arrays, t0_x, baseline_y, pt_per_s, pt_per_mv):
    """Join ONE lead's pieces into a (t, mV) series. Consecutive pieces may share their
    join point; repeated x (joins, vertical strokes) are averaged. Pieces that overlap
    in time by any further point are two traces drawn over the same moment
    (e.g. a lead running into its neighbour's column) and are refused."""
    arrays = sorted(point_arrays, key=lambda a: a[0, 0])
    steps = np.concatenate([np.diff(a[:, 0]) for a in arrays if len(a) > 1])
    step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
    for prev, nxt in zip(arrays, arrays[1:]):
        if prev[-1, 0] - nxt[0, 0] > 0.5 * step:   # a shared join point overlaps by 0
            raise EcgPdfError("trace pieces overlap in time (a lead runs into its neighbour)")
    pts = np.concatenate(arrays)
    order = np.argsort(pts[:, 0], kind="stable")
    x, y = pts[order, 0], pts[order, 1]
    ux, inv = np.unique(np.round(x, 6), return_inverse=True)
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    return (ux - t0_x) / pt_per_s, (uy - baseline_y) / pt_per_mv


def _split_columns(pieces, t0_x, pt_per_s, n_cols=4):
    """Point arrays per column. Leads change at 2.5 s boundaries, where the last point
    of one lead and the first of the next can share the same time, so points are
    never pooled across columns: a piece whose interior lies in one column goes there
    whole; a piece drawn across a boundary is cut at it (the boundary vertex starts
    the next lead)."""
    cols = [[] for _ in range(n_cols)]
    for p in pieces:
        if np.any(np.diff(p.points[:, 0]) < -1e-6):
            raise EcgPdfError("a trace runs backwards in time")
        t = (p.points[:, 0] - t0_x) / pt_per_s
        inner = t[1:-1] if len(t) > 2 else t
        idx = np.clip(np.floor(inner / COLUMN_SECONDS + 1e-9).astype(int), 0, n_cols - 1)
        if idx.min() == idx.max():
            cols[int(idx[0])].append(p.points)
            continue
        which = np.clip(np.floor(t / COLUMN_SECONDS + 1e-6).astype(int), 0, n_cols - 1)
        for c in np.unique(which):
            cols[int(c)].append(p.points[which == c])
    return cols


def _check_coverage(t, t_lo, t_hi, what):
    gaps = np.diff(t)
    if t[0] > t_lo + 0.02 or t[-1] < t_hi - 0.02 or (gaps.size and gaps.max() > 0.02):
        raise EcgPdfError(f"{what} does not cover {t_lo:g}-{t_hi:g} s without gaps")


def _on_grid(t, v, t_lo, t_hi, fs=1000.0):
    grid = np.arange(t_lo, t_hi, 1.0 / fs)
    return np.interp(grid, t, v)


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def open_pdf(path: str) -> PdfReader:
    """
    Open a PDF for reading, after the checks that need no parsing of its pages: a regular file
    of at most MAX_PDF_BYTES, a "%PDF-" header, readable by pypdf, not encrypted (an
    owner-password-only PDF, which opens with the empty password, is fine; AES needs the
    `cryptography` package) and 1 to MAX_PDF_PAGES pages.

    Raises:
        EcgPdfError: if any check fails.
    """
    try:
        info = os.stat(path)
        if not stat.S_ISREG(info.st_mode):     # e.g. a FIFO, whose open() would block
            raise EcgPdfError("not a regular file")
        size = info.st_size
        with open(path, "rb") as f:
            head = f.read(PDF_HEADER_WINDOW)
    except EcgPdfError:
        raise
    except (OSError, ValueError) as e:         # ValueError: e.g. a NUL byte in the path
        raise EcgPdfError(f"cannot read the file ({type(e).__name__})") from None
    if size > MAX_PDF_BYTES:
        raise EcgPdfError(f"PDF is {size} bytes (> {MAX_PDF_BYTES})")
    if b"%PDF-" not in head:
        raise EcgPdfError("not a PDF file")
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            try:                               # an owner-password-only PDF opens with ""
                opened = bool(reader.decrypt(""))
            except Exception:
                opened = False
            if not opened:
                raise EcgPdfError("PDF is encrypted and could not be opened")
        n_pages = len(reader.pages)
    except EcgPdfError:
        raise
    except Exception as e:                     # pypdf raises many types on bad files
        raise EcgPdfError(f"cannot open PDF ({type(e).__name__})") from None
    if n_pages < 1:
        raise EcgPdfError("PDF has no pages")
    if n_pages > MAX_PDF_PAGES:
        raise EcgPdfError(f"PDF has {n_pages} pages (> {MAX_PDF_PAGES})")
    return reader


def check_pdf(path: str) -> None:
    """The checks of open_pdf only (no page is parsed): cheap enough to run on every PDF
    before inference. Raises EcgPdfError."""
    with _pypdf_limits():
        open_pdf(path)


def read_ecg_pdf(path: str) -> EcgPdf:
    """Read a single-page vector ECG PDF into per-lead (t, mV) series.

    Multi-page files: exactly one page may look like an ECG, i.e. carry a red millimetre grid
    or a trace (a dark polyline of 20+ points), and it must carry both; pages with neither
    (e.g. a text report) are ignored. 0 or 2+ such pages are refused, never chosen between.
    """
    with _pypdf_limits():
        return _read_ecg_pdf(path)


def _read_ecg_pdf(path: str) -> EcgPdf:
    reader = open_pdf(path)
    budget = _Budget()                         # for the whole file
    candidates = []
    for i, pg in enumerate(reader.pages):
        st, texts = _walk_page(reader, pg, budget)
        has_grid = sum(_is_grid(s) for s in st) >= GRID_MIN_LINES
        has_trace = any(_is_dark(s) and len(s.points) >= 20 for s in st)
        if has_grid or has_trace:
            candidates.append((st, texts, has_grid and has_trace))
    if len(candidates) != 1:
        raise EcgPdfError(f"expected exactly one ECG page, found {len(candidates)} of {len(reader.pages)} pages")
    strokes, texts, complete = candidates[0]
    if not complete:
        raise EcgPdfError("the ECG page lacks a red millimetre grid or traces (polylines of 20+ points)")
    labels, speeds, gains = read_labels_and_scale(texts)

    # ---- scale -------------------------------------------------------------
    if len(speeds) != 1 or len(gains) != 1:     # counts only: the values come from the page text
        raise EcgPdfError(f"printed scale not found or ambiguous ({len(speeds)} speeds, {len(gains)} gains)")
    speed, gain = speeds.pop(), gains.pop()
    pt_per_mm, grid_spacings = _grid_pt_per_mm(strokes)
    pt_per_s, pt_per_mv = speed * pt_per_mm, gain * pt_per_mm

    pulses = _calibration_pulses(strokes, pt_per_mm)
    traces = [s for s in strokes if _is_dark(s) and len(s.points) >= 20]
    if not traces:
        raise EcgPdfError("no ECG traces found")
    if len(pulses) != 4:
        raise EcgPdfError(f"expected 4 calibration pulses (3 lead rows + rhythm), found {len(pulses)}")
    heights_mm = [float(np.ptp(p.points[:, 1])) / pt_per_mm for p in pulses]
    if any(abs(h - gain) > 0.05 * gain for h in heights_mm):
        raise EcgPdfError(f"calibration pulses are {np.round(heights_mm, 2).tolist()} mm, not the printed gain")
    for pl in pulses:   # a stepped pulse (e.g. chest leads at half gain) is not one gain
        y = pl.points[:, 1]
        lo, hi = y.min(), y.max()
        tol = 0.03 * (hi - lo)
        if not np.all((np.abs(y - lo) <= tol) | (np.abs(y - hi) <= tol)):
            raise EcgPdfError("a calibration pulse has more than two levels (mixed gains are not supported)")

    # ---- rows (top to bottom): baseline = foot of each row's pulse ----------
    baselines = sorted((float(p.points[:, 1].min()) for p in pulses), reverse=True)
    rows = [[] for _ in baselines]
    for s in traces:
        mid = float(np.median(s.points[:, 1]))
        rows[int(np.argmin([abs(mid - b) for b in baselines]))].append(s)
    if any(not r for r in rows):
        raise EcgPdfError("a calibrated row has no trace")
    t0_x = min(float(s.points[:, 0].min()) for s in traces)

    # ---- labels -> layout ----------------------------------------------------
    col_x = [t0_x + k * COLUMN_SECONDS * pt_per_s for k in range(4)]
    found = {}
    for name, lx, ly in labels:
        r = int(np.argmin([abs(ly - b) for b in baselines]))
        c = int(np.argmin([abs(lx - cx) for cx in col_x]))
        if abs(lx - col_x[c]) <= 5 * pt_per_mm:
            found.setdefault((r, c), name)
    expected = {(r, c): STANDARD_ROWS[r][c] for r in range(3) for c in range(4)}
    expected[(3, 0)] = RHYTHM_LEAD
    # The lead relations below check the limb leads and the rhythm strip, but nothing in
    # the signals can tell V1 from V4: without readable labels the layout is a guess.
    if not found:
        raise EcgPdfError("lead labels could not be read, so the chest-lead order cannot be verified")
    wrong = {k: v for k, v in found.items() if expected.get(k) != v}
    if wrong or set(found) != set(expected):
        raise EcgPdfError("lead labels do not match the standard 3x4 + lead-II rhythm layout")
    labels_verified = True

    # ---- leads ----------------------------------------------------------------
    leads = {}
    for r in range(3):
        cols = _split_columns(rows[r], t0_x, pt_per_s)
        for c, name in enumerate(STANDARD_ROWS[r]):
            if not cols[c]:
                raise EcgPdfError(f"no trace for {name}")
            leads[name] = _series(cols[c], t0_x, baselines[r], pt_per_s, pt_per_mv)
            _check_coverage(leads[name][0], c * COLUMN_SECONDS, (c + 1) * COLUMN_SECONDS, name)
    rhythm = _series([p.points for p in rows[3]], t0_x, baselines[3], pt_per_s, pt_per_mv)
    _check_coverage(rhythm[0], 0.0, RECORD_SECONDS, "the rhythm strip")

    # ---- the signals must agree with the layout --------------------------------
    w1 = (0.05, COLUMN_SECONDS - 0.05)
    lead_i, lead_ii, lead_iii = (_on_grid(*leads[n], *w1) for n in ("I", "II", "III"))
    einthoven = _rms(lead_ii - (lead_i + lead_iii))
    w2 = (COLUMN_SECONDS + 0.05, 2 * COLUMN_SECONDS - 0.05)
    avr, avl, avf = (_on_grid(*leads[n], *w2) for n in ("aVR", "aVL", "aVF"))
    goldberger = _rms(avr + avl + avf)
    ii_w2 = _on_grid(*rhythm, *w2)
    limb = _rms(avl - (1.5 * ii_w2 - 2.0 * avf))          # aVL = 1.5 II - 2 aVF
    rhythm_vs_ii = _rms(_on_grid(*rhythm, *w1) - lead_ii)
    qc = {
        "speed_mm_s": speed, "gain_mm_mv": gain, "pt_per_mm": round(pt_per_mm, 5),
        "grid_spacings_pt": [round(s, 4) for s in grid_spacings],
        "labels_verified": labels_verified,
        "einthoven_rms_mv": round(einthoven, 4), "goldberger_rms_mv": round(goldberger, 4),
        "limb_rms_mv": round(limb, 4), "rhythm_vs_lead_ii_rms_mv": round(rhythm_vs_ii, 4),
    }
    failed = [k for k in ("einthoven_rms_mv", "goldberger_rms_mv", "limb_rms_mv", "rhythm_vs_lead_ii_rms_mv")
              if not qc[k] <= RELATION_TOL_MV]
    if failed:
        raise EcgPdfError(f"lead relations fail ({', '.join(f'{k}={qc[k]}' for k in failed)}); "
                          f"leads are mislabelled or the layout is not standard")
    return EcgPdf(leads=leads, rhythm=rhythm, rhythm_lead=RHYTHM_LEAD, speed_mm_s=speed,
                  gain_mm_mv=gain, pt_per_mm=pt_per_mm, qc=qc)


def to_mhi_array(ecg: EcgPdf, fs: int = 250) -> np.ndarray:
    """(fs*10, 12) array in mV, MHI lead order, sample n at t = n/fs.

    Each lead fills only the window it was printed in; lead II is the full 10 s
    rhythm strip. That is exactly what the training renderer draws (it reads lead I
    from 0-2.5 s, aVR from 2.5-5 s, ..., and lead II for the rhythm row), so samples
    outside a lead's window never reach the image. Polylines are interpolated onto a
    1 kHz grid, then decimated with an anti-aliasing filter.
    """
    n_total = int(round(RECORD_SECONDS * fs))
    out = np.zeros((n_total, len(LEAD_ORDER)), dtype=np.float64)
    hi_fs = 4 * fs

    def place(lead_idx, t, v, t_lo, t_hi):
        grid = np.arange(int(round(t_lo * hi_fs)), int(round(t_hi * hi_fs))) / hi_fs
        dense = np.interp(grid, t, v)
        low = resample_poly(dense, 1, 4, padtype="line")
        start = int(round(t_lo * fs))
        out[start:start + len(low), lead_idx] = low[: n_total - start]

    for r in range(3):
        for c, name in enumerate(STANDARD_ROWS[r]):
            if name == RHYTHM_LEAD:
                continue
            place(LEAD_ORDER.index(name), *ecg.leads[name], c * COLUMN_SECONDS, (c + 1) * COLUMN_SECONDS)
    place(LEAD_ORDER.index(ecg.rhythm_lead), *ecg.rhythm, 0.0, RECORD_SECONDS)
    return out


def extract_png_from_pdf(path: str) -> bytes:
    """
    The ECG of a vector ECG PDF, redrawn as a DeepECG-PNG training page, as PNG bytes.

    read_ecg_pdf -> to_mhi_array -> the training renderer (utils/ecg_page_renderer.py),
    saved as the training PNGs were (PNG, 240 dpi) into memory: no file is written. The
    page is always PAGE_SIZE, whatever the PDF declares.

    Raises:
        EcgPdfError: for any PDF that cannot be read this way. Errors from pypdf, numpy or
            the renderer are re-raised as EcgPdfError naming only their type, so no message
            quotes the file's path or content.
    """
    try:
        counts = to_mhi_array(read_ecg_pdf(path)) / MV_PER_COUNT
        buffer = io.BytesIO()
        save_training_page(render_training_page(counts), buffer)
    except EcgPdfError:
        raise
    except LimitReachedError:
        raise EcgPdfError(f"PDF exceeds a limit of the PDF reader (per stream: {MAX_PDF_CONTENT_BYTES} "
                          f"bytes decoded, {MAX_STREAM_PARSING_WORK} operators and operands)") from None
    except Exception as e:                     # MemoryError included: one row, not the run
        raise EcgPdfError(f"cannot read the ECG of this PDF ({type(e).__name__})") from None
    return buffer.getvalue()
