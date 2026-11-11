"""PDF inputs for the image models (issue #100), on CPU, with synthetic vector ECG PDFs only.

  * utils/ecg_pdf.py reads the leads of a vector ECG PDF back exactly and refuses, instead of
    guessing, every PDF it cannot read without guessing (tests ported from DeepECG-PNG
    tests/test_ecg_pdf_pipeline.py, branch issue_215/ecg-pdf-pipeline);
  * the limits checked before a large allocation (file size, pages, drawing operations, nested
    or repeated Form XObjects) and the errors, which never quote the file's path or content;
  * a PDF gives the same model input as the PNG of its extracted page, bit for bit;
  * the page is extracted lazily, one row at a time, by the image DataLoader (num_workers 0 with
    PDFs): a PDF that cannot be read is left out of its batch and listed, the other rows still run;
  * main(): .pdf rows go to the image models with no flag, next to PNG, JPEG, XML and NPY rows in
    one CSV, and nothing is written for a PDF (no PNG, Base64 or embedding file).
"""
import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

pypdf = pytest.importorskip("pypdf")

from pypdf import PdfReader, PdfWriter  # noqa: E402
from pypdf.generic import (  # noqa: E402
    ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject,
)

from test_deepecg_png_pipeline import (  # noqa: E402  (the stub-weights world of the PNG tests)
    PREPROCESSING, _one, pipeline, run, task_csv, world,  # noqa: F401  (world: a pytest fixture)
)
from data import image_dataset  # noqa: E402
from data.image_dataset import PAGE_ERRORS, create_image_dataloader  # noqa: E402
from utils import analysis_pipeline, ecg_pdf  # noqa: E402
from utils.constants import IMAGE, SIGNAL, input_type_of, source_format_of  # noqa: E402
from utils.ecg_page_renderer import render_training_page  # noqa: E402
from utils.ecg_pdf import (  # noqa: E402
    LEAD_ORDER, MAX_PDF_OPERATIONS, MV_PER_COUNT, PAGE_SIZE, STANDARD_ROWS, EcgPdfError, check_pdf, extract_png_from_pdf,
    read_ecg_pdf, to_mhi_array,
)
from utils.image_preprocessing import (  # noqa: E402
    check_ecg_page, load_ecg_image, load_ecg_page, open_png_bytes, preprocess_ecg_image,
)

PT_PER_MM = 72 / 25.4
FS_DRAW = 500
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The training renderer's page of a fixed signal (tests/test_signals_to_png.py of DeepECG_Docker #82)
RENDERER_GOLDEN_PIXELS_SHA256 = 'c4fba3cc04ec22c3d577dc9c8c883e3f1a3f3e6bd41e4474cba235afbd8aefc7'


# ---------------------------------------------------------------------------
# Synthetic vector ECG PDFs (DeepECG-PNG tests/test_ecg_pdf_pipeline.py)
# ---------------------------------------------------------------------------
def synthetic_leads(seed=0):
    """12 leads (mV) on a 500 Hz grid over 10 s; limb leads obey Einthoven/Goldberger."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 10.0, 1 / FS_DRAW)

    def wave():
        out = np.zeros_like(t)
        for f in rng.uniform(0.7, 6.0, size=4):
            out += rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        return out

    lead_i, lead_ii = wave(), wave()
    leads = {"I": lead_i, "II": lead_ii, "III": lead_ii - lead_i,
             "aVR": -(lead_i + lead_ii) / 2, "aVL": lead_i - lead_ii / 2, "aVF": lead_ii - lead_i / 2}
    for v in ("V1", "V2", "V3", "V4", "V5", "V6"):
        leads[v] = wave()
    return t, leads


def make_ecg_pdf(path, t, leads, *, rotate=90, gain=10.0, speed=25.0, labels=True, scale_text=True,
                 label_swap=None, trace_swap=None, pulse_mm=None, continuous_rows=False,
                 stepped_pulse=False, overrun=0, extra_page=None, header_text=None, extra_ops=(),
                 label_runs="separate", in_form_cm=None, extra_fonts=None):
    """Draw leads in the standard 3x4 + lead-II rhythm layout, the way the vendor PDF
    does: display coordinates are mapped onto a rotated page so the parser has to undo
    /Rotate; each row is split into 1 s / column pieces.

    extra_page: "ecg" (the same page again), "grey_ecg" (again, with a grey grid), "grid_only"
    (just the red grid), "red_rules" (a report page ruled by 25 red lines) or "text";
    extra_ops: content operations added to the ECG page; label_runs "row": the labels of a row
    in one text object, placed with Td; in_form_cm: the page drawn as a Form XObject under this
    cm; extra_fonts: {name: (ToUnicode bytes or None, /Differences list or None)} added to the
    ECG page's fonts."""
    W, H = (595.0, 792.0) if rotate == 90 else (792.0, 595.0)

    def user(X, Y):  # display (time right, voltage up) -> user space
        return (W - Y, X) if rotate == 90 else (X, Y)

    ops = []
    t0, bases = 60.0, [470.0, 365.0, 260.0, 155.0]
    pt_s, pt_mv = speed * PT_PER_MM, gain * PT_PER_MM

    def line(points, rgb, width):
        u = [user(x, y) for x, y in points]
        ops.append(f"q {rgb[0]} {rgb[1]} {rgb[2]} RG {width} w {u[0][0]:.4f} {u[0][1]:.4f} m "
                   + " ".join(f"{x:.4f} {y:.4f} l" for x, y in u[1:]) + " S Q")

    x_lo, x_hi, y_lo, y_hi = 30.0, 770.0, 100.0, 520.0
    for k in range(int((x_hi - x_lo) / PT_PER_MM) + 1):
        x = x_lo + k * PT_PER_MM
        major = k % 5 == 0
        line([(x, y_lo), (x, y_hi)], (0.883, 0.6, 0.557) if major else (1.0, 0.752, 0.797), 0.24 if major else 0.048)
    for k in range(int((y_hi - y_lo) / PT_PER_MM) + 1):
        y = y_lo + k * PT_PER_MM
        major = k % 5 == 0
        line([(x_lo, y), (x_hi, y)], (0.883, 0.6, 0.557) if major else (1.0, 0.752, 0.797), 0.24 if major else 0.048)

    height = (pulse_mm if pulse_mm is not None else gain) * PT_PER_MM
    for b in bases:
        x0 = t0 - 20
        if stepped_pulse:   # half gain then full gain: two gains on one page
            line([(x0, b), (x0 + 1, b), (x0 + 1, b + height / 2), (x0 + 3 * PT_PER_MM, b + height / 2),
                  (x0 + 3 * PT_PER_MM, b + height), (x0 + 6 * PT_PER_MM, b + height),
                  (x0 + 6 * PT_PER_MM, b), (x0 + 7 * PT_PER_MM, b)], (0, 0, 0), 0.48)
            continue
        line([(x0, b), (x0 + 1, b), (x0 + 1, b + height), (x0 + 6 * PT_PER_MM, b + height),
              (x0 + 6 * PT_PER_MM, b), (x0 + 7 * PT_PER_MM, b)], (0, 0, 0), 0.48)

    rows = [list(r) for r in STANDARD_ROWS]
    if trace_swap:
        a, b = trace_swap
        for r in rows:
            for c, n in enumerate(r):
                r[c] = b if n == a else a if n == b else n

    def piece(sig, b, lo, hi):
        sel = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        line([(t0 + tt * pt_s, b + vv * pt_mv) for tt, vv in zip(t[sel], sig[sel])], (0, 0, 0), 0.48)

    for r in range(3):
        if continuous_rows:   # one polyline per row; each lead's first vertex is on its boundary
            pts = []
            for c in range(4):
                sel = (t >= c * 2.5 - 1e-9) & (t < (c + 1) * 2.5 - 1e-9)
                pts += [(t0 + tt * pt_s, bases[r] + vv * pt_mv) for tt, vv in zip(t[sel], leads[rows[r][c]][sel])]
            line(pts, (0, 0, 0), 0.48)
            continue
        for c in range(4):
            sig = leads[rows[r][c]]
            for k in range(3):
                lo = c * 2.5 + k
                hi = min(lo + 1.0, (c + 1) * 2.5 - 1 / FS_DRAW)
                if k == 2 and c < 3:
                    hi += overrun / FS_DRAW        # this lead runs into the next column
                piece(sig, bases[r], lo, hi)
    for k in range(10):
        piece(leads["II"], bases[3], k, min(k + 1.0, 10.0 - 1 / FS_DRAW))

    def text(X, Y, s):
        ux, uy = user(X, Y)
        ops.append(f"BT /F1 9 Tf 1 0 0 1 {ux:.3f} {uy:.3f} Tm ({s}) Tj ET")

    if labels:
        names = {(r, c): STANDARD_ROWS[r][c] for r in range(3) for c in range(4)}
        names[(3, 0)] = "II"
        if label_swap:
            inv = {v: k for k, v in names.items() if k != (3, 0)}
            ka, kb = inv[label_swap[0]], inv[label_swap[1]]
            names[ka], names[kb] = names[kb], names[ka]
        if label_runs == "row":
            for r in range(4):
                row = [(c, names[(r, c)]) for c in range(4) if (r, c) in names]
                pts = [user(t0 + c * 2.5 * pt_s + 2, bases[r] + 0.6 * pt_mv) for c, _ in row]
                parts = [f"1 0 0 1 {pts[0][0]:.3f} {pts[0][1]:.3f} Tm ({row[0][1]}) Tj"]
                for (_, name), (x0, y0), (x1, y1) in zip(row[1:], pts, pts[1:]):
                    parts.append(f"{x1 - x0:.3f} {y1 - y0:.3f} Td ({name}) Tj")
                ops.append("BT /F1 9 Tf " + " ".join(parts) + " ET")
        else:
            for (r, c), s in names.items():
                text(t0 + c * 2.5 * pt_s + 2, bases[r] + 0.6 * pt_mv, s)
    if scale_text:
        text(40, 60, f"{speed:g} mm/s {gain:g} mm/mV 0.05-150 Hz")
    if header_text:
        text(40, 560, header_text)

    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                                NameObject("/Subtype"): NameObject("/Type1"),
                                                NameObject("/BaseFont"): NameObject("/Helvetica")}))

    fonts = DictionaryObject({NameObject("/F1"): font})
    for name, (to_unicode, differences) in (extra_fonts or {}).items():
        spec = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        if to_unicode is not None:
            cmap = DecodedStreamObject()
            cmap.set_data(to_unicode)
            spec[NameObject("/ToUnicode")] = writer._add_object(cmap)
        if differences is not None:
            spec[NameObject("/Encoding")] = DictionaryObject({NameObject("/Differences"): ArrayObject(
                [NumberObject(d) if isinstance(d, int) else NameObject(d) for d in differences])})
        fonts[NameObject(name)] = writer._add_object(spec)

    def stream_of(content_ops):
        stream = DecodedStreamObject()
        stream.set_data("\n".join(content_ops).encode("latin-1"))
        return stream

    def add_page(content_ops):
        page = writer.add_blank_page(width=W, height=H)
        if rotate:
            page.rotate(rotate)
        resources = DictionaryObject({NameObject("/Font"): fonts})
        if in_form_cm is not None:
            form = stream_of(content_ops)
            form.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Form"),
                         NameObject("/BBox"): ArrayObject([NumberObject(0), NumberObject(0),
                                                           NumberObject(int(W)), NumberObject(int(H))]),
                         NameObject("/Resources"): resources})
            resources = DictionaryObject({NameObject("/XObject"): DictionaryObject(
                {NameObject("/Fm0"): writer._add_object(form)})})
            content_ops = [f"q {in_form_cm} cm /Fm0 Do Q"]
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = writer._add_object(stream_of(content_ops))

    grid_colours = ("0.883 0.6 0.557 RG", "1.0 0.752 0.797 RG")
    add_page(ops + list(extra_ops))
    if extra_page == "ecg":
        add_page(ops)
    elif extra_page == "grey_ecg":
        add_page([o.replace(grid_colours[0], "0.7 0.7 0.7 RG").replace(grid_colours[1], "0.7 0.7 0.7 RG")
                  for o in ops])
    elif extra_page == "grid_only":
        add_page([o for o in ops if any(c in o for c in grid_colours)])
    elif extra_page == "red_rules":
        add_page([f"q 0.9 0.3 0.3 RG 0.5 w 72 {100 + 20 * k} m 520 {100 + 20 * k} l S Q" for k in range(25)]
                 + ["BT /F1 12 Tf 1 0 0 1 72 650 Tm (Measurements) Tj ET"])
    elif extra_page == "text":
        add_page(["BT /F1 12 Tf 1 0 0 1 72 500 Tm (Interpretation page) Tj ET"])
    with open(path, "wb") as fh:
        writer.write(fh)
    return str(path)


def make_form_pdf(path, *, form_ops=300, calls=100, self_reference=False, body=None, subtype="/Form",
                  n_forms=1, compress=False, form_font_to_unicode=None):
    """A page that draws Form XObjects `calls` times each (or a form that draws itself).

    body: the forms' content (default: form_ops drawing operations); n_forms distinct forms;
    compress: Flate-compress their content (a small file that decodes to a lot)."""
    import zlib
    from pypdf.generic import EncodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    if body is None:
        body = "0 0 m 1 1 l S\n" * (form_ops // 3) + ("/Fm0 Do\n" if self_reference else "")
    xobjects = DictionaryObject()
    for k in range(n_forms):
        if compress:
            form = EncodedStreamObject()
            form._data = zlib.compress(body.encode("latin-1"), 9)
            form[NameObject("/Filter")] = NameObject("/FlateDecode")
        else:
            form = DecodedStreamObject()
            form.set_data(body.encode("latin-1"))
        form.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject(subtype),
                     NameObject("/BBox"): ArrayObject([NumberObject(0), NumberObject(0),
                                                       NumberObject(200), NumberObject(200)])})
        xobjects[NameObject(f"/Fm{k}")] = writer._add_object(form)
        if self_reference:
            form[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): xobjects})
        if form_font_to_unicode is not None:
            cmap = DecodedStreamObject()
            cmap.set_data(form_font_to_unicode)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica"),
                                     NameObject("/ToUnicode"): writer._add_object(cmap)})
            form[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(
                {NameObject("/Fx"): writer._add_object(font)})})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): xobjects})
    content = DecodedStreamObject()
    content.set_data("".join(f"/Fm{k} Do\n" for k in range(n_forms) for _ in range(calls)).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)
    with open(path, "wb") as fh:
        writer.write(fh)
    return str(path)


def encrypt(src, dst, user_password, owner_password=None):
    w = PdfWriter(clone_from=PdfReader(str(src)))
    w.encrypt(user_password=user_password, owner_password=owner_password, algorithm="RC4-128")
    with open(dst, "wb") as fh:
        w.write(fh)
    return str(dst)


@pytest.fixture(scope="module")
def signals():
    return synthetic_leads()


@pytest.fixture(scope="module")
def good_pdf(tmp_path_factory, signals):
    t, leads = signals
    return make_ecg_pdf(tmp_path_factory.mktemp("pdf") / "ecg.pdf", t, leads)


# ---------------------------------------------------------------------------
# Reading the leads (ported)
# ---------------------------------------------------------------------------
def test_round_trip_recovers_every_lead(good_pdf, signals):
    t, leads = signals
    ecg = read_ecg_pdf(good_pdf)
    assert ecg.qc["labels_verified"] is True
    assert (ecg.speed_mm_s, ecg.gain_mm_mv) == (25.0, 10.0)
    assert ecg.pt_per_mm == pytest.approx(PT_PER_MM, rel=1e-3)
    for r in range(3):
        for c, name in enumerate(STANDARD_ROWS[r]):
            tt, vv = ecg.leads[name]
            assert tt[0] == pytest.approx(c * 2.5, abs=1e-3) and tt[-1] < (c + 1) * 2.5
            truth = np.interp(tt, t, leads[name])
            assert np.abs(vv - truth).max() < 2e-3, name
    tt, vv = ecg.rhythm
    assert np.abs(vv - np.interp(tt, t, leads["II"])).max() < 2e-3
    for k in ("einthoven_rms_mv", "goldberger_rms_mv", "limb_rms_mv", "rhythm_vs_lead_ii_rms_mv"):
        assert ecg.qc[k] < 0.005, k


def test_rows_drawn_as_one_continuous_line(tmp_path, signals):
    t, leads = signals
    ecg = read_ecg_pdf(make_ecg_pdf(tmp_path / "ecg.pdf", t, leads, continuous_rows=True))
    for r in range(3):
        for c, name in enumerate(STANDARD_ROWS[r]):
            tt, vv = ecg.leads[name]
            assert tt[0] == pytest.approx(c * 2.5, abs=1e-3)
            assert np.abs(vv - np.interp(tt, t, leads[name])).max() < 2e-3, name   # no neighbour points


def test_unrotated_page_and_half_gain(tmp_path, signals):
    t, leads = signals
    ecg = read_ecg_pdf(make_ecg_pdf(tmp_path / "ecg.pdf", t, leads, rotate=0, gain=5.0))
    assert ecg.gain_mm_mv == 5.0
    tt, vv = ecg.leads["V3"]
    assert np.abs(vv - np.interp(tt, t, leads["V3"])).max() < 2e-3   # mV, not mm


def test_a_shared_join_point_at_a_column_boundary_is_fine(tmp_path, signals):
    t, leads = signals   # overrun=1: a lead's last point sits on the boundary
    ecg = read_ecg_pdf(make_ecg_pdf(tmp_path / "ecg.pdf", t, leads, overrun=1))
    tt, vv = ecg.leads["aVR"]
    assert np.abs(vv - np.interp(tt, t, leads["aVR"])).max() < 2e-3


def test_a_text_page_next_to_the_ecg_page_is_fine(tmp_path, signals):
    """Multi-page rule: the one page with an ECG is used, pages without one are ignored."""
    t, leads = signals
    ecg = read_ecg_pdf(make_ecg_pdf(tmp_path / "ecg.pdf", t, leads, extra_page="text"))
    assert ecg.qc["labels_verified"] is True


@pytest.mark.parametrize("kwargs,needle", [
    ({"label_swap": ("aVL", "aVF")}, "lead labels do not match"),
    ({"labels": False}, "lead labels could not be read"),       # V leads unverifiable
    ({"trace_swap": ("aVL", "aVF")}, "lead relations fail"),
    ({"trace_swap": ("I", "II")}, "lead relations fail"),
    ({"scale_text": False}, "printed scale not found"),
    ({"pulse_mm": 5.0}, "calibration pulses"),
    ({"stepped_pulse": True}, "more than two levels"),
    ({"overrun": 3}, "overlap in time"),
    ({"overrun": 2}, "overlap in time"),          # one sample past the shared join point
    ({"extra_page": "ecg"}, "exactly one ECG page"),   # multi-page rule: 2 ECG pages are ambiguous
])
def test_refuses_instead_of_guessing(tmp_path, signals, kwargs, needle):
    t, leads = signals
    with pytest.raises(EcgPdfError, match=needle):
        read_ecg_pdf(make_ecg_pdf(tmp_path / "ecg.pdf", t, leads, **kwargs))


def test_mhi_array_places_each_lead_in_its_window(good_pdf, signals):
    t, leads = signals
    arr = to_mhi_array(read_ecg_pdf(good_pdf))
    assert arr.shape == (2500, 12)
    n = np.arange(2500) / 250.0
    for r in range(3):
        for c, name in enumerate(STANDARD_ROWS[r]):
            j = LEAD_ORDER.index(name)
            inner = slice(c * 625 + 10, (c + 1) * 625 - 10)       # away from filter edges
            truth = np.interp(n[inner], t, leads[name])
            assert np.abs(arr[inner, j] - truth).max() < 0.02, name
            if name != "II":
                outside = np.ones(2500, bool)
                outside[c * 625:(c + 1) * 625] = False
                assert np.all(arr[outside, j] == 0.0), name
    j = LEAD_ORDER.index("II")
    assert np.abs(arr[10:-10, j] - np.interp(n[10:-10], t, leads["II"])).max() < 0.02


# ---------------------------------------------------------------------------
# Opening a PDF: limits and clean errors
# ---------------------------------------------------------------------------
def test_password_protected_pdf_is_a_clean_error(tmp_path, good_pdf):
    locked = encrypt(good_pdf, tmp_path / "locked.pdf", user_password="secret")
    for call in (check_pdf, read_ecg_pdf, extract_png_from_pdf):
        with pytest.raises(EcgPdfError, match="encrypted"):
            call(locked)


def test_owner_password_only_pdf_opens(tmp_path, good_pdf):
    """No user password: readers open it with the empty password (print/copy restrictions only)."""
    restricted = encrypt(good_pdf, tmp_path / "restricted.pdf", user_password="", owner_password="owner")
    assert PdfReader(restricted).is_encrypted
    assert extract_png_from_pdf(restricted) == extract_png_from_pdf(good_pdf)


@pytest.mark.parametrize("content,needle", [
    (b"", "not a PDF file"),
    (b"this is not a pdf", "not a PDF file"),
    (b"\x89PNG\r\n\x1a\n" + b"\0" * 64, "not a PDF file"),
    (b"%PDF-1.7\n%garbage only, no objects and no xref\n", "cannot open PDF"),
])
def test_not_a_pdf(tmp_path, content, needle):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(content)
    with pytest.raises(EcgPdfError, match=needle):
        check_pdf(str(bad))
    with pytest.raises(EcgPdfError, match=needle):
        extract_png_from_pdf(str(bad))


def test_truncated_pdf_is_a_clean_error(tmp_path, good_pdf):
    data = open(good_pdf, "rb").read()
    for n in (64, len(data) // 2, len(data) - 40):
        bad = tmp_path / f"truncated_{n}.pdf"
        bad.write_bytes(data[:n])
        with pytest.raises(EcgPdfError):
            extract_png_from_pdf(str(bad))


def test_a_pdf_without_pages_is_refused(tmp_path):
    with open(tmp_path / "empty.pdf", "wb") as fh:
        PdfWriter().write(fh)
    with pytest.raises(EcgPdfError, match="no pages"):
        check_pdf(str(tmp_path / "empty.pdf"))


def test_a_missing_file_is_a_clean_error(tmp_path):
    with pytest.raises(EcgPdfError, match="cannot read the file") as e:
        check_pdf(str(tmp_path / "MRN123_not_there.pdf"))
    assert "MRN123" not in str(e.value)


def test_size_and_page_limits_are_checked_before_the_pages_are_parsed(tmp_path, signals, good_pdf, monkeypatch):
    t, leads = signals
    two_pages = make_ecg_pdf(tmp_path / "two.pdf", t, leads, extra_page="text")
    parsed = []
    monkeypatch.setattr(ecg_pdf, "read_strokes", lambda *a, **k: parsed.append(1) or [])
    monkeypatch.setattr(ecg_pdf, "MAX_PDF_BYTES", os.path.getsize(good_pdf) - 1)
    with pytest.raises(EcgPdfError, match=r"bytes \(>"):
        extract_png_from_pdf(good_pdf)
    monkeypatch.setattr(ecg_pdf, "MAX_PDF_BYTES", 20_000_000)
    monkeypatch.setattr(ecg_pdf, "MAX_PDF_PAGES", 1)
    with pytest.raises(EcgPdfError, match=r"2 pages \(> 1\)"):
        extract_png_from_pdf(two_pages)
    assert not parsed


def test_drawing_operations_are_capped_for_the_whole_file(tmp_path, signals, good_pdf, monkeypatch):
    t, leads = signals
    reader = PdfReader(good_pdf)
    operations = pypdf.generic.ContentStream(reader.pages[0].get_contents(), reader).operations
    monkeypatch.setattr(ecg_pdf, "MAX_PDF_OPERATIONS", sum(1 + len(operands) for operands, _ in operations))
    read_ecg_pdf(good_pdf)                                   # exactly at the cap: fine
    two = make_ecg_pdf(tmp_path / "two.pdf", t, leads, extra_page="text")
    with pytest.raises(EcgPdfError, match="drawing operations"):
        read_ecg_pdf(two)                                    # the text page's operations count too


def test_a_form_drawn_many_times_counts_every_time(tmp_path, monkeypatch):
    monkeypatch.setattr(ecg_pdf, "MAX_PDF_OPERATIONS", 10_000)
    path = make_form_pdf(tmp_path / "forms.pdf", form_ops=300, calls=100)   # 60k operators + operands drawn
    with pytest.raises(EcgPdfError, match="drawing operations"):
        read_ecg_pdf(path)
    with pytest.raises(EcgPdfError, match="drawing operations"):
        extract_png_from_pdf(path)


def test_a_form_that_draws_itself_is_refused(tmp_path):
    path = make_form_pdf(tmp_path / "loop.pdf", form_ops=3, calls=1, self_reference=True)
    with pytest.raises(EcgPdfError, match="nested too deeply"):
        extract_png_from_pdf(path)


def _refused_quickly(path, needle, seconds=15):
    import time
    start = time.perf_counter()
    with pytest.raises(EcgPdfError, match=needle):
        extract_png_from_pdf(path)
    assert time.perf_counter() - start < seconds


def test_a_large_form_drawn_many_times_is_refused_quickly(tmp_path):
    """Each time a form is drawn its decoded content is charged, comments and whitespace included."""
    path = make_form_pdf(tmp_path / "comment.pdf", body="% " + "x" * 2_000_000 + "\n", calls=100, compress=True)
    assert os.path.getsize(path) < 20_000
    _refused_quickly(path, "bytes of drawing content")


def test_distinct_compressed_forms_are_capped_in_total(tmp_path):
    """Small file, many distinct forms that each decode to a lot (pypdf keeps them decoded)."""
    path = make_form_pdf(tmp_path / "forms.pdf", body=" " * 5_000_000, calls=1, n_forms=40, compress=True)
    assert os.path.getsize(path) < 1_000_000
    _refused_quickly(path, "bytes of drawing content")


def test_a_stream_that_decodes_past_the_cap_is_refused(tmp_path, monkeypatch):
    path = make_form_pdf(tmp_path / "bomb.pdf", body=" " * 3_000_000, calls=1, compress=True)
    monkeypatch.setattr(ecg_pdf, "MAX_PDF_CONTENT_BYTES", 1_000_000)   # also pypdf's cap per stream
    _refused_quickly(path, r"limit of the PDF reader|bytes of drawing content")


def test_operands_count_towards_the_operation_limit(tmp_path):
    """One operator with ~2M operands, drawn 10 times: refused during its first parse."""
    path = make_form_pdf(tmp_path / "operands.pdf", body="1 " * 1_999_000 + "n\n", calls=10, compress=True)
    _refused_quickly(path, r"limit of the PDF reader|drawing operations and operands", seconds=8)


def test_text_is_capped(tmp_path, signals):
    t, leads = signals
    long_text = make_ecg_pdf(tmp_path / "long.pdf", t, leads, extra_ops=(
        ["BT /F1 9 Tf 1 0 0 1 72 72 Tm"] + ["[(" + "x" * 1000 + ") 5] TJ"] * 2100 + ["ET"]))
    _refused_quickly(long_text, "bytes of text")
    many = make_ecg_pdf(tmp_path / "many.pdf", t, leads,
                        extra_ops=["BT /F1 9 Tf 1 0 0 1 72 72 Tm"] + ["(x) Tj"] * 100_001 + ["ET"])
    _refused_quickly(many, "text operations")


def _cmap(pairs):
    body = "".join(f"<{src:02X}> <{dst}>\n" for src, dst in pairs)
    return (f"/CIDInit /ProcSet findresource begin begincmap\n{len(pairs)} beginbfchar\n{body}"
            "endbfchar\nendcmap\n").encode("ascii")


def test_fonts_that_are_not_used_are_never_decoded(tmp_path, signals, good_pdf):
    """A huge ToUnicode map on a font no text uses costs nothing (pypdf's extract_text built them all)."""
    t, leads = signals
    huge = b"% " + b"<1> <2>\n" * 2_000_000
    path = make_ecg_pdf(tmp_path / "fonts.pdf", t, leads, extra_fonts={"/Fz": (huge, None)})
    import time
    start = time.perf_counter()
    assert extract_png_from_pdf(path) == extract_png_from_pdf(good_pdf)
    assert time.perf_counter() - start < 15


def test_a_used_font_with_a_huge_map_is_refused(tmp_path, signals):
    t, leads = signals
    huge = b"1 beginbfchar\n" + b"<01> <0041>\n" * 100_000 + b"endbfchar\n"
    path = make_ecg_pdf(tmp_path / "used.pdf", t, leads, extra_fonts={"/Fz": (huge, None)},
                        extra_ops=["BT /Fz 9 Tf 1 0 0 1 72 72 Tm (a) Tj ET"])
    _refused_quickly(path, "ToUnicode map is too large")


def test_a_font_is_decoded_once_per_file(tmp_path, monkeypatch):
    """A form with its own font, drawn 500 times: the font map is parsed once."""
    parsed = []
    real = ecg_pdf._parse_to_unicode
    monkeypatch.setattr(ecg_pdf, "_parse_to_unicode", lambda data: parsed.append(1) or real(data))
    cmap = _cmap([(k, f"{0x41 + k % 26:04X}") for k in range(256)])
    path = make_form_pdf(tmp_path / "fontform.pdf", body="BT /Fx 9 Tf 1 0 0 1 9 9 Tm (abc) Tj ET\n",
                         calls=500, form_font_to_unicode=cmap)
    _refused_quickly(path, "exactly one ECG page, found 0")
    assert len(parsed) == 1


@pytest.mark.parametrize("expanding", ["to_unicode", "differences"])
def test_one_code_never_expands_into_long_text(tmp_path, signals, expanding):
    t, leads = signals
    if expanding == "to_unicode":
        font = (_cmap([(k, "0041" * 256) for k in range(256)]), None)
    else:
        font = (None, [0] + ["/" + "x" * 4000] * 256)
    path = make_ecg_pdf(tmp_path / "amp.pdf", t, leads, extra_fonts={"/Fz": font},
                        extra_ops=["BT /Fz 9 Tf 1 0 0 1 72 72 Tm"] + ["(" + "a" * 40 + ") Tj"] * 4900 + ["ET"])
    import time
    start = time.perf_counter()
    extract_png_from_pdf(path)                                    # read, and fast
    assert time.perf_counter() - start < 15
    texts = ecg_pdf._walk_page(PdfReader(path), PdfReader(path).pages[0])[1]
    assert max(len(text) for text, *_ in texts) <= 4 * 40 * 4900


def test_tj_array_elements_count_towards_the_operation_limit(tmp_path, signals):
    t, leads = signals
    dense = "BT /F1 9 Tf [" + " 1" * 99_990 + "] TJ ET"
    path = make_ecg_pdf(tmp_path / "dense.pdf", t, leads, extra_ops=[dense] * 9)
    reader = PdfReader(path)
    budget = ecg_pdf._Budget()
    with ecg_pdf._pypdf_limits():
        ecg_pdf._walk_page(reader, reader.pages[0], budget)
    assert MAX_PDF_OPERATIONS - budget.left["operations"] > 9 * 99_990


def make_text_pdf(path, content, fonts):
    """One page with this content; fonts: {name: ToUnicode bytes or None}."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=400)
    font_dict = DictionaryObject()
    for name, to_unicode in fonts.items():
        spec = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        if to_unicode is not None:
            cmap = DecodedStreamObject()
            cmap.set_data(to_unicode)
            spec[NameObject("/ToUnicode")] = writer._add_object(cmap)
        font_dict[NameObject(name)] = writer._add_object(spec)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): font_dict})
    stream = DecodedStreamObject()
    stream.set_data(content.encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with open(path, "wb") as fh:
        writer.write(fh)
    return str(path)


def _texts(path):
    reader = PdfReader(path)
    return [text for text, *_ in ecg_pdf._walk_page(reader, reader.pages[0])[1]]


def test_q_and_Q_save_and_restore_the_text_font(tmp_path):
    swap = _cmap([(0x31, "0032"), (0x32, "0031")] + [(c, f"{c:04X}") for c in (0x56,)])   # '1' <-> '2'
    path = make_text_pdf(tmp_path / "q.pdf", "BT /F1 9 Tf ET q BT /F9 9 Tf 1 0 0 1 9 9 Tm (x) Tj ET Q "
                                             "BT 1 0 0 1 50 50 Tm (V1) Tj ET", {"/F1": None, "/F9": swap})
    assert _texts(path)[-1] == "V1"                       # shown in /F1 again after Q


def test_overlapping_ranges_count_towards_the_font_map_limit():
    cmap = b"3 beginbfrange\n" + b"<0000> <7FFF> <0041>\n" * 3 + b"endbfrange\n"
    with pytest.raises(EcgPdfError, match="ToUnicode map is too large"):
        ecg_pdf._parse_to_unicode(cmap)


def test_one_long_piece_of_text_grows_linearly(tmp_path, signals):
    t, leads = signals
    four = _cmap([(0x41, "0041004100410041")])
    path = make_ecg_pdf(tmp_path / "long_piece.pdf", t, leads, extra_fonts={"/Fz": (four, None)},
                        extra_ops=["BT /Fz 9 Tf 1 0 0 1 72 72 Tm"] + ["(" + "A" * 20 + ") Tj"] * 99_000 + ["ET"])
    import time
    start = time.perf_counter()
    extract_png_from_pdf(path)
    assert time.perf_counter() - start < 15


def test_nested_operands_are_counted_at_every_depth():
    # outer list 1 + [1, 2, [3]] (1 + 2 + [3]: 2) + {"/a": [4, 5]} (1 + [4, 5]: 3)
    assert ecg_pdf._operand_size([[1, 2, [3]], {"/a": [4, 5]}]) == 1 + 5 + 4


def test_a_scale_split_into_runs_is_read(tmp_path, signals):
    t, leads = signals
    path = make_ecg_pdf(tmp_path / "split.pdf", t, leads, scale_text=False, rotate=0, extra_ops=[
        "BT /F1 9 Tf 1 0 0 1 40 60 Tm (25) Tj 12 0 Td (mm/s) Tj 30 0 Td (10 mm/mV) Tj ET"])
    ecg = read_ecg_pdf(path)
    assert (ecg.speed_mm_s, ecg.gain_mm_mv) == (25.0, 10.0)


def test_the_scale_patterns_are_linear_on_long_digit_runs():
    import time
    for text in ("1" * 200_000, "1 " * 100_000, "25" + " " * 100_000 + "mm/s"):
        start = time.perf_counter()
        assert not ecg_pdf._SPEED.findall(text) and not ecg_pdf._GAIN.findall(text)
        assert time.perf_counter() - start < 2
    assert ecg_pdf._SPEED.findall("25 mm/s 0,5 mm/s 12.5mm / s") == ["25", "0,5", "12.5"]
    assert ecg_pdf._GAIN.findall("10 mm/mV, 5mm/mV") == ["10", "5"]


def test_malformed_text_operators_are_ignored(tmp_path, signals):
    t, leads = signals
    path = make_ecg_pdf(tmp_path / "malformed.pdf", t, leads,
                        extra_ops=["BT 1 0 0 1 Tm ET", "BT 5 Td ET", "BT Tf ET", "/x TL", "BT /a /b /c /d /e /f Tm ET"])
    read_ecg_pdf(path)


def test_xobjects_other_than_forms_and_images_are_refused(tmp_path):
    path = make_form_pdf(tmp_path / "ps.pdf", body=" " * 1000, calls=3, subtype="/PS")
    _refused_quickly(path, "unsupported XObject type")


@pytest.mark.parametrize("extra_page", ["grey_ecg", "grid_only", "ecg"])
def test_a_second_page_that_looks_like_an_ecg_is_ambiguous(tmp_path, signals, extra_page):
    """Multi-page rule: a page with a grid or a trace is an ECG candidate; never choose among two."""
    t, leads = signals
    with pytest.raises(EcgPdfError, match="exactly one ECG page, found 2 of 2"):
        read_ecg_pdf(make_ecg_pdf(tmp_path / "two.pdf", t, leads, extra_page=extra_page))


def test_red_table_rules_on_a_report_page_are_not_a_grid(tmp_path, signals):
    t, leads = signals
    read_ecg_pdf(make_ecg_pdf(tmp_path / "rules.pdf", t, leads, extra_page="red_rules"))


def test_labels_of_a_row_in_one_text_object_are_read(tmp_path, signals, good_pdf):
    t, leads = signals
    for rotate in (0, 90):
        one_run = make_ecg_pdf(tmp_path / f"row_{rotate}.pdf", t, leads, label_runs="row", rotate=rotate)
        separate = make_ecg_pdf(tmp_path / f"sep_{rotate}.pdf", t, leads, rotate=rotate)
        assert extract_png_from_pdf(one_run) == extract_png_from_pdf(separate)


@pytest.mark.parametrize("cm", ["1 0 0 1 0 0", "1 0 0 1 0 -40", "0.5 0 0 0.5 0 0"])
def test_an_ecg_drawn_inside_a_form_is_read(tmp_path, signals, cm):
    t, leads = signals
    ecg = read_ecg_pdf(make_ecg_pdf(tmp_path / "form.pdf", t, leads, in_form_cm=cm, rotate=0))
    tt, vv = ecg.leads["V3"]
    assert np.abs(vv - np.interp(tt, t, leads["V3"])).max() < 2e-3


def test_scale_numbers_from_the_page_text_are_never_quoted(tmp_path, signals):
    t, leads = signals
    path = make_ecg_pdf(tmp_path / "gain.pdf", t, leads, scale_text=False, header_text="25 mm/s  MRN 4711 mm/mV")
    with pytest.raises(EcgPdfError) as e:
        read_ecg_pdf(path)
    assert "4711" not in str(e.value)


def test_a_path_with_a_nul_byte_is_a_clean_error():
    with pytest.raises(EcgPdfError, match="cannot read the file"):
        check_pdf("bad\0name.pdf")


def test_rotation_must_be_a_multiple_of_90(tmp_path, good_pdf):
    from pypdf.generic import FloatObject
    w = PdfWriter(clone_from=PdfReader(good_pdf))
    w.pages[0][NameObject("/Rotate")] = FloatObject(90.5)
    with open(tmp_path / "rot.pdf", "wb") as fh:
        w.write(fh)
    with pytest.raises(EcgPdfError, match="not a multiple of 90"):
        read_ecg_pdf(str(tmp_path / "rot.pdf"))


def test_a_fifo_is_refused_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "pipe.pdf")
    with pytest.raises(EcgPdfError, match="not a regular file"):
        check_pdf(str(tmp_path / "pipe.pdf"))


def test_a_memory_error_is_an_error_of_its_row(good_pdf, monkeypatch):
    def out_of_memory(path):
        raise MemoryError()
    monkeypatch.setattr(ecg_pdf, "read_ecg_pdf", out_of_memory)
    with pytest.raises(EcgPdfError, match=r"\(MemoryError\)"):
        extract_png_from_pdf(good_pdf)


@pytest.mark.parametrize("algorithm", ["AES-128", "AES-256"])
def test_aes_owner_password_only_pdf_opens(tmp_path, good_pdf, algorithm):
    pytest.importorskip("cryptography")          # pypdf[crypto] in requirements.txt
    w = PdfWriter(clone_from=PdfReader(good_pdf))
    w.encrypt(user_password="", owner_password="owner", algorithm=algorithm)
    with open(tmp_path / "aes.pdf", "wb") as fh:
        w.write(fh)
    check_pdf(str(tmp_path / "aes.pdf"))
    assert extract_png_from_pdf(str(tmp_path / "aes.pdf")) == extract_png_from_pdf(good_pdf)


def test_scale_errors_do_not_quote_numbers_from_the_page_text(tmp_path, signals):
    t, leads = signals
    path = make_ecg_pdf(tmp_path / "mrn.pdf", t, leads, header_text="DOE JOHN  MRN 4711 mm/s")
    with pytest.raises(EcgPdfError, match="printed scale not found or ambiguous") as e:
        read_ecg_pdf(path)
    assert "4711" not in str(e.value)


def test_errors_never_quote_the_path_or_the_page(tmp_path, signals, monkeypatch):
    t, leads = signals
    folder = tmp_path / "Doe_John_MRN123"
    folder.mkdir()
    bad = make_ecg_pdf(folder / "Doe_John_MRN123.pdf", t, leads, label_swap=("aVL", "aVF"),
                       header_text="Name: DOE JOHN  MRN 123")
    with pytest.raises(EcgPdfError) as e:
        extract_png_from_pdf(bad)
    good = make_ecg_pdf(folder / "ok.pdf", t, leads, header_text="Name: DOE JOHN  MRN 123")

    def leak(*args, **kwargs):
        raise RuntimeError(f"failed on {good}: DOE JOHN")
    monkeypatch.setattr(ecg_pdf, "render_training_page", leak)
    with pytest.raises(EcgPdfError, match=r"\(RuntimeError\)") as e2:
        extract_png_from_pdf(good)
    for message in (str(e.value), str(e2.value)):
        assert "MRN" not in message and "DOE" not in message and str(tmp_path) not in message


# ---------------------------------------------------------------------------
# The extracted page and the model input
# ---------------------------------------------------------------------------
def test_pdf_model_input_equals_the_png_of_its_page(tmp_path, good_pdf):
    png = extract_png_from_pdf(good_pdf)
    assert png == extract_png_from_pdf(good_pdf)                 # deterministic
    with open_png_bytes(png) as image:
        assert image.size == PAGE_SIZE == (1250, 625) and image.format == "PNG"
    reference = tmp_path / "reference.png"
    reference.write_bytes(png)                                   # the independently saved page
    from_pdf = load_ecg_page(good_pdf, PREPROCESSING)
    assert torch.equal(from_pdf, load_ecg_image(str(reference), PREPROCESSING))
    # and the PNG round trip is lossless: the same as preprocessing the rendered page directly
    page = render_training_page(to_mhi_array(read_ecg_pdf(good_pdf)) / MV_PER_COUNT)
    assert torch.equal(from_pdf, preprocess_ecg_image(page, PREPROCESSING))
    rgb = dict(PREPROCESSING, color_mode='rgb')
    assert torch.equal(load_ecg_page(good_pdf, rgb), load_ecg_image(str(reference), rgb))


def test_the_extracted_page_size_is_checked(good_pdf, monkeypatch):
    small = Image.new("RGBA", (100, 50), "white")
    import io
    buffer = io.BytesIO()
    small.save(buffer, format="PNG")
    monkeypatch.setattr(ecg_pdf, "extract_png_from_pdf", lambda path: buffer.getvalue())
    with pytest.raises(ValueError, match="expected 1250x625"):
        load_ecg_page(good_pdf, PREPROCESSING)


def test_png_and_jpeg_pages_load_as_before(tmp_path):
    page = Image.new("RGBA", (125, 62), "white")
    page.save(tmp_path / "page.png")
    with Image.open(tmp_path / "page.png") as image:
        assert torch.equal(load_ecg_page(str(tmp_path / "page.png"), PREPROCESSING),
                           preprocess_ecg_image(image, PREPROCESSING))
    page.convert("RGB").save(tmp_path / "page.jpg")
    assert torch.equal(load_ecg_page(str(tmp_path / "page.jpg"), PREPROCESSING),
                       load_ecg_image(str(tmp_path / "page.jpg"), PREPROCESSING))


def test_check_ecg_page_does_not_extract_the_pdf(good_pdf, monkeypatch):
    monkeypatch.setattr(ecg_pdf, "read_ecg_pdf", lambda path: pytest.fail("parsed in the preflight"))
    check_ecg_page(good_pdf)


# ---------------------------------------------------------------------------
# Routing and the DataLoader
# ---------------------------------------------------------------------------
def test_pdf_is_an_image_input_and_a_pdf_source():
    for name in ('a.pdf', 'B.PDF', 'dir.v2/report.Pdf'):
        assert input_type_of(name) == IMAGE and source_format_of(name) == 'pdf'
    assert [source_format_of(n) for n in ('a.xml', 'a.npy', 'a.png', 'a.jpg', 'a.JPEG', 'a.tif', 'pdf')] == \
        ['xml', 'npy', 'png', 'jpeg', 'jpeg', None, None]
    for name in ('a.xml', 'a.npy', 'a.pdf.npy'):
        assert input_type_of(name) == SIGNAL


def _frame(paths):
    return pd.DataFrame({'diagnosis': [1.0] * len(paths), 'ecg_path': paths,
                         'file_name': [os.path.basename(p) for p in paths]})


def test_pdfs_are_extracted_lazily_one_row_at_a_time(tmp_path, good_pdf, signals, monkeypatch):
    t, leads = signals
    bad = make_ecg_pdf(tmp_path / "bad.pdf", t, leads, label_swap=("aVL", "aVF"))
    page = Image.new("RGBA", (125, 62), "white")
    page.save(tmp_path / "page.png")
    calls = []
    real = ecg_pdf.extract_png_from_pdf

    def counting(path):
        calls.append(os.path.basename(path))
        return real(path)
    monkeypatch.setattr(ecg_pdf, "extract_png_from_pdf", counting)
    PAGE_ERRORS.clear()
    rows = [good_pdf, bad, str(tmp_path / "page.png"), bad, bad, good_pdf]
    loader = create_image_dataloader(_frame(rows), PREPROCESSING, batch_size=3, num_workers=4)
    assert loader.num_workers == 0                      # PDFs: one at a time, in the main process
    assert calls == []                                  # nothing extracted before the DataLoader runs
    batches = []
    for batch in loader:
        batches.append(batch)
        # by the time a batch arrives, only its own rows (and the earlier ones) were extracted
        assert len(calls) == {1: 2, 2: 3}[len(batches)]   # batch 2: the bad PDF is not read again
    assert calls == ['ecg.pdf', 'bad.pdf', 'ecg.pdf']
    assert [b['file_name'] for b in batches] == [['ecg.pdf', 'page.png'], ['ecg.pdf']]
    assert batches[0]['ecg_signal'].shape == (2, 3, 32, 32)
    assert PAGE_ERRORS == {bad: "EcgPdfError: lead labels do not match the standard 3x4 + lead-II rhythm layout"}
    PAGE_ERRORS.clear()

    only_bad = create_image_dataloader(_frame([bad, bad]), PREPROCESSING, batch_size=2)
    assert list(only_bad) == [None]                     # a batch with no readable row
    PAGE_ERRORS.clear()


def test_png_only_dataloaders_keep_their_workers_and_batches(tmp_path):
    for k in range(3):
        Image.new("RGBA", (125, 62), (k, 255, 255, 255)).save(tmp_path / f"p{k}.png")
    paths = [str(tmp_path / f"p{k}.png") for k in range(3)]
    loader = create_image_dataloader(_frame(paths), PREPROCESSING, batch_size=2, num_workers=2)
    assert loader.num_workers == 2
    batch = next(iter(create_image_dataloader(_frame(paths), PREPROCESSING, batch_size=3)))
    want = torch.stack([load_ecg_image(p, PREPROCESSING) for p in paths])
    assert torch.equal(batch['ecg_signal'], want) and batch['file_name'] == ['p0.png', 'p1.png', 'p2.png']
    assert batch['diagnosis'].dtype == torch.float64


# ---------------------------------------------------------------------------
# main(): mixed CSVs
# ---------------------------------------------------------------------------
def _files(root):
    return {os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs}


@pytest.fixture
def pdf_world(world, signals, good_pdf):
    """The PNG world plus a readable PDF, the PNG of its page, a PDF whose ECG cannot be read
    without guessing (found while the model reads it) and a locked PDF (found before)."""
    t, leads = signals
    pages = world / 'pngs'
    shutil.copy(good_pdf, pages / 'report.pdf')
    (pages / 'report_page.png').write_bytes(extract_png_from_pdf(good_pdf))
    make_ecg_pdf(pages / 'mislabelled.pdf', t, leads, label_swap=("aVL", "aVF"))
    encrypt(good_pdf, pages / 'locked.pdf', user_password="secret")
    Image.open(pages / 'ecg_2.png').convert('RGB').save(pages / 'photo.jpg')
    pipeline.UNREADABLE_IMAGES.clear()
    PAGE_ERRORS.clear()
    return world


def test_pdf_rows_run_with_png_and_jpeg_rows_and_nothing_is_written(pdf_world, caplog):
    files = ['ecg_0.png', 'report.pdf', 'photo.jpg', 'mislabelled.pdf', 'locked.pdf', 'ecg_1.png']
    before = _files(pdf_world)
    with caplog.at_level('INFO'):
        out = run(pdf_world, task_csv(files), 'pdfmix')     # no flag; returns (no error exit)
    good = ['ecg_0.png', 'report.pdf', 'photo.jpg', 'ecg_1.png']
    for task in ('shd', 'lvef_40', 'lvef_50', 'afib_5y', 'ecg_machine_diagnosis'):
        probs = pd.read_csv(_one(out, f'_{task}_probabilities.csv'))
        assert probs['file_name'].tolist() == good, task
    bad = pd.read_csv(_one(out, 'unreadable_images_'))
    reasons = dict(zip((os.path.basename(p) for p in bad['ecg_path']), bad['error']))
    assert reasons == {'locked.pdf': 'EcgPdfError: PDF is encrypted and could not be opened',
                       'mislabelled.pdf': 'EcgPdfError: lead labels do not match the standard 3x4 '
                                          '+ lead-II rhythm layout'}
    # nothing written but the outputs: no PNG, Base64 or embedding file for a PDF
    new = _files(pdf_world) - before - {'key.json', 'pdfmix.csv'}     # the test's own inputs
    assert new and all(f.startswith('out_pdfmix' + os.sep) for f in new), sorted(new)
    assert all(f.endswith(('.csv', '.json')) for f in new), sorted(new)
    assert not os.listdir(pdf_world / 'prep_pdfmix')
    # the log counts the skipped PDFs, and never names a file
    assert "PDF row(s) without a readable ECG skipped" in caplog.text
    assert 'mislabelled' not in caplog.text and 'locked' not in caplog.text


def test_a_pdf_predicts_exactly_like_the_png_of_its_page(pdf_world):
    # Same batch position in two runs: on CPU a row's output also depends on its position in the
    # batch (the same PNG twice in one batch differs by up to ~2e-7), which is not what is tested here.
    files = ['ecg_0.png', 'ecg_1.png', 'ecg_2.png']
    out_pdf = run(pdf_world, task_csv(files + ['report.pdf']), 'frompdf')
    out_png = run(pdf_world, task_csv(files + ['report_page.png']), 'frompng')
    for task in ('shd', 'lvef_40', 'lvef_50', 'afib_5y', 'ecg_machine_diagnosis'):
        a = pd.read_csv(_one(out_pdf, f'_{task}_probabilities.csv')).drop(columns='file_name')
        b = pd.read_csv(_one(out_png, f'_{task}_probabilities.csv')).drop(columns='file_name')
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=task)


class FakeSignalModel:
    """Stands in for a signal model (EfficientNetV2): a constant probability."""

    def __init__(self):
        self.model = self

    def eval(self):
        return self

    def __call__(self, ecg):
        return torch.full((ecg.shape[0], 1), 0.25)


def test_one_csv_mixes_xml_npy_png_jpeg_and_pdf(pdf_world, monkeypatch):
    pages = pdf_world / 'pngs'
    shutil.copy(os.path.join(REPO, 'ecg_signals', 'dummy_ecg.xml'), pages / 'signal_a.xml')
    shutil.copy(os.path.join(REPO, 'ecg_signals', 'dummy_ecg.npy'), pages / 'signal_b.npy')
    factory = analysis_pipeline.HeartWiseModelFactory

    class Factory:
        @staticmethod
        def create_model(model_config):
            if 'efficientnet' in model_config['model_name'].lower():
                return FakeSignalModel()
            return factory.create_model(model_config)
    monkeypatch.setattr(analysis_pipeline, 'HeartWiseModelFactory', Factory)
    files = ['signal_a.xml', 'ecg_0.png', 'report.pdf', 'signal_b.npy', 'photo.jpg']
    df = pd.DataFrame({'afib_5y': [0, 1, 0, 1, 1], 'afib_ecg_file_name': files})
    out = run(pdf_world, df, 'allformats',
              models={'signal_models': ['efficientnet'], 'image_models': ['deepecg_siglip2']})
    image = pd.read_csv(_one(out, 'deepecg_siglip2_afib_5y_', '_probabilities.csv'))
    assert image['file_name'].tolist() == ['ecg_0.png', 'report.pdf', 'photo.jpg']
    signal = [f for f in os.listdir(out) if f.endswith('_afib_5y_probabilities.csv') and 'siglip2' not in f]
    assert len(signal) == 1
    assert sorted(pd.read_csv(os.path.join(out, signal[0]))['file_name']) == ['signal_a.base64', 'signal_b.base64']
    # only the signals were preprocessed
    assert sorted(os.listdir(pdf_world / 'prep_allformats')) == ['signal_a.base64', 'signal_b.base64']


def test_a_task_whose_pdfs_all_fail_writes_no_outputs(pdf_world):
    df = pd.DataFrame({'afib_5y': [0, 1], 'afib_ecg_file_name': ['mislabelled.pdf', 'mislabelled.pdf']})
    out = run(pdf_world, df, 'allbad')
    assert not any(f.endswith(('_probabilities.csv', '_afib_5y.json', '_afib_5y.csv')) for f in os.listdir(out))
    bad = pd.read_csv(_one(out, 'unreadable_images_'))
    assert [os.path.basename(p) for p in bad['ecg_path']] == ['mislabelled.pdf']


def test_a_bad_pdf_is_read_once_per_run(pdf_world, monkeypatch):
    calls = []
    real = ecg_pdf.extract_png_from_pdf
    monkeypatch.setattr(ecg_pdf, "extract_png_from_pdf", lambda path: calls.append(os.path.basename(path)) or real(path))
    run(pdf_world, task_csv(['mislabelled.pdf', 'report.pdf']), 'once')
    assert calls.count('mislabelled.pdf') == 1 and calls.count('report.pdf') == 5   # 5 tasks


def test_a_failed_pdf_that_changes_on_disk_is_read_again(tmp_path, good_pdf, signals):
    t, leads = signals
    path = tmp_path / "x.pdf"
    make_ecg_pdf(path, t, leads, label_swap=("aVL", "aVF"))
    PAGE_ERRORS.clear()
    assert list(create_image_dataloader(_frame([str(path)]), PREPROCESSING)) == [None]
    assert list(create_image_dataloader(_frame([str(path)]), PREPROCESSING)) == [None]   # unchanged: skipped
    shutil.copy(good_pdf, path)                                  # fixed on disk
    os.utime(path, ns=(1, 1))                                    # a different mtime even on coarse clocks
    [batch] = list(create_image_dataloader(_frame([str(path)]), PREPROCESSING))
    assert batch['file_name'] == ['x.pdf']
    assert str(path) in PAGE_ERRORS                              # still listed: it failed for an earlier task
    PAGE_ERRORS.clear()


def test_main_leaves_no_page_errors_behind(pdf_world):
    run(pdf_world, task_csv(['mislabelled.pdf', 'ecg_0.png']), 'leaves')
    assert PAGE_ERRORS == {} and image_dataset._FAILED_AS == {}


def test_a_pdf_that_fails_again_keeps_its_first_reason(tmp_path, signals):
    t, leads = signals
    path = make_ecg_pdf(tmp_path / "x.pdf", t, leads, label_swap=("aVL", "aVF"))
    image_dataset.clear_page_errors()
    list(create_image_dataloader(_frame([path]), PREPROCESSING))
    os.remove(path)                                              # changed (gone): read again, fails again
    list(create_image_dataloader(_frame([path]), PREPROCESSING))
    assert PAGE_ERRORS[path].startswith("EcgPdfError: lead labels do not match")
    image_dataset.clear_page_errors()


def test_page_errors_are_cleared_between_runs(pdf_world):
    run(pdf_world, task_csv(['mislabelled.pdf', 'ecg_0.png']), 'first')
    out = run(pdf_world, task_csv(['ecg_0.png']), 'second')
    assert not any(f.startswith('unreadable_images_') for f in os.listdir(out))


def test_page_errors_module_state_is_the_one_main_reads():
    assert pipeline.PAGE_ERRORS is image_dataset.PAGE_ERRORS


def test_the_renderer_draws_the_verified_pixels():
    """The page of a fixed signal, as drawn when the renderer was verified pixel-identical to the
    training PNGs (DeepECG_Docker #82: matplotlib 3.10.9, Pillow 12.3.0, numpy 2.2.6; the same
    pixels with Pillow 13.0.0). A different hash means the renderer or a pinned package changed
    its drawing: re-verify against the training PNGs."""
    t = np.arange(2500) / 250.0
    leads = []
    for k in range(12):
        beat = sum(np.exp(-((t - c) ** 2) / (2 * 0.012 ** 2)) for c in np.arange(0.4, 10, 0.85))
        sign = 1 if k % 3 else -1
        leads.append(np.round(sign * (150 + 20 * k) * beat + 40 * np.sin(2 * np.pi * (0.3 + 0.05 * k) * t)))
    page = render_training_page(np.stack(leads, axis=1))
    assert hashlib.sha256(np.asarray(page).tobytes()).hexdigest() == RENDERER_GOLDEN_PIXELS_SHA256

