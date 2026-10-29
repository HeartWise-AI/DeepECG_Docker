import os
import struct

import numpy as np
import torch
from PIL import Image

# ECG pages are ~1-3 Mpx: anything far above is refused before Pillow allocates it.
MAX_IMAGE_PIXELS = 50_000_000
MAX_PNG_PIXELS = MAX_IMAGE_PIXELS
MAX_PNG_CHUNKS = 10_000        # chunks allowed before the image data (real pages: a handful)
MAX_JPEG_SEGMENTS = 10_000     # segments allowed before the image data (real files: a handful)
MAX_JPEG_SCANS = 100           # progressive scans (real pages: ~10); each one costs a full decode pass
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# JPEG start-of-frame markers (baseline, extended, progressive, lossless; Huffman / arithmetic)
JPEG_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _check_size(kind: str, width: int, height: int) -> None:
    if width == 0 or height == 0:
        raise ValueError(f"{kind} without an image size")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"{kind} declares {width}x{height} pixels (> {MAX_IMAGE_PIXELS})")


def open_png(path: str) -> Image.Image:
    """
    Open a PNG with Pillow's PNG decoder only, after checking its header.

    Before Pillow sees the file: it must be a PNG (signature + one IHDR), declare at most
    MAX_IMAGE_PIXELS pixels, not be animated (no acTL chunk before the image data) and reach its
    image data within MAX_PNG_CHUNKS chunks.
    Pillow 12 allocates an APNG's full declared canvas inside Image.open, before its own
    size check, so a tiny crafted APNG could otherwise exhaust memory.

    Raises:
        ValueError: if the file is not an acceptable PNG.
    """
    with open(path, "rb") as f:
        if f.read(8) != PNG_SIGNATURE:
            raise ValueError("not a PNG file")
        header = f.read(8)
        if len(header) != 8 or header[4:] != b"IHDR":
            raise ValueError("PNG without an IHDR header")
        width, height = struct.unpack(">II", f.read(8))
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(f"PNG declares {width}x{height} pixels (> {MAX_IMAGE_PIXELS})")
        f.seek(8 + 8 + struct.unpack(">I", header[:4])[0] + 4)   # past IHDR data + CRC
        for _ in range(MAX_PNG_CHUNKS):                            # chunks before the image data
            chunk = f.read(8)
            if len(chunk) < 8:
                break                                              # truncated: Pillow refuses it
            length, kind = struct.unpack(">I", chunk[:4])[0], chunk[4:]
            if kind == b"acTL":
                raise ValueError("animated PNG (APNG) is not an ECG page")
            if kind == b"IHDR":                                    # Pillow would take the later size
                raise ValueError("PNG with more than one IHDR header")
            if kind in (b"IDAT", b"IEND"):
                break
            f.seek(length + 4, 1)
        else:
            raise ValueError(f"PNG with more than {MAX_PNG_CHUNKS} chunks before its image data")
    return Image.open(path, formats=["PNG"])


def open_jpeg(path: str) -> Image.Image:
    """
    Open a JPEG with Pillow's JPEG decoder only, after checking its header.

    Before Pillow sees the file: it must be a JPEG (start-of-image marker), have exactly one
    frame header declaring at most MAX_IMAGE_PIXELS pixels before its image data, at most
    MAX_JPEG_SCANS scans (a crafted progressive file can cost minutes of CPU per decode) and an
    end-of-image marker (a truncated file would decode silently as grey). Pillow must then read it
    as one plain JPEG frame of the same size: a multi-picture file (MPO), whose multi-frame loader
    has known Pillow 12 bugs, is refused, while a phone photo with a gain-map MPF segment that
    Pillow reads as a single JPEG is kept.

    Raises:
        ValueError: if the file is not an acceptable JPEG.
    """
    size = None
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError("not a JPEG file")
        for _ in range(MAX_JPEG_SEGMENTS):
            if f.read(1) != b"\xff":
                raise ValueError("malformed or truncated JPEG header")
            marker = f.read(1)
            while marker == b"\xff":                               # fill bytes
                marker = f.read(1)
            if not marker:
                raise ValueError("truncated JPEG header")
            m = marker[0]
            if m == 0x01 or 0xD0 <= m <= 0xD7:                      # markers without a length
                continue
            if m in (0xD8, 0xD9):
                raise ValueError("JPEG header without image data")
            raw = f.read(2)
            if len(raw) < 2 or struct.unpack(">H", raw)[0] < 2:
                raise ValueError("malformed or truncated JPEG header")
            data = f.read(struct.unpack(">H", raw)[0] - 2)
            if m in JPEG_SOF_MARKERS:
                if size is not None:
                    raise ValueError("JPEG with more than one frame header")
                if len(data) < 6:
                    raise ValueError("malformed JPEG frame header")
                height, width = struct.unpack(">HH", data[1:5])
                _check_size("JPEG", width, height)
                size = (width, height)
            if m == 0xDA:                                           # start of the image data
                if size is None:
                    raise ValueError("JPEG image data before its frame header")
                break
        else:
            raise ValueError(f"JPEG with more than {MAX_JPEG_SEGMENTS} segments before its image data")
        # Entropy-coded data never contains FF DA / FF D9 (an FF data byte is followed by 00), so
        # these byte pairs are the remaining start-of-scan and end-of-image markers.
        scans, end_seen, tail = 1, False, b""
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            window = tail + block
            scans += window.count(b"\xff\xda") - tail.count(b"\xff\xda")
            end_seen = end_seen or b"\xff\xd9" in window
            if scans > MAX_JPEG_SCANS:
                raise ValueError(f"JPEG with more than {MAX_JPEG_SCANS} scans")
            tail = block[-1:]
        if not end_seen:
            raise ValueError("truncated JPEG (no end-of-image marker)")
    image = Image.open(path, formats=["JPEG"])
    if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1 or image.size != size:
        image.close()
        raise ValueError("JPEG that Pillow does not read as one plain frame of its header's size (e.g. MPO)")
    return image


def open_ecg_image(path: str) -> Image.Image:
    """
    Open an ECG page image by its extension (.png, .jpg / .jpeg), with that format's decoder
    only and its header checked first. A file whose content is not its extension's format is
    refused, so no other Pillow decoder (TIFF, GIF, PSD, EPS, ...) is ever reached.

    Raises:
        ValueError: if the file is not an acceptable PNG / JPEG.
    """
    extension = os.path.splitext(str(path))[1].lower()
    if extension == ".png":
        return open_png(path)
    if extension in (".jpg", ".jpeg"):
        return open_jpeg(path)
    raise ValueError(f"unsupported image type {extension!r} (only .png, .jpg, .jpeg)")


def load_ecg_image(path: str, preprocessing: dict) -> torch.Tensor:
    """
    Load an ECG page image and preprocess it exactly as DeepECG-PNG was trained
    (DeepECG-PNG utils.image.process_image, no augmentation):

    1. convert to grayscale ("L"; color_mode "rgb" keeps RGB),
    2. bilinear resize to (height, width), stretching the page (torchvision's v2.Resize
       on a PIL image is Pillow's BILINEAR resize),
    3. three identical channels (v2.Grayscale(num_output_channels=3)),
    4. uint8 -> float32 in [0, 1] (v2.ToDtype(scale=True) multiplies by 1/255),
    5. per-channel (x - mean) / std (v2.Normalize).

    Args:
        path: Path to the PNG or JPEG page.
        preprocessing: The weights' config.json "preprocessing" block
            (resize [h, w], color_mode, mean [3], std [3]).

    Returns:
        Float32 tensor of shape (3, height, width).
    """
    height, width = (int(v) for v in preprocessing["resize"])
    gray = str(preprocessing.get("color_mode", "grayscale")).lower().startswith("gray")
    # header checked first, then only the decoder of the file's own format (see open_ecg_image)
    with open_ecg_image(path) as image:
        image = image.convert("L" if gray else "RGB")
        image = image.resize((width, height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    array = np.repeat(array[None], 3, axis=0) if gray else array.transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32).mul_(1.0 / 255)
    mean = torch.tensor(preprocessing["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(preprocessing["std"], dtype=torch.float32).view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)
