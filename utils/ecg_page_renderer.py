"""Draw a 12-lead ECG page exactly as the DeepECG-PNG training images were drawn.

This is the DeepECG-PNG training renderer, ported line for line (only the file handling is left out):
  * ``utils/waveform_to_image/ecg_plotter.py`` (``ECGPlotter`` with ``dataset="MHI"``), DeepECG-PNG
    commit 778f7f3;
  * the three settings of ``utils/waveform_to_image/render_array.py`` (DeepECG-PNG branch
    ``issue_215/ecg-pdf-pipeline``, commit a8d0376) that the training PNGs (rendered in May 2024)
    differ by: the tick at the start of each lead is matplotlib's default blue (C0) instead of black,
    it is drawn under the traces, and the lead labels are fontsize 44 instead of 32.
With these, DeepECG-PNG re-rendered training ECGs from their MUSE XML on 100.000% of the stored
training pixels (encoder features: cosine 1.00000); the plotter as it is in that repository moved
probe outputs by up to 0.21. matplotlib 3.8.4 / 3.9.0 / 3.10.3 and Pillow 10.3 / 12.3 gave the same
pixels; so do the pinned matplotlib 3.10.9 with Pillow 12.3.0 and 13.0.0 (golden pixel hash in
tests/test_pdf_inputs.py).

Input: raw MUSE counts (4.88 uV per count; the plotter's MHI amplitude factor), 2500 samples
(10 s at 250 Hz) of the 12 leads in the order I, II, III, aVR, aVL, aVF, V1-V6.
Page: leads in 3 rows x 4 columns of 2.5 s (I/aVR/V1/V4, II/aVL/V2/V5, III/aVF/V3/V6) and a 10 s
lead II rhythm strip, each after a 1 mV calibration pulse, on a red grid (1 mV = 10 grid units);
1250x625 RGBA, saved at 240 dpi like the training PNGs.
"""
import io

import matplotlib

matplotlib.use("Agg")   # as render_array.py: no display

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

UV_PER_COUNT = 4.88     # MUSE resolution: the plotter's MHI amplitude factor
SAMPLES = 2500          # 10 s at 250 Hz
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PAGE_WIDTH = 1250       # training PNGs are 1250x625
PAGE_DPI = (240, 240)   # as the plotter saved them
TICK_COLOR = "C0"       # matplotlib default blue, as in every training PNG
TICK_ZORDER = 1.9       # under the traces (2), over the grid
LABEL_FONTSIZE = 44
# rcParams ecg_plotter.py sets when it is imported, on top of matplotlib's defaults
_STYLE = {"figure.facecolor": "white", "axes.facecolor": "white"}


def _plot_leads(ax, lead_dict):
    # ECGPlotter._plot_leads with amplitude_factor = 4.88 (MHI)
    amplitude_factor = UV_PER_COUNT
    activation = [0] * 5 + [10] * 50 + [0] * 5

    # Generate the panels
    pannel_1_y = (
        [i + 50 for i in activation]
        + [((i * amplitude_factor) / 100) + 50 for i in lead_dict["I"][60:625]]
        + [((i * amplitude_factor) / 100) + 50 for i in lead_dict["aVR"][625:1250]]
        + [((i * amplitude_factor) / 100) + 50 for i in lead_dict["V1"][1250:1875]]
        + [((i * amplitude_factor) / 100) + 50 for i in lead_dict["V4"][1875:2500]]
    )
    pannel_2_y = (
        [i + 15 for i in activation]
        + [((i * amplitude_factor) / 100) + 15 for i in lead_dict["II"][60:625]]
        + [((i * amplitude_factor) / 100) + 15 for i in lead_dict["aVL"][625:1250]]
        + [((i * amplitude_factor) / 100) + 15 for i in lead_dict["V2"][1250:1875]]
        + [((i * amplitude_factor) / 100) + 15 for i in lead_dict["V5"][1875:2500]]
    )
    pannel_3_y = (
        [i - 15 for i in activation]
        + [((i * amplitude_factor) / 100) - 15 for i in lead_dict["III"][60:625]]
        + [((i * amplitude_factor) / 100) - 15 for i in lead_dict["aVF"][625:1250]]
        + [((i * amplitude_factor) / 100) - 15 for i in lead_dict["V3"][1250:1875]]
        + [((i * amplitude_factor) / 100) - 15 for i in lead_dict["V6"][1875:2500]]
    )
    pannel_4_y = [i - 50 for i in activation] + [
        ((i * amplitude_factor) / 100) - 50 for i in lead_dict["II"][60:]
    ]

    ax.axis([0 - 100, 2500 + 100, min(pannel_4_y) - 10, max(pannel_1_y) + 10])
    x = [pos for pos in range(0, len(pannel_1_y))]
    ax.plot(x, pannel_1_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_2_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_3_y, linewidth=3, color="#000000")
    ax.plot(x, pannel_4_y, linewidth=3, color="#000000")


def _add_lead_labels(ax):
    # ECGPlotter._add_lead_labels with the tick colour, tick zorder and label size of the training PNGs
    labels = [
        (60, -10, "III"), (625, -10, "aVF"), (1250, -10, "V3"), (1875, -10, "V6"),
        (60, 20, "II"), (625, 20, "aVL"), (1250, 20, "V2"), (1875, 20, "V5"),
        (60, 55, "I"), (625, 55, "aVR"), (1250, 55, "V1"), (1875, 55, "V4"),
        (60, -45, "II"),
    ]
    for x, y, label in labels:
        ax.vlines(x, y - 10, y, linewidth=4, color=TICK_COLOR, zorder=TICK_ZORDER)
        ax.text(x, y, label, fontsize=LABEL_FONTSIZE, color="black")


def _setup_plot():
    # ECGPlotter._setup_plot / _setup_grid / _setup_axes
    fig, ax = plt.subplots(figsize=(40, 20))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(ls="-", color="red", linewidth=1.2)
    ax.grid(which="minor", ls=":", color="red", linewidth=1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig, ax


def render_training_page(counts: np.ndarray) -> Image.Image:
    """
    Draw one ECG page as the DeepECG-PNG training images were drawn.

    Args:
        counts: (2500, 12) raw MUSE counts (4.88 uV per count), leads in LEAD_ORDER.

    Returns:
        The page, an RGBA PIL image of 1250x625.

    Raises:
        ValueError: if counts is not a finite (2500, 12) array.
    """
    # float64, as the plotter read the training .npy files (float32 values would be drawn in float32)
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (SAMPLES, len(LEAD_ORDER)):
        raise ValueError(f"expected a ({SAMPLES}, {len(LEAD_ORDER)}) array, got {counts.shape}")
    if not np.isfinite(counts).all():
        raise ValueError("the signal has non-finite values")
    lead_dict = dict(zip(LEAD_ORDER, np.swapaxes(counts, 0, 1)))

    # ECGPlotter.plot_ecg (no title) and _save_or_show_plot, minus the file write
    with plt.style.context("default"), plt.rc_context(_STYLE):
        fig, ax = _setup_plot()
        try:
            _plot_leads(ax, lead_dict)
            _add_lead_labels(ax)
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", transparent=False, facecolor="white", edgecolor="none")
        finally:
            plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer)
    new_height = int(PAGE_WIDTH * (image.height / image.width))
    return image.resize((PAGE_WIDTH, new_height), Image.LANCZOS)


def save_training_page(image: Image.Image, path: str) -> None:
    """Save a page as the training PNGs were saved (PNG, 240 dpi)."""
    image.save(path, format="PNG", dpi=PAGE_DPI)
