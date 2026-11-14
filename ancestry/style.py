"""Shared plotting style for the publication figures, matched to the manuscript's own figures.

Every value here is read off the manuscript figures (v49, 2026-08-04):
panel letters bold and lower-case outside the axes at top left, value labels above bars, black capped error
bars, a dotted grey reference line at chance, top and right spines removed, a legend of colour squares under
the panel, and a sans-serif face throughout.

PALETTE PROVENANCE AND ONE DEVIATION
The cohort palette (UKB blue, MHI rust, CLSA green) is the manuscript's and passes every check of the
dataviz validator: worst adjacent CVD deltaE 9.0 (deutan), normal-vision 24.3, all three above 3:1 on the
surface. It is used unchanged.

The manuscript's ancestry-subgroup palette (EUR #5BA3D0, SAS #4CB391, AFR #E89BC0, EAS #E8B04B) does NOT
pass. The validator reports: two hues outside the lightness band, the blue below the chroma floor (it reads
grey), a normal-vision floor of deltaE 11.9 between SAS and EUR (below the hard floor of 15, so those two are
hard to tell apart even with full colour vision), a tritan separation of 2.4, and all four below 3:1 contrast
against the surface. ANCESTRY_PAPER reproduces it exactly so the figure matches the manuscript;
ANCESTRY_ACCESSIBLE is the nearest passing alternative (all checks PASS, worst adjacent CVD deltaE 8.7,
normal-vision 21.6) and is used for the `_accessible` variants.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- manuscript palette -------------------------------------------------------------------------
BLUE = '#3272B5'        # UKB / the reference model / ancestry bars
BLUE_LIGHT = '#92C0DE'  # ECG+disease
RUST = '#D95F2B'        # MHI
GREEN = '#3A9C6E'       # CLSA / polygenic score
AMBER = '#E8A838'       # MHI in Fig 3d, supervised models in Fig 1c
PURPLE = '#9B59A8'      # disease-matched
GREY = '#A6A6A6'        # comparators and controls
GREY_DARK = '#6E6E6E'
INK = '#1a1a1a'

COHORT = {'UKB': BLUE, 'MHI': RUST, 'CLSA': GREEN}
ANCESTRY_PAPER = {'EUR': '#5BA3D0', 'SAS': '#4CB391', 'AFR': '#E89BC0', 'EAS': '#E8B04B'}
ANCESTRY_ACCESSIBLE = {'EUR': '#2E75B6', 'SAS': '#3E9B4F', 'AFR': '#C75D96', 'EAS': '#D89426'}


def use_paper_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'axes.edgecolor': INK,
        'text.color': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'legend.frameon': False,
    })


def panel(ax, letter, dx=-0.16, dy=1.06, size=15):
    """Bold lower-case panel letter outside the axes at top left, as the manuscript sets them."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight='bold',
            va='top', ha='left', color=INK)


def despine(ax, keep_left=True):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)


def chance_line(ax, y=0.5, style=':'):
    ax.axhline(y, ls=style, lw=1.0, color='#9a9a9a', zorder=0)


def bar_value_labels(ax, xs, heights, errs=None, fmt='{:.2f}', dy=0.012, size=10, weight='normal'):
    for x, h in zip(xs, heights):
        e = 0 if errs is None else (errs[list(xs).index(x)] if hasattr(errs, '__len__') else errs)
        ax.text(x, h + e + dy, fmt.format(h), ha='center', va='bottom', fontsize=size,
                fontweight=weight, color=INK)


def square_legend(ax, labels_colors, ncol=None, y=-0.22, size=9.5):
    """The manuscript's legend: filled squares with labels, centred under the panel, no frame."""
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='none') for _, c in labels_colors]
    ax.legend(handles, [l for l, _ in labels_colors], loc='upper center',
              bbox_to_anchor=(0.5, y), ncol=ncol or len(labels_colors), frameon=False,
              handlelength=1.1, handleheight=1.1, handletextpad=0.5, columnspacing=1.6, fontsize=size)


ERRORBAR = dict(ecolor=INK, capsize=3.5, elinewidth=1.1, capthick=1.1)
