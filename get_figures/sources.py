"""
CET-DET corpus: segments per ocean basin (stacked vertical bars,
                log-scaled height with linearly-proportional colour splits)
               + segments per deployment platform (lollipop chart, log scale).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.patches import Patch

# ---- palette --------------------------------------------------------------
TEAL      = "#2f9292"   # cetacean vocalisations
DEEPBLUE  = "#185FA5"   # anthropogenic & ambient noise
AMBER     = "#EF9F27"   # coastal (small boat)
VIOLET    = "#7b5bc9"   # fishing gear
CORAL     = "#D85A30"   # captive
GREY      = "#bbc5cc"   # unknown
INK       = "#23303a"
MUTED     = "#8a9ba5"
PANEL     = "#eef3f3"
GRID      = "#d8dee2"

# ---- basin data -----------------------------------------------------------
# (label, cetacean, background)
basins = [
    ("NE Pacific",                21696, 150725),
    ("NE Atlantic / N Sea",        5661,  30383),
    ("Adriatic (captive)",         4238,  12622),
    ("W Mediterranean",            4046,   1291),
    ("Mediterranean (captive)",    1241,   1645),
    ("Gulf of Mexico (captive)",    786,   2056),
    ("Multi-ocean (global)",       1877,      0),
    ("Adriatic Sea",                942,    238),
    ("Bay of Biscay",              1150,      0),
    ("E Tropical Pacific",          889,    153),
    ("Unknown (aggregated)",        617,      0),
    ("Indian Ocean (Australia)",    332,      0),
]
basins = sorted(basins, key=lambda r: r[1] + r[2], reverse=True)

# ---- platform data --------------------------------------------------------
# (label, color, count)
platforms = [
    ("Surface vessel (shipping)",  TEAL,   157024),
    ("Aquarium pool (enclosed)",   CORAL,   17657),
    ("Towed / mobile array",       TEAL,    15543),
    ("Surface vessel (small boat)",AMBER,   13624),
    ("Fishing gear (FAD/net)",     VIOLET,   2850),
    ("Autonomous glider",          TEAL,     2300),
    ("Fixed marine infrastructure",TEAL,      319),
    ("Ambient (weather)",          GREY,       32),
]
platforms = sorted(platforms, key=lambda r: r[2], reverse=True)

# ---- style ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": INK, "text.color": INK, "figure.dpi": 300,
})

fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.38)
ax_l = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1])

# ===== LEFT: proportional-split bars on a log-height linear axis ===========
ax_l.set_facecolor(PANEL)
n_b  = len(basins)
xpos = list(range(n_b))

for xi, (label, cet, noise) in enumerate(basins):
    total = cet + noise
    if total == 0:
        continue

    log_h = np.log10(total)
    h_cet   = (cet   / total) * log_h
    h_noise = (noise / total) * log_h

    if h_noise > 0:
        ax_l.bar(xi, h_noise, bottom=0,       color=DEEPBLUE,
                 edgecolor="white", linewidth=0.8, zorder=3, width=0.6)
    if h_cet > 0:
        ax_l.bar(xi, h_cet,   bottom=h_noise, color=TEAL,
                 edgecolor="white", linewidth=0.8, zorder=3, width=0.6)

    ax_l.text(xi, log_h + 0.04, f"{total:,}", ha="center", va="bottom",
              fontsize=8.5, fontweight="bold", color=INK, zorder=4, rotation=45)

tick_vals = [10, 100, 1_000, 10_000, 100_000]
ax_l.set_yticks([np.log10(v) for v in tick_vals])
ax_l.set_yticklabels([f"{v:,}" for v in tick_vals], color=MUTED)
ax_l.set_ylim(0, np.log10(300_000))
ax_l.set_ylabel("Number of 5\u2009s segments  (log scale)", fontsize=11, color=INK)
ax_l.set_xticks(xpos)
ax_l.set_xticklabels([r[0] for r in basins], rotation=35, ha="right",
                      fontsize=9.5, fontstyle="italic", color=INK)
ax_l.set_title("a · segments per ocean basin (log scale)",
               fontsize=11, color=INK, pad=10, loc="left", fontweight="bold")
ax_l.grid(axis="y", which="major", color="white", linewidth=2.0, zorder=0)
ax_l.set_axisbelow(True)
for s in ["top", "right"]:
    ax_l.spines[s].set_visible(False)
ax_l.spines["bottom"].set_color(GRID)
ax_l.spines["left"].set_color(GRID)
ax_l.tick_params(axis="x", length=0)
ax_l.tick_params(axis="y", length=0)

left_legend = [
    Patch(facecolor=TEAL,     label="Cetacean vocalisations"),
    Patch(facecolor=DEEPBLUE, label="Anthropogenic & ambient noise"),
]
ax_l.legend(handles=left_legend, loc="upper right", frameon=False,
            fontsize=10, handlelength=1.2, labelspacing=0.5)

# ===== RIGHT: lollipop chart per platform (log scale) =====================
ax_r.set_facecolor(PANEL)
n_p  = len(platforms)
ypos = list(range(n_p))[::-1]

for yi, (label, col, count) in zip(ypos, platforms):
    # stem line from 1 to count
    ax_r.hlines(yi, 1, count, colors=MUTED, linewidth=1.5, zorder=2)
    # dot at the end
    ax_r.scatter(count, yi, color=col, s=90, zorder=4, linewidths=0)
    # count label
    ax_r.text(count * 1.18, yi, f"{count:,}", va="center", ha="left",
              fontsize=10, fontweight="bold", color=INK, zorder=4)

ax_r.set_yticks(ypos)
ax_r.set_yticklabels([r[0] for r in platforms], fontsize=10,
                      fontstyle="italic", color=INK)
ax_r.set_xscale("log")
ax_r.set_xlim(1, 2_000_000)
ax_r.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
ax_r.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
ax_r.set_xlabel("Number of 5\u2009s segments  (log scale)", fontsize=11, color=INK)
ax_r.set_title("b · segments per deployment platform (log scale)",
               fontsize=11, color=INK, pad=10, loc="left", fontweight="bold")
ax_r.set_ylim(-0.6, n_p - 0.4)
ax_r.grid(axis="x", which="major", color="white", linewidth=2.0, zorder=0)
ax_r.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax_r.spines[s].set_visible(False)
ax_r.spines["bottom"].set_color(GRID)
ax_r.tick_params(axis="y", length=0)
ax_r.tick_params(axis="x", colors=MUTED)

right_legend = [
    Patch(facecolor=TEAL,   label="open ocean / wild"),
    Patch(facecolor=AMBER,  label="coastal (small boat)"),
    Patch(facecolor=VIOLET, label="fishing gear"),
    Patch(facecolor=CORAL,  label="captive"),
    Patch(facecolor=GREY,   label="unknown"),
]
ax_r.legend(handles=right_legend, loc="lower right", frameon=False,
            fontsize=10, handlelength=1.2, labelspacing=0.5)

fig.savefig("/data2/mromaniuc/cet-det/get_figures/imgs/fig_recording_context.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print("saved")