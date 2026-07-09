"""
Cetacean corpus: per-species vocalisation-type split (log x-axis).
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.patches import Patch

# ---- palette --------------------------------------------------------------
TEAL     = "#2f9292"
VIOLET   = "#7b5bc9"
GREY     = "#5a6b73"
AMBER    = "#EF9F27"
DEEPBLUE = "#0c6dcf"
INK      = "#23303a"
MUTED    = "#bbc5cc"
PANEL    = "#eef3f3"
GRID     = "#d8dee2"

VOC_COL = {
    "whistles":    TEAL,
    "clicks":      VIOLET,
    "feeding":     DEEPBLUE,
    "bursting":    AMBER,
    "unspecified": MUTED,
}
VOC_LABEL = {
    "whistles":    "Whistles / tonal",
    "clicks":      "Clicks / echolocation",
    "feeding":     "Feeding buzzes",
    "bursting":    "Bursting pulses",
    "unspecified": "Unspecified call-type",
}
VOC_ORDER = ["whistles", "clicks", "feeding", "bursting", "unspecified"]

# ---- data -----------------------------------------------------------------
species = [
    ("Orcinus orca",               "O. orca",          21809, {}),
    ("Tursiops truncatus",         "T. truncatus",      7362,
        {"whistles": 5120, "clicks": 1367, "feeding": 285, "bursting": 492}),
    ("Delphinus delphis",          "D. delphis",        1292, {"whistles": 1150}),
    ("Physeter macrocephalus",     "P. macrocephalus",  1089, {"clicks": 179}),
    ("Globicephala melas",         "G. melas",           853, {"clicks": 218, "whistles": 55}),
    ("Delphinidae (indet.)",       "Delphinid (unid.)",  815, {}),
    ("Balaenoptera physalus",      "B. physalus",        539, {}),
    ("Grampus griseus",            "G. griseus",         219, {"clicks": 45}),
    ("Stenella coeruleoalba",      "S. coeruleoalba",    144, {"clicks": 15}),
    ("Balaenoptera acutorostrata", "B. acutorostrata",    17, {}),
]
species = sorted(species, key=lambda r: r[2], reverse=True)

def split_for(total, raw):
    """Always returns a list — species with no raw data get 100% unspecified."""
    d = {k: raw.get(k, 0) for k in VOC_ORDER}
    d["unspecified"] += max(total - sum(d.values()), 0)
    return [(k, d[k]) for k in VOC_ORDER if d[k] > 0]

# ---- style ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.edgecolor": INK, "text.color": INK, "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor(PANEL)

# ---- bars -----------------------------------------------------------------
n = len(species)
ypos = list(range(n))[::-1]
BASE = 1.0

for yi, (latin, abbr, total, raw) in zip(ypos, species):
    segs = split_for(total, raw)
    left = BASE
    for k, c in segs:
        ax.barh(yi, c, height=0.45, left=left, color=VOC_COL[k],
                edgecolor="white", linewidth=0.8, zorder=3)
        left += c
    ax.text(total * 1.08, yi, f"{total:,}", va="center", ha="left",
            fontsize=12, fontweight="bold", color=INK, zorder=4)

# ---- axes -----------------------------------------------------------------
ax.set_yticks(ypos)
ax.set_yticklabels([r[1] for r in species], fontsize=13,
                   fontstyle="italic", color=INK)
ax.set_xscale("log")
ax.set_xlim(1, 40000)
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
ax.set_xlabel("Number of 5\u2009s segments  (log scale)", fontsize=13, color=INK)
ax.set_ylim(-0.6, n - 0.4)
ax.grid(axis="x", which="major", color="white", linewidth=1.4, zorder=0)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED)

# ---- legend ---------------------------------------------------------------
legend_items = [Patch(facecolor=VOC_COL[k], label=VOC_LABEL[k]) for k in VOC_ORDER]
ax.legend(handles=legend_items, loc="lower right", frameon=False,
          fontsize=11, handlelength=1.2, labelspacing=0.5, borderaxespad=1.0)

fig.savefig("/data2/mromaniuc/cet-det/get_figures/imgs/fig_species_voctype.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print("saved")