"""
plot_sample_selection_figure.py

Illustrates the automated training/threshold-region selection process
(Section 3.3-3.4) for one example reservoir, Pozzillo (Sicily), side by
side for the two detectors:
  (a) SVM: JRC-derived water (occurrence >=95%, fallback >=80%) and land
      (500-2000 m ring, WorldCover non-water, zero JRC occurrence) training
      points, exact pipeline logic/parameters (seed 42), with the 500 m and
      2000 m ring boundaries drawn thin for reference.
  (b) Otsu: the 500 m land-ringed buffer used to build the per-scene VV
      backscatter histogram.

Both panels share a Sentinel-1 VV median backdrop (2020) for visual context.
The plotted region is the lake's own bounding box expanded by 2500 m, so the
whole reservoir and the full 2000 m ring are always in frame regardless of
lake shape/size (fixed a bug where a fixed-radius region clipped one arm of
Rosamarina's dendritic shape).
Geometries/points pulled once via Earth Engine and cached locally (see
conversation 9 Jul 2026); this script only does the local plotting so it can
be re-run without re-hitting EE.

Inputs (pre-fetched):
  scratchpad/pozzillo_samples.json  (region, lakePoly, ring500, ring2000,
                                      otsuBuffer, waterPts, landPts)
  scratchpad/pozzillo_s1_vv.png     (S1 VV median backdrop, 1000x1000)
Output:
  manuscript/figures/sample_selection_pozzillo.png/.pdf
"""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.image import imread

SCRATCH = (r"C:/Users/Unipa/AppData/Local/Temp/claude/"
           r"c--Users-Unipa-Documents-reservoirs-s1-svm/"
           r"db674b5e-ba1b-4e18-8572-ecd885395938/scratchpad")
OUTDIR = "C:/Users/Unipa/Documents/reservoirs_s1_svm/manuscript/figures"

data = json.load(open(f"{SCRATCH}/pozzillo_samples.json"))
img = imread(f"{SCRATCH}/pozzillo_s1_vv.png")

# region bounds used for the S1 thumbnail (lake bbox + 2500 m, from the fetch script)
_region_coords = data["region"]["coordinates"][0]
LON_MIN = min(c[0] for c in _region_coords); LON_MAX = max(c[0] for c in _region_coords)
LAT_MIN = min(c[1] for c in _region_coords); LAT_MAX = max(c[1] for c in _region_coords)
EXTENT = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

WATER_COLOR = "#2f6fb0"
LAND_COLOR = "#c0522d"
LAKE_EDGE = "#ffffff"
BUFFER_EDGE = "#ffd400"


def poly_xy(geom):
    """Return list of (x, y) arrays for each ring of a Polygon/MultiPolygon geometry."""
    rings = []
    if geom["type"] == "Polygon":
        rings = geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            rings.extend(poly)
    return [np.asarray(r) for r in rings]


def points_xy(geom):
    xs, ys = [], []
    if geom["type"] == "MultiPoint":
        for c in geom["coordinates"]:
            xs.append(c[0]); ys.append(c[1])
    elif geom["type"] == "GeometryCollection":
        for g in geom["geometries"]:
            if g["type"] == "Point":
                xs.append(g["coordinates"][0]); ys.append(g["coordinates"][1])
    return np.asarray(xs), np.asarray(ys)


lake_rings = poly_xy(data["lakePoly"])
buffer_rings = poly_xy(data["otsuBuffer"])
ring500_rings = poly_xy(data["ring500"])
ring2000_rings = poly_xy(data["ring2000"])
wx, wy = points_xy(data["waterPts"])
lx, ly = points_xy(data["landPts"])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))

for ax in axes:
    ax.imshow(img, extent=EXTENT, cmap="gray", origin="upper", zorder=0)
    for r in lake_rings:
        ax.plot(r[:, 0], r[:, 1], color=LAKE_EDGE, lw=1.8, zorder=3)
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1 / np.cos(np.radians((LAT_MIN + LAT_MAX) / 2)))
    ax.set_xticks([]); ax.set_yticks([])

# (a) SVM training points
axL = axes[0]
for r in ring500_rings:
    axL.plot(r[:, 0], r[:, 1], color="#ffd400", lw=0.9, zorder=3, ls="--")
for r in ring2000_rings:
    axL.plot(r[:, 0], r[:, 1], color="#ffd400", lw=0.9, zorder=3, ls=":")
axL.scatter(wx, wy, s=5, c=WATER_COLOR, alpha=0.85, linewidths=0, zorder=4,
            label=f"water training (n={len(wx)})")
axL.scatter(lx, ly, s=5, c=LAND_COLOR, alpha=0.85, linewidths=0, zorder=4,
            label=f"land training (n={len(lx)})")
axL.plot([], [], color="#ffd400", lw=0.9, ls="--", label="500 m ring")
axL.plot([], [], color="#ffd400", lw=0.9, ls=":", label="2000 m ring")
axL.set_title("(a) SVM training samples\nJRC water occurrence + WorldCover land, 500-2000 m ring",
              fontsize=10)
axL.legend(fontsize=7.5, loc="lower right", markerscale=3, frameon=True)

# (b) Otsu histogram buffer
axR = axes[1]
for r in buffer_rings:
    axR.plot(r[:, 0], r[:, 1], color=BUFFER_EDGE, lw=1.8, zorder=3, ls="--")
axR.fill(buffer_rings[0][:, 0], buffer_rings[0][:, 1], color=BUFFER_EDGE, alpha=0.12, zorder=2)
axR.set_title("(b) Otsu histogram buffer\n500 m land-ringed buffer, VV backscatter histogram",
              fontsize=10)
from matplotlib.lines import Line2D
axR.legend(handles=[
    Line2D([0], [0], color=LAKE_EDGE, lw=1.8, label="JRC max-extent polygon"),
    Line2D([0], [0], color=BUFFER_EDGE, lw=1.8, ls="--", label="500 m buffer"),
], fontsize=8, loc="lower right", frameon=True)

fig.suptitle("Automated sample/region selection at Pozzillo (Sicily), "
             "Sentinel-1 VV backdrop (2020 median)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/sample_selection_pozzillo.{ext}", dpi=200, bbox_inches="tight")
print(f"Saved: {OUTDIR}/sample_selection_pozzillo.png")
