"""Study-area / coverage map: the FULL reservoir dataset as one set, coloured by A/P class.

No cohort split (v3/v4/coverage all belong to the chosen dataset), no point labels.
Reservoirs + A/P from bestof_kge.csv (the analysed universe, incl. the 4 Sicilian);
coordinates from exportGlobalPilotV4.js (+ the 4 Sicilian, not in that JS).
Basemap = Natural Earth 110m.
Output: analysis/method_comparison_output/coverage_current_plus_new.png
"""
import re, json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
GEOJSON = (r"C:/Users/Unipa/AppData/Local/Temp/claude/"
           r"c--Users-Unipa-Documents-reservoirs-s1-svm/"
           r"db674b5e-ba1b-4e18-8572-ecd885395938/scratchpad/ne_110m_countries.geojson")
OUT = os.path.join(HERE, "method_comparison_output", "coverage_current_plus_new.png")

LOW_MAX, HIGH_MIN = 120.0, 250.0
CLASS_COLOR = {"Low": "#f88f4d", "Medium": "#d64a02", "High": "#8a2d04"}
LAND, LAND_EDGE, SURFACE = "#e7e6e1", "#ffffff", "#ffffff"
INK, MUTED = "#1b1b1b", "#6b6b6b"

def ap_class(ap):
    return "Low" if ap < LOW_MAX else ("Medium" if ap < HIGH_MIN else "High")

# ── coordinates: from the export JS + the 4 Sicilian ──────────────────────────
js = open(os.path.join(HERE, "exportGlobalPilotV4.js"), encoding="utf-8").read()
coord = {}
for m in re.finditer(r"\[\s*'([A-Za-z_]+)'\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", js):
    coord.setdefault(m.group(1), (float(m.group(2)), float(m.group(3))))
coord.update({"Ancipa": (37.887, 14.565), "Poma": (37.994, 13.090),
              "Pozzillo": (37.700, 14.530), "Rosamarina": (37.944, 13.640)})

# ── the full dataset (analysed universe) ──────────────────────────────────────
df = pd.read_csv(os.path.join(HERE, "bestof_kge.csv"))[["name", "ap_m"]].dropna(subset=["ap_m"])
df["lat"] = df["name"].map(lambda n: coord.get(n, (np.nan, np.nan))[0])
df["lon"] = df["name"].map(lambda n: coord.get(n, (np.nan, np.nan))[1])
miss = df[df.lat.isna()]["name"].tolist()
if miss:
    print("WARN missing coords:", miss)
df = df.dropna(subset=["lat", "lon"])
df["cls"] = df["ap_m"].map(ap_class)
print(f"N = {len(df)} | classes {df.cls.value_counts().to_dict()} | A/P {df.ap_m.min():.0f}-{df.ap_m.max():.0f} m")

# ── figure ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

gj = json.load(open(GEOJSON, encoding="utf-8"))
def draw(cs):
    for ring in cs:
        a = np.asarray(ring)
        ax.fill(a[:, 0], a[:, 1], facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.4, zorder=1)
for feat in gj["features"]:
    g = feat["geometry"]
    if g["type"] == "Polygon":
        draw(g["coordinates"])
    elif g["type"] == "MultiPolygon":
        for poly in g["coordinates"]:
            draw(poly)
for x in range(-120, 181, 30):
    ax.axvline(x, color="#f0f0ee", lw=0.6, zorder=0)
for y in range(-40, 61, 20):
    ax.axhline(y, color="#f0f0ee", lw=0.6, zorder=0)

for cls in ["Low", "Medium", "High"]:
    s = df[df.cls == cls]
    ax.scatter(s.lon, s.lat, marker="o", s=80, c=CLASS_COLOR[cls],
               edgecolors="white", linewidths=0.8, zorder=5, alpha=0.95)

ax.set_xlim(-130, 160); ax.set_ylim(-45, 62); ax.set_aspect("equal")
ax.set_xlabel("Longitude ($^{\\circ}$)", color=MUTED, fontsize=8)
ax.set_ylabel("Latitude ($^{\\circ}$)", color=MUTED, fontsize=8)
ax.tick_params(colors=MUTED, labelsize=7)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

leg = [Line2D([0], [0], marker="o", ls="", mfc=CLASS_COLOR[c], mec="white", ms=10, label=lab)
       for c, lab in [("Low", "Low   (< 120 m)"),
                      ("Medium", "Medium   (120–250 m)"),
                      ("High", "High   (≥ 250 m)")]]
ax.legend(handles=leg, title="Shoreline A/P (static)", loc="lower left", frameon=False,
          fontsize=8.5, title_fontsize=9, bbox_to_anchor=(0.005, 0.02))

ax.set_title(f"Global reservoir dataset (n = {len(df)}) by shoreline compactness",
             color=INK, fontsize=11.5, loc="left", pad=8)
ax.text(0.005, -0.13, "A/P classes: Low < 120 m  ·  Medium 120–250 m  ·  High ≥ 250 m. "
        "Basemap: Natural Earth.", transform=ax.transAxes, fontsize=6.8, color=MUTED)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("saved ->", OUT)
