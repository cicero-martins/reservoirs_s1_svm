"""Study-area map (Paper 2): the nine validated Sicilian reservoirs.

Coloured by shoreline area-to-perimeter class (the reliability axis); star marker =
independent survey ground truth (Garcia echo-sounder; Poma/Rosamarina/Arancio survey
curves), circle = cross-sensor / scalar only. Basemap = FAO GAUL 'Sicilia'
(analysis/geo/sicily_gaul.geojson, reused from Paper 1). Output → manuscript_paper2/figures/.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SICILY = os.path.join(HERE, "geo", "sicily_gaul.geojson")
OUTDIR = os.path.normpath(os.path.join(HERE, "..", "manuscript_paper2", "figures"))
os.makedirs(OUTDIR, exist_ok=True)

LOW_MAX, HIGH_MIN = 100.0, 200.0
CLASS_COLOR = {"Low": "#f88f4d", "Medium": "#d64a02", "High": "#8a2d04"}
LAND = "#e7e6e1"; LAND_EDGE = "#ffffff"; SURFACE = "#ffffff"; INK = "#1b1b1b"; MUTED = "#6b6b6b"
def ap_class(ap): return "Low" if ap < LOW_MAX else ("Medium" if ap < HIGH_MIN else "High")

# name, lat, lon (dam coords, sicilia_dighe_anagrafica.csv), A/P (m), survey ground truth?
R = pd.DataFrame([
    ("Poma",       38.011037, 13.056135, 190, True),
    ("Rosamarina", 37.960336, 13.654665, 187, True),
    ("Garcia",     37.793124, 13.098185, 168, True),
    ("Arancio",    37.634491, 13.065184, 182, True),
    ("Castello",   37.582494, 13.420304, 127, False),
    ("Ancipa",     37.836222, 14.562873,  90, False),
    ("Pozzillo",   37.674037, 14.610613, 240, False),
    ("Nicoletti",  37.604822, 14.346314, 120, False),
    ("Olivo",      37.405048, 14.286604,  51, False),
], columns=["name", "lat", "lon", "ap_m", "survey"])
R["cls"] = R.ap_m.map(ap_class)

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 6.4), dpi=300)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

def draw(coords):
    for ring in coords:
        a = np.asarray(ring)
        ax.fill(a[:, 0], a[:, 1], facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.5, zorder=1)
gj = json.load(open(SICILY, encoding="utf-8"))
geom = gj["geometry"] if gj.get("type") == "Feature" else gj
if geom["type"] == "Polygon":
    draw(geom["coordinates"])
else:
    for poly in geom["coordinates"]:
        draw(poly)
for x in np.arange(12.5, 15.6, 0.5): ax.axvline(x, color="#f2f1ef", lw=0.6, zorder=0)
for y in np.arange(36.5, 38.6, 0.5): ax.axhline(y, color="#f2f1ef", lw=0.6, zorder=0)

# label offsets (dx, dy, ha) to reduce overlap
OFF = {"Poma": (0.03, 0.05, "left"), "Rosamarina": (0.05, 0.02, "left"),
       "Garcia": (-0.05, -0.07, "right"), "Arancio": (-0.05, -0.02, "right"),
       "Castello": (0.05, -0.02, "left"), "Ancipa": (0.05, 0.04, "left"),
       "Pozzillo": (0.05, -0.02, "left"), "Nicoletti": (-0.05, 0.03, "right"),
       "Olivo": (0.05, -0.04, "left")}
for _, r in R.iterrows():
    mk = "*" if r.survey else "o"
    sz = 340 if r.survey else 130
    ax.scatter(r.lon, r.lat, marker=mk, s=sz, c=CLASS_COLOR[r.cls],
               edgecolors="white", linewidths=1.0, zorder=5)
    dx, dy, ha = OFF[r["name"]]
    ax.annotate(f"{r['name']}\nA/P {r.ap_m:.0f}", (r.lon, r.lat), (r.lon + dx, r.lat + dy),
                ha=ha, va="center", fontsize=8, color=INK, zorder=6,
                linespacing=0.95)

ax.set_xlim(12.35, 15.35); ax.set_ylim(36.62, 38.32); ax.set_aspect("equal")
ax.set_xlabel("Longitude ($^{\\circ}$E)", color=MUTED, fontsize=9)
ax.set_ylabel("Latitude ($^{\\circ}$N)", color=MUTED, fontsize=9)
ax.tick_params(colors=MUTED, labelsize=8)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.set_title("Study area: nine validated Sicilian reservoirs", color=INK, fontsize=13,
             loc="left", pad=8)

handles = [Line2D([0], [0], marker="o", ls="", mfc=CLASS_COLOR[c], mec="white", ms=10,
                  label=f"{c} A/P") for c in ["Low", "Medium", "High"]]
handles += [Line2D([0], [0], marker="*", ls="", mfc="#888", mec="white", ms=15, label="Survey ground truth"),
            Line2D([0], [0], marker="o", ls="", mfc="#888", mec="white", ms=9, label="Cross-sensor / scalar")]
ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9, ncol=1)
ax.text(0.005, -0.13, "A/P classes: Low < %d m · Medium %d–%d m · High ≥ %d m. "
        "Basemap: FAO GAUL." % (LOW_MAX, LOW_MAX, HIGH_MIN, HIGH_MIN),
        transform=ax.transAxes, fontsize=7, color=MUTED)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTDIR, f"study_area.{ext}"), bbox_inches="tight")
plt.close(fig)
print("N =", len(R), "| classes:", R.cls.value_counts().to_dict())
print("Saved:", os.path.join(OUTDIR, "study_area.png"))
