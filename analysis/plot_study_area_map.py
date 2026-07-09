"""Study-area world map: the global-pilot reservoirs coloured by A/P class.

Reservoir set = the JRC-scored reservoirs in biome_kge.csv (final global set).
Coords / country / climate from global_pilot_v4_candidates.csv (fallback: parse
exportGlobalPilotV4.js). Colour = static A/P (m), 3 ordered classes (sequential
orange). Outputs PNG (300 dpi) + PDF to ../manuscript/figures/.
"""
import re, json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = (r"C:/Users/Unipa/AppData/Local/Temp/claude/"
           r"c--Users-Unipa-Documents-reservoirs-s1-svm/"
           r"db674b5e-ba1b-4e18-8572-ecd885395938/scratchpad")
GEOJSON = os.path.join(SCRATCH, "ne_110m_countries.geojson")
SICILY_GEOJSON = os.path.join(HERE, "geo", "sicily_gaul.geojson")
OUTDIR = os.path.normpath(os.path.join(HERE, "..", "manuscript", "figures"))
os.makedirs(OUTDIR, exist_ok=True)

LOW_MAX, HIGH_MIN = 100.0, 200.0
CLASS_COLOR = {"Low": "#f88f4d", "Medium": "#d64a02", "High": "#8a2d04"}
LAND = "#e7e6e1"; LAND_EDGE = "#ffffff"; SURFACE = "#ffffff"; INK = "#1b1b1b"; MUTED = "#6b6b6b"
def ap_class(ap): return "Low" if ap < LOW_MAX else ("Medium" if ap < HIGH_MIN else "High")

# ---- the scored reservoirs + A/P ----
df = pd.read_csv(os.path.join(HERE, "biome_kge.csv"))[["name", "ap_m"]].copy()

# ---- coords from the candidate table (fallback: export script) ----
cand = pd.read_csv(os.path.join(HERE, "global_pilot_v4_candidates.csv"))
coord = {r["name"]: (r["lat"], r["lon"]) for _, r in cand.iterrows()}
js = open(os.path.join(HERE, "exportGlobalPilotV4.js"), encoding="utf-8").read()
for m in re.finditer(r"\[\s*'([A-Za-z_]+)'\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", js):
    coord.setdefault(m.group(1), (float(m.group(2)), float(m.group(3))))

df["lat"] = df.name.map(lambda n: coord.get(n, (np.nan, np.nan))[0])
df["lon"] = df.name.map(lambda n: coord.get(n, (np.nan, np.nan))[1])
miss = df[df.lat.isna()].name.tolist()
if miss:
    print("WARN missing coords:", miss)
df = df.dropna(subset=["lat", "lon"])
df["cls"] = df.ap_m.map(ap_class)
print("N =", len(df), "| classes:", df.cls.value_counts().to_dict(),
      "| A/P range %.0f-%.0f" % (df.ap_m.min(), df.ap_m.max()))

# ---- Sicily near-truth reservoirs (not in the global/JRC set, hardcoded: coords
# from exportSicilyPlanet.js, A/P from bestof_kge.csv) ----
sicily = pd.DataFrame([
    {"name": "Ancipa",     "lat": 37.887, "lon": 14.565, "ap_m": 90.0},
    {"name": "Pozzillo",   "lat": 37.700, "lon": 14.530, "ap_m": 240.0},
    {"name": "Poma",       "lat": 37.994, "lon": 13.090, "ap_m": 190.0},
    {"name": "Rosamarina", "lat": 37.944, "lon": 13.640, "ap_m": 198.0},
])
sicily["cls"] = sicily.ap_m.map(ap_class)

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(13, 5.4), dpi=300)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 2, width_ratios=[4.3, 1.15], height_ratios=[1, 1],
                       hspace=0.06, wspace=0.06)
ax = fig.add_subplot(gs[:, 0])
ax_sic = fig.add_subplot(gs[0, 1])
ax_leg = fig.add_subplot(gs[1, 1])
ax.set_facecolor(SURFACE); ax_sic.set_facecolor(SURFACE)
gj = json.load(open(GEOJSON, encoding="utf-8"))
def draw(coords, axis):
    for ring in coords:
        a = np.asarray(ring)
        axis.fill(a[:, 0], a[:, 1], facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.4, zorder=1)
for feat in gj["features"]:
    g = feat["geometry"]
    if g["type"] == "Polygon":
        draw(g["coordinates"], ax)
    elif g["type"] == "MultiPolygon":
        for poly in g["coordinates"]:
            draw(poly, ax)
for x in range(-120, 181, 30): ax.axvline(x, color="#f0f0ee", lw=0.6, zorder=0)
for y in range(-40, 61, 20): ax.axhline(y, color="#f0f0ee", lw=0.6, zorder=0)

# Sicily inset: the 110m world basemap is too coarse to render the island
# (it blurs into the Calabrian coast), so this panel uses a dedicated,
# higher-resolution regional boundary (FAO GAUL level-1 "Sicilia", pulled via
# Earth Engine and cached at analysis/geo/sicily_gaul.geojson) instead.
sic_gj = json.load(open(SICILY_GEOJSON, encoding="utf-8"))
if sic_gj["type"] == "Polygon":
    draw(sic_gj["coordinates"], ax_sic)
elif sic_gj["type"] == "MultiPolygon":
    for poly in sic_gj["coordinates"]:
        draw(poly, ax_sic)
for x in np.arange(12.5, 15.6, 0.5): ax_sic.axvline(x, color="#f0f0ee", lw=0.5, zorder=0)
for y in np.arange(36.5, 38.6, 0.5): ax_sic.axhline(y, color="#f0f0ee", lw=0.5, zorder=0)

for cls in ["Low", "Medium", "High"]:
    s = df[df.cls == cls]
    ax.scatter(s.lon, s.lat, marker="o", s=78, c=CLASS_COLOR[cls],
               edgecolors="white", linewidths=0.8, zorder=5, alpha=0.95)

ax.set_xlim(-125, 180); ax.set_ylim(-52, 62); ax.set_aspect("equal")
ax.set_xlabel("Longitude ($^{\\circ}$)", color=MUTED, fontsize=8, labelpad=16)
ax.set_ylabel("Latitude ($^{\\circ}$)", color=MUTED, fontsize=8)
ax.tick_params(colors=MUTED, labelsize=7)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.set_title("Global pilot reservoirs (n = %d) by shoreline compactness" % len(df),
             color=INK, fontsize=11, loc="left", pad=8)
ax.text(0.005, -0.22, "A/P classes: Low < %d m  ·  Medium %d–%d m  ·  High ≥ %d m. "
        "Basemap: Natural Earth." % (LOW_MAX, LOW_MAX, HIGH_MIN, HIGH_MIN),
        transform=ax.transAxes, fontsize=6.8, color=MUTED)

# ---- Sicily inset ----
for cls in ["Low", "Medium", "High"]:
    s = sicily[sicily.cls == cls]
    ax_sic.scatter(s.lon, s.lat, marker="o", s=60, c=CLASS_COLOR[cls],
                   edgecolors="white", linewidths=0.8, zorder=5, alpha=0.95)
LABEL_OFFSET = {"Pozzillo": (3.5, -7.5)}
for _, r in sicily.iterrows():
    dx, dy = LABEL_OFFSET.get(r["name"], (3.5, 3.5))
    ax_sic.annotate(r["name"], (r.lon, r.lat), fontsize=6.3, color=INK,
                     xytext=(dx, dy), textcoords="offset points", zorder=6)
ax_sic.set_xlim(12.3, 15.8); ax_sic.set_ylim(36.5, 38.6); ax_sic.set_aspect("equal")
ax_sic.set_title("Sicilian near-truth set (n = 4)", color=INK, fontsize=8.5,
                  loc="left", pad=4)
ax_sic.tick_params(colors=MUTED, labelsize=6)
ax_sic.set_xticks([13, 14, 15]); ax_sic.set_yticks([37, 38])
for sp in ["top", "right"]: ax_sic.spines[sp].set_visible(False)

# ---- legend panel (shares the right-hand column with the Sicily inset) ----
ax_leg.axis("off")
leg = [Line2D([0], [0], marker="o", ls="", mfc=CLASS_COLOR[c], mec="white", ms=10, label=lab)
       for c, lab in [("Low", "Low   (< 100 m)"), ("Medium", "Medium   (100–200 m)"),
                      ("High", "High   (≥ 200 m)")]]
ax_leg.legend(handles=leg, title="Shoreline A/P (static)", loc="upper left",
              frameon=False, fontsize=8.5, title_fontsize=9)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTDIR, f"study_area_map.{ext}"), dpi=300,
                bbox_inches="tight", facecolor=SURFACE)
print("saved ->", os.path.join(OUTDIR, "study_area_map.{png,pdf}"))
