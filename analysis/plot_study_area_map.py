"""Study-area world map: global pilot reservoirs coloured by A/P class.

Core set (28, v4)      -> circles ; from pilot_kge_compare.csv
Complementary set (10) -> triangles ; from pilot_kge_v3.csv (high-A/P tail + tropics)
Colour = static A/P (m), 3 ordered classes (sequential orange ramp).

Coordinates are parsed from exportGlobalPilotV4.js so there is a single source
of truth. Basemap = Natural Earth 110m countries (downloaded to scratchpad).
Outputs PNG (300 dpi) + PDF to ../manuscript/figures/.
"""
import re, json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = (r"C:/Users/Unipa/AppData/Local/Temp/claude/"
           r"c--Users-Unipa-Documents-reservoirs-s1-svm/"
           r"db674b5e-ba1b-4e18-8572-ecd885395938/scratchpad")
GEOJSON = os.path.join(SCRATCH, "ne_110m_countries.geojson")
OUTDIR = os.path.normpath(os.path.join(HERE, "..", "manuscript", "figures"))
os.makedirs(OUTDIR, exist_ok=True)

# ── A/P class thresholds (m) and sequential palette (validated, CVD ok) ──────
LOW_MAX, HIGH_MIN = 120.0, 250.0          # <120 low | 120-250 med | >=250 high
CLASS_COLOR = {"Low": "#f88f4d", "Medium": "#d64a02", "High": "#8a2d04"}
LAND = "#e7e6e1"; LAND_EDGE = "#ffffff"; SURFACE = "#ffffff"
INK = "#1b1b1b"; MUTED = "#6b6b6b"

def ap_class(ap):
    return "Low" if ap < LOW_MAX else ("Medium" if ap < HIGH_MIN else "High")

# ── 1. coordinates from the export script ───────────────────────────────────
js = open(os.path.join(HERE, "exportGlobalPilotV4.js"), encoding="utf-8").read()
coord = {}
for m in re.finditer(r"\[\s*'([A-Za-z_]+)'\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", js):
    coord.setdefault(m.group(1), (float(m.group(2)), float(m.group(3))))  # first hit wins

# ── 2. A/P + attributes for the two sets ────────────────────────────────────
core = pd.read_csv(os.path.join(HERE, "pilot_kge_compare.csv"))[["name", "ap_m", "country", "climate"]]
core["group"] = "Core (28)"
comp = pd.read_csv(os.path.join(HERE, "pilot_kge_v3.csv"))[["name", "ap_m"]]
comp = comp[comp["name"].isin(["Yesa","Caia","Forggen","Garcia","Hubbard_Creek",
                               "Harlan_County","Umbuluzi","Erfenis","Paraibuna","Contas"])].copy()
comp["group"] = "Complementary (10)"
df = pd.concat([core, comp], ignore_index=True)

df["lat"] = df["name"].map(lambda n: coord.get(n, (np.nan, np.nan))[0])
df["lon"] = df["name"].map(lambda n: coord.get(n, (np.nan, np.nan))[1])
missing = df[df["lat"].isna()]["name"].tolist()
if missing:
    print("WARN missing coords:", missing)
df["cls"] = df["ap_m"].map(ap_class)

print("N =", len(df), "| classes:", df["cls"].value_counts().to_dict())
print("A/P range: %.0f-%.0f m" % (df.ap_m.min(), df.ap_m.max()))

# ── 3. figure ───────────────────────────────────────────────────────────────
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": "#cccccc", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=300)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

gj = json.load(open(GEOJSON, encoding="utf-8"))
def draw(coords):
    for ring in coords:
        arr = np.asarray(ring)
        ax.fill(arr[:, 0], arr[:, 1], facecolor=LAND, edgecolor=LAND_EDGE,
                linewidth=0.4, zorder=1)
for feat in gj["features"]:
    g = feat["geometry"];
    if g["type"] == "Polygon":
        draw(g["coordinates"])
    elif g["type"] == "MultiPolygon":
        for poly in g["coordinates"]:
            draw(poly)

# graticule (recessive)
for x in range(-120, 181, 30):
    ax.axvline(x, color="#f0f0ee", lw=0.6, zorder=0)
for y in range(-40, 61, 20):
    ax.axhline(y, color="#f0f0ee", lw=0.6, zorder=0)

for cls in ["Low", "Medium", "High"]:
    s = df[df.cls == cls]
    ax.scatter(s.lon, s.lat, marker="o", s=85, c=CLASS_COLOR[cls],
               edgecolors="white", linewidths=0.9, zorder=5, alpha=0.95)

ax.set_xlim(-130, 160); ax.set_ylim(-45, 60); ax.set_aspect("equal")
ax.set_xlabel("Longitude ($^{\\circ}$)", color=MUTED, fontsize=8)
ax.set_ylabel("Latitude ($^{\\circ}$)", color=MUTED, fontsize=8)
ax.tick_params(colors=MUTED, labelsize=7)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# single legend: colour = static A/P class
leg1 = [Line2D([0], [0], marker="o", ls="", mfc=CLASS_COLOR[c], mec="white",
               ms=10, label=lab)
        for c, lab in [("Low", "Low   (< 120 m)"),
                       ("Medium", "Medium   (120–250 m)"),
                       ("High", "High   (≥ 250 m)")]]
ax.legend(handles=leg1, title="Shoreline A/P (static)", loc="lower left",
          frameon=False, fontsize=8.5, title_fontsize=9,
          bbox_to_anchor=(0.005, 0.02))

ax.set_title("Global pilot reservoirs (n = %d) by shoreline compactness"
             % len(df), color=INK, fontsize=11, loc="left", pad=8)
ax.text(0.005, -0.13, "A/P classes: Low < %d m  ·  Medium %d–%d m  ·  "
        "High ≥ %d m. Basemap: Natural Earth." % (LOW_MAX, LOW_MAX, HIGH_MIN, HIGH_MIN),
        transform=ax.transAxes, fontsize=6.8, color=MUTED)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTDIR, f"study_area_map.{ext}"), dpi=300,
                bbox_inches="tight", facecolor=SURFACE)
print("saved ->", os.path.join(OUTDIR, "study_area_map.{png,pdf}"))
