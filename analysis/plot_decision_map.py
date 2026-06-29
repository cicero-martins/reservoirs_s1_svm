"""
plot_decision_map.py  —  the 2D decision map (synthesis figure).

Two orthogonal axes for SAR reservoir monitoring:
  x = A/P static (geometry — universal, sensor-agnostic mixed-pixel mechanism)
  y = wind exposure (ERA5 p90 wind speed — radiometric, SAR-specific)

Each reservoir is colored by which method is adequate (KGE >= 0.5):
  - neither      : below the geometric floor (low A/P) — no method works
  - VV enough    : adequate with cheap VV-only Otsu — complexity unjustified
  - dual needed  : only VV+VH SVM is adequate — complexity justified (typically high wind)

Reads:
  analysis/pilot_kge_compare.csv          (kge_dual, kge_vv, ap_m)
  raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind/Era5Wind_*.csv  (wind exposure)

Output: analysis/schwatke_output/decision_map.png

Skeleton: guards missing VV / wind data.
"""

import pathlib
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

CMP_CSV   = pathlib.Path('analysis/pilot_kge_compare.csv')
WIND_DIR  = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind')
OUT_PNG   = pathlib.Path('analysis/schwatke_output/decision_map.png')
KGE_THRESH = 0.5
AP_THRESH  = 118     # Youden-J from the v3+v4 classification figure

if not CMP_CSV.exists():
    sys.exit('Run compute_kge_compare.py first.')

df = pd.read_csv(CMP_CSV)

have_vv   = df['kge_vv'].notna().any()
have_wind = WIND_DIR.exists() and any(WIND_DIR.glob('Era5Wind_*.csv'))
if not (have_vv and have_wind):
    print('[notice] need both VV_OTSU KGE and ERA5 wind.')
    print(f'  VV KGE present: {have_vv}')
    print(f'  wind at {WIND_DIR}: {"OK" if have_wind else "MISSING"}')
    sys.exit(0)


def wind_p90(name):
    p = WIND_DIR / f'Era5Wind_{name}.csv'
    if not p.exists():
        return np.nan
    try:
        w = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return np.nan
    if 'wind_ms' not in w.columns or w.empty:
        return np.nan
    return float(w['wind_ms'].quantile(0.90))


df['wind_p90'] = df['name'].apply(wind_p90)
df = df.dropna(subset=['ap_m', 'kge_dual', 'kge_vv', 'wind_p90']).reset_index(drop=True)

# ── Regime per reservoir ──────────────────────────────────────────────────────
def regime(row):
    dual_ok = row['kge_dual'] >= KGE_THRESH
    vv_ok   = row['kge_vv']   >= KGE_THRESH
    if vv_ok:
        return 'VV enough'
    if dual_ok:
        return 'dual needed'
    return 'neither'

df['regime'] = df.apply(regime, axis=1)
COLORS = {'VV enough': '#2ca02c', 'dual needed': '#1f77b4', 'neither': '#d62728'}

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(10, 7))

for reg, grp in df.groupby('regime'):
    ax.scatter(grp['ap_m'], grp['wind_p90'], s=70, color=COLORS[reg],
               edgecolors='white', linewidths=0.6, alpha=0.9, zorder=4, label=reg)

for _, r in df.iterrows():
    ax.annotate(r['name'].replace('_', ' '), (r['ap_m'], r['wind_p90']),
                fontsize=6, xytext=(4, 3), textcoords='offset points', color='#444')

ax.axvline(AP_THRESH, color='navy', lw=1.5, ls='--', alpha=0.7,
           label=f'A/P geometric floor ({AP_THRESH} m)')

ax.set_xlabel('A/P static (m)  —  geometry (universal)', fontsize=10)
ax.set_ylabel('ERA5 wind exposure, p90 (m/s)  —  radiometric (SAR-specific)', fontsize=10)
ax.set_title('Decision map: when is dual-pol SAR worth its cost?', fontsize=11, fontweight='bold')

handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=9, label=k)
           for k, c in COLORS.items()]
handles.append(Line2D([0], [0], color='navy', lw=1.5, ls='--', label=f'A/P floor ({AP_THRESH} m)'))
ax.legend(handles=handles, fontsize=9, loc='best')
ax.grid(alpha=0.25)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_PNG}')
print(f'\nRegime counts:\n{df["regime"].value_counts().to_string()}')
