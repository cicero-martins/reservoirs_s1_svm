"""
analyze_wind_divergence.py  —  Experiment C (wind mechanism).

Tests whether the VV-only vs dual-pol classification divergence grows with wind.
Per acquisition (same date, same selected orbit for both methods), treating the
wind-robust dual-pol as the reference:

    rel_div(t) = (area_vv(t) − area_dual(t)) / area_dual(t)

Physical expectation: as wind rises, water VV backscatter rises (Bragg), Otsu loses the
water mode → VV under-detects → rel_div becomes increasingly negative with wind speed.

Reads:
  dual : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4/SAR_area_*.csv
  vv   : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu/SAR_area_*.csv
  wind : raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind/Era5Wind_*.csv

Output:
  analysis/wind_divergence_pooled.csv
  analysis/schwatke_output/wind_divergence.png

Skeleton: guards missing VV/wind data and exits cleanly with a notice.
"""

import pathlib
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

SAR_DUAL_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
SAR_VV_DIR   = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu')
WIND_DIR     = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind')
OUT_CSV      = pathlib.Path('analysis/wind_divergence_pooled.csv')
OUT_PNG      = pathlib.Path('analysis/schwatke_output/wind_divergence.png')

EXCLUDE  = {'Oued_Makhazine', 'Guajaraz', 'Antero', 'Miyagase', 'Welbedacht', 'Tzaneen'}
AREA_MIN = {'Saint_Cassien': 200}
SAR_MIN_FRAC = 0.02

cand  = pd.read_csv('analysis/global_pilot_v4_candidates.csv')
NAMES = [n for n in cand['name'].tolist() if n not in EXCLUDE]


def load_acq(name, sar_dir):
    """Per-acquisition area (raw, no smoothing) keyed by date."""
    p = sar_dir / f'SAR_area_{name}.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    df = df[df['area_ha'] > 0].copy()
    if name in AREA_MIN:
        df = df[df['area_ha'] >= AREA_MIN[name]].copy()
    if df.empty:
        return None
    p99 = df['area_ha'].quantile(0.99)
    df  = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    return df[['date', 'area_ha']].rename(columns={'area_ha': 'area'})


def load_wind(name):
    p = WIND_DIR / f'Era5Wind_{name}.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    if df.empty or 'wind_ms' not in df.columns:
        return None
    # one wind value per date (mean if duplicate overpasses)
    return df.groupby('date')['wind_ms'].mean().reset_index()


have_vv   = SAR_VV_DIR.exists() and any(SAR_VV_DIR.glob('SAR_area_*.csv'))
have_wind = WIND_DIR.exists()   and any(WIND_DIR.glob('Era5Wind_*.csv'))
if not (have_vv and have_wind):
    print('[notice] need both VV_OTSU and ERA5 wind data.')
    print(f'  VV at   {SAR_VV_DIR}: {"OK" if have_vv else "MISSING"}')
    print(f'  wind at {WIND_DIR}: {"OK" if have_wind else "MISSING"}')
    sys.exit(0)

# ── Pool per-acquisition divergence + wind across reservoirs ──────────────────
pool = []
for name in NAMES:
    dual = load_acq(name, SAR_DUAL_DIR)
    vv   = load_acq(name, SAR_VV_DIR)
    wind = load_wind(name)
    if dual is None or vv is None or wind is None:
        continue
    m = (dual.rename(columns={'area': 'dual'})
         .merge(vv.rename(columns={'area': 'vv'}), on='date')
         .merge(wind, on='date').dropna())
    if m.empty:
        continue
    m['rel_div'] = (m['vv'] - m['dual']) / m['dual']
    m['name'] = name
    pool.append(m[['name', 'date', 'wind_ms', 'dual', 'vv', 'rel_div']])

if not pool:
    sys.exit('No reservoirs with all three series joined.')

allp = pd.concat(pool, ignore_index=True)
allp.to_csv(OUT_CSV, index=False)
print(f'Pooled {len(allp)} acquisitions across {allp["name"].nunique()} reservoirs -> {OUT_CSV}')

r, pval = stats.pearsonr(allp['wind_ms'], allp['rel_div'])
print(f'\nPearson r(wind, rel_div) = {r:+.3f}  p = {pval:.2e}')
slope, icpt, *_ = stats.linregress(allp['wind_ms'], allp['rel_div'])
print(f'Slope = {slope:+.4f} per m/s  (rel. VV under-detection per unit wind)')

# Binned medians
bins = [0, 2, 4, 6, 8, 100]
allp['wbin'] = pd.cut(allp['wind_ms'], bins)
print('\nMedian rel_div by wind bin:')
print(allp.groupby('wbin')['rel_div'].agg(['median', 'count']).to_string())

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(allp['wind_ms'], allp['rel_div'], s=6, alpha=0.15, color='#1565C0',
           linewidths=0, zorder=2)
xf = np.linspace(allp['wind_ms'].min(), allp['wind_ms'].max(), 100)
ax.plot(xf, slope * xf + icpt, 'k-', lw=2, zorder=4,
        label=f'linear: r={r:.2f}, p={pval:.1e}')
# binned medians
med = allp.groupby('wbin')['rel_div'].median()
cnt = allp.groupby('wbin')['rel_div'].count()
centers = [iv.mid for iv in med.index]
ax.plot(centers, med.values, 'o-', color='#C62828', ms=8, lw=2, zorder=5,
        label='median per wind bin')
ax.axhline(0, color='gray', lw=1, ls=':', alpha=0.7)
ax.set_xlabel('ERA5 10 m wind speed at overpass (m/s)')
ax.set_ylabel('rel. divergence  (area$_{VV}$ − area$_{dual}$) / area$_{dual}$')
ax.set_title('Experiment C — VV-only under-detection grows with wind\n'
             '(dual-pol as wind-robust reference)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.set_ylim(-1.0, 0.6)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT_PNG}')
