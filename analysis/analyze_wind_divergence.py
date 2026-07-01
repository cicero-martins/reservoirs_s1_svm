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
  analysis/method_comparison_output/wind_divergence.png

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
OUT_PNG      = pathlib.Path('analysis/method_comparison_output/wind_divergence.png')

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

# Binned medians. Edges follow the real data range (ERA5 overpass winds top out
# ~10 m/s here); an open-ended top bin like (8, 100] is avoided because its
# midpoint (54) would plot a phantom marker far beyond any observed wind.
wmax = allp['wind_ms'].max()
bins = [b for b in [0, 1, 2, 3, 4, 6, 8, 10, 12] if b < wmax] + [np.ceil(wmax)]
allp['wbin'] = pd.cut(allp['wind_ms'], bins, include_lowest=True)
binned = allp.groupby('wbin', observed=True)['rel_div'].agg(['median', 'count'])
print('\nMedian rel_div by wind bin:')
print(binned.to_string())

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
# binned medians — x is the ACTUAL median wind within each bin (data-driven),
# not the interval midpoint, so markers always sit on observed wind values.
grp     = allp.groupby('wbin', observed=True)
med     = grp['rel_div'].median()
centers = grp['wind_ms'].median().values
ax.plot(centers, med.values, 'o-', color='#C62828', ms=8, lw=2, zorder=5,
        label='median per wind bin')
ax.axhline(0, color='gray', lw=1, ls=':', alpha=0.7)
ax.set_xlabel('ERA5 10 m wind speed at overpass (m/s)')
ax.set_ylabel('rel. divergence  (area$_{VV}$ − area$_{dual}$) / area$_{dual}$')
ax.set_title('Experiment C — no systematic VV/dual divergence with wind\n'
             f'(pooled r={r:.2f}, p={pval:.2f}, NS; dual-pol as reference)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.25)
ax.set_ylim(-1.0, 0.6)
ax.set_xlim(0, np.ceil(allp['wind_ms'].max()))
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT_PNG}')

# ── Per-reservoir panel ───────────────────────────────────────────────────────
# Within-reservoir Pearson r(wind, rel_div): tests the Bragg under-detection
# hypothesis site by site (it predicts r < 0). A pooled null could hide per-site
# effects, so this resolves them. Significant r<0 = hypothesis supported there.
OUT_PNG2 = pathlib.Path('analysis/method_comparison_output/wind_divergence_per_reservoir.png')
MIN_N_PR = 10
prs = []
for name, g in allp.groupby('name'):
    if len(g) < MIN_N_PR or g['wind_ms'].std() == 0:
        continue
    rr, pp = stats.pearsonr(g['wind_ms'], g['rel_div'])
    prs.append({'name': name, 'n': len(g), 'wmax': g['wind_ms'].max(),
                'r': rr, 'p': pp})
prdf = pd.DataFrame(prs).sort_values('r').reset_index(drop=True)

n_neg = int(((prdf['r'] < 0) & (prdf['p'] < 0.05)).sum())
n_pos = int(((prdf['r'] > 0) & (prdf['p'] < 0.05)).sum())

def _bar_color(row):
    if row['p'] >= 0.05:
        return '#9e9e9e'                       # not significant
    return '#C62828' if row['r'] < 0 else '#2E7D32'  # red=Bragg, green=opposite

colors = prdf.apply(_bar_color, axis=1)
ypos   = np.arange(len(prdf))

fig2, ax2 = plt.subplots(figsize=(9, 9))
ax2.barh(ypos, prdf['r'], color=colors, edgecolor='white', height=0.7, zorder=3)
ax2.axvline(0, color='k', lw=1, zorder=4)
ax2.set_yticks(ypos)
ax2.set_yticklabels([f"{n.replace('_', ' ')}  (n={nn}, w$_{{max}}$={w:.1f})"
                     for n, nn, w in zip(prdf['name'], prdf['n'], prdf['wmax'])],
                    fontsize=8)
ax2.set_ylim(-0.6, len(prdf) - 0.4)
ax2.set_xlabel('within-reservoir Pearson r(wind, rel. divergence)', fontsize=10)
ax2.set_title('Per-reservoir wind effect on VV/dual divergence\n'
              'Bragg hypothesis predicts r < 0 (red) — observed in '
              f'{n_neg}/{len(prdf)}; opposite sign (green) in {n_pos}/{len(prdf)}',
              fontsize=10, fontweight='bold')
# significance stars
for y, (_, row) in zip(ypos, prdf.iterrows()):
    if row['p'] < 0.05:
        dx = 0.012 if row['r'] >= 0 else -0.012
        ax2.text(row['r'] + dx, y, '*', va='center',
                 ha='left' if row['r'] >= 0 else 'right', fontsize=12, color='#222')
from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(color='#C62828', label='r<0 sig. (supports Bragg)'),
    Patch(color='#2E7D32', label='r>0 sig. (opposite)'),
    Patch(color='#9e9e9e', label='not significant (p≥0.05)')],
    fontsize=8, loc='lower right')
ax2.grid(axis='x', alpha=0.25)
fig2.tight_layout()
fig2.savefig(OUT_PNG2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {OUT_PNG2}')
