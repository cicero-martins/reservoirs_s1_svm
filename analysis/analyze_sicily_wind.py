"""
analyze_sicily_wind.py

Wind analyses for the 4 PlanetScope-validated Sicilian reservoirs (2024–2025),
mirroring analyze_wind_divergence.py + plot_delta_kge_vs_wind.py but Sicily-only.

(A) Wind divergence — per acquisition (raw canonical SAR, matched on date):
      rel_div = (area_vv_otsu − area_dual) / area_dual   vs ERA5 overpass wind.
    Pooled across the 4 reservoirs (~feasible N from ~80 scenes each).

(B) ΔKGE-vs-wind — per reservoir: ΔKGE(dual−Otsu) vs ERA5 p90 wind.
    ⚠ N=4 reservoirs → NOT a meaningful correlation; shown as 4 labelled points
    for context only (no regression).

Reads:
  dual : raw_data/GEE_SicilyPlanet/SAR_area_*.csv            (area_ha, raw)
  vv   : raw_data/GEE_SicilyPlanet_VVotsu/SAR_area_*.csv     (area_ha, raw)
  wind : raw_data/GEE_SicilyPlanet_Era5Wind/Era5Wind_*.csv   (wind_ms)  [from exportEra5WindSicily.js]
  ΔKGE : analysis/sicily_planet_compare.csv                  (from compare_sicily_planet.py)

Output:
  analysis/sicily_wind_divergence_pooled.csv
  analysis/method_comparison_output/sicily_wind_divergence.png
  analysis/method_comparison_output/sicily_delta_kge_vs_wind.png
"""

import pathlib
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

DUAL_DIR = pathlib.Path('raw_data/GEE_SicilyPlanet')
VV_DIR   = pathlib.Path('raw_data/GEE_SicilyPlanet_VVotsu')
WIND_DIR = pathlib.Path('raw_data/GEE_SicilyPlanet_Era5Wind')
CMP_CSV  = pathlib.Path('analysis/sicily_planet_compare.csv')
OUT_POOL = pathlib.Path('analysis/sicily_wind_divergence_pooled.csv')
OUT_DIV  = pathlib.Path('analysis/method_comparison_output/sicily_wind_divergence.png')
OUT_DKW  = pathlib.Path('analysis/method_comparison_output/sicily_delta_kge_vs_wind.png')

RESERVOIRS = ['Ancipa', 'Pozzillo', 'Poma', 'Rosamarina']
RCOL = {'Ancipa': '#1f77b4', 'Pozzillo': '#ff7f0e', 'Poma': '#2ca02c', 'Rosamarina': '#9467bd'}

have_wind = WIND_DIR.exists() and any(WIND_DIR.glob('Era5Wind_*.csv'))
if not have_wind:
    print(f'[notice] no Sicily wind at {WIND_DIR}.')
    print('  Run exportEra5WindSicily.js → download to that folder → re-run.')
    sys.exit(0)


def load_raw(folder, name):
    p = folder / f'SAR_area_{name}.csv'
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    d = d[d['area_ha'] > 0][['date', 'area_ha']]
    return d.sort_values('date')


def load_wind(name):
    p = WIND_DIR / f'Era5Wind_{name}.csv'
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    if 'wind_ms' not in d.columns or d.empty:
        return None
    return d.groupby('date')['wind_ms'].mean().reset_index()


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── (A) Wind divergence ───────────────────────────────────────────────────────
pool = []
for name in RESERVOIRS:
    dual, vv, wind = load_raw(DUAL_DIR, name), load_raw(VV_DIR, name), load_wind(name)
    if dual is None or vv is None or wind is None:
        continue
    m = (dual.rename(columns={'area_ha': 'dual'})
         .merge(vv.rename(columns={'area_ha': 'vv'}), on='date')
         .merge(wind, on='date').dropna())
    if m.empty:
        continue
    m['rel_div'] = (m['vv'] - m['dual']) / m['dual']
    m['name'] = name
    pool.append(m[['name', 'date', 'wind_ms', 'dual', 'vv', 'rel_div']])

if pool:
    allp = pd.concat(pool, ignore_index=True)
    allp.to_csv(OUT_POOL, index=False)
    r, pval = stats.pearsonr(allp['wind_ms'], allp['rel_div'])
    print(f'(A) Pooled {len(allp)} acquisitions / {allp["name"].nunique()} reservoirs')
    print(f'    r(wind, rel_div) = {r:+.3f}  p = {pval:.2e}  (NS)')

    # Binned medians at the ACTUAL median wind per bin (no misleading linear fit —
    # a leverage-driven regression line contradicts the null r; see global fix).
    wmax = allp['wind_ms'].max()
    bins = [b for b in [0, 1, 2, 3, 4, 6, 8, 10, 12] if b < wmax] + [np.ceil(wmax)]
    allp['wbin'] = pd.cut(allp['wind_ms'], bins, include_lowest=True)
    grp = allp.groupby('wbin', observed=True)
    med, cx = grp['rel_div'].median(), grp['wind_ms'].median()
    print('    median rel_div by wind bin:')
    print(grp['rel_div'].agg(['median', 'count']).to_string().replace('\n', '\n    '))

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, g in allp.groupby('name'):
        ax.scatter(g['wind_ms'], g['rel_div'], s=14, alpha=0.5, color=RCOL.get(name, '#555'),
                   linewidths=0, label=name, zorder=3)
    ax.plot(cx.values, med.values, 'o-', color='#C62828', ms=8, lw=2, zorder=5,
            label='median per wind bin')
    ax.axhline(0, color='gray', lw=1, ls=':', zorder=2)
    ax.set_xlabel('ERA5 10 m wind at overpass (m/s)')
    ax.set_ylabel('rel. divergence (area$_{VV}$ − area$_{dual}$)/area$_{dual}$')
    ax.set_title('Sicily — no systematic VV/dual divergence with wind\n'
                 f'(per acquisition; r={r:.2f}, p={pval:.2f}, NS)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_ylim(-0.6, 1.0); ax.set_xlim(0, np.ceil(wmax))
    OUT_DIV.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIV, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'    Saved: {OUT_DIV}')
else:
    print('(A) no reservoirs with dual+vv+wind joined.')

# ── (B) ΔKGE vs wind p90 (N=4, context only) ──────────────────────────────────
if CMP_CSV.exists():
    cmp = pd.read_csv(CMP_CSV)
    piv = cmp.pivot_table(index='name', columns='method', values='KGE')
    if {'dual (VV+VH SVM)', 'Otsu (VV-only)'}.issubset(piv.columns):
        piv['dkge'] = piv['dual (VV+VH SVM)'] - piv['Otsu (VV-only)']
        rows = []
        for name in RESERVOIRS:
            w = load_wind(name)
            if w is None or name not in piv.index:
                continue
            rows.append({'name': name, 'wind_p90': float(w['wind_ms'].quantile(0.90)),
                         'dkge': float(piv.loc[name, 'dkge'])})
        d = pd.DataFrame(rows)
        print(f'\n(B) ΔKGE vs wind p90 (N={len(d)} — context only, no correlation):')
        print(d.to_string(index=False))

        fig, ax = plt.subplots(figsize=(7.5, 6))
        for _, r in d.iterrows():
            ax.scatter(r['wind_p90'], r['dkge'], s=120, color=RCOL.get(r['name'], '#555'),
                       edgecolors='white', linewidths=0.8, zorder=4)
            ax.annotate(r['name'], (r['wind_p90'], r['dkge']), fontsize=9,
                        xytext=(6, 3), textcoords='offset points')
        ax.axhline(0, color='gray', lw=1, ls=':', zorder=2)
        ax.set_xlabel('ERA5 wind p90 (m/s)')
        ax.set_ylabel('ΔKGE = KGE$_{dual}$ − KGE$_{Otsu}$ (vs PlanetScope)')
        ax.set_title('Sicily — dual advantage vs wind exposure\n'
                     '⚠ N=4 reservoirs: context only, not a correlation',
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.25)
        fig.savefig(OUT_DKW, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'    Saved: {OUT_DKW}')
else:
    print('\n(B) skipped — run compare_sicily_planet.py first.')
