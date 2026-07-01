"""
compare_sicily_planet.py

Sicily method comparison anchored to PlanetScope 3 m (near-truth), NOT JRC 30 m —
this breaks the inter-product circularity of the 28-reservoir global pilot.

For each of the 4 PlanetScope-validated Sicilian reservoirs, three SAR water-area
series (already exported, 2024-05–2025-05) are matched to the nearest PlanetScope
acquisition (±TOL_DAYS) and scored against it:
  Otsu      = VV-only per-scene Otsu        (Tier 1)   area_Otsu_Invaso_*.csv
  dual      = VV+VH SVM                      (Tier 3)   area_SVM_VVpVH_Invaso_*.csv
  vv_svm    = VV-only SVM                    (extra)    area_SVM_VVonly_Invaso_*.csv

Reference (ground truth):
  validation_data/statistics/area_statistics/{name}Planet.csv   (cols: data, area[ha])

Metrics per method per reservoir: KGE, Pearson r, RMSE (ha), bias (mean sim−obs).
ΔKGE = KGE_dual − KGE_otsu mirrors compute_kge_compare.py but vs PlanetScope.

Output:
  analysis/sicily_planet_compare.csv
  analysis/method_comparison_output/sicily_planet_validation.png
"""

import pathlib
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

SAR_DIR    = pathlib.Path('validation_data/GEEvalidation-extracted')   # legacy (raw, off-pipeline)
CANON_DUAL = pathlib.Path('raw_data/GEE_SicilyPlanet')                 # canonical SVM (if exported)
CANON_VV   = pathlib.Path('raw_data/GEE_SicilyPlanet_VVotsu')          # canonical Otsu (if exported)
PLANET_DIR = pathlib.Path('validation_data/statistics/area_statistics')
OUT_CSV    = pathlib.Path('analysis/sicily_planet_compare.csv')
OUT_PNG    = pathlib.Path('analysis/method_comparison_output/sicily_planet_validation.png')

TOL_DAYS = 6   # match each SAR scene to nearest PlanetScope obs within ±this

RESERVOIRS = ['Ancipa', 'Pozzillo', 'Poma', 'Rosamarina']
# label : (legacy filename token, legacy area column)
METHODS = {
    'Otsu (VV-only)':   ('Otsu',      'areaOtsu_ha'),
    'dual (VV+VH SVM)': ('SVM_VVpVH', 'areaSVM_ha'),
    'VV-only SVM':      ('SVM_VVonly', 'areaVVonly_ha'),
}
MCOL = {'Otsu (VV-only)': '#2ca02c', 'dual (VV+VH SVM)': '#1f77b4', 'VV-only SVM': '#9467bd'}

# Prefer the canonical re-export (same pipeline as the 28-reservoir global pilot)
# when present; it is clean+smoothed to match the global ΔKGE. Otherwise fall back
# to the legacy GEEvalidation series (raw, off-pipeline) with a loud notice.
USE_CANON = CANON_DUAL.exists() and any(CANON_DUAL.glob('SAR_area_*.csv'))


# ── clean+smooth (identical to compute_kge_compare.py) — applied to canonical SAR ─
def _remove_global(s, threshold=2.0):
    m, sd = s.mean(), s.std()
    return s[np.abs(s - m) <= threshold * sd]

def _remove_local(s, window=5, threshold=1.5):
    arr, idx = s.values.copy(), s.index.tolist()
    keep, half = [], window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        win = arr[lo:hi]
        m, sd = win.mean(), win.std()
        if sd == 0 or abs(arr[i] - m) <= threshold * sd:
            keep.append(idx[i])
    return s.loc[keep]

def _lowess(dates, values, window_days=20, bandwidth=7):
    out = []
    for t0 in dates:
        dd = np.abs((dates - t0).dt.total_seconds().values / 86400)
        mask = dd <= window_days
        w = np.exp(-(dd[mask] / bandwidth) ** 2)
        out.append(float((values[mask] * w).sum() / w.sum()))
    return np.array(out)

def clean_and_smooth(df):
    s = df.set_index('date')['sar'].copy().reset_index(drop=True)
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 10, 1.5)
    dates_s = df['date'].reset_index(drop=True).loc[s.index].reset_index(drop=True)
    sm = _lowess(dates_s, s.reset_index(drop=True), 20, 7)
    return pd.DataFrame({'date': dates_s, 'sar': sm})


def kge(obs, sim):
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan, np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), r


def parse_dates(series):
    """PlanetScope files use mixed formats: ISO 'YYYY-MM-DD' (Rosamarina) and
    'DD-Mon-YY' (the other three). Do NOT pass dayfirst=True — it silently fails
    on ISO dates whose day > 12 (e.g. '2024-05-17' → NaT), which dropped 26 of
    Rosamarina's 38 points. Default parsing handles ISO correctly AND 'DD-Mon-YY'
    (the month name is unambiguous); a %d-%b-%y pass backfills any stragglers."""
    dt = pd.to_datetime(series, errors='coerce', format='ISO8601')
    if dt.isna().any():
        dt2 = pd.to_datetime(series, errors='coerce', format='%d-%b-%y')
        dt = dt.fillna(dt2)
    return dt


def load_planet(name):
    p = PLANET_DIR / f'{name.lower()}Planet.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d.columns = [c.strip().lower().replace('﻿', '') for c in d.columns]
    d['date'] = parse_dates(d['data'])
    d = d.dropna(subset=['date']).sort_values('date')
    return d[['date', 'area']].rename(columns={'area': 'planet'})


def _load_canon(folder, name):
    """Canonical export (global-pipeline format): cols date, area_ha → clean+smooth."""
    p = folder / f'SAR_area_{name}.csv'
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    d = d[d['area_ha'] > 0][['date', 'area_ha']].rename(columns={'area_ha': 'sar'})
    d = d.sort_values('date').reset_index(drop=True)
    if len(d) < 5:
        return d
    return clean_and_smooth(d)


def load_sar(name, label, token, col):
    """Prefer canonical re-export for the two experiment methods (dual, Otsu);
    VV-only SVM exists only in the legacy set. Legacy series are returned raw."""
    if USE_CANON:
        if label == 'dual (VV+VH SVM)':
            c = _load_canon(CANON_DUAL, name)
            if c is not None and not c.empty:
                return c
        elif label == 'Otsu (VV-only)':
            c = _load_canon(CANON_VV, name)
            if c is not None and not c.empty:
                return c
    p = SAR_DIR / f'area_{token}_Invaso_{name}.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d['date'] = pd.to_datetime(d['data'], errors='coerce')
    d = d.dropna(subset=['date']).sort_values('date')
    return d[['date', col]].rename(columns={col: 'sar'})


def match_nearest(sar, planet, tol_days):
    """One-to-one date match, NO monthly averaging: each PlanetScope truth obs is
    paired to its single nearest SAR scene within ±tol_days (Planet is the left
    table, so every truth point in 2024-25 is evaluated; unmatched ones are dropped).
    direction='nearest' + tolerance enforces a 1:1 pairing per Planet date."""
    p = planet.copy(); s = sar.copy()
    p['date'] = p['date'].astype('datetime64[ns]')   # unify resolution (Planet [s] vs SAR [us])
    s['date'] = s['date'].astype('datetime64[ns]')
    m = pd.merge_asof(p.sort_values('date'), s.sort_values('date'),
                      on='date', direction='nearest',
                      tolerance=pd.Timedelta(days=tol_days)).dropna(subset=['sar'])
    return m


if USE_CANON:
    print('[source] CANONICAL re-export (raw_data/GEE_SicilyPlanet*) — clean+smoothed, '
          'pipeline-consistent with the 28-reservoir global pilot.\n')
else:
    print('[source] ⚠ LEGACY validation_data/GEEvalidation-extracted — RAW, off-pipeline '
          '(not clean+smoothed; different AOI/orbit likely). Run exportSicilyPlanet.js '
          '(SVM + VV_OTSU), download to raw_data/GEE_SicilyPlanet*, then re-run for '
          'pipeline-consistent numbers.\n')

rows = []
matched = {}   # (name, method) -> merged df for plotting
for name in RESERVOIRS:
    planet = load_planet(name)
    if planet is None or planet.empty:
        print(f'[skip] no PlanetScope for {name}')
        continue
    for label, (token, col) in METHODS.items():
        sar = load_sar(name, label, token, col)
        if sar is None or sar.empty:
            continue
        m = match_nearest(sar, planet, TOL_DAYS)
        if len(m) < 3:
            continue
        k, r = kge(m['planet'].values, m['sar'].values)
        rmse = float(np.sqrt(np.mean((m['sar'] - m['planet']) ** 2)))
        bias = float(np.mean(m['sar'] - m['planet']))
        rows.append({'name': name, 'method': label, 'n_pairs': len(m),
                     'KGE': round(k, 3), 'r': round(r, 3),
                     'RMSE_ha': round(rmse, 2), 'bias_ha': round(bias, 2)})
        matched[(name, label)] = m

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)
print(f'Saved {len(res)} rows -> {OUT_CSV}\n')

# ── per-reservoir ΔKGE (dual − Otsu) vs PlanetScope ───────────────────────────
piv = res.pivot_table(index='name', columns='method', values='KGE')
if 'dual (VV+VH SVM)' in piv and 'Otsu (VV-only)' in piv:
    piv['ΔKGE(dual−Otsu)'] = piv['dual (VV+VH SVM)'] - piv['Otsu (VV-only)']
print('KGE vs PlanetScope (near-truth):')
print(piv.round(3).to_string())
print(f"\nmean KGE  Otsu={res[res.method=='Otsu (VV-only)']['KGE'].mean():.3f}  "
      f"dual={res[res.method=='dual (VV+VH SVM)']['KGE'].mean():.3f}  "
      f"VVsvm={res[res.method=='VV-only SVM']['KGE'].mean():.3f}")

# ── Figure: per-reservoir timeseries (PlanetScope + 3 SAR methods) ────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Only the two experiment methods (drop legacy VV-only SVM — it's raw/off-pipeline noise).
PLOT_METHODS = ['Otsu (VV-only)', 'dual (VV+VH SVM)']

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.ravel()
for ax, name in zip(axes, RESERVOIRS):
    planet = load_planet(name)
    if planet is not None and not planet.empty:
        ax.plot(planet['date'], planet['planet'], 'o-', color='#d62728', ms=4,
                lw=1.3, zorder=6, label='PlanetScope (3 m, truth)')
    for label in PLOT_METHODS:
        token, col = METHODS[label]
        sar = load_sar(name, label, token, col)
        if sar is None or sar.empty:
            continue
        ax.plot(sar['date'], sar['sar'], '.-', color=MCOL[label], ms=4, lw=1.0,
                alpha=0.85, zorder=4, label=label)
    sub = res[(res.name == name) & (res.method.isin(PLOT_METHODS))]
    txt = '\n'.join(f"{r.method.split(' ')[0]:>5}: KGE={r.KGE:+.2f}"
                    for r in sub.itertuples())
    if txt:
        ax.text(0.015, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='#bbb', alpha=0.85))
    # Focus the x-axis on the PlanetScope-covered window (±2 weeks margin).
    if planet is not None and not planet.empty:
        pad = pd.Timedelta(days=14)
        ax.set_xlim(planet['date'].min() - pad, planet['date'].max() + pad)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_ylabel('area (ha)', fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(axis='x', labelsize=8)
axes[0].legend(fontsize=8, loc='lower right')
fig.suptitle('Sicily SAR methods vs PlanetScope 3 m (near-truth), 2024–2025 — '
             'breaks the JRC inter-product circularity',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT_PNG}')

# ── Scatter panels: Otsu×Planet, dual×Planet, Otsu×dual (pooled, colored by lake) ──
OUT_SC = pathlib.Path('analysis/method_comparison_output/sicily_planet_scatter.png')
RCOL = {'Ancipa': '#1f77b4', 'Pozzillo': '#ff7f0e', 'Poma': '#2ca02c', 'Rosamarina': '#9467bd'}

# (a)/(b) reuse matched Planet↔SAR pairs; (c) merge Otsu & dual SAR on common dates.
def pooled_vs_planet(label):
    out = []
    for name in RESERVOIRS:
        m = matched.get((name, label))
        if m is not None and not m.empty:
            out.append(pd.DataFrame({'x': m['planet'], 'y': m['sar'], 'name': name}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def pooled_otsu_dual():
    out = []
    for name in RESERVOIRS:
        o = load_sar(name, 'Otsu (VV-only)',  'Otsu',      'areaOtsu_ha')
        d = load_sar(name, 'dual (VV+VH SVM)', 'SVM_VVpVH', 'areaSVM_ha')
        if o is None or d is None or o.empty or d.empty:
            continue
        mm = pd.merge(d.rename(columns={'sar': 'x'}), o.rename(columns={'sar': 'y'}), on='date')
        if not mm.empty:
            out.append(pd.DataFrame({'x': mm['x'], 'y': mm['y'], 'name': name}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

PANELS = [
    (pooled_vs_planet('Otsu (VV-only)'),   'PlanetScope area (ha)', 'VV-only Otsu area (ha)', '(a) VV-only vs PlanetScope'),
    (pooled_vs_planet('dual (VV+VH SVM)'), 'PlanetScope area (ha)', 'VV+VH SVM area (ha)',    '(b) dual vs PlanetScope'),
    (pooled_otsu_dual(),                   'VV+VH SVM area (ha)',   'VV-only Otsu area (ha)', '(c) VV-only vs dual'),
]
allv = np.concatenate([np.r_[d['x'], d['y']] for d, *_ in PANELS if not d.empty])
lo, hi = allv.min() * 0.8, allv.max() * 1.2

fig2, axes2 = plt.subplots(1, 3, figsize=(16.5, 5.6))
for ax, (d, xlab, ylab, title) in zip(axes2, PANELS):
    if d.empty:
        ax.set_visible(False); continue
    for name, g in d.groupby('name'):
        ax.scatter(g['x'], g['y'], s=22, color=RCOL.get(name, '#555'), alpha=0.7,
                   edgecolors='none', label=name, zorder=3)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.1, alpha=0.7, zorder=4, label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    r, _ = stats.pearsonr(np.log10(d['x']), np.log10(d['y']))
    ratio = np.median(d['y'] / d['x'])
    ax.text(0.04, 0.96, f'N={len(d)}\nr(log)={r:.3f}\nmed y/x={ratio:.2f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='#bbb', alpha=0.85))
    ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2, which='both'); ax.legend(fontsize=7, loc='lower right')
fig2.suptitle('Sicily — area agreement vs PlanetScope 3 m (pooled, log-log, by reservoir)',
              fontsize=13, fontweight='bold')
fig2.savefig(OUT_SC, dpi=140, bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {OUT_SC}')
