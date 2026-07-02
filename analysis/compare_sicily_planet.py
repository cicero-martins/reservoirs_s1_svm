"""
compare_sicily_planet.py

Sicily method comparison anchored to PlanetScope 3 m (near-truth), NOT JRC 30 m —
this breaks the inter-product circularity of the 28-reservoir global pilot, and
mirrors the global four-way comparison (compute_kge_4way.py) against real truth.

For each of the 4 PlanetScope-validated Sicilian reservoirs, four canonical SAR
water-area series (all exported through the SAME pipeline, 2024-01..2025-05) are
matched to the nearest PlanetScope acquisition (±TOL_DAYS) and scored against it:
  Otsu   = VV-only per-scene Otsu                 GEE_SicilyPlanet_VVotsu
  dual   = VV+VH SVM, FIXED 2023 training          GEE_SicilyPlanet
  adapt  = VV+VH SVM, PER-SCENE retraining         GEE_SicilyPlanet_SVMadapt
  fast   = VV-only Otsu, NO vectorisation          GEE_SicilyPlanet_VVfast

Two questions (same as the global 4-way, but vs near-truth instead of JRC):
  Q1  adapt vs dual — does per-scene retraining help/hurt against truth?
  Q2  fast  vs Otsu — does dropping vectorisation (the cost lever) degrade area?

Reference (ground truth):
  validation_data/statistics/area_statistics/{name}Planet.csv   (cols: data, area[ha])

Metrics per method per reservoir: KGE, Pearson r, RMSE (ha), bias (mean sim−obs).

Output:
  analysis/sicily_planet_compare.csv
  analysis/method_comparison_output/sicily_planet_validation.png
  analysis/method_comparison_output/sicily_planet_scatter.png
"""

import pathlib
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

PLANET_DIR = pathlib.Path('validation_data/statistics/area_statistics')
OUT_CSV    = pathlib.Path('analysis/sicily_planet_compare.csv')
OUT_PNG    = pathlib.Path('analysis/method_comparison_output/sicily_planet_validation.png')

TOL_DAYS = 6   # match each PlanetScope truth obs to nearest SAR scene within ±this

RESERVOIRS = ['Ancipa', 'Pozzillo', 'Poma', 'Rosamarina']

# label : canonical export folder (all pipeline-consistent, cols date, area_ha)
METHODS = {
    'Otsu (VV-only)':      pathlib.Path('raw_data/GEE_SicilyPlanet_VVotsu'),
    'dual (SVM fixed)':    pathlib.Path('raw_data/GEE_SicilyPlanet'),
    'adapt (SVM per-scene)': pathlib.Path('raw_data/GEE_SicilyPlanet_SVMadapt'),
    'fast (Otsu no-vec)':  pathlib.Path('raw_data/GEE_SicilyPlanet_VVfast'),
}
MCOL = {'Otsu (VV-only)': '#2ca02c', 'dual (SVM fixed)': '#1f77b4',
        'adapt (SVM per-scene)': '#ff7f0e', 'fast (Otsu no-vec)': '#9467bd'}
# dashed for the two "lever/baseline" variants so the two headline methods read clearly
MLS  = {'Otsu (VV-only)': '-', 'dual (SVM fixed)': '--',
        'adapt (SVM per-scene)': '-', 'fast (Otsu no-vec)': '--'}


# ── clean+smooth (identical to compute_kge_4way.py) ───────────────────────────
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


def load_sar(name, folder):
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
        return d if not d.empty else None
    return clean_and_smooth(d)


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


missing = [lab for lab, d in METHODS.items() if not (d.exists() and any(d.glob('SAR_area_*.csv')))]
if missing:
    print(f'[notice] missing canonical folders for: {missing} — those methods skipped.\n')

rows = []
matched = {}   # (name, method) -> merged df for plotting
for name in RESERVOIRS:
    planet = load_planet(name)
    if planet is None or planet.empty:
        print(f'[skip] no PlanetScope for {name}')
        continue
    for label, folder in METHODS.items():
        sar = load_sar(name, folder)
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

# ── KGE table + the two headline deltas ───────────────────────────────────────
piv = res.pivot_table(index='name', columns='method', values='KGE')
if {'adapt (SVM per-scene)', 'dual (SVM fixed)'}.issubset(piv.columns):
    piv['ΔKGE Q1 (adapt−dual)'] = piv['adapt (SVM per-scene)'] - piv['dual (SVM fixed)']
if {'fast (Otsu no-vec)', 'Otsu (VV-only)'}.issubset(piv.columns):
    piv['ΔKGE Q2 (fast−Otsu)'] = piv['fast (Otsu no-vec)'] - piv['Otsu (VV-only)']
if {'adapt (SVM per-scene)', 'Otsu (VV-only)'}.issubset(piv.columns):
    piv['adapt−Otsu'] = piv['adapt (SVM per-scene)'] - piv['Otsu (VV-only)']
print('KGE vs PlanetScope (near-truth):')
print(piv.round(3).to_string())

print('\nmean KGE per method:')
for label in METHODS:
    sub = res[res.method == label]['KGE']
    if not sub.empty:
        print(f'  {label:<22} mean KGE={sub.mean():+.3f}  (n_resv={len(sub)})')

# ── Figure: per-reservoir timeseries (PlanetScope + 4 SAR methods) ────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.ravel()
for ax, name in zip(axes, RESERVOIRS):
    planet = load_planet(name)
    if planet is not None and not planet.empty:
        ax.plot(planet['date'], planet['planet'], 'o-', color='#d62728', ms=4,
                lw=1.5, zorder=6, label='PlanetScope (3 m, truth)')
    for label, folder in METHODS.items():
        sar = load_sar(name, folder)
        if sar is None or sar.empty:
            continue
        ax.plot(sar['date'], sar['sar'], MLS[label], color=MCOL[label], ms=3, lw=1.1,
                alpha=0.85, zorder=4, label=label)
    sub = res[res.name == name]
    txt = '\n'.join(f"{r.method.split(' ')[0]:>5}: KGE={r.KGE:+.2f}"
                    for r in sub.itertuples())
    if txt:
        ax.text(0.015, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='#bbb', alpha=0.85))
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

# ── Scatter panels: adapt×Planet, Otsu×Planet, adapt×Otsu (pooled, by lake) ───
OUT_SC = pathlib.Path('analysis/method_comparison_output/sicily_planet_scatter.png')
RCOL = {'Ancipa': '#1f77b4', 'Pozzillo': '#ff7f0e', 'Poma': '#2ca02c', 'Rosamarina': '#9467bd'}

def pooled_vs_planet(label):
    out = []
    for name in RESERVOIRS:
        m = matched.get((name, label))
        if m is not None and not m.empty:
            out.append(pd.DataFrame({'x': m['planet'], 'y': m['sar'], 'name': name}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def pooled_pair(label_x, label_y):
    """Merge two SAR methods on common dates (per reservoir) for a method-vs-method scatter."""
    out = []
    for name in RESERVOIRS:
        dx = load_sar(name, METHODS[label_x])
        dy = load_sar(name, METHODS[label_y])
        if dx is None or dy is None or dx.empty or dy.empty:
            continue
        mm = pd.merge(dx.rename(columns={'sar': 'x'}), dy.rename(columns={'sar': 'y'}), on='date')
        if not mm.empty:
            out.append(pd.DataFrame({'x': mm['x'], 'y': mm['y'], 'name': name}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

PANELS = [
    (pooled_vs_planet('Otsu (VV-only)'),      'PlanetScope area (ha)', 'VV-only Otsu area (ha)',   '(a) Otsu vs PlanetScope'),
    (pooled_vs_planet('adapt (SVM per-scene)'), 'PlanetScope area (ha)', 'VV+VH SVM (per-scene) (ha)', '(b) adapt-SVM vs PlanetScope'),
    (pooled_pair('adapt (SVM per-scene)', 'Otsu (VV-only)'), 'VV+VH SVM (per-scene) (ha)', 'VV-only Otsu area (ha)', '(c) Otsu vs adapt-SVM'),
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
