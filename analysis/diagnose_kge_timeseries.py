"""Diagnostic figure: SAR area vs reference time series for all 20 pilot reservoirs.

Sicily (n=4):  SAR-SVM vs PlanetScope (daily, 2024-2025)
Global (n=16): SAR monthly mean vs JRC monthly area (2014-2021)

Sorted by A/P (ascending). Each panel shows both series + KGE + N pairs.
"""

import sys, csv, warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE    = Path('validation_data')
SAR_DIR = BASE / 'GROWL_SAR_pilot'
OUT_DIR = Path('analysis/schwatke_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_FRAC   = 0.90
SAR_MIN_FRAC = 0.02
N_MIN        = 4

# ---------------------------------------------------------------------------
def kge(sim, obs):
    if len(sim) < N_MIN or np.std(obs) < 1e-6 or np.std(sim) < 1e-6:
        return np.nan
    r, _ = pearsonr(sim, obs)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def load_gee_csv(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append({k: v for k, v in row.items()
                         if k not in ('.geo', 'system:index')})
    return rows


# ---------------------------------------------------------------------------
# Sicily: SVM vs PlanetScope
# ---------------------------------------------------------------------------
SIC_NAMES = {
    'Ancipa':     ('ancipaSVM.csv',     'ancipaPlanet.csv'),
    'Rosamarina': ('rosamarinaSVM.csv', 'rosamarinaPlanet.csv'),
    'Poma':       ('pomaSVM.csv',       'pomaPlanet.csv'),
    'Pozzillo':   ('pozzilloSVM.csv',   'pozzilloPlanet.csv'),
}
SIC_AREA_DIR = BASE / 'statistics/area_statistics'

def load_sicily(name):
    svm_f, pl_f = SIC_NAMES[name]
    def read_csv(f):
        df = pd.read_csv(SIC_AREA_DIR / f, encoding='utf-8-sig')
        df.columns = [c.strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['data'], dayfirst=False, errors='coerce')
        df = df.dropna(subset=['date']).rename(columns={'area': 'area_ha'})
        return df[['date', 'area_ha']].sort_values('date').reset_index(drop=True)
    return read_csv(svm_f), read_csv(pl_f)


# ---------------------------------------------------------------------------
# Global: monthly SAR mean vs monthly JRC
# ---------------------------------------------------------------------------
EXCLUDED = {
    'Panam', 'Kakki', 'Ry_de_Rome_BE', 'OShannassy_AU',
    'Leech_US', 'Winnibigoshish_US', 'Upper_Coliban_AU', 'Minilla_ES',
}

def load_global(sar_path, jrc_path):
    # SAR → aggregate to monthly mean
    sar_rows = load_gee_csv(sar_path)
    aoi_ha = None
    try:
        aoi_ha = float(sar_rows[0]['area_aoi_ha'])
    except Exception:
        pass
    sar_mo = {}
    for row in sar_rows:
        try:
            dt   = pd.to_datetime(row['date'])
            area = float(row['area_ha'])
            if aoi_ha and area / aoi_ha < SAR_MIN_FRAC:
                continue
            key = (dt.year, dt.month)
            sar_mo.setdefault(key, []).append(area)
        except Exception:
            pass
    sar_df = pd.DataFrame([
        {'date': pd.Timestamp(y, m, 15), 'sar_ha': np.mean(v)}
        for (y, m), v in sorted(sar_mo.items()) if v
    ])

    # JRC → monthly, filter by valid_frac
    jrc_rows = load_gee_csv(jrc_path)
    jrc_df = pd.DataFrame([
        {'date': pd.to_datetime(row['date']),
         'jrc_ha': float(row['jrc_area_ha']),
         'vf': float(row.get('valid_frac', 1.0))}
        for row in jrc_rows
        if row.get('jrc_area_ha') and row.get('date')
    ])
    jrc_df = jrc_df[jrc_df['vf'] >= VALID_FRAC].copy()

    # Merge on year-month
    if sar_df.empty or jrc_df.empty:
        return sar_df, jrc_df, pd.DataFrame()
    sar_df['ym'] = sar_df['date'].dt.to_period('M')
    jrc_df['ym'] = jrc_df['date'].dt.to_period('M')
    merged = sar_df.merge(jrc_df[['ym', 'jrc_ha']], on='ym', how='inner')
    return sar_df, jrc_df, merged


# ---------------------------------------------------------------------------
# Load pilot_kge_results for reference KGE values
# ---------------------------------------------------------------------------
kge_ref = pd.read_csv(
    BASE / 'morphometric_analysis/shoreline_compactness/pilot_kge_results.csv')
kge_ref = kge_ref.dropna(subset=['ap_m', 'kge'])
kge_map = dict(zip(kge_ref['name'], zip(kge_ref['kge'], kge_ref['ap_m'],
                                         kge_ref['country'])))

# AP from all_reservoirs file (includes Sicily)
ap_all = pd.read_csv(
    BASE / 'morphometric_analysis/shoreline_compactness/AP_all_reservoirs.csv')
ap_sic = {r: float(a) for r, a in zip(ap_all['reservoir'], ap_all['AP_m'])
           if r in SIC_NAMES}

# ---------------------------------------------------------------------------
# Build all panels
# ---------------------------------------------------------------------------
panels = []

# Sicily panels
for name in SIC_NAMES:
    kge_val, ap_val, country = kge_map.get(name, (np.nan, ap_sic.get(name, np.nan), 'Italy'))
    svm_df, pl_df = load_sicily(name)
    # match to nearest date ±5 days for scatter KGE
    pairs = []
    for _, row in svm_df.iterrows():
        delta = (pl_df['date'] - row['date']).dt.days.abs()
        idx = delta.idxmin()
        if delta[idx] <= 5:
            pairs.append({'sar': row['area_ha'], 'ref': pl_df.loc[idx, 'area_ha']})
    p = pd.DataFrame(pairs)
    if len(p) >= N_MIN:
        sim = p['sar'].values; obs = p['ref'].values
        sim_n = sim / sim.max() if sim.max() > 0 else sim
        obs_n = obs / obs.max() if obs.max() > 0 else obs
        kge_computed = kge(sim_n, obs_n)
    else:
        kge_computed = np.nan
    panels.append({
        'name': name, 'country': country, 'ap': ap_val,
        'kge_ref': kge_val, 'kge_comp': kge_computed,
        'type': 'sicily',
        'sar_full': svm_df, 'ref_full': pl_df,
        'merged': p, 'n': len(p),
        'ref_label': 'PlanetScope',
    })

# Global panels
sar_files = sorted(SAR_DIR.glob('SAR_area_*.csv'))
jrc_files = {f.stem.replace('JRC_area_', ''): f
             for f in sorted(SAR_DIR.glob('JRC_area_*.csv'))}

for sar_path in sar_files:
    key  = sar_path.stem.replace('SAR_area_', '')
    parts = key.rsplit('_', 1)
    name  = parts[0]
    if name in EXCLUDED:
        continue
    jrc_path = jrc_files.get(key)
    if not jrc_path:
        continue

    kge_val, ap_val, country = kge_map.get(name, (np.nan, np.nan, ''))
    if np.isnan(ap_val):
        continue   # not in pilot results

    sar_df, jrc_df, merged = load_global(sar_path, jrc_path)
    # Normalize by own maximum (matching original compute_pilot_kge.py)
    if len(merged) >= N_MIN:
        sim = merged['sar_ha'].values; obs = merged['jrc_ha'].values
        sim_n = sim / sim.max() if sim.max() > 0 else sim
        obs_n = obs / obs.max() if obs.max() > 0 else obs
        kge_computed = kge(sim_n, obs_n)
    else:
        kge_computed = np.nan

    panels.append({
        'name': name, 'country': country, 'ap': ap_val,
        'kge_ref': kge_val, 'kge_comp': kge_computed,
        'type': 'global',
        'sar_full': sar_df.rename(columns={'sar_ha': 'area_ha'}) if not sar_df.empty else pd.DataFrame(),
        'ref_full': jrc_df.rename(columns={'jrc_ha': 'area_ha'}) if not jrc_df.empty else pd.DataFrame(),
        'merged': merged, 'n': len(merged),
        'ref_label': 'JRC monthly',
    })

# Sort by A/P ascending
panels.sort(key=lambda p: p['ap'])
print(f"Total panels: {len(panels)}")
for p in panels:
    print(f"  {p['name']:30s}  AP={p['ap']:.0f}  KGE_ref={p['kge_ref']:.3f}  "
          f"KGE_comp={p['kge_comp']:.3f}  N={p['n']}  ref={p['ref_label']}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
ncols = 4
nrows = (len(panels) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
axes = axes.flatten()

for i, p in enumerate(panels):
    ax = axes[i]
    sar = p['sar_full']
    ref = p['ref_full']

    kge_use = p['kge_comp']   # recomputed from raw data
    color_kge = ('#2ca02c' if kge_use >= 0.5
                 else '#d62728' if not np.isnan(kge_use)
                 else 'gray')

    if p['type'] == 'sicily':
        # time series — both SVM and Planet by date
        if not ref.empty:
            ax.plot(ref['date'], ref['area_ha'],
                    'o', ms=4, color='darkorange', alpha=0.8, label='PlanetScope')
        if not sar.empty:
            ax.plot(sar['date'], sar['area_ha'],
                    's', ms=3.5, lw=1, color='steelblue', alpha=0.8, label='SAR-SVM')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%y'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('Area (ha)', fontsize=7)
    else:
        # time series — plot normalized (fraction of max) matching KGE computation
        mg = p['merged']
        if not mg.empty:
            ref_norm = mg['jrc_ha'] / mg['jrc_ha'].max()
            sar_norm = mg['sar_ha'] / mg['sar_ha'].max()
            dates_mg = pd.to_datetime(mg['date'])
            ax.plot(dates_mg, ref_norm,
                    '-o', ms=3, lw=1, color='gray', alpha=0.8,
                    label=f'{p["ref_label"]} (norm.)')
            ax.plot(dates_mg, sar_norm,
                    '-', lw=1.2, color='steelblue', alpha=0.9, label='SAR (norm.)')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('Fraction of max', fontsize=7)
        ax.set_ylim(-0.05, 1.15)

    # Title
    kge_str = f'{kge_use:.2f}' if not np.isnan(kge_use) else 'N/A'
    disc = '✓' if kge_use >= 0.5 else ('✗' if not np.isnan(kge_use) else '?')
    ax.set_title(
        f'{p["name"].replace("_", " ")} ({p["country"]})\n'
        f'AP={p["ap"]:.0f} m  KGE={kge_str} {disc}  N={p["n"]}',
        fontsize=8, color=color_kge, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6.5, loc='upper left')

# Hide unused axes
for j in range(len(panels), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    'Diagnostic: SAR area vs reference — all 20 pilot reservoirs (sorted by A/P)',
    fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT_DIR / 'pilot_kge_diagnostic.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out}')
