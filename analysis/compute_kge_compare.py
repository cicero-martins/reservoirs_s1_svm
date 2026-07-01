"""
compute_kge_compare.py

Method comparison: VV+VH SVM (Tier 3, dual) vs VV-only Otsu (Tier 1) against the
same JRC reference. Core metric is ΔKGE = KGE_dual − KGE_vv, computed on the
COMMON months (same dates for both methods) so it isolates the classifier effect
from coverage differences. Independent KGE (each method on its own months) and
coverage counts are also reported.

Reads:
  SAR dual : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4/SAR_area_*.csv      (existing)
  SAR VV   : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu/SAR_area_*.csv (new run)
  JRC      : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC/JRC_area_*.csv

Output:
  analysis/pilot_kge_compare.csv
  analysis/method_comparison_output/kge_compare_dual_vs_vv.png   (only if VV data present)

Runs safely BEFORE the VV_OTSU export exists: dual columns are filled, VV columns
are NaN, and a clear notice is printed.
"""

import pathlib
import re as _re
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')


# ── Clean-and-smooth pipeline (identical to compute_kge_v4.py) ────────────────
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
    smoothed = []
    for t0 in dates:
        diff_d = np.abs((dates - t0).dt.total_seconds().values / 86400)
        mask   = diff_d <= window_days
        w      = np.exp(-(diff_d[mask] / bandwidth) ** 2)
        smoothed.append(float((values[mask] * w).sum() / w.sum()))
    return np.array(smoothed)

def clean_and_smooth(df, col='area_ha'):
    s = df[col].copy()
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 10, 1.5)
    dates_s  = df.loc[s.index, 'date'].reset_index(drop=True)
    smoothed = _lowess(dates_s, s.reset_index(drop=True), 20, 7)
    return pd.DataFrame({'date': dates_s, 'area_ha': smoothed})


# ── Config (mirrors compute_kge_v4.py) ────────────────────────────────────────
SAR_DUAL_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
SAR_VV_DIR   = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu')
JRC_DIR      = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
OUT_CSV      = pathlib.Path('analysis/pilot_kge_compare.csv')
OUT_PNG      = pathlib.Path('analysis/method_comparison_output/kge_compare_dual_vs_vv.png')

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
MIN_PAIRS      = 12       # min months for independent KGE
MIN_COMMON     = 10       # min shared months for ΔKGE

EXCLUDE  = {'Oued_Makhazine', 'Guajaraz', 'Antero', 'Miyagase', 'Welbedacht', 'Tzaneen'}
AREA_MIN = {'Saint_Cassien': 200}

cand  = pd.read_csv('analysis/global_pilot_v4_candidates.csv')
NAMES = cand['name'].tolist()


def kge(obs, sim):
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2), r, alpha, beta


def _jrc_path(name):
    cands = sorted(JRC_DIR.glob(f'JRC_area_{name}*.csv'))
    plain = [p for p in cands if not _re.search(r'\s*\(\d+\)', p.stem)]
    return plain[0] if plain else (cands[0] if cands else None)


def load_sar_monthly(name, sar_dir):
    """Monthly SAR series after clean+smooth. Returns (monthly_df, ap_m) or (None, nan)."""
    p = sar_dir / f'SAR_area_{name}.csv'
    if not p.exists():
        return None, np.nan
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None, np.nan
    if df.empty:
        return None, np.nan

    ap_m = float(df['ap_m'].iloc[0]) if 'ap_m' in df.columns else np.nan

    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if name in AREA_MIN:
        df = df[df['area_ha'] >= AREA_MIN[name]].copy()
    if df.empty:
        return None, ap_m

    df  = df[df['date'] <= '2021-12-31'].copy()
    p99 = df['area_ha'].quantile(0.99)
    df  = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    if df.empty:
        return None, ap_m

    df = clean_and_smooth(df.reset_index(drop=True))
    if df.empty:
        return None, ap_m

    df['ym'] = df['date'].dt.to_period('M')
    monthly  = df.groupby('ym')['area_ha'].mean().reset_index()
    monthly.columns = ['ym', 'sar_area_ha']
    return monthly, ap_m


def load_jrc_monthly(name):
    p = _jrc_path(name)
    if p is None:
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    df = df.sort_values('date').reset_index(drop=True)
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= VALID_FRAC_MIN].copy()
    if df.empty:
        return None
    m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
    if sd > 0:
        df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * sd].copy()
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()


# ── Main loop ─────────────────────────────────────────────────────────────────
have_vv = SAR_VV_DIR.exists() and any(SAR_VV_DIR.glob('SAR_area_*.csv'))
if not have_vv:
    print(f'[notice] VV_OTSU data not found at {SAR_VV_DIR}')
    print('         Run the export with CLASSIFIER="VV_OTSU"; dual columns written, VV=NaN.\n')

rows = []
for name in NAMES:
    if name in EXCLUDE:
        continue

    jrc_m = load_jrc_monthly(name)
    if jrc_m is None or jrc_m.empty:
        continue

    dual_m, ap_m = load_sar_monthly(name, SAR_DUAL_DIR)
    vv_m,   _    = load_sar_monthly(name, SAR_VV_DIR)

    if dual_m is None or dual_m.empty:
        continue

    # Independent KGE (each method vs JRC on its own months)
    md = pd.merge(dual_m, jrc_m, on='ym', how='inner').dropna()
    kge_dual = kge(md['jrc_area_ha'].values, md['sar_area_ha'].values)[0] \
               if len(md) >= MIN_PAIRS else np.nan

    kge_vv = np.nan
    n_vv = 0
    if vv_m is not None and not vv_m.empty:
        mv = pd.merge(vv_m, jrc_m, on='ym', how='inner').dropna()
        n_vv = len(mv)
        if n_vv >= MIN_PAIRS:
            kge_vv = kge(mv['jrc_area_ha'].values, mv['sar_area_ha'].values)[0]

    # Common-months ΔKGE (same dates → isolates classifier)
    kge_dc = kge_vc = delta = np.nan
    r_d = r_v = a_d = a_v = b_d = b_v = np.nan
    n_common = 0
    if vv_m is not None and not vv_m.empty:
        trio = (dual_m.rename(columns={'sar_area_ha': 'dual'})
                .merge(vv_m.rename(columns={'sar_area_ha': 'vv'}), on='ym')
                .merge(jrc_m, on='ym').dropna())
        n_common = len(trio)
        if n_common >= MIN_COMMON:
            obs = trio['jrc_area_ha'].values
            kge_dc, r_d, a_d, b_d = kge(obs, trio['dual'].values)
            kge_vc, r_v, a_v, b_v = kge(obs, trio['vv'].values)
            delta = kge_dc - kge_vc

    rows.append({
        'name':        name,
        'ap_m':        round(ap_m, 1) if not np.isnan(ap_m) else np.nan,
        'country':     cand.loc[cand['name'] == name, 'country'].iloc[0],
        'climate':     cand.loc[cand['name'] == name, 'climate_zone'].iloc[0],
        'n_dual':      len(md),
        'n_vv':        n_vv,
        'n_common':    n_common,
        'kge_dual':    round(kge_dual, 4) if not np.isnan(kge_dual) else np.nan,
        'kge_vv':      round(kge_vv,   4) if not np.isnan(kge_vv)   else np.nan,
        'kge_dual_c':  round(kge_dc, 4) if not np.isnan(kge_dc) else np.nan,
        'kge_vv_c':    round(kge_vc, 4) if not np.isnan(kge_vc) else np.nan,
        'delta_kge':   round(delta, 4) if not np.isnan(delta) else np.nan,
        'r_dual':      round(r_d, 4) if not np.isnan(r_d) else np.nan,
        'r_vv':        round(r_v, 4) if not np.isnan(r_v) else np.nan,
        'alpha_dual':  round(a_d, 4) if not np.isnan(a_d) else np.nan,
        'alpha_vv':    round(a_v, 4) if not np.isnan(a_v) else np.nan,
        'beta_dual':   round(b_d, 4) if not np.isnan(b_d) else np.nan,
        'beta_vv':     round(b_v, 4) if not np.isnan(b_v) else np.nan,
    })

df_out = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df_out.to_csv(OUT_CSV, index=False)
print(f'Saved {len(df_out)} reservoirs -> {OUT_CSV}')

# ── Summary + figure (only when VV present) ───────────────────────────────────
if have_vv and df_out['delta_kge'].notna().any():
    valid = df_out.dropna(subset=['delta_kge'])
    print(f'\nΔKGE (dual − vv), common months, N={len(valid)}:')
    print(f'  mean ΔKGE   = {valid["delta_kge"].mean():+.3f}')
    print(f'  median ΔKGE = {valid["delta_kge"].median():+.3f}')
    print(f'  dual wins (ΔKGE>0.02): {(valid["delta_kge"] > 0.02).sum()}')
    print(f'  tie (|ΔKGE|≤0.02):     {(valid["delta_kge"].abs() <= 0.02).sum()}')
    print(f'  vv wins (ΔKGE<-0.02):  {(valid["delta_kge"] < -0.02).sum()}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'wspace': 0.3})

    # (a) KGE_dual vs KGE_vv, 1:1
    ax = axes[0]
    sc = ax.scatter(valid['kge_vv_c'], valid['kge_dual_c'],
                    c=valid['ap_m'], cmap='viridis', s=55,
                    edgecolors='white', linewidths=0.5, zorder=4)
    lims = [-0.6, 1.05]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1 (sem ganho)')
    for _, r in valid.iterrows():
        ax.annotate(r['name'].replace('_', ' '), (r['kge_vv_c'], r['kge_dual_c']),
                    fontsize=5, xytext=(3, 2), textcoords='offset points', color='#444')
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel('KGE  VV-only Otsu (Tier 1)')
    ax.set_ylabel('KGE  VV+VH SVM (Tier 3)')
    ax.set_title('(a) Dual vs VV-only — meses comuns')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.colorbar(sc, ax=ax, label='A/P (m)')

    # (b) ΔKGE vs A/P
    ax = axes[1]
    colors = np.where(valid['delta_kge'] > 0.02, '#2ca02c',
             np.where(valid['delta_kge'] < -0.02, '#d62728', '#999'))
    ax.scatter(valid['ap_m'], valid['delta_kge'], c=colors, s=55,
               edgecolors='white', linewidths=0.5, zorder=4)
    ax.axhline(0, color='k', lw=1, alpha=0.5)
    for _, r in valid.iterrows():
        if abs(r['delta_kge']) > 0.1:
            ax.annotate(r['name'].replace('_', ' '), (r['ap_m'], r['delta_kge']),
                        fontsize=5.5, xytext=(3, 2), textcoords='offset points', color='#444')
    ax.set_xlabel('A/P estático (m)')
    ax.set_ylabel('ΔKGE = KGE$_{dual}$ − KGE$_{vv}$')
    ax.set_title('(b) Ganho do dual-pol vs geometria\n(verde: dual vence; vermelho: VV vence)')
    ax.grid(alpha=0.25)

    fig.suptitle(f'Comparação de métodos: VV+VH SVM vs VV-only Otsu  (N={len(valid)})',
                 fontsize=11, fontweight='bold')
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {OUT_PNG}')
else:
    print('\n[figure skipped] needs VV_OTSU data with ≥10 common months per reservoir.')
