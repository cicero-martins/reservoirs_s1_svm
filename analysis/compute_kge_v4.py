"""
compute_kge_v4.py

Full-period KGE for the 34-reservoir global pilot v4 set.
Uses GEE_GlobalPilotV4b exports:
  SAR: raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4/SAR_area_*.csv
  JRC: raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC/JRC_area_*.csv

Pipeline identical to compute_kge_v3.py (clean_and_smooth + monthly KGE).
Also captures ap_m_dynamic (mean per-acquisition dynamic A/P).

Output: analysis/pilot_kge_v4.csv
"""

import pathlib
import re as _re
import numpy as np
import pandas as pd
from scipy import stats


# ── Clean-and-smooth pipeline (mirrors original.js cleanAndSmooth) ────────────
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
    """removeOutliers(2) + detectAndRemoveLocal(5,1.5)x2 + (10,1.5) + lowess(20,7)."""
    s = df[col].copy()
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 10, 1.5)
    dates_s  = df.loc[s.index, 'date'].reset_index(drop=True)
    smoothed = _lowess(dates_s, s.reset_index(drop=True), 20, 7)
    return pd.DataFrame({'date': dates_s, 'area_ha': smoothed})


# ── Config ────────────────────────────────────────────────────────────────────
SAR_DIR  = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
JRC_DIR  = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
OUT_CSV  = pathlib.Path('analysis/pilot_kge_v4.csv')

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02   # drop acquisitions below 2% of p99 area (noise floor)
MIN_PAIRS      = 12

# Oued_Makhazine: SAR export empty (GDW polygon 19.7 km off)
# Guajaraz: only 14 acquisitions (2015-05 to 2016-08; GDW polygon 12.9 km off)
# Antero: SAR classification systematically wrong (high-altitude Colorado, winter ice)
# Miyagase: two bridges bisect the lake, removing major arms from classified polygon
# Welbedacht: aquatic vegetation covers large fraction — SAR/JRC detect land, not water
# Tzaneen: orbit-footprint artefact — SAR returns a near-constant inflated area
#   (mean 864 ha, CV 8%) regardless of true level (JRC 540 ha, CV 38%); the classifier
#   fills the swath footprint instead of tracking water (alpha=0.26, beta=1.6). Not an A/P signal.
EXCLUDE = {'Oued_Makhazine', 'Guajaraz', 'Antero', 'Miyagase', 'Welbedacht', 'Tzaneen'}

# Per-reservoir minimum SAR area (ha) applied before clean/smooth.
# Saint_Cassien: a bridge bisects the lake; images detecting only the lower section
# (~130-200 ha) must be excluded so LOWESS sees the full lake area (~280 ha mean).
AREA_MIN = {'Saint_Cassien': 200}

# ── Reservoir list from candidates CSV ───────────────────────────────────────
cand = pd.read_csv('analysis/global_pilot_v4_candidates.csv')
NAMES = cand['name'].tolist()


# ── KGE ───────────────────────────────────────────────────────────────────────
def kge(obs, sim):
    if obs.std() == 0 or sim.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = sim.std()  / obs.std()
    beta  = sim.mean() / obs.mean()
    return 1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2), r, alpha, beta


# ── Loaders ───────────────────────────────────────────────────────────────────
def _sar_path(name):
    p = SAR_DIR / f'SAR_area_{name}.csv'
    return p if p.exists() else None

def _jrc_path(name):
    # handle Google Drive "(1)" duplicates — prefer plain file
    candidates = sorted(JRC_DIR.glob(f'JRC_area_{name}*.csv'))
    plain = [p for p in candidates if not _re.search(r'\s*\(\d+\)', p.stem)]
    return plain[0] if plain else (candidates[0] if candidates else None)


def load_sar_monthly(name):
    p = _sar_path(name)
    if p is None:
        return None, np.nan, np.nan

    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None, np.nan, np.nan

    if df.empty:
        return None, np.nan, np.nan

    ap_m         = float(df['ap_m'].iloc[0])         if 'ap_m'         in df.columns else np.nan
    ap_m_dynamic = float(df['ap_m_dynamic'].mean())  if 'ap_m_dynamic' in df.columns else np.nan

    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if name in AREA_MIN:
        df = df[df['area_ha'] >= AREA_MIN[name]].copy()
    if df.empty:
        return None, ap_m, ap_m_dynamic

    df  = df[df['date'] <= '2021-12-31'].copy()
    p99 = df['area_ha'].quantile(0.99)
    df  = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    if df.empty:
        return None, ap_m, ap_m_dynamic

    df = clean_and_smooth(df.reset_index(drop=True))
    if df.empty:
        return None, ap_m, ap_m_dynamic

    df['ym']  = df['date'].dt.to_period('M')
    monthly   = df.groupby('ym')['area_ha'].mean().reset_index()
    monthly.columns = ['ym', 'sar_area_ha']
    return monthly, ap_m, ap_m_dynamic


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
    # Sigma-clip JRC series: removes artefacts that pass the valid_frac threshold
    # (e.g. Yamba 2018-05 = 0 ha, Blyde 2018-11 = 39 ha, Saguaro 2017-11 = 361 ha)
    m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
    if sd > 0:
        df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * sd].copy()
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()


# ── Main loop ─────────────────────────────────────────────────────────────────
rows, skipped = [], []

hdr = (f'{"Name":<22}  {"A/P":>6}  {"AP_dyn":>6}  {"n":>4}  '
       f'{"KGE":>6}  {"r":>5}  {"alpha":>5}  {"beta":>5}  '
       f'{"SAR_ha":>7}  {"JRC_ha":>7}')
print(hdr)
print('-' * len(hdr))

for name in NAMES:
    if name in EXCLUDE:
        skipped.append((name, 'excluded'))
        continue

    sar_m, ap_m, ap_dyn = load_sar_monthly(name)
    jrc_m               = load_jrc_monthly(name)

    if sar_m is None or jrc_m is None or sar_m.empty or jrc_m.empty:
        skipped.append((name, 'no data'))
        continue

    merged = pd.merge(sar_m, jrc_m, on='ym', how='inner').dropna()

    if len(merged) < MIN_PAIRS:
        skipped.append((name, f'only {len(merged)} pairs'))
        continue

    obs = merged['jrc_area_ha'].values
    sim = merged['sar_area_ha'].values
    kge_v, r, alpha, beta = kge(obs, sim)

    ap_str  = f'{ap_m:6.0f}'  if not np.isnan(ap_m)  else '   nan'
    dyn_str = f'{ap_dyn:6.0f}' if not np.isnan(ap_dyn) else '   nan'
    print(f'  {name:<22}  {ap_str}  {dyn_str}  {len(merged):>4d}  '
          f'{kge_v:>6.3f}  {r:>5.3f}  {alpha:>5.3f}  {beta:>5.3f}  '
          f'{sim.mean():>7.1f}  {obs.mean():>7.1f}')

    rows.append({
        'name':        name,
        'ap_m':        round(ap_m,   1) if not np.isnan(ap_m)   else np.nan,
        'ap_m_dynamic':round(ap_dyn, 1) if not np.isnan(ap_dyn) else np.nan,
        'ap_expected': cand.loc[cand['name'] == name, 'ap_expected'].iloc[0],
        'country':     cand.loc[cand['name'] == name, 'country'].iloc[0],
        'climate':     cand.loc[cand['name'] == name, 'climate_zone'].iloc[0],
        'n_pairs':     len(merged),
        'kge':         round(kge_v, 4),
        'r':           round(r,     4),
        'alpha':       round(alpha, 4),
        'beta':        round(beta,  4),
        'mean_sar_ha': round(float(sim.mean()), 1),
        'mean_jrc_ha': round(float(obs.mean()), 1),
    })

print('-' * len(hdr))

if skipped:
    print('\nSkipped:')
    for n, reason in skipped:
        print(f'  {n}: {reason}')

df_out = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df_out.to_csv(OUT_CSV, index=False)

print(f'\nSaved {len(df_out)} reservoirs -> {OUT_CSV}')
print(f'KGE > 0.5 : {(df_out["kge"] > 0.5).sum()} / {len(df_out)}')
print(f'KGE > 0.0 : {(df_out["kge"] > 0.0).sum()} / {len(df_out)}')
print(f'KGE <= 0.0: {(df_out["kge"] <= 0.0).sum()} / {len(df_out)}')
print(f'A/P range : {df_out["ap_m"].min():.0f} - {df_out["ap_m"].max():.0f} m')
print(f'KGE range : {df_out["kge"].min():.3f} - {df_out["kge"].max():.3f}')
