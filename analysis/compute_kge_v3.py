"""
compute_kge_v3.py

Full-period KGE (2014-10-01 to 2021-12-31) for the 22-reservoir pilot_v3 set.
Uses GEE_GlobalPilotV2c exports:
  SAR: raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2/SAR_area_*.csv
  JRC: raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC/JRC_area_*.csv

Pipeline corrections vs v2:
  - fillCoverageGaps applied in GEE export (fixes Harlan County partial swath)
  - No APP_OVERRIDES needed (all reservoirs use their own export CSV)
  - No S1C filter (data ends 2021-12-31, before S1C launch Dec 2024)
  - 4 new reservoirs: Kerkini, Caballo, Curwensville, Erfenis

Output: analysis/pilot_kge_v3.csv
"""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats


# ── App-equivalent cleaning pipeline (mirrors original.js cleanAndSmooth) ────
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
    for t0, _ in zip(dates, values):
        diff_d = np.abs((dates - t0).dt.total_seconds().values / 86400)
        mask   = diff_d <= window_days
        w      = np.exp(-(diff_d[mask] / bandwidth) ** 2)
        smoothed.append(float((values[mask] * w).sum() / w.sum()))
    return np.array(smoothed)

def clean_and_smooth(df, col='area_ha'):
    """removeOutliers(2) + detectAndRemoveLocal(5,1.5)×2 + (10,1.5) + lowess(20,7)."""
    s = df[col].copy()
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 10, 1.5)
    dates_s  = df.loc[s.index, 'date'].reset_index(drop=True)
    smoothed = _lowess(dates_s, s.reset_index(drop=True), 20, 7)
    return pd.DataFrame({'date': dates_s, 'area_ha': smoothed})

SAR_DIR  = pathlib.Path('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2')
JRC_DIR  = pathlib.Path('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
OUT_CSV  = pathlib.Path('analysis/pilot_kge_v3.csv')

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
MIN_PAIRS      = 12

EXCLUDE   = {'Eder', 'Curwensville', 'Hugo_Lake', 'Caballo'}
NO_SMOOTH = {'Hubbard_Creek'}  # raw signal already clean; LOWESS over-attenuates

# App-extracted SAR series override: {name: (path, date_col, area_col)}
APP_OVERRIDES = {
    'Ancipa': (
        pathlib.Path('C:/Users/Unipa/Documents/GEE/Results/fractaldim/area_ancipa_2014-25.csv'),
        'date', 'areaLago',
    ),
}


def kge(obs, sim):
    if obs.std() == 0 or sim.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = sim.std()  / obs.std()
    beta  = sim.mean() / obs.mean()
    return 1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2), r, alpha, beta


def load_sar_monthly(name, smooth=True):
    # ap_m always from batch export if available
    batch_p = SAR_DIR / f'SAR_area_{name}.csv'
    ap_m = np.nan
    if batch_p.exists():
        _b = pd.read_csv(batch_p)
        if not _b.empty and 'ap_m' in _b.columns:
            ap_m = float(_b['ap_m'].iloc[0])

    if name in APP_OVERRIDES:
        path, dcol, acol = APP_OVERRIDES[name]
        df = pd.read_csv(path, parse_dates=[dcol])
        df = df.rename(columns={dcol: 'date', acol: 'area_ha'})
    else:
        if not batch_p.exists():
            return None, np.nan
        df = pd.read_csv(batch_p, parse_dates=['date'])

    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if df.empty:
        return None, ap_m
    # limit to JRC period
    df = df[df['date'] <= '2021-12-31'].copy()
    p99 = df['area_ha'].quantile(0.99)
    df  = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    if df.empty:
        return None, ap_m
    # clean and smooth (mirrors original.js pipeline) — skip for NO_SMOOTH set
    if smooth:
        df = clean_and_smooth(df.reset_index(drop=True))
    if df.empty:
        return None, ap_m
    df['ym'] = df['date'].dt.to_period('M')
    monthly  = df.groupby('ym')['area_ha'].mean().reset_index()
    monthly.columns = ['ym', 'sar_area_ha']
    return monthly, ap_m


def load_jrc_monthly(name):
    p = JRC_DIR / f'JRC_area_{name}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= VALID_FRAC_MIN].copy()
    if df.empty:
        return None
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()


# ── Reservoir list (ordered as in exportGlobalPilotV2.js) ────────────────────
NAMES = [
    'Ancipa', 'Poma', 'Pozzillo', 'Rosamarina', 'Garcia',
    'Yesa', 'Puente_Nuevo', 'Alcantara', 'Caia', 'Kerkini',
    'Eder', 'Forggen',
    'Caballo', 'Curwensville', 'Hugo_Lake', 'Hubbard_Creek', 'Harlan_County',
    'Umbuluzi', 'Erfenis',
    'Vani_Vilasa',
    'Paraibuna', 'Contas',
]

rows    = []
skipped = []

hdr = (f'{"Name":<22}  {"A/P":>5}  {"n":>4}  '
       f'{"KGE":>6}  {"r":>5}  {"alpha":>5}  {"beta":>5}  '
       f'{"SAR mean":>8}  {"JRC mean":>8}')
print(hdr)
print('-' * len(hdr))

for name in NAMES:
    if name in EXCLUDE:
        skipped.append((name, 'excluded'))
        continue
    sar_m, ap_m = load_sar_monthly(name, smooth=(name not in NO_SMOOTH))
    jrc_m       = load_jrc_monthly(name)

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

    rows.append({
        'name':        name,
        'ap_m':        round(ap_m, 1) if not np.isnan(ap_m) else np.nan,
        'n_pairs':     len(merged),
        'kge':         round(kge_v, 4),
        'r':           round(r, 4),
        'alpha':       round(alpha, 4),
        'beta':        round(beta, 4),
        'mean_sar_ha': round(float(sim.mean()), 1),
        'mean_jrc_ha': round(float(obs.mean()), 1),
    })

    print(f'  {name:<22}  {ap_m:>5.0f}  {len(merged):>4d}  '
          f'{kge_v:>6.3f}  {r:>5.3f}  {alpha:>5.3f}  {beta:>5.3f}  '
          f'{sim.mean():>8.1f}  {obs.mean():>8.1f}')

print('-' * len(hdr))

if skipped:
    print('\nSkipped:')
    for n, reason in skipped:
        print(f'  {n}: {reason}')

df_out = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df_out.to_csv(OUT_CSV, index=False)

print(f'\nSaved {len(df_out)} reservoirs: {OUT_CSV}')
print(f'KGE > 0.5 : {(df_out["kge"] > 0.5).sum()} / {len(df_out)}')
print(f'KGE > 0.0 : {(df_out["kge"] > 0.0).sum()} / {len(df_out)}')
print(f'KGE <= 0.0: {(df_out["kge"] <= 0.0).sum()} / {len(df_out)}')
print(f'A/P range : {df_out["ap_m"].min():.0f} – {df_out["ap_m"].max():.0f} m')
print(f'KGE range : {df_out["kge"].min():.3f} – {df_out["kge"].max():.3f}')
