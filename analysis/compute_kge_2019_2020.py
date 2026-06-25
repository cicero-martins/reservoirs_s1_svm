"""
compute_kge_2019_2020.py

KGE computation for the 2019-2020 reference window.

Special cases:
  Harlan_County — uses app-extracted series (raw_data/ee-chart_HarlanCounty.csv)
                  which went through the full app pipeline (fillCoverageGaps +
                  cleanAndSmooth). Area column has comma-formatted numbers.
  Elwell        — excluded (GEE user-memory crash during rendering)
  Sterkfontein  — excluded (GEE user-memory crash during rendering)

All other reservoirs use the standard GEE_GlobalPilotV2b CSVs filtered to 2019-2020.

Output: analysis/pilot_kge_2019_2020.csv
"""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR   = pathlib.Path('raw_data/GEE_GlobalPilotV2b')
APP_DIR    = pathlib.Path('raw_data')   # ee-chart_*.csv files
OUT_CSV    = pathlib.Path('analysis/pilot_kge_2019_2020.csv')

PERIOD_START   = '2019-01-01'
PERIOD_END     = '2020-12-31'
VALID_FRAC_MIN = 0.80
S1C_DATE       = pd.Timestamp('2024-12-01')   # not active in 2019-2020, included for safety
SAR_MIN_FRAC   = 0.02
MIN_PAIRS      = 8    # lower minimum for 2-year window

EXCLUDE = {'Elwell', 'Sterkfontein'}

# Reservoirs with a manually app-extracted CSV (columns: date, area_ha[, area_ha_smoothed])
APP_OVERRIDES = {
    'Harlan_County': APP_DIR / 'ee-chart_HarlanCounty.csv',
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def kge(obs, sim):
    """KGE (Gupta et al. 2009). obs=JRC, sim=SAR."""
    if obs.std() == 0 or sim.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = sim.std()  / obs.std()
    beta  = sim.mean() / obs.mean()
    return 1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2), r, alpha, beta


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sar_monthly_export(name):
    """Standard export CSV → filter 2019-2020 → monthly mean. Returns (df_monthly, ap_m)."""
    p = DATA_DIR / f'SAR_area_{name}.csv'
    if not p.exists():
        return None, np.nan
    df = pd.read_csv(p, parse_dates=['date'])
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    ap_m = float(df['ap_m'].iloc[0]) if not df.empty else np.nan

    df = df[(df['date'] >= PERIOD_START) & (df['date'] <= PERIOD_END)].copy()
    df = df[df['date'] < S1C_DATE].copy()
    if df.empty:
        return None, ap_m

    p99 = df['area_ha'].quantile(0.99)
    df = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    m, s = df['area_ha'].mean(), df['area_ha'].std()
    if s > 0:
        df = df[np.abs(df['area_ha'] - m) <= 3.0 * s].copy()
    if df.empty:
        return None, ap_m

    df['ym'] = df['date'].dt.to_period('M')
    monthly  = df.groupby('ym')['area_ha'].mean().reset_index()
    monthly.columns = ['ym', 'sar_area_ha']
    return monthly, ap_m


def load_sar_monthly_app(name, csv_path):
    """App-extracted ee-chart CSV → parse comma-formatted numbers → monthly mean."""
    df = pd.read_csv(csv_path)
    # First two columns: date and area_ha (third is smoothed — use raw)
    df.columns = ['date', 'area_ha'] + [f'_c{i}' for i in range(len(df.columns)-2)]
    df['date']    = pd.to_datetime(df['date'])
    df['area_ha'] = (df['area_ha'].astype(str)
                     .str.replace(',', '', regex=False)
                     .astype(float))
    df = df[(df['date'] >= PERIOD_START) & (df['date'] <= PERIOD_END)].copy()
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if df.empty:
        return None

    # A/P not in the app CSV — read from the export CSV if available
    exp_p = DATA_DIR / f'SAR_area_{name}.csv'
    ap_m  = np.nan
    if exp_p.exists():
        try:
            ap_m = float(pd.read_csv(exp_p, nrows=1)['ap_m'].iloc[0])
        except Exception:
            pass

    df['ym'] = df['date'].dt.to_period('M')
    monthly  = df.groupby('ym')['area_ha'].mean().reset_index()
    monthly.columns = ['ym', 'sar_area_ha']
    return monthly, ap_m


def load_jrc_monthly(name):
    """JRC export CSV filtered to 2019-2020 with valid_frac >= threshold."""
    p = DATA_DIR / f'JRC_area_{name}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = df[(df['date'] >= PERIOD_START) & (df['date'] <= PERIOD_END)].copy()
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= VALID_FRAC_MIN].copy()
    if df.empty:
        return None
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()


# ── Main ──────────────────────────────────────────────────────────────────────

sar_names = {p.stem[len('SAR_area_'):] for p in DATA_DIR.glob('SAR_area_*.csv')}
jrc_names = {p.stem[len('JRC_area_'):] for p in DATA_DIR.glob('JRC_area_*.csv')}
# Add app-override names (they have JRC CSVs in DATA_DIR)
all_names = sorted((sar_names | set(APP_OVERRIDES)) & (jrc_names | set(APP_OVERRIDES)))
all_names = [n for n in all_names if n not in EXCLUDE]

rows = []
skipped = []

hdr = (f'{"Name":<22}  {"A/P":>5}  {"src":>4}  {"n":>4}  '
       f'{"KGE":>6}  {"r":>5}  {"alpha":>5}  {"beta":>5}  '
       f'{"SAR mean":>8}  {"JRC mean":>8}')
print(hdr)
print('-' * len(hdr))

for name in all_names:
    # SAR
    if name in APP_OVERRIDES:
        result = load_sar_monthly_app(name, APP_OVERRIDES[name])
        if result is None:
            skipped.append((name, 'app CSV empty in 2019-2020'))
            continue
        sar_m, ap_m = result
        src = 'app'
    else:
        sar_m, ap_m = load_sar_monthly_export(name)
        src = 'exp'

    jrc_m = load_jrc_monthly(name)

    if sar_m is None or jrc_m is None or sar_m.empty or jrc_m.empty:
        skipped.append((name, 'no data in 2019-2020'))
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
        'source':      src,
        'n_pairs':     len(merged),
        'kge':         round(kge_v, 4),
        'r':           round(r, 4),
        'alpha':       round(alpha, 4),
        'beta':        round(beta, 4),
        'mean_sar_ha': round(float(sim.mean()), 1),
        'mean_jrc_ha': round(float(obs.mean()), 1),
    })

    print(f'  {name:<22}  {ap_m:>5.0f}  {src:>4}  {len(merged):>4d}  '
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
print(f'Period: {PERIOD_START} to {PERIOD_END}')
print(f'KGE > 0.5 : {(df_out["kge"] > 0.5).sum()} / {len(df_out)}')
print(f'KGE > 0.0 : {(df_out["kge"] > 0.0).sum()} / {len(df_out)}')
print(f'KGE <= 0.0: {(df_out["kge"] <= 0.0).sum()} / {len(df_out)}')
print(f'\nA/P range : {df_out["ap_m"].min():.0f} - {df_out["ap_m"].max():.0f} m')
print(f'KGE range : {df_out["kge"].min():.3f} - {df_out["kge"].max():.3f}')
