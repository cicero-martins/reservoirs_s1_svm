"""
compute_kge.py

Compute Kling-Gupta Efficiency (KGE) between SAR water area and JRC monthly area
for all pilot_v2 reservoirs in raw_data/GEE_GlobalPilotV2b.

Temporal alignment: SAR observations are aggregated to monthly mean, then matched
to JRC by year-month. Only pre-S1C SAR data (before 2024-12-01) is used.

Output: analysis/pilot_kge_v2.csv
"""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV2b')
OUT_CSV  = pathlib.Path('analysis/pilot_kge_v2.csv')

VALID_FRAC_MIN = 0.80       # JRC cloud-cover filter
S1C_DATE       = pd.Timestamp('2024-12-01')
SAR_MIN_FRAC   = 0.02       # 2% of p99 area — removes wind misclassification
MIN_PAIRS      = 12         # minimum co-located months to compute KGE


# ── Metrics ───────────────────────────────────────────────────────────────────

def kge(obs, sim):
    """Kling-Gupta Efficiency (Gupta et al. 2009). obs=JRC reference, sim=SAR."""
    if obs.std() == 0 or sim.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = sim.std()  / obs.std()
    beta  = sim.mean() / obs.mean()
    kge_v = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge_v, r, alpha, beta


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sar_monthly(name):
    """SAR: filter noise + spikes, aggregate to monthly mean. Returns (df_monthly, ap_m)."""
    p = DATA_DIR / f'SAR_area_{name}.csv'
    if not p.exists():
        return None, np.nan
    df = pd.read_csv(p, parse_dates=['date'])
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    ap_m = float(df['ap_m'].iloc[0]) if not df.empty else np.nan

    # S1A/B only
    df = df[df['date'] < S1C_DATE].copy()
    if df.empty:
        return None, ap_m

    # Wind-misclassification filter
    p99 = df['area_ha'].quantile(0.99)
    df = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()

    # Global 3σ spike filter
    m, s = df['area_ha'].mean(), df['area_ha'].std()
    if s > 0:
        df = df[np.abs(df['area_ha'] - m) <= 3.0 * s].copy()

    if df.empty:
        return None, ap_m

    df['ym'] = df['date'].dt.to_period('M')
    monthly = df.groupby('ym')['area_ha'].mean().reset_index()
    monthly.columns = ['ym', 'sar_area_ha']
    return monthly, ap_m


def load_jrc_monthly(name):
    """JRC: apply valid_frac filter, return year-month + area."""
    p = DATA_DIR / f'JRC_area_{name}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= VALID_FRAC_MIN].copy()
    if df.empty:
        return None
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()


# ── Main ──────────────────────────────────────────────────────────────────────

sar_names = {p.stem[len('SAR_area_'):] for p in DATA_DIR.glob('SAR_area_*.csv')}
jrc_names = {p.stem[len('JRC_area_'):] for p in DATA_DIR.glob('JRC_area_*.csv')}
names = sorted(sar_names & jrc_names)

rows = []
skipped = []

hdr = f'{"Name":<22}  {"A/P":>5}  {"n":>4}  {"KGE":>6}  {"r":>5}  {"alpha":>5}  {"beta":>5}  {"SAR mean":>8}  {"JRC mean":>8}'
print(hdr)
print('-' * len(hdr))

for name in names:
    sar_m, ap_m = load_sar_monthly(name)
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
        'name':       name,
        'ap_m':       round(ap_m, 1) if not np.isnan(ap_m) else np.nan,
        'n_pairs':    len(merged),
        'kge':        round(kge_v, 4),
        'r':          round(r, 4),
        'alpha':      round(alpha, 4),
        'beta':       round(beta, 4),
        'mean_sar_ha': round(float(sim.mean()), 1),
        'mean_jrc_ha': round(float(obs.mean()), 1),
    })

    print(f'  {name:<22}  {ap_m:>5.0f}  {len(merged):>4d}  {kge_v:>6.3f}'
          f'  {r:>5.3f}  {alpha:>5.3f}  {beta:>5.3f}'
          f'  {sim.mean():>8.1f}  {obs.mean():>8.1f}')

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
print(f'\nA/P range: {df_out["ap_m"].min():.0f} – {df_out["ap_m"].max():.0f} m')
print(f'KGE range: {df_out["kge"].min():.3f} – {df_out["kge"].max():.3f}')
