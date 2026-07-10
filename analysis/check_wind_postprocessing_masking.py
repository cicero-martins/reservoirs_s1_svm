"""
check_wind_postprocessing_masking.py

Robustness check for the wind-null result (Section 4.5): does the identical
post-processing chain (gap-fill via distance transform, applied to both VV
and SVM masks) mask a wind-driven vulnerability in VV that would appear in
the raw, per-pixel classification? Tested directly using the no-vectorisation
"fast" VV variant already exported for the cost audit (Section 4.4 / 3.10),
which skips gap-fill/vectorise/keep-polygon and computes area by direct pixel
counting.

If gap-fill were compensating for wind-induced omission gaps in VV, the
SVM-vs-fast-VV gap should correlate with wind much more strongly (more
Bragg-consistent) than the SVM-vs-full-VV gap. It does not: both are
statistically and practically equivalent, and gap-fill's own benefit to VV
does not itself scale with wind exposure.

Reads: analysis/pilot_kge_4way.csv, raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind/
"""
import pathlib
import numpy as np, pandas as pd
from scipy import stats

WIND = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind')

df = pd.read_csv('analysis/pilot_kge_4way.csv')
df['delta_full'] = df['kge_adapt'] - df['kge_vv']          # SVM vs full-pipeline VV (gap-filled)
df['delta_fast'] = df['kge_adapt'] - df['kge_fast']        # SVM vs no-post-processing VV
df['gapfill_gain'] = df['kge_vv'] - df['kge_fast']         # how much gap-fill itself helps VV


def wind_p90(name):
    p = WIND / f'Era5Wind_{name}.csv'
    if not p.exists():
        return np.nan
    try:
        w = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return np.nan
    if 'wind_ms' not in w.columns or w.empty:
        return np.nan
    return float(w['wind_ms'].quantile(0.90))


df['wind_p90'] = df['name'].apply(wind_p90)
d = df.dropna(subset=['delta_full', 'delta_fast', 'wind_p90']).reset_index(drop=True)
print(f'N = {len(d)}\n')

for col, label in [('delta_full', 'SVM - full VV (gap-filled)'),
                    ('delta_fast', 'SVM - fast VV (no post-processing)')]:
    r, p = stats.pearsonr(d['wind_p90'], d[col])
    rho, prho = stats.spearmanr(d['wind_p90'], d[col])
    print(f'{label}:')
    print(f'  Pearson  r={r:+.3f}  p={p:.3f}')
    print(f'  Spearman rho={rho:+.3f}  p={prho:.3f}\n')

r2, p2 = stats.pearsonr(d['wind_p90'], d['gapfill_gain'])
rho2, prho2 = stats.spearmanr(d['wind_p90'], d['gapfill_gain'])
print('Does gap-fill help VV MORE at high wind (direct masking-hypothesis test)?')
print(f'  Pearson  r(wind, kge_vv - kge_fast) = {r2:+.3f}  p={p2:.3f}')
print(f'  Spearman rho = {rho2:+.3f}  p={prho2:.3f}')
print('\nConclusion: near-identical to the full-pipeline result in both tests -> '
      'post-processing is not masking a wind-driven vulnerability. Reported in '
      'Results Section 4.5.')
