"""Re-test: does wind predict which per-scene detector (adapt SVM vs VV Otsu)
is more accurate, on the CURRENT full global 4-way set (N=50), now that wind
data has been extended beyond the original 28-reservoir subsample?

delta = kge_adapt - kge_vv  (>0 -> dual-pol more accurate; <0 -> VV-only more accurate)
Bragg prediction: wind -> VV worse -> delta more positive.

Reads analysis/pilot_kge_4way.csv + raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind/*.csv
(now extended to the full set by fetch_era5_wind_missing.py).
Output: analysis/method_comparison_output/delta_kge_vs_wind_global.png
"""
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

WIND = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind')
OUT = pathlib.Path('analysis/method_comparison_output/delta_kge_vs_wind_global.png')

df = pd.read_csv('analysis/pilot_kge_4way.csv')
df['delta'] = df['kge_adapt'] - df['kge_vv']


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
d = df.dropna(subset=['delta', 'wind_p90', 'ap_m']).reset_index(drop=True)
print(f"N with wind data = {len(d)} of {len(df)}")
ids = pd.read_csv('analysis/reservoir_ids.csv')[['name', 'id']].set_index('name')
d['id'] = d['name'].map(ids['id']).fillna(d['name'].str.replace('_', ' '))

r, p = stats.pearsonr(d['wind_p90'], d['delta'])
rho, p_rho = stats.spearmanr(d['wind_p90'], d['delta'])
slope, icpt, *_ = stats.linregress(d['wind_p90'], d['delta'])
print(f"Pearson  r(wind_p90, delta) = {r:+.3f}  p = {p:.3f}")
print(f"Spearman rho              = {rho:+.3f}  p = {p_rho:.3f}")
print(f"(delta>0 -> dual/adapt more accurate; Bragg predicts r<0)")

# partial check controlling for A/P (since wind exposure and A/P may correlate)
rho_ap_wind, _ = stats.spearmanr(d['ap_m'], d['wind_p90'])
print(f"Spearman(A/P, wind_p90) = {rho_ap_wind:+.3f}  (confound check)")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text

fig, ax = plt.subplots(figsize=(11, 7.5))
sc = ax.scatter(d['wind_p90'], d['delta'], c=d['ap_m'], cmap='viridis',
                 s=70, edgecolors='white', linewidths=0.6, zorder=4)
xf = np.linspace(d['wind_p90'].min(), d['wind_p90'].max(), 50)
ax.plot(xf, slope * xf + icpt, 'k-', lw=1.6, alpha=0.8, zorder=3,
        label=f'linear: r={r:.2f}, p={p:.2f}')
ax.axhline(0, color='gray', lw=1, ls=':', zorder=2)
texts = [ax.text(row['wind_p90'], row['delta'], row['id'], fontsize=8,
                  color='#444', zorder=10) for _, row in d.iterrows()]
adjust_text(texts, x=d['wind_p90'].values, y=d['delta'].values, ax=ax,
            expand=(1.6, 1.8), force_static=(0.7, 0.8), force_text=(0.3, 0.4),
            arrowprops=dict(arrowstyle='-', color='#999', lw=0.5))
ax.set_xlabel('ERA5 wind exposure, p90 (m/s)')
ax.set_ylabel(r'$\Delta$KGE = KGE$_{adapt}$ - KGE$_{vv}$ (common months)')
ax.set_title(f'Wind vs accuracy gap, full global set (n={len(d)})', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.25)
plt.colorbar(sc, ax=ax, label='A/P static (m)')
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT}")
