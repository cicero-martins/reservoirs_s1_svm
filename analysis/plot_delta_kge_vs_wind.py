"""
plot_delta_kge_vs_wind.py

Accuracy view (not divergence): does wind predict WHICH method is more accurate
against JRC?  ΔKGE = KGE_dual − KGE_vv per reservoir (common months).
  ΔKGE > 0 → dual-pol SVM more accurate;  ΔKGE < 0 → VV-only Otsu more accurate.

Bragg prediction: wind → VV worse → ΔKGE more positive. Tested here vs ERA5 p90
wind, colored by A/P. Companion to analyze_wind_divergence.py (which tests area
divergence). Reads analysis/pilot_kge_compare.csv + GEE_Era5Wind/*.csv.

Output: analysis/method_comparison_output/delta_kge_vs_wind.png
"""

import pathlib
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

WIND = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind')
OUT  = pathlib.Path('analysis/method_comparison_output/delta_kge_vs_wind.png')

df = pd.read_csv('analysis/pilot_kge_compare.csv')


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
d = df.dropna(subset=['delta_kge', 'wind_p90', 'ap_m']).reset_index(drop=True)

r, p   = stats.pearsonr(d['wind_p90'], d['delta_kge'])
slope, icpt, *_ = stats.linregress(d['wind_p90'], d['delta_kge'])
print(f'r(wind_p90, ΔKGE) = {r:+.3f}  p = {p:.3f}  (ΔKGE>0 → dual better)')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9.5, 6.5))
sc = ax.scatter(d['wind_p90'], d['delta_kge'], c=d['ap_m'], cmap='viridis',
                s=70, edgecolors='white', linewidths=0.6, zorder=4)
xf = np.linspace(d['wind_p90'].min(), d['wind_p90'].max(), 50)
ax.plot(xf, slope * xf + icpt, 'k-', lw=1.6, alpha=0.8, zorder=3,
        label=f'linear: r={r:.2f}, p={p:.2f} (NS)')
ax.axhline(0, color='gray', lw=1, ls=':', zorder=2)
for _, row in d.iterrows():
    ax.annotate(row['name'].replace('_', ' '), (row['wind_p90'], row['delta_kge']),
                fontsize=6, xytext=(4, 3), textcoords='offset points', color='#444')
ax.text(0.02, 0.97, 'dual-pol mais preciso', transform=ax.transAxes,
        va='top', fontsize=8, color='#1f77b4', fontweight='bold')
ax.text(0.02, 0.03, 'VV-only mais preciso', transform=ax.transAxes,
        va='bottom', fontsize=8, color='#2ca02c', fontweight='bold')
ax.set_xlabel('ERA5 wind exposure, p90 (m/s)  —  radiometric')
ax.set_ylabel('ΔKGE = KGE$_{dual}$ − KGE$_{vv}$  (common months)')
ax.set_title('Accuracy: does wind decide whether dual-pol beats VV-only?\n'
             'No — wind is not a significant predictor; A/P (color) is',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.25)
plt.colorbar(sc, ax=ax, label='A/P static (m)')
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
