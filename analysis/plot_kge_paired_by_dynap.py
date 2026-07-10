"""
plot_kge_paired_by_dynap.py

Paired scatter (KGE_VV vs KGE_SVM-adapt) for the 62-reservoir clean set,
coloured by each reservoir's median per-scene DYNAMIC A/P (not the static
JRC-max-extent A/P used everywhere else in the paper), to visually check
whether points above/below the 1:1 line cluster by dynamic A/P class.
Companion figure to the Wilcoxon signed-rank test (see
dynamic_ap_switch_experiment.py / conversation 9 Jul 2026): if SVM's edge
were concentrated at low dynamic A/P, the "Low" points should sit
systematically above the 1:1 line and "High" points below it. Confirms/
refutes that visually alongside the formal test.

Output: analysis/method_comparison_output/kge_paired_by_dynap.png
"""
import pandas as pd, numpy as np
from scipy import stats

LOW_MAX, HIGH_MIN = 100.0, 200.0
def ap_class(ap): return 'Low' if ap < LOW_MAX else ('Medium' if ap < HIGH_MIN else 'High')
CLASS_COLOR = {'Low': '#f88f4d', 'Medium': '#d64a02', 'High': '#8a2d04'}

clean_names = pd.read_csv('analysis/pilot_kge_apcurve.csv')['name'].tolist()
best = pd.read_csv('analysis/bestof_kge.csv')
df = best[best.name.isin(clean_names)].dropna(subset=['kge_adapt', 'kge_vv']).copy()

dynap = pd.read_csv('analysis/_dyn_ap_median_per_reservoir.csv')
df = df.merge(dynap, on='name', how='left').dropna(subset=['ap_dyn_med'])
df['cls'] = df['ap_dyn_med'].map(ap_class)
df['delta'] = df['kge_adapt'] - df['kge_vv']

print(f'N = {len(df)}')
print(df['cls'].value_counts())

wstat, wp = stats.wilcoxon(df['kge_adapt'], df['kge_vv'])
print(f'\nWilcoxon signed-rank (paired, kge_adapt vs kge_vv): W={wstat:.1f}  p={wp:.3f}')
for c in ['Low', 'Medium', 'High']:
    sub = df[df.cls == c]
    n_svm = int((sub.delta > 0).sum())
    print(f'  {c:7s} n={len(sub):2d}  SVM wins {n_svm}/{len(sub)}  '
          f'median delta={sub.delta.median():+.3f}')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 7))
lo, hi = -1.05, 1.05
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.7, zorder=2, label='1:1 (VV = SVM)')
ax.axhline(0, color='#ddd', lw=0.8, zorder=1); ax.axvline(0, color='#ddd', lw=0.8, zorder=1)
for c in ['Low', 'Medium', 'High']:
    sub = df[df.cls == c]
    ax.scatter(sub.kge_vv, sub.kge_adapt, s=70, c=CLASS_COLOR[c], edgecolors='white',
               linewidths=0.8, alpha=0.9, zorder=4,
               label=f'{c} dynamic A/P (n={len(sub)})')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
ax.set_xlabel('KGE, VV-Otsu (vs JRC)')
ax.set_ylabel('KGE, SVM per-scene (vs JRC)')
ax.set_title(f'Paired KGE by reservoir, coloured by median dynamic A/P\n'
             f'N={len(df)}, Wilcoxon p={wp:.2f} (NS) -- points above the line: SVM wins',
             fontsize=10.5)
ax.legend(fontsize=9, loc='lower right', frameon=True)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig('analysis/method_comparison_output/kge_paired_by_dynap.png', dpi=160)
print('\nSaved: analysis/method_comparison_output/kge_paired_by_dynap.png')
