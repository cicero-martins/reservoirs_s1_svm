"""
plot_monthly_error_pairs_by_dynap.py

Month-by-month paired comparison of |err_VV| vs |err_SVM| (log-ratio area
error against JRC that month), split into 3 panels by THAT MONTH's own
dynamic A/P (Otsu's own per-scene geometry, low/medium/high), pooled across
all reservoirs. Finer-grained than the per-reservoir paired-KGE scatter
(kge_paired_by_dynap.png): every reservoir contributes many points (one per
valid month), and each point is classified by the A/P condition AT THAT
MONTH, not by the reservoir's overall/median A/P. Direct visual test of
whether the point cloud shifts below the 1:1 line (SVM more accurate) in the
low-A/P panel specifically.

Reuses analysis/dynamic_ap_switch_pooled.csv (from dynamic_ap_switch_experiment.py,
9 Jul 2026): columns name, ym, ap_m_dynamic, area_vv, area_svm, area_jrc.

Output: analysis/method_comparison_output/monthly_error_pairs_by_dynap.png
"""
import numpy as np, pandas as pd
from scipy import stats

LOW_MAX, HIGH_MIN = 100.0, 200.0
def ap_class(ap): return 'Low' if ap < LOW_MAX else ('Medium' if ap < HIGH_MIN else 'High')
CLASS_COLOR = {'Low': '#f88f4d', 'Medium': '#d64a02', 'High': '#8a2d04'}

d = pd.read_csv('analysis/dynamic_ap_switch_pooled.csv')
d['err_vv'] = np.log(d['area_vv'] / d['area_jrc']).abs()
d['err_svm'] = np.log(d['area_svm'] / d['area_jrc']).abs()
d['cls'] = d['ap_m_dynamic'].map(ap_class)

print(f'N = {len(d)} reservoir-months across {d.name.nunique()} reservoirs')
print(d['cls'].value_counts())
print()

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
lo, hi = 0.0, min(1.2, max(d.err_vv.quantile(0.99), d.err_svm.quantile(0.99)))
for ax, c in zip(axes, ['Low', 'Medium', 'High']):
    sub = d[d.cls == c]
    below = int((sub.err_svm < sub.err_vv).sum())  # SVM more accurate that month
    n = len(sub)
    ax.scatter(sub.err_vv, sub.err_svm, s=10, c=CLASS_COLOR[c], alpha=0.35, linewidths=0, zorder=3)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.7, zorder=4, label='1:1')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_title(f'{c} dynamic A/P  (n={n})\nSVM more accurate in {below}/{n} months ({100*below/n:.0f}%)',
                 fontsize=10.5)
    ax.set_xlabel('|err$_{VV}$| = |log(area$_{VV}$/area$_{JRC}$)|', fontsize=9)
    ax.grid(alpha=0.2); ax.legend(fontsize=8, loc='upper right')
axes[0].set_ylabel('|err$_{SVM}$| = |log(area$_{SVM}$/area$_{JRC}$)|', fontsize=9)

rng = f'<{LOW_MAX:.0f}m / {LOW_MAX:.0f}-{HIGH_MIN:.0f}m / >={HIGH_MIN:.0f}m'
fig.suptitle(f"Monthly paired area error, VV-Otsu vs SVM, split by that month's own dynamic A/P ({rng})\n"
             f'points below the 1:1 line = SVM more accurate that month', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('analysis/method_comparison_output/monthly_error_pairs_by_dynap.png', dpi=160)
print('Saved: analysis/method_comparison_output/monthly_error_pairs_by_dynap.png')

print()
for c in ['Low', 'Medium', 'High']:
    sub = d[d.cls == c]
    w, p = stats.wilcoxon(sub.err_vv, sub.err_svm)
    below = int((sub.err_svm < sub.err_vv).sum())
    print(f'{c:7s} n={len(sub):4d}  SVM more accurate: {below}/{len(sub)} ({100*below/len(sub):.1f}%)  '
          f'median |err_vv|={sub.err_vv.median():.3f} median |err_svm|={sub.err_svm.median():.3f}  '
          f'Wilcoxon p={p:.2e}')
