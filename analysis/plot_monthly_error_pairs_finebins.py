"""
plot_monthly_error_pairs_finebins.py

Same paired-error idea as plot_monthly_error_pairs_by_dynap.py, but with 10
quantile bins instead of 3 Low/Medium/High classes, to see the narrow
51-70m crossover (found via rolling-window scan, 9 Jul 2026) directly as a
panel-by-panel scatter instead of only as a table.

Output: analysis/method_comparison_output/monthly_error_pairs_finebins.png
"""
import numpy as np, pandas as pd

NBINS = 10
d = pd.read_csv('analysis/dynamic_ap_switch_pooled.csv')
d['err_vv'] = np.log(d['area_vv'] / d['area_jrc']).abs()
d['err_svm'] = np.log(d['area_svm'] / d['area_jrc']).abs()
d['bin'] = pd.qcut(d['ap_m_dynamic'], NBINS, duplicates='drop')
bins = sorted(d['bin'].unique(), key=lambda b: b.left)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ncols = 5
nrows = int(np.ceil(len(bins) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.7 * nrows), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
lo, hi = 0.0, 0.55

for ax, b in zip(axes, bins):
    sub = d[d['bin'] == b]
    n = len(sub)
    below = int((sub.err_svm < sub.err_vv).sum())
    pct = 100 * below / n
    color = '#2E7D32' if pct >= 50 else '#C62828'
    ax.scatter(sub.err_vv, sub.err_svm, s=12, c='#1565C0', alpha=0.4, linewidths=0, zorder=3)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.1, alpha=0.7, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_title(f'A/P {b.left:.0f}-{b.right:.0f}m (n={n})\nSVM better: {pct:.0f}%',
                 fontsize=9.5, color=color, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7.5)

for ax in axes[len(bins):]:
    ax.axis('off')

fig.supxlabel('|err$_{VV}$| = |log(area$_{VV}$/area$_{JRC}$)|', fontsize=10)
fig.supylabel('|err$_{SVM}$| = |log(area$_{SVM}$/area$_{JRC}$)|', fontsize=10)
fig.suptitle(f'Monthly paired area error by dynamic A/P decile ({NBINS} quantile bins)\n'
             'green title = SVM more accurate in >=50% of months that bin; red = VV more accurate\n'
             'points below the 1:1 line = SVM more accurate that month', fontsize=11.5, y=0.985)
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.06, top=0.82, hspace=0.55, wspace=0.15)
fig.savefig('analysis/method_comparison_output/monthly_error_pairs_finebins.png', dpi=155)
print('Saved: analysis/method_comparison_output/monthly_error_pairs_finebins.png')
for b in bins:
    sub = d[d['bin'] == b]
    below = int((sub.err_svm < sub.err_vv).sum())
    print(f'  {b.left:6.1f}-{b.right:6.1f}m  n={len(sub):4d}  SVM better {100*below/len(sub):5.1f}%')
