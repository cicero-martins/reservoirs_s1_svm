"""Save the A/P-quintile robustness check (per-scene SVM-adapt vs VV-Otsu gap)
as a table + figure, so it isn't console-only. Confirms no A/P regime shows a
reliable dual-polarisation advantage.
"""
import numpy as np
import pandas as pd
from scipy import stats

fw = pd.read_csv('analysis/pilot_kge_4way.csv')
fw['d'] = fw.kge_adapt - fw.kge_vv   # >0 -> dual/adapt SVM more accurate
fw['q'] = pd.qcut(fw.ap_m, 5)

g = fw.groupby('q', observed=True).agg(
    n=('d', 'size'), median_d=('d', 'median'), mean_d=('d', 'mean'),
    dual_win=('d', lambda x: (x > 0.02).sum()),
    vv_win=('d', lambda x: (x < -0.02).sum()),
    ap_lo=('ap_m', 'min'), ap_hi=('ap_m', 'max'),
)
g.to_csv('analysis/ap_quintile_robustness.csv')
print(g.to_string())

rho, p = stats.spearmanr(fw.ap_m, fw.d)
print(f"\nSpearman(A/P, adapt-vv) = {rho:.3f}  p = {p:.3f}  (overall, N={len(fw)})")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6))
labels = [f"{int(r.ap_lo)}-{int(r.ap_hi)} m\n(n={int(r.n)})" for _, r in g.iterrows()]
colors = ['#1f77b4' if m > 0 else '#d62728' for m in g.median_d]
ax.bar(range(len(g)), g.median_d, color=colors, edgecolor='white', width=0.6, zorder=3)
ax.axhline(0, color='black', lw=1, zorder=2)
ax.set_xticks(range(len(g)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(r'Median $\Delta$KGE = KGE$_{adapt}$ - KGE$_{vv}$')
ax.set_title('Dual-pol advantage by A/P quintile (blue = dual wins, red = VV wins)\n'
             f'No monotonic pattern; overall Spearman(A/P, gap) rho={rho:.2f} p={p:.2f}',
             fontsize=10, fontweight='bold')
ax.grid(alpha=0.25, axis='y')
out = 'analysis/method_comparison_output/ap_quintile_robustness.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved table -> analysis/ap_quintile_robustness.csv")
print(f"Saved figure -> {out}")
