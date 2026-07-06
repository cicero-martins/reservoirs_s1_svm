"""
compute_kge_apcurve.py  (v2 — best-of, de-spiked JRC)

The headline A/P → monitorability curve, now on the FINAL metric: best-of(adapt, vv)
KGE vs the de-spiked JRC reference (from bestof_kge.csv → compute_bestof_kge.py). The
retired dual-FIXED method and the 4 flat-JRC "chapados" are excluded; the 4 Sicilian
(JRC period = dual-only) drop out automatically (best=NaN). Points are coloured by the
winning method (adapt vs VV-Otsu) so the "1-band is enough / better" story is visible on
the same axes as the geometric-predictor story.

Reads:  analysis/bestof_kge.csv
Output: analysis/pilot_kge_apcurve.csv
        analysis/method_comparison_output/ap_kge_curve_pooled.png
"""
import pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('analysis/bestof_kge.csv')
df = df.dropna(subset=['best'])                     # drops the 4 Sicilian (dual-only)
n_all = len(df)
df = df[~df['chapado']].copy()                      # drop flat-JRC chapados
df = df.sort_values('ap_m').reset_index(drop=True)
df.to_csv('analysis/pilot_kge_apcurve.csv', index=False)

r, p = stats.spearmanr(df['ap_m'], df['best'])
print(f'A/P → best-of KGE (de-spiked JRC), N={len(df)} (from {n_all}, −{n_all-len(df)} chapados)')
print(f'  Spearman ρ = {r:+.3f}  p = {p:.2e}   median KGE = {df["best"].median():.3f}')
lo, hi = df[df.ap_m < 100], df[df.ap_m >= 200]
print(f'  A/P<100  (n={len(lo)}): median {lo["best"].median():.3f}')
print(f'  A/P>=200 (n={len(hi)}): median {hi["best"].median():.3f}')
print(f'  winner: VV-Otsu {int((df.winner=="vv").sum())} | adapt {int((df.winner=="adapt").sum())}')

# ── figure ────────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WCOL = {'vv': '#2ca02c', 'adapt': '#1f77b4'}
fig, ax = plt.subplots(figsize=(11, 7))
for w, c in WCOL.items():
    s = df[df.winner == w]
    ax.scatter(s['ap_m'], s['best'], s=70, color=c, alpha=0.85, edgecolors='white',
               linewidths=0.7, zorder=4, label=f'best = {"VV-Otsu" if w=="vv" else "adapt SVM"} '
               f'(n={len(s)})')
for _, r_ in df.iterrows():
    ax.annotate(r_['name'].replace('_', ' '), (r_['ap_m'], r_['best']), fontsize=5.5,
                xytext=(3, 2), textcoords='offset points', color='#666')

edges = [df.ap_m.min() - 1, 90, 130, 180, 260, df.ap_m.max() + 1]
df['bin'] = pd.cut(df['ap_m'], edges)
g = df.groupby('bin', observed=True)
ax.plot(g['ap_m'].median().values, g['best'].median().values, 'k-o', lw=2, ms=7, zorder=6,
        label='binned median')

ax.axhline(0.5, color='#999', ls=':', lw=1, zorder=2)
ax.set_xlabel('A/P — static area/perimeter (m)', fontsize=11)
ax.set_ylabel('KGE — best of {adapt, VV-Otsu} vs JRC', fontsize=11)
ax.set_title(f'A/P → monitorability (best-of method, de-spiked JRC), N={len(df)}\n'
             f'single continuous trend (Spearman ρ={r:+.2f}, p={p:.1e}); '
             f'VV-Otsu best in {int((df.winner=="vv").sum())}/{len(df)}',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.25); ax.legend(fontsize=9, loc='lower right'); ax.set_ylim(-0.1, 1.02)
OUT = pathlib.Path('analysis/method_comparison_output/ap_kge_curve_pooled.png')
fig.savefig(OUT, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {OUT}')
