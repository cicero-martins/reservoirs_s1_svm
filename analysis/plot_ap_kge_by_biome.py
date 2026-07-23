"""Complementary view: does the A/P->KGE relationship hold WITHIN each biome,
or is it only a pooled/confounded effect? Scatter of A/P vs KGE coloured by
biome family, plus within-biome Spearman correlations.
"""
import numpy as np
import pandas as pd
from scipy import stats

bi_all = pd.read_csv('analysis/biome_kge.csv')
order = ['Mediterranean', 'Semi-arid/arid', 'Temperate/continental', '(Sub)tropical']
bcol = {'Mediterranean': '#1f77b4', 'Semi-arid/arid': '#d62728',
        'Temperate/continental': '#2ca02c', '(Sub)tropical': '#ff7f0e'}

# Restrict to the 4 named families up front (matches explore_biome_kge.py's convention)
# so the printed/plotted N is never out of step with what the legend actually sums to.
excluded = sorted(bi_all[~bi_all.biome.isin(order)].name)
bi = bi_all[bi_all.biome.isin(order)].copy()
if excluded:
    print(f"[note] {len(excluded)} reservoir(s) with unclassified biome excluded from "
          f"this view: {excluded}")
print(f"N = {len(bi)} (of {len(bi_all)} total)\n")
print(f"{'biome':<24}{'n':>4}{'rho':>8}{'p':>8}{'median KGE':>12}")
print('-' * 56)
for b in order:
    s = bi[bi.biome == b]
    if len(s) < 4:
        rho, p = float('nan'), float('nan')
    else:
        rho, p = stats.spearmanr(s.ap_m, s.kge)
    print(f"{b:<24}{len(s):>4}{rho:>8.2f}{p:>8.3f}{s.kge.median():>12.3f}")

# pooled for reference
rho_all, p_all = stats.spearmanr(bi.ap_m, bi.kge)
print(f"\n{'POOLED (all biomes)':<24}{len(bi):>4}{rho_all:>8.2f}{p_all:>8.1e}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6.5))
for b in order:
    s = bi[bi.biome == b]
    if s.empty:
        continue
    ax.scatter(s.ap_m, s.kge, s=65, color=bcol[b], edgecolors='white',
               linewidths=0.6, zorder=4, label=f'{b} (n={len(s)})')
ax.axhline(0.5, color='gray', lw=1, ls=':', zorder=2)
ax.set_ylim(-0.05, 1.0)   # matches the ap_kge_curve_pooled y-range for a consistent view
ax.set_xlabel('Static A/P (m)')
ax.set_ylabel('KGE (best-of adapt/vv vs JRC)')
title_n = 'n=%d' % len(bi) if len(bi) == len(bi_all) else 'n=%d of %d' % (len(bi), len(bi_all))
ax.set_title('A/P vs KGE within each climate zone (%s)' % title_n,
              fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
          title='Biome (Spearman rho, p in console)', borderaxespad=0)
ax.grid(alpha=0.25)
out = 'analysis/method_comparison_output/ap_kge_by_biome.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
