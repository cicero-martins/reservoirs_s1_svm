"""
explore_biome_kge.py

Does BIOME/climate relate to monitorability (KGE), or is A/P the whole story?

The paper's thesis is that A/P is a UNIVERSAL (climate-independent) predictor. If a
biome predicted KGE *independently of A/P*, that would undercut universality; if any
biome signal is really just A/P (some climates skew to certain geometries) or the known
radiometric second axis, universality stands. This tests it directly.

KGE = dual-pol VV+VH SVM vs JRC on the pooled 42-reservoir set (pilot_kge_apcurve.csv).
Climate = Köppen-family zone from global_pilot_v4_candidates.csv (+ the 4 Sicilian, Med).

Coarse biome families (raw Köppen zones are too thin, many n=1):
  Mediterranean          (Mediterranean, Mediterranean highland)
  Semi-arid/arid         (Semi-arid, Semi-arid continental/highland)
  Temperate/continental  (Humid temperate, Humid continental)
  (Sub)tropical          (Humid subtropical/tropical, Tropical highland/savanna)

Outputs:
  analysis/biome_kge.csv
  analysis/method_comparison_output/biome_kge.png
"""

import pathlib
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

KGE = pd.read_csv('analysis/pilot_kge_apcurve.csv').rename(columns={'best': 'kge'})  # best-of, de-spiked, chapados excluded
CAND = pd.read_csv('analysis/global_pilot_v4_candidates.csv')[['name', 'climate_zone']]

# 4 Sicilian are on the PlanetScope track, not in the v4 candidate file → all Mediterranean.
SICILY_CLIM = {n: 'Mediterranean' for n in ['Ancipa', 'Poma', 'Pozzillo', 'Rosamarina']}

df = KGE.merge(CAND, on='name', how='left')
df['climate_zone'] = df.apply(
    lambda r: SICILY_CLIM.get(r['name'], r['climate_zone']), axis=1)

missing = df[df['climate_zone'].isna()]['name'].tolist()
if missing:
    print(f'[warn] no climate for: {missing}')

def family(z):
    z = str(z)
    if 'Mediterranean' in z:                       return 'Mediterranean'
    if 'Semi-arid' in z or 'arid' in z:            return 'Semi-arid/arid'
    if 'temperate' in z or 'continental' in z:     return 'Temperate/continental'
    if any(k in z for k in ('subtropical', 'tropical', 'Tropical')): return '(Sub)tropical'
    return 'Other'

df['biome'] = df['climate_zone'].map(family)
df.to_csv('analysis/biome_kge.csv', index=False)

ORDER = ['Mediterranean', 'Semi-arid/arid', 'Temperate/continental', '(Sub)tropical']
df = df[df['biome'].isin(ORDER)].copy()

print(f'N = {len(df)} reservoirs with biome + KGE\n')
print(f'{"biome":<24}{"n":>3}{"med KGE":>9}{"med A/P":>9}{"mean KGE":>10}')
print('-' * 55)
groups = []
for b in ORDER:
    s = df[df['biome'] == b]
    if s.empty:
        continue
    groups.append(s['kge'].values)
    print(f'{b:<24}{len(s):>3}{s["kge"].median():>9.3f}{s["ap_m"].median():>9.0f}'
          f'{s["kge"].mean():>10.3f}')

# 1) Does KGE differ across biomes at all? (Kruskal-Wallis, non-parametric ANOVA)
H, p_kw = stats.kruskal(*groups)
print(f'\nKruskal-Wallis  KGE ~ biome:  H={H:.2f}, p={p_kw:.3f}  '
      f'→ {"biomes differ" if p_kw < 0.05 else "NO significant KGE difference by biome"}')

# 2) The confound check: do biomes also differ in A/P? If a biome looks worse only
#    because it sits at low A/P, then A/P — not climate — is the driver.
apg = [df[df['biome'] == b]['ap_m'].values for b in ORDER if (df['biome'] == b).any()]
H2, p_ap = stats.kruskal(*apg)
print(f'Kruskal-Wallis  A/P ~ biome:  H={H2:.2f}, p={p_ap:.3f}  '
      f'→ {"biomes differ in geometry too" if p_ap < 0.05 else "biomes share similar A/P"}')

# 3) Is climate STILL predictive once A/P is accounted for? Regress KGE on A/P, then
#    test whether the residuals differ by biome (i.e. biome effect BEYOND geometry).
x = df['ap_m'].values
y = df['kge'].values
sl, ic, r, p_lin, se = stats.linregress(x, y)
df['resid'] = y - (sl * x + ic)
resg = [df[df['biome'] == b]['resid'].values for b in ORDER if (df['biome'] == b).any()]
H3, p_res = stats.kruskal(*resg)
print(f'\nA/P→KGE linear fit: slope={sl:+.2e}/m, r={r:+.2f}, p={p_lin:.1e}')
print(f'Kruskal-Wallis  (KGE − A/P fit) residual ~ biome:  H={H3:.2f}, p={p_res:.3f}')
print(f'  → {"climate adds signal BEYOND A/P" if p_res < 0.05 else "NO climate signal beyond A/P — A/P is the whole story (supports universality)"}')

# ── figure: (a) KGE by biome boxplot, (b) A/P–KGE scatter coloured by biome ────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BCOL = {'Mediterranean': '#1f77b4', 'Semi-arid/arid': '#d62728',
        'Temperate/continental': '#2ca02c', '(Sub)tropical': '#ff7f0e'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.25]})

# (a) boxplot + jittered points
present = [b for b in ORDER if (df['biome'] == b).any()]
data = [df[df['biome'] == b]['kge'].values for b in present]
bp = ax1.boxplot(data, tick_labels=[b.replace('/', '/\n') for b in present],
                 patch_artist=True, widths=0.6, showmeans=True,
                 medianprops=dict(color='black', lw=1.6),
                 meanprops=dict(marker='D', mfc='white', mec='black', ms=6))
for patch, b in zip(bp['boxes'], present):
    patch.set(facecolor=BCOL[b], alpha=0.35)
rng = np.random.default_rng(0)
for i, b in enumerate(present):
    yv = df[df['biome'] == b]['kge'].values
    ax1.scatter(np.full(len(yv), i + 1) + rng.uniform(-0.13, 0.13, len(yv)), yv,
                color=BCOL[b], s=28, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=3)
ax1.set_ylabel('KGE — best of {adapt, VV-Otsu} vs JRC', fontsize=11)
ax1.set_title(f'(a) KGE by biome (N={len(df)})\n'
              f'Kruskal-Wallis p={p_kw:.2f} '
              f'({"differ" if p_kw < 0.05 else "no sig. difference"})',
              fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.25)
ax1.tick_params(axis='x', labelsize=8)

# (b) the confound view: A/P vs KGE, coloured by biome, with the shared A/P trend line
for b in present:
    s = df[df['biome'] == b]
    ax2.scatter(s['ap_m'], s['kge'], color=BCOL[b], s=60, alpha=0.85,
                edgecolors='white', linewidths=0.6, label=f'{b} (n={len(s)})', zorder=4)
xs = np.linspace(df['ap_m'].min(), df['ap_m'].max(), 50)
ax2.plot(xs, sl * xs + ic, 'k--', lw=1.4, alpha=0.7, zorder=3,
         label=f'shared A/P trend (r={r:+.2f})')
ax2.set_xlabel('A/P — static area/perimeter (m)', fontsize=11)
ax2.set_ylabel('KGE best-of vs JRC', fontsize=11)
ax2.set_title('(b) Biomes intermix along the SAME A/P trend\n'
              f'residual-by-biome p={p_res:.2f} '
              f'({"climate adds signal" if p_res < 0.05 else "no signal beyond A/P"})',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(alpha=0.25)

fig.suptitle('Biome vs monitorability — is climate a predictor, or just A/P?',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = pathlib.Path('analysis/method_comparison_output/biome_kge.png')
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT}')
