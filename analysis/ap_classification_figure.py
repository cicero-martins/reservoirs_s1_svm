"""A/P classification figure for SAR reservoir monitoring adequacy.

Panels:
  (a) KGE bars sorted by A/P, colored by fail mode (r / alpha / beta issue)
  (b) KGE components (r, alpha, beta) scatter vs A/P
  (c) ROC curve with AUC and optimal threshold marked
  (d) KGE boxplot for two A/P classes (< vs >= threshold)
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu, pearsonr
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('analysis/schwatke_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('analysis/pilot_kge_2019_2020.csv')
df = df.dropna(subset=['ap_m', 'kge']).reset_index(drop=True)

KGE_THRESH = 0.5   # adequacy threshold

df['adequate'] = df['kge'] >= KGE_THRESH
df = df.sort_values('ap_m').reset_index(drop=True)

# Optimal A/P threshold via Youden-J index from ROC
auc = roc_auc_score(df['adequate'].astype(int), df['ap_m'])
fpr, tpr, thresholds = roc_curve(df['adequate'].astype(int), df['ap_m'])
j_idx = np.argmax(tpr - fpr)
AP_OPT = float(thresholds[j_idx])

df['high_ap'] = df['ap_m'] >= AP_OPT

# ---- ROC -----------------------------------------------------------------
auc = roc_auc_score(df['adequate'].astype(int), df['ap_m'])
fpr, tpr, thresholds = roc_curve(df['adequate'].astype(int), df['ap_m'])

def clean_name(n):
    n = n.replace('_', ' ')
    for suf in [' ES', ' US', ' AU', ' MX', ' IN']:
        if n.endswith(suf):
            n = n[:-3]
    return n

df['label'] = df['name'].apply(clean_name)

# ---- Fail mode classification (for bar colors) ---------------------------
# Based on KGE components: identify dominant failure mode per reservoir.
# alpha >> 1 → SAR too noisy (over-dispersed)
# alpha << 1 → SAR too flat (under-dispersed)
# beta << 0.7 → systematic underestimation
# r < 0.5 → temporal dynamics mismatch (most fundamental failure)
def fail_mode(row):
    if row['adequate']:
        return 'adequate'
    if row['alpha'] > 1.5:
        return 'noisy'     # SAR variance >> JRC variance
    if row['alpha'] < 0.5:
        return 'flat'      # SAR barely varies while JRC does
    if row['beta'] < 0.6:
        return 'bias'      # SAR systematically underestimates
    return 'low_r'         # poor temporal correlation

df['fail'] = df.apply(fail_mode, axis=1)

COLOR_MAP = {
    'adequate': '#2ca02c',  # green
    'noisy':    '#d62728',  # red   (alpha >> 1)
    'flat':     '#ff7f0e',  # orange (alpha << 1)
    'bias':     '#9467bd',  # purple (beta << 1)
    'low_r':    '#8c564b',  # brown  (low r)
}
df['bar_color'] = df['fail'].map(COLOR_MAP)

# ---- Figure ---------------------------------------------------------------
fig = plt.figure(figsize=(18, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)
axes = [fig.add_subplot(gs[0, 0]),   # (a) bars
        fig.add_subplot(gs[0, 1]),   # (b) components scatter
        fig.add_subplot(gs[1, 0]),   # (c) ROC
        fig.add_subplot(gs[1, 1])]   # (d) boxplot
fig.suptitle('A/P ratio as a pre-screening metric for SAR reservoir monitoring  '
             f'(pilot v2, 2019-2020, N={len(df)}, excl. Elwell/Sterkfontein)',
             fontsize=12, fontweight='bold')

# =========================================================================
# (a) KGE bars sorted by A/P
# =========================================================================
ax = axes[0]
x  = np.arange(len(df))
bars = ax.bar(x, df['kge'], color=df['bar_color'],
              edgecolor='white', linewidth=0.5, zorder=3)

split_idx = df.index[df['ap_m'] >= AP_OPT]
split_x   = (split_idx[0] - 0.5) if len(split_idx) else len(df) - 0.5
ax.axvline(split_x, color='navy', lw=1.8, ls='--', zorder=4,
           label=f'A/P threshold ({AP_OPT:.0f} m)')
ax.axhline(KGE_THRESH, color='gray', lw=1.2, ls=':', zorder=2,
           label=f'KGE = {KGE_THRESH}')

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x[::2])
ax2.set_xticklabels([f'{v:.0f}' for v in df['ap_m'].iloc[::2]],
                    fontsize=6, rotation=45, ha='left')
ax2.set_xlabel('A/P (m)', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(df['label'], rotation=55, ha='right', fontsize=7)
ax.set_ylabel('KGE')
ax.set_title('(a) KGE sorted by A/P ratio')
ax.set_ylim(df['kge'].min() - 0.3, 1.10)
ax.grid(axis='y', alpha=0.3, zorder=0)

n_low  = (df['ap_m'] < AP_OPT).sum()
n_high = (df['ap_m'] >= AP_OPT).sum()
prec_high = ((df['adequate']) & (df['high_ap'])).sum() / df['high_ap'].sum() if n_high else 0
prec_low  = ((df['adequate']) & (~df['high_ap'])).sum() / n_low if n_low else 0
ax.text(split_x / 2, ax.get_ylim()[1] - 0.05,
        f'A/P < {AP_OPT:.0f} m\n{prec_low*100:.0f}% adequate\n(n={n_low})',
        ha='center', va='top', fontsize=8.5, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax.text(split_x + n_high / 2, ax.get_ylim()[1] - 0.05,
        f'A/P >= {AP_OPT:.0f} m\n{prec_high*100:.0f}% adequate\n(n={n_high})',
        ha='center', va='top', fontsize=8.5, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

legend_patches = [
    mpatches.Patch(fc='#2ca02c', label='KGE >= 0.5 (adequate)'),
    mpatches.Patch(fc='#d62728', label='Fail: alpha >> 1 (noisy SAR)'),
    mpatches.Patch(fc='#ff7f0e', label='Fail: alpha << 1 (flat SAR)'),
    mpatches.Patch(fc='#9467bd', label='Fail: beta < 0.6 (bias)'),
    mpatches.Patch(fc='#8c564b', label='Fail: low r'),
]
ax.legend(handles=legend_patches, fontsize=6.5, loc='lower right')

# =========================================================================
# (b) KGE components vs A/P
# =========================================================================
ax = axes[1]
ap = df['ap_m'].values

comp_cfg = [
    ('r',     '#1f77b4', 'r  (temporal correlation)'),
    ('alpha', '#ff7f0e', 'alpha (variability ratio)'),
    ('beta',  '#2ca02c', 'beta (bias ratio)'),
]
for col, color, label in comp_cfg:
    ax.scatter(ap, df[col], s=25, color=color, alpha=0.7, zorder=3)
    # trend line
    z = np.polyfit(ap, df[col], 1)
    xfit = np.linspace(ap.min(), ap.max(), 100)
    r_pearson, p_pearson = pearsonr(ap, df[col])
    ax.plot(xfit, np.polyval(z, xfit), color=color, lw=1.5, alpha=0.6,
            label=f'{label}  (r={r_pearson:.2f}, p={p_pearson:.2f})')

ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('A/P (m)')
ax.set_ylabel('Component value')
ax.set_title('(b) KGE components vs A/P')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)
# label outlier reservoirs
for _, row in df.iterrows():
    if row['alpha'] > 1.5 or row['alpha'] < 0.4 or row['beta'] < 0.5:
        ax.text(row['ap_m'] + 4, row['alpha'], row['label'],
                fontsize=5.5, color='#555', va='center')

# =========================================================================
# (c) ROC curve
# =========================================================================
ax = axes[2]
ax.plot(fpr, tpr, 'steelblue', lw=2, label=f'ROC (AUC = {auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

J = tpr - fpr
ax.plot(fpr[j_idx], tpr[j_idx], 'ro', ms=8, zorder=5,
        label=f'Optimal A/P = {AP_OPT:.0f} m\nJ = {J[j_idx]:.2f}')
ax.annotate(f'TPR={tpr[j_idx]:.2f}\nFPR={fpr[j_idx]:.2f}',
            xy=(fpr[j_idx], tpr[j_idx]),
            xytext=(fpr[j_idx] + 0.12, tpr[j_idx] - 0.10),
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('(c) ROC — A/P as adequacy classifier (KGE >= 0.5)')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
ax.text(0.97, 0.05, f'AUC = {auc:.2f}', transform=ax.transAxes,
        fontsize=11, ha='right', va='bottom', color='steelblue', fontweight='bold')

# =========================================================================
# (d) KGE boxplot by A/P class
# =========================================================================
ax = axes[3]
groups_data = [df[~df['high_ap']]['kge'].values,
               df[df['high_ap']]['kge'].values]
group_labels = [f'A/P < {AP_OPT:.0f} m\n(n={n_low})',
                f'A/P >= {AP_OPT:.0f} m\n(n={n_high})']

bp = ax.boxplot(groups_data, patch_artist=True, widths=0.45,
                medianprops=dict(color='black', lw=2))
for patch, color in zip(bp['boxes'], ['#ffb3b3', '#b3e6b3']):
    patch.set_facecolor(color); patch.set_alpha(0.8)

rng = np.random.default_rng(42)
for i, (d, col) in enumerate(zip(groups_data, ['#d62728', '#2ca02c']), start=1):
    jitter = rng.uniform(-0.12, 0.12, size=len(d))
    ax.scatter(np.full_like(d, i) + jitter, d,
               s=30, color=col, alpha=0.7, zorder=4)

# Label notable outliers
low_df  = df[~df['high_ap']].reset_index(drop=True)
high_df = df[df['high_ap']].reset_index(drop=True)
jitter_l = rng.uniform(-0.12, 0.12, size=len(low_df))
jitter_h = rng.uniform(-0.12, 0.12, size=len(high_df))
for j, (_, row) in enumerate(low_df.iterrows()):
    if abs(row['kge']) > 0.6 or row['kge'] < -0.3:
        ax.text(1 + jitter_l[j] + 0.07, row['kge'],
                row['label'], fontsize=6, va='center', color='#333')
for j, (_, row) in enumerate(high_df.iterrows()):
    if row['kge'] < -0.1 or row['kge'] > 0.85:
        ax.text(2 + jitter_h[j] + 0.07, row['kge'],
                row['label'], fontsize=6, va='center', color='#333')

ax.axhline(KGE_THRESH, color='gray', lw=1.2, ls=':', label=f'KGE = {KGE_THRESH}')

stat, pval = mannwhitneyu(groups_data[1], groups_data[0], alternative='greater')
ax.text(0.97, 0.97, f'Mann-Whitney U\np = {pval:.3f}',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))

ax.set_xticks([1, 2]); ax.set_xticklabels(group_labels, fontsize=9)
ax.set_ylabel('KGE'); ax.set_title('(d) KGE distribution by A/P class')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

out = OUT_DIR / 'AP_classification_2019_2020.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ---- Summary stats -------------------------------------------------------
print(f'\nN = {len(df)} reservoirs')
print(f'KGE > 0.5: {df["adequate"].sum()} / {len(df)}')
print(f'AUC = {auc:.3f}')
print(f'Youden-J optimal: A/P = {AP_OPT:.0f} m')
print(f'  At A/P >= {AP_OPT:.0f}: precision = {prec_high:.2f}, '
      f'TPR = {tpr[j_idx]:.2f}, FPR = {fpr[j_idx]:.2f}')
print(f'\nKGE low-AP:  median={np.median(groups_data[0]):.2f}, '
      f'IQR=[{np.percentile(groups_data[0],25):.2f}, {np.percentile(groups_data[0],75):.2f}]')
print(f'KGE high-AP: median={np.median(groups_data[1]):.2f}, '
      f'IQR=[{np.percentile(groups_data[1],25):.2f}, {np.percentile(groups_data[1],75):.2f}]')
print(f'Mann-Whitney U (one-sided high>low): p = {pval:.4f}')

print('\nPearson r(A/P, KGE component):')
for col, _, label in comp_cfg:
    r_p, p_p = pearsonr(df['ap_m'], df[col])
    print(f'  {label}: r = {r_p:.3f}, p = {p_p:.3f}')
