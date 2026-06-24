"""A/P classification figure for SAR reservoir monitoring adequacy.

Panels:
  (a) KGE bars sorted by A/P, colored adequate/inadequate, threshold line
  (b) ROC curve with AUC and optimal threshold marked
  (c) KGE boxplot for two A/P classes (< vs >= threshold)
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('validation_data/morphometric_analysis/shoreline_compactness')

df = pd.read_csv(OUT_DIR / 'pilot_kge_results.csv')
df = df.dropna(subset=['ap_m', 'kge']).reset_index(drop=True)

KGE_THRESH = 0.5       # adequacy threshold
AP_OPT     = 332.6     # Youden-J optimal (pre-computed)

df['adequate']  = df['kge'] >= KGE_THRESH
df['high_ap']   = df['ap_m'] >= AP_OPT
df = df.sort_values('ap_m').reset_index(drop=True)

# ---- ROC -----------------------------------------------------------------
auc = roc_auc_score(df['adequate'].astype(int), df['ap_m'])
fpr, tpr, thresholds = roc_curve(df['adequate'].astype(int), df['ap_m'])

# ---- Colors & labels -----------------------------------------------------
# Quadrant colors
def bar_color(row):
    if row['adequate'] and row['high_ap']:   return '#2ca02c'   # TP green
    if row['adequate'] and not row['high_ap']: return '#98df8a' # FN light green
    if not row['adequate'] and row['high_ap']: return '#d62728' # FP red
    return '#ff9896'                                             # TN light red

df['color'] = df.apply(bar_color, axis=1)

def clean_name(n):
    """Shorten reservoir name for x-axis labels."""
    n = n.replace('_', ' ')
    # remove country suffix patterns like _ES _US _AU _MX
    for suf in [' ES', ' US', ' AU', ' MX', ' IN']:
        if n.endswith(suf):
            n = n[:-3]
    return n

df['label'] = df['name'].apply(clean_name)

# ---- Figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle(
    'A/P ratio as a pre-screening classifier for SAR reservoir monitoring',
    fontsize=12, fontweight='bold')

# =========================================================================
# (a) KGE bars sorted by A/P
# =========================================================================
ax = axes[0]
x  = np.arange(len(df))
bars = ax.bar(x, df['kge'], color=df['color'], edgecolor='white', linewidth=0.5, zorder=3)

# Mark bars below KGE threshold with hatch
for i, row in df.iterrows():
    if row['kge'] < KGE_THRESH:
        bars[i].set_hatch('//')

# A/P threshold vertical divider
# find position between last low-AP and first high-AP bar
split_x = df.index[df['ap_m'] >= AP_OPT][0] - 0.5
ax.axvline(split_x, color='navy', lw=1.8, ls='--', zorder=4, label=f'AP threshold ({AP_OPT:.0f} m)')
ax.axhline(KGE_THRESH, color='gray', lw=1.2, ls=':', zorder=2, label=f'KGE = {KGE_THRESH}')

# Annotate A/P values on secondary x
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x[::2])
ax2.set_xticklabels([f'{v:.0f}' for v in df['ap_m'].iloc[::2]],
                    fontsize=6.5, rotation=45, ha='left')
ax2.set_xlabel('A/P (m)', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(df['label'], rotation=55, ha='right', fontsize=7.5)
ax.set_ylabel('KGE')
ax.set_title('(a) KGE sorted by A/P ratio')
ax.set_ylim(df['kge'].min() - 0.3, 1.10)
ax.grid(axis='y', alpha=0.3, zorder=0)

# Zone labels
n_low  = (df['ap_m'] < AP_OPT).sum()
n_high = (df['ap_m'] >= AP_OPT).sum()
prec_high = (df['adequate'] & df['high_ap']).sum() / df['high_ap'].sum()
prec_low  = (df['adequate'] & ~df['high_ap']).sum() / (~df['high_ap']).sum()
ax.text(split_x / 2, ax.get_ylim()[1] - 0.05,
        f'AP < {AP_OPT:.0f} m\n{prec_low*100:.0f}% adequate\n(n={n_low})',
        ha='center', va='top', fontsize=8.5, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax.text(split_x + n_high / 2, ax.get_ylim()[1] - 0.05,
        f'AP ≥ {AP_OPT:.0f} m\n{prec_high*100:.0f}% adequate\n(n={n_high})',
        ha='center', va='top', fontsize=8.5, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

legend_patches = [
    mpatches.Patch(fc='#2ca02c', label='TP – high AP, KGE ≥ 0.5'),
    mpatches.Patch(fc='#98df8a', label='FN – low AP, KGE ≥ 0.5'),
    mpatches.Patch(fc='#d62728', label='FP – high AP, KGE < 0.5'),
    mpatches.Patch(fc='#ff9896', label='TN – low AP, KGE < 0.5'),
]
ax.legend(handles=legend_patches, fontsize=7, loc='lower right')

# =========================================================================
# (b) ROC curve
# =========================================================================
ax = axes[1]
ax.plot(fpr, tpr, 'steelblue', lw=2, label=f'ROC (AUC = {auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

# Youden-J point
J = tpr - fpr
j_idx = np.argmax(J)
ax.plot(fpr[j_idx], tpr[j_idx], 'ro', ms=8, zorder=5,
        label=f'Optimal (AP={AP_OPT:.0f} m)\nJ={J[j_idx]:.2f}')
ax.annotate(f'TPR={tpr[j_idx]:.2f}\nFPR={fpr[j_idx]:.2f}',
            xy=(fpr[j_idx], tpr[j_idx]),
            xytext=(fpr[j_idx] + 0.12, tpr[j_idx] - 0.10),
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='red', lw=1))

ax.set_xlabel('False Positive Rate (1 – Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('(b) ROC curve — A/P as KGE classifier\n(KGE ≥ 0.5 = adequate)')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
ax.text(0.97, 0.05, f'AUC = {auc:.2f}', transform=ax.transAxes,
        fontsize=11, ha='right', va='bottom', color='steelblue', fontweight='bold')

# =========================================================================
# (c) KGE boxplot by AP class
# =========================================================================
ax = axes[2]

groups = {
    f'AP < {AP_OPT:.0f} m\n(n={n_low})': df[~df['high_ap']]['kge'].values,
    f'AP ≥ {AP_OPT:.0f} m\n(n={n_high})': df[df['high_ap']]['kge'].values,
}
labels = list(groups.keys())
data   = list(groups.values())

bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                medianprops=dict(color='black', lw=2))
for patch, color in zip(bp['boxes'], ['#ff9896', '#2ca02c']):
    patch.set_facecolor(color); patch.set_alpha(0.7)

# Scatter individual points
for i, (d, col) in enumerate(zip(data, ['#d62728', '#006400']), start=1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
    ax.scatter(np.full_like(d, i) + jitter, d,
               s=30, color=col, alpha=0.7, zorder=4)

# Mark country for outliers
for i, (d, vals_df) in enumerate(
        [(df[~df['high_ap']], df[~df['high_ap']]),
         (df[df['high_ap']],  df[df['high_ap']])], start=1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
    for j, (_, row) in enumerate(d.iterrows()):
        if row['kge'] < -0.5 or row['kge'] > 0.8:
            ax.text(i + jitter[j] + 0.06, row['kge'],
                    row['label'], fontsize=6.5, va='center', color='#333')

ax.axhline(KGE_THRESH, color='gray', lw=1.2, ls=':', label=f'KGE = {KGE_THRESH}')

# Mann-Whitney U
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(data[1], data[0], alternative='greater')
ax.text(0.97, 0.97, f'Mann–Whitney\np = {pval:.3f}',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))

ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('KGE'); ax.set_title('(c) KGE distribution by A/P class')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
out = OUT_DIR / 'AP_classification_analysis.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ---- Summary stats -------------------------------------------------------
print(f'\nKGE  low-AP: median={np.median(data[0]):.2f}, IQR=[{np.percentile(data[0],25):.2f}, {np.percentile(data[0],75):.2f}]')
print(f'KGE high-AP: median={np.median(data[1]):.2f}, IQR=[{np.percentile(data[1],25):.2f}, {np.percentile(data[1],75):.2f}]')
print(f'Mann-Whitney U (one-sided high>low): p={pval:.4f}')
print(f'AUC = {auc:.3f}')
print(f'At AP >= {AP_OPT:.0f} m: precision={prec_high:.2f}, recall (sens)={tpr[j_idx]:.2f}')
