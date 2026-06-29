"""A/P classification figure — combined pilot v3 (N=14) + pilot v4 (N=32).

Panels:
  (a) KGE bars sorted by A/P, colored by fail mode, hatched for v3
  (b) KGE scatter vs A/P with v3/v4 markers, trend line + Pearson r
  (c) ROC curve (A/P as adequacy classifier, KGE >= 0.5)
  (d) KGE boxplot for two A/P classes (< vs >= Youden-J threshold)
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

# ── Load and merge ────────────────────────────────────────────────────────────
v3 = pd.read_csv('analysis/pilot_kge_v3.csv')
v3['source'] = 'v3'
v3['ap_m_dynamic'] = np.nan

v4 = pd.read_csv('analysis/pilot_kge_v4.csv')
v4['source'] = 'v4'

keep_cols = ['name', 'ap_m', 'ap_m_dynamic', 'n_pairs', 'kge', 'r', 'alpha', 'beta',
             'mean_sar_ha', 'mean_jrc_ha', 'source']
df = pd.concat([v3[keep_cols], v4[keep_cols]], ignore_index=True)
df = df.dropna(subset=['ap_m', 'kge']).reset_index(drop=True)

KGE_THRESH = 0.5

df['adequate'] = df['kge'] >= KGE_THRESH
df = df.sort_values('ap_m').reset_index(drop=True)

# ── ROC + optimal threshold ───────────────────────────────────────────────────
auc = roc_auc_score(df['adequate'].astype(int), df['ap_m'])
fpr, tpr, thresholds = roc_curve(df['adequate'].astype(int), df['ap_m'])
j_idx  = np.argmax(tpr - fpr)
AP_OPT = float(thresholds[j_idx])
df['high_ap'] = df['ap_m'] >= AP_OPT

# ── Labels + fail mode ────────────────────────────────────────────────────────
def clean_name(n):
    n = n.replace('_', ' ')
    for suf in [' ES', ' US', ' AU', ' MX', ' IN']:
        if n.endswith(suf):
            n = n[:-3]
    return n

df['label'] = df['name'].apply(clean_name)

def fail_mode(row):
    if row['adequate']:
        return 'adequate'
    if row['alpha'] > 1.5:
        return 'noisy'
    if row['alpha'] < 0.5:
        return 'flat'
    if row['beta'] < 0.6:
        return 'bias'
    return 'low_r'

df['fail']      = df.apply(fail_mode, axis=1)
COLOR_MAP = {
    'adequate': '#2ca02c',
    'noisy':    '#d62728',
    'flat':     '#ff7f0e',
    'bias':     '#9467bd',
    'low_r':    '#8c564b',
}
df['bar_color'] = df['fail'].map(COLOR_MAP)

n_low  = (~df['high_ap']).sum()
n_high = df['high_ap'].sum()
prec_high = ((df['adequate']) & (df['high_ap'])).sum()  / n_high  if n_high else 0
prec_low  = ((df['adequate']) & (~df['high_ap'])).sum() / n_low   if n_low  else 0

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
axes = [fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])]

n_v3 = (df['source'] == 'v3').sum()
n_v4 = (df['source'] == 'v4').sum()
fig.suptitle(
    f'A/P ratio as a pre-screening metric for SAR reservoir monitoring  '
    f'(v3 N={n_v3} + v4 N={n_v4} = {len(df)}, KGE threshold={KGE_THRESH}, 2014-2021)',
    fontsize=11, fontweight='bold')

# =========================================================================
# (a) KGE bars sorted by A/P
# =========================================================================
ax = axes[0]
x  = np.arange(len(df))
bars = ax.bar(x, df['kge'], color=df['bar_color'],
              edgecolor='white', linewidth=0.4, zorder=3)
# hatch v3 bars to distinguish them
for i, (bar, src) in enumerate(zip(bars, df['source'])):
    if src == 'v3':
        bar.set_hatch('//')
        bar.set_edgecolor('#333')
        bar.set_linewidth(0.6)

split_idx = df.index[df['ap_m'] >= AP_OPT].tolist()
split_x   = (split_idx[0] - 0.5) if split_idx else len(df) - 0.5
ax.axvline(split_x, color='navy', lw=1.8, ls='--', zorder=4,
           label=f'A/P threshold ({AP_OPT:.0f} m)')
ax.axhline(KGE_THRESH, color='gray', lw=1.2, ls=':', zorder=2,
           label=f'KGE = {KGE_THRESH}')

# A/P value on top axis
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
step = max(1, len(df) // 14)
ax2.set_xticks(x[::step])
ax2.set_xticklabels([f'{v:.0f}' for v in df['ap_m'].iloc[::step]],
                    fontsize=5.5, rotation=45, ha='left')
ax2.set_xlabel('A/P (m)', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(df['label'], rotation=55, ha='right', fontsize=5.5)
ax.set_ylabel('KGE')
ax.set_title('(a) KGE sorted by A/P ratio')
ymin = min(df['kge'].min() - 0.2, -0.5)
ax.set_ylim(ymin, 1.12)
ax.grid(axis='y', alpha=0.3, zorder=0)

ax.text(split_x / 2, ax.get_ylim()[1] - 0.05,
        f'A/P < {AP_OPT:.0f} m\n{prec_low*100:.0f}% adequate\n(n={n_low})',
        ha='center', va='top', fontsize=8, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax.text(split_x + n_high / 2, ax.get_ylim()[1] - 0.05,
        f'A/P >= {AP_OPT:.0f} m\n{prec_high*100:.0f}% adequate\n(n={n_high})',
        ha='center', va='top', fontsize=8, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

legend_patches = [
    mpatches.Patch(fc='#2ca02c', label=f'KGE >= {KGE_THRESH} (adequate)'),
    mpatches.Patch(fc='#d62728', label='Fail: alpha >> 1 (noisy)'),
    mpatches.Patch(fc='#ff7f0e', label='Fail: alpha << 1 (flat)'),
    mpatches.Patch(fc='#9467bd', label='Fail: beta < 0.6 (bias)'),
    mpatches.Patch(fc='#8c564b', label='Fail: low r'),
    mpatches.Patch(fc='#aaa', hatch='//', label='pilot v3 (hatched)'),
]
ax.legend(handles=legend_patches, fontsize=6, loc='lower right')

# =========================================================================
# (b) KGE scatter vs A/P
# =========================================================================
ax = axes[1]
ap = df['ap_m'].values

# scatter: circles = v4, triangles = v3, colored by fail mode
for _, row in df.iterrows():
    marker = '^' if row['source'] == 'v3' else 'o'
    color  = COLOR_MAP[row['fail']]
    ax.scatter(row['ap_m'], row['kge'],
               s=40 if row['source'] == 'v3' else 28,
               marker=marker, color=color,
               edgecolors='white', linewidths=0.4, alpha=0.85, zorder=4)

# trend line + Pearson r for all points
z  = np.polyfit(ap, df['kge'], 1)
xf = np.linspace(ap.min(), ap.max(), 200)
r_all, p_all = pearsonr(ap, df['kge'])
ax.plot(xf, np.polyval(z, xf), 'k-', lw=1.4, alpha=0.6,
        label=f'trend (r={r_all:.2f}, p={p_all:.3f})')

ax.axhline(KGE_THRESH, color='gray', lw=1.0, ls=':', alpha=0.6,
           label=f'KGE = {KGE_THRESH}')
ax.axvline(AP_OPT, color='navy', lw=1.0, ls='--', alpha=0.6,
           label=f'A/P = {AP_OPT:.0f} m')

# label notable outliers
for _, row in df.iterrows():
    if row['kge'] < -0.4 or (row['kge'] > 0.85 and row['ap_m'] < 150):
        ax.annotate(row['label'], (row['ap_m'], row['kge']),
                    fontsize=5, xytext=(4, 2), textcoords='offset points', color='#444')

legend_elems = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#666',
           markersize=6, label='pilot v4'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#666',
           markersize=7, label='pilot v3'),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + legend_elems, fontsize=7, loc='lower right')

ax.set_xlabel('A/P (m)')
ax.set_ylabel('KGE')
ax.set_title('(b) KGE vs A/P (all reservoirs)')
ax.grid(alpha=0.3)

# =========================================================================
# (c) ROC curve
# =========================================================================
ax = axes[2]
ax.plot(fpr, tpr, 'steelblue', lw=2, label=f'ROC (AUC = {auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

J = tpr - fpr
ax.plot(fpr[j_idx], tpr[j_idx], 'ro', ms=8, zorder=5,
        label=f'Optimal A/P = {AP_OPT:.0f} m  (J={J[j_idx]:.2f})')
ax.annotate(f'TPR={tpr[j_idx]:.2f}\nFPR={fpr[j_idx]:.2f}',
            xy=(fpr[j_idx], tpr[j_idx]),
            xytext=(fpr[j_idx] + 0.12, tpr[j_idx] - 0.12),
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.set_xlabel('False Positive Rate  (1 - Specificity)')
ax.set_ylabel('True Positive Rate  (Sensitivity)')
ax.set_title(f'(c) ROC — A/P as adequacy classifier (KGE >= {KGE_THRESH})')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
ax.text(0.97, 0.05, f'AUC = {auc:.2f}', transform=ax.transAxes,
        fontsize=12, ha='right', va='bottom', color='steelblue', fontweight='bold')

# =========================================================================
# (d) KGE boxplot by A/P class
# =========================================================================
ax = axes[3]
low_data  = df[~df['high_ap']]['kge'].values
high_data = df[ df['high_ap']]['kge'].values

bp = ax.boxplot([low_data, high_data], patch_artist=True, widths=0.45,
                medianprops=dict(color='black', lw=2))
for patch, color in zip(bp['boxes'], ['#ffb3b3', '#b3e6b3']):
    patch.set_facecolor(color); patch.set_alpha(0.8)

rng = np.random.default_rng(42)
for k, (data, col) in enumerate(zip([low_data, high_data], ['#d62728', '#2ca02c'])):
    jitter = rng.uniform(-0.12, 0.12, size=len(data))
    ax.scatter(np.full_like(data, k + 1) + jitter, data,
               s=28, color=col, alpha=0.75, zorder=4)

# Label v3 points distinctly
for k, sub_df in enumerate([df[~df['high_ap']], df[df['high_ap']]]):
    sub_df = sub_df.reset_index(drop=True)
    jitter = rng.uniform(-0.12, 0.12, size=len(sub_df))
    for j, row in sub_df.iterrows():
        if row['source'] == 'v3' or row['kge'] < -0.3 or row['kge'] > 0.88:
            ax.text(k + 1 + jitter[j] + 0.08, row['kge'],
                    row['label'], fontsize=5.5, va='center', color='#333')

ax.axhline(KGE_THRESH, color='gray', lw=1.2, ls=':', label=f'KGE = {KGE_THRESH}')

stat, pval = mannwhitneyu(high_data, low_data, alternative='greater')
ax.text(0.97, 0.97, f'Mann-Whitney U\np = {pval:.4f}',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))

ax.set_xticks([1, 2])
ax.set_xticklabels([f'A/P < {AP_OPT:.0f} m\n(n={n_low})',
                    f'A/P >= {AP_OPT:.0f} m\n(n={n_high})'], fontsize=9)
ax.set_ylabel('KGE')
ax.set_title('(d) KGE distribution by A/P class')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

out = OUT_DIR / 'AP_classification_v4.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f'\nCombined dataset: N = {len(df)} ({n_v3} v3 + {n_v4} v4)')
print(f'KGE >= {KGE_THRESH}: {df["adequate"].sum()} / {len(df)}  '
      f'({df["adequate"].mean()*100:.0f}%)')
print(f'AUC = {auc:.3f}')
print(f'Youden-J optimal A/P threshold = {AP_OPT:.0f} m')
print(f'  A/P >= {AP_OPT:.0f}: {prec_high*100:.0f}% adequate  '
      f'TPR={tpr[j_idx]:.2f}  FPR={fpr[j_idx]:.2f}  J={J[j_idx]:.2f}')
print(f'  A/P <  {AP_OPT:.0f}: {prec_low*100:.0f}% adequate')
print(f'\nKGE low-AP:  median={np.median(low_data):.2f}  '
      f'IQR=[{np.percentile(low_data,25):.2f}, {np.percentile(low_data,75):.2f}]')
print(f'KGE high-AP: median={np.median(high_data):.2f}  '
      f'IQR=[{np.percentile(high_data,25):.2f}, {np.percentile(high_data,75):.2f}]')
print(f'Mann-Whitney U (one-sided): p = {pval:.4f}')
print(f'\nPearson r(A/P, KGE) = {r_all:.3f}  p = {p_all:.4f}')
