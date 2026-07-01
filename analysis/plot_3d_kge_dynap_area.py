"""
plot_3d_kge_dynap_area.py

3D scatter (dynamic A/P, reservoir area, KGE) with a fitted regression plane
KGE ~ b0 + b1*dyn_ap + b2*log10(area).

Outputs:
  analysis/method_comparison_output/3d_kge_dynap_area.html   (plotly, interactive)
  analysis/method_comparison_output/3d_kge_dynap_area.png    (matplotlib, 4 viewpoints)
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR  = Path('analysis/method_comparison_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)
KGE_THRESH = 0.5

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('analysis/pilot_kge_v4.csv')
df = df.dropna(subset=['ap_m_dynamic', 'kge', 'mean_jrc_ha']).reset_index(drop=True)
df['adequate'] = df['kge'] >= KGE_THRESH
df['log_area'] = np.log10(df['mean_jrc_ha'])
df['label']    = df['name'].str.replace('_', ' ')

def fail_mode(row):
    if row['adequate']:     return 'adequate'
    if row['alpha'] > 1.5: return 'noisy'
    if row['alpha'] < 0.5: return 'flat'
    if row['beta']  < 0.6: return 'bias'
    return 'low_r'

COLOR_MAP = {
    'adequate': '#2ca02c',
    'noisy':    '#d62728',
    'flat':     '#ff7f0e',
    'bias':     '#9467bd',
    'low_r':    '#8c564b',
}
COLOR_PLOTLY = {
    'adequate': 'green',
    'noisy':    'crimson',
    'flat':     'darkorange',
    'bias':     'mediumpurple',
    'low_r':    'saddlebrown',
}

df['fail']  = df.apply(fail_mode, axis=1)
df['color'] = df['fail'].map(COLOR_MAP)
df['cplotly'] = df['fail'].map(COLOR_PLOTLY)

# ── Fit regression plane: KGE ~ dyn_ap + log10(area) ─────────────────────────
X = df[['ap_m_dynamic', 'log_area']].values
y = df['kge'].values

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2     = r2_score(y, y_pred)
b0, b1, b2 = model.intercept_, model.coef_[0], model.coef_[1]

print(f'Regression plane: KGE = {b0:.3f} + {b1:.4f}*dyn_AP + {b2:.3f}*log10(area)')
print(f'R² = {r2:.3f}  (N={len(df)})')
print(f'  Pearson r(dyn_AP, KGE)   = {pearsonr(df["ap_m_dynamic"], df["kge"])[0]:+.3f}')
print(f'  Pearson r(log_area, KGE) = {pearsonr(df["log_area"],     df["kge"])[0]:+.3f}')

# ── Mesh for the regression plane ─────────────────────────────────────────────
dp_min, dp_max = df['ap_m_dynamic'].min()-5,  df['ap_m_dynamic'].max()+5
la_min, la_max = df['log_area'].min()-0.1,    df['log_area'].max()+0.1

dp_grid, la_grid = np.meshgrid(np.linspace(dp_min, dp_max, 40),
                                np.linspace(la_min, la_max, 40))
kge_grid = b0 + b1*dp_grid + b2*la_grid

# Clamp plane to visible KGE range
kge_grid = np.clip(kge_grid, -0.5, 1.1)

# KGE=0.5 contour on the plane (where the plane crosses adequacy threshold)
# b0 + b1*dp + b2*la = 0.5 → dp = (0.5 - b0 - b2*la) / b1
la_line = np.linspace(la_min, la_max, 200)
dp_line = (KGE_THRESH - b0 - b2*la_line) / b1
# only keep within visible range
valid = (dp_line >= dp_min) & (dp_line <= dp_max)

# ── Helper: tick labels for log10(area) axis ──────────────────────────────────
LA_TICKS = [2.0, 2.3, 2.6, 3.0]   # log10(100), log10(200), log10(400), log10(1000)
LA_LBLS  = ['100', '200', '400', '1000']


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLOTLY interactive
# ══════════════════════════════════════════════════════════════════════════════
fig_p = go.Figure()

# Regression plane
fig_p.add_trace(go.Surface(
    x=dp_grid, y=la_grid, z=kge_grid,
    colorscale=[[0,'#FFCDD2'],[0.5,'#FFECB3'],[1,'#C8E6C9']],
    opacity=0.45,
    showscale=False,
    name='Regression plane',
    hoverinfo='skip',
))

# KGE=0.5 reference plane (horizontal)
dp_ref = np.array([[dp_min, dp_max], [dp_min, dp_max]])
la_ref = np.array([[la_min, la_min], [la_max, la_max]])
kge_ref = np.full_like(dp_ref, KGE_THRESH)
fig_p.add_trace(go.Surface(
    x=dp_ref, y=la_ref, z=kge_ref,
    colorscale=[[0,'rgba(100,100,200,0.15)'],[1,'rgba(100,100,200,0.15)']],
    opacity=0.25, showscale=False, name=f'KGE={KGE_THRESH}', hoverinfo='skip',
))

# Adequacy contour on regression plane
if valid.any():
    kge_on_line = b0 + b1*dp_line[valid] + b2*la_line[valid]
    fig_p.add_trace(go.Scatter3d(
        x=dp_line[valid], y=la_line[valid], z=kge_on_line,
        mode='lines',
        line=dict(color='navy', width=4, dash='dash'),
        name=f'Adequacy boundary (plane ∩ KGE={KGE_THRESH})',
    ))

# Scatter per fail mode
for fail, grp in df.groupby('fail'):
    symbol = 'circle' if fail == 'adequate' else 'cross'
    fig_p.add_trace(go.Scatter3d(
        x=grp['ap_m_dynamic'],
        y=grp['log_area'],
        z=grp['kge'],
        mode='markers+text',
        text=grp['label'],
        textposition='top center',
        textfont=dict(size=8),
        marker=dict(
            size=7,
            color=grp['cplotly'],
            symbol=symbol,
            line=dict(color='white', width=0.5),
        ),
        name=fail,
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Dyn A/P: %{x:.0f} m<br>'
            'Area: %{customdata:.0f} ha<br>'
            'KGE: %{z:.3f}<extra></extra>'
        ),
        customdata=grp['mean_jrc_ha'],
    ))

fig_p.update_layout(
    title=dict(
        text=(f'KGE vs Dynamic A/P vs Reservoir Area — pilot v4 (N={len(df)})<br>'
              f'<sub>Regression plane: KGE = {b0:.2f} + {b1:.4f}·dyn_AP '
              f'+ {b2:.2f}·log₁₀(area)  |  R² = {r2:.2f}</sub>'),
        x=0.5, font=dict(size=13)),
    scene=dict(
        xaxis=dict(title='Dynamic A/P (m)'),
        yaxis=dict(title='log₁₀(Area ha)',
                   tickvals=LA_TICKS, ticktext=LA_LBLS),
        zaxis=dict(title='KGE', range=[-0.5, 1.1]),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    ),
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
    margin=dict(l=0, r=0, t=80, b=0),
    width=950, height=700,
)

out_html = OUT_DIR / '3d_kge_dynap_area.html'
fig_p.write_html(str(out_html))
print(f'Saved: {out_html}')


# ══════════════════════════════════════════════════════════════════════════════
# 2. MATPLOTLIB static — 4 viewpoints
# ══════════════════════════════════════════════════════════════════════════════
VIEWS = [(25, -60), (25, 30), (10, -90), (70, -60)]
TITLES = ['Vista oblíqua (principal)', 'Vista lateral direita',
          'Vista frontal (A/P axis)', 'Vista superior']

fig_m, axes_m = plt.subplots(2, 2, figsize=(14, 11),
                              subplot_kw={'projection': '3d'})
axes_m = axes_m.flatten()

for ax, (elev, azim), title in zip(axes_m, VIEWS, TITLES):
    # Regression plane
    ax.plot_surface(dp_grid, la_grid, kge_grid,
                    alpha=0.25, color='#FFF9C4', edgecolor='none', zorder=1)

    # KGE=0.5 reference plane
    ax.plot_surface(dp_ref, la_ref, kge_ref,
                    alpha=0.12, color='steelblue', edgecolor='none', zorder=0)

    # Adequacy contour
    if valid.any():
        kge_on_line = b0 + b1*dp_line[valid] + b2*la_line[valid]
        ax.plot(dp_line[valid], la_line[valid], kge_on_line,
                'navy', lw=2, ls='--', zorder=5, label='Adequacy boundary')

    # Scatter
    for fail, grp in df.groupby('fail'):
        ax.scatter(grp['ap_m_dynamic'], grp['log_area'], grp['kge'],
                   c=grp['color'], s=45,
                   marker='o' if fail == 'adequate' else 'X',
                   edgecolors='white', linewidths=0.4,
                   alpha=0.88, zorder=6, label=fail)

    ax.set_xlabel('Dyn A/P (m)', fontsize=7, labelpad=4)
    ax.set_ylabel('log₁₀(Area)', fontsize=7, labelpad=4)
    ax.set_zlabel('KGE', fontsize=7, labelpad=3)
    ax.set_yticks(LA_TICKS); ax.set_yticklabels(LA_LBLS, fontsize=5)
    ax.tick_params(axis='both', labelsize=5)
    ax.set_zlim(-0.5, 1.1)
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

# Legend on first panel only
handles = [
    mpatches.Patch(fc='#2ca02c', label='adequate (KGE≥0.5)'),
    mpatches.Patch(fc='#d62728', label='noisy (α>1.5)'),
    mpatches.Patch(fc='#ff7f0e', label='flat (α<0.5)'),
    mpatches.Patch(fc='#9467bd', label='bias (β<0.6)'),
    mpatches.Patch(fc='#8c564b', label='low r'),
]
axes_m[0].legend(handles=handles, fontsize=6.5, loc='upper left',
                 bbox_to_anchor=(0.0, 1.0))

fig_m.suptitle(
    f'KGE ~ Dynamic A/P + log₁₀(Area)   |   '
    f'Plane: {b0:.2f} + {b1:.4f}·dyn_AP + {b2:.2f}·log₁₀(area)   |   '
    f'R² = {r2:.2f}   (v4, N={len(df)})',
    fontsize=9, fontweight='bold')
fig_m.tight_layout(rect=[0, 0, 1, 0.95])

out_png = OUT_DIR / '3d_kge_dynap_area.png'
fig_m.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig_m)
print(f'Saved: {out_png}')
