"""
plot_pilot_timeseries.py

SAR area (scatter, coloured by pass direction) and JRC monthly area (line)
for all pilot_v2 reservoirs from GEE_GlobalPilotV2b.

SAR cleaning mirrors GEE app cleanAndSmooth:
  1x global ±2σ outlier removal
  3x local window=5 ±1.5σ outlier removal
  1x local window=10 ±1.5σ outlier removal
  Gaussian kernel smooth (sigma=7 d, window=±20 d) — displayed as line only

Outputs:
  analysis/method_comparison_output/pilot_v2_ts/ts_{name}.png  — individual (8x3")
  analysis/method_comparison_output/pilot_v2_ts/overview.png   — 5-column grid
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

DATA_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV2b')
OUT_DIR  = pathlib.Path('analysis/method_comparison_output/pilot_v2_ts')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_COLOR      = {'ASCENDING': '#1565C0', 'DESCENDING': '#E65100'}  # blue / orange
VALID_FRAC_MIN = 0.80
S1C_DATE       = pd.Timestamp('2024-12-01')
SAR_MIN_FRAC   = 0.02   # 2% of p99 area = minimum threshold (wind misclassification)


# ── Spike filter (light global 3σ — catches instrument spikes, not seasonality) ──
#
# The GEE app's 4-pass local 1.5σ cleaning is designed for pre-composited data
# (fillCoverageGaps ±6 d). On raw export data it removes seasonal transitions
# (50–60% of points). Here we use a single global 3σ pass instead: only removes
# values that sit >3 standard deviations outside the annual mean — genuine spikes,
# not seasonal extremes.

def clean_sar_outliers(df):
    """Single global 3σ spike filter. Returns (df_clean, df_removed)."""
    if df.empty or len(df) < 4:
        return df.copy(), df.iloc[0:0].copy()
    df = df.sort_values('date').copy()
    m, s = df['area_ha'].mean(), df['area_ha'].std()
    keep = (s == 0) or (np.abs(df['area_ha'].values - m) <= 3.0 * s)
    if isinstance(keep, bool):
        keep = np.ones(len(df), dtype=bool)
    df_clean   = df[keep].reset_index(drop=True)
    df_removed = df[~keep].copy()
    return df_clean, df_removed


def gaussian_smooth(df, sigma_days=7, window_days=20):
    """Gaussian kernel smoother on irregular time series. Returns df with added 'area_smooth'."""
    if len(df) < 2:
        result = df.copy()
        result['area_smooth'] = result['area_ha']
        return result
    df = df.sort_values('date').reset_index(drop=True)
    t = (df['date'] - pd.Timestamp('2000-01-01')).dt.days.values.astype(float)
    a = df['area_ha'].values
    smoothed = np.empty_like(a)
    for i in range(len(t)):
        dt = np.abs(t - t[i])
        mask = dt <= window_days
        w = np.exp(-0.5 * (dt[mask] / sigma_days) ** 2)
        smoothed[i] = np.dot(w, a[mask]) / w.sum()
    result = df.copy()
    result['area_smooth'] = smoothed
    return result


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_sar(name):
    """Returns (df_valid, df_s1c, df_noise, df_outliers, ap_m).
    df_valid   : S1A/B, above SAR_MIN_FRAC, 4-pass outlier cleaned
    df_s1c     : post S1C_DATE entries (calibration uncertainty)
    df_noise   : pre-S1C entries below SAR_MIN_FRAC (wind misclassification)
    df_outliers: passed SAR_MIN_FRAC but removed by 4-pass cleaning
    """
    p = DATA_DIR / f'SAR_area_{name}.csv'
    if not p.exists():
        return None, None, None, None, np.nan
    df = pd.read_csv(p, parse_dates=['date'])
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    ap_m = float(df['ap_m'].iloc[0]) if not df.empty else np.nan

    pre_s1c  = df[df['date'] < S1C_DATE].copy()
    post_s1c = df[df['date'] >= S1C_DATE].copy()

    p99 = pre_s1c['area_ha'].quantile(0.99) if not pre_s1c.empty else df['area_ha'].quantile(0.99)
    min_area = SAR_MIN_FRAC * p99

    above_min = pre_s1c[pre_s1c['area_ha'] >= min_area].copy()
    df_noise  = pre_s1c[pre_s1c['area_ha'] <  min_area].copy()

    df_valid, df_outliers = clean_sar_outliers(above_min)

    return df_valid, post_s1c, df_noise, df_outliers, ap_m


def load_jrc(name):
    """Returns (df_all, df_valid) where df_valid has valid_frac >= VALID_FRAC_MIN."""
    p = DATA_DIR / f'JRC_area_{name}.csv'
    if not p.exists():
        return None, None
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    has_vf = 'valid_frac' in df.columns
    df_valid = df[df['valid_frac'] >= VALID_FRAC_MIN].copy() if has_vf else df.copy()
    return df, df_valid


# ── Plot ──────────────────────────────────────────────────────────────────────

def draw_panel(ax, name, sar_valid, sar_s1c, sar_noise, sar_outliers, ap_m,
               jrc_all, jrc_valid):
    has_sar_v = sar_valid    is not None and not sar_valid.empty
    has_sar_c = sar_s1c     is not None and not sar_s1c.empty
    has_sar_n = sar_noise    is not None and not sar_noise.empty
    has_sar_o = sar_outliers is not None and not sar_outliers.empty
    has_jrc_a = jrc_all     is not None and not jrc_all.empty
    has_jrc_v = jrc_valid   is not None and not jrc_valid.empty

    # JRC rejected months (low valid_frac) — grey dots
    if has_jrc_a and 'valid_frac' in jrc_all.columns:
        rejected = jrc_all[jrc_all['valid_frac'] < VALID_FRAC_MIN]
        if not rejected.empty:
            ax.scatter(rejected['date'], rejected['jrc_area_ha'],
                       s=5, color='#BDBDBD', alpha=0.5, linewidths=0, zorder=1)

    # JRC valid months — red line
    if has_jrc_v:
        ax.plot(jrc_valid['date'], jrc_valid['jrc_area_ha'],
                color='#C62828', lw=1.3, alpha=0.9, zorder=2)

    # SAR noise (below SAR_MIN_FRAC) + 4-pass outliers — grey markers
    for df_grey, mkr in [(sar_noise, 'x'), (sar_outliers, 'D')]:
        if df_grey is not None and not df_grey.empty:
            ax.scatter(df_grey['date'], df_grey['area_ha'],
                       s=7, color='#BDBDBD', marker=mkr, linewidths=0.6,
                       alpha=0.6, zorder=2)

    # SAR valid (cleaned S1A/B) — coloured dots + Gaussian smooth line
    if has_sar_v:
        for direction, grp in sar_valid.groupby('passDirection'):
            ax.scatter(grp['date'], grp['area_ha'],
                       s=8, color=DIR_COLOR.get(direction, '#666'),
                       alpha=0.75, linewidths=0, zorder=3)
        df_sm = gaussian_smooth(sar_valid)
        ax.plot(df_sm['date'], df_sm['area_smooth'],
                color='#263238', lw=1.0, alpha=0.65, zorder=4)

    # S1C data (post Dec 2024) — open circles
    if has_sar_c:
        for direction, grp in sar_s1c.groupby('passDirection'):
            ax.scatter(grp['date'], grp['area_ha'],
                       s=12, facecolors='none',
                       edgecolors=DIR_COLOR.get(direction, '#999'),
                       linewidths=0.8, alpha=0.8, zorder=3)

    ax.axvline(S1C_DATE, color='#795548', lw=0.7, ls='--', alpha=0.5)

    label = (f'{name.replace("_", " ")}  A/P={ap_m:.0f} m'
             if not np.isnan(ap_m) else name.replace("_", " "))
    ax.set_title(label, fontsize=7.5, fontweight='bold', pad=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=6)
    ax.set_ylabel('Area (ha)', fontsize=6, labelpad=2)
    ax.grid(True, alpha=0.2, linewidth=0.4)

    if not has_sar_v and not has_jrc_a:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', color='gray', fontsize=9)


# ── Reservoir list ────────────────────────────────────────────────────────────
sar_names = {p.stem[len('SAR_area_'):] for p in DATA_DIR.glob('SAR_area_*.csv')}
jrc_names = {p.stem[len('JRC_area_'):] for p in DATA_DIR.glob('JRC_area_*.csv')}
all_names = sorted(sar_names | jrc_names)
print(f'{len(all_names)} reservoirs  (SAR={len(sar_names)}, JRC={len(jrc_names)})')

LEGEND_HANDLES = [
    Line2D([0], [0], color='#C62828', lw=1.3, label=f'JRC (vf>={VALID_FRAC_MIN:.0%})'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#BDBDBD',
           markersize=5, label='JRC low coverage'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0',
           markersize=6, label='SAR asc (cleaned)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E65100',
           markersize=6, label='SAR desc (cleaned)'),
    Line2D([0], [0], color='#263238', lw=1.0, alpha=0.65, label='SAR Gauss smooth'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#1565C0', markersize=6, label='SAR S1C (post Dec-24)'),
    Line2D([0], [0], marker='x', color='#BDBDBD', markersize=5,
           lw=0, label=f'SAR removed (noise + outliers)'),
]

# ── Individual PNGs ───────────────────────────────────────────────────────────
for name in all_names:
    sar_valid, sar_s1c, sar_noise, sar_outliers, ap_m = load_sar(name)
    jrc_all, jrc_valid                                 = load_jrc(name)

    fig, ax = plt.subplots(figsize=(8, 3))
    draw_panel(ax, name, sar_valid, sar_s1c, sar_noise, sar_outliers, ap_m, jrc_all, jrc_valid)
    ax.legend(handles=LEGEND_HANDLES, fontsize=6, loc='upper left', framealpha=0.7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'ts_{name}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    n_v  = len(sar_valid)    if sar_valid    is not None else 0
    n_c  = len(sar_s1c)     if sar_s1c      is not None else 0
    n_n  = len(sar_noise)   if sar_noise    is not None else 0
    n_o  = len(sar_outliers) if sar_outliers is not None else 0
    n_jv = len(jrc_valid)   if jrc_valid    is not None else 0
    n_ja = len(jrc_all)     if jrc_all      is not None else 0
    print(f'  {name:<22} A/P={ap_m:>5.0f}  SAR clean={n_v:3d}  S1C={n_c:2d}'
          f'  noise={n_n:2d}  outliers={n_o:2d}  JRC={n_ja:2d}({n_jv}ok)')

# ── Overview grid ─────────────────────────────────────────────────────────────
NCOLS = 5
NROWS = -(-len(all_names) // NCOLS)
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4.2, NROWS * 2.8))
axes_flat = axes.flatten()

for i, name in enumerate(all_names):
    sar_valid, sar_s1c, sar_noise, sar_outliers, ap_m = load_sar(name)
    jrc_all, jrc_valid                                 = load_jrc(name)
    draw_panel(axes_flat[i], name, sar_valid, sar_s1c, sar_noise, sar_outliers, ap_m,
               jrc_all, jrc_valid)

for j in range(len(all_names), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.legend(handles=LEGEND_HANDLES, loc='lower right',
           bbox_to_anchor=(0.99, 0.01), fontsize=8, framealpha=0.85)
fig.suptitle(
    f'Pilot v2 - SAR area (cleaned + Gauss smooth) vs JRC (valid_frac>={VALID_FRAC_MIN:.0%})',
    fontsize=12, fontweight='bold', y=1.002)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT_DIR / 'overview.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print(f'\nSaved overview: {OUT_DIR}/overview.png')
print(f'Individual PNGs: {OUT_DIR}/ts_*.png')
