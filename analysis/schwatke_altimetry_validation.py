"""Altimetry validation and multi-source hypsometric comparison.

For Poma and Rosamarina (the two reservoirs with DAHITI coverage):

Figure 1 — Data coherence (per reservoir, 2×2):
  (a) DAHITI WL vs SAR area (full SAR dataset, colored by S1A/S1C era)
  (b) Gauge WL  vs SAR area (full SAR dataset, colored by era)
  (c) DAHITI vs gauge — time series (plateau periods flagged)
  (d) DAHITI vs gauge — scatter excl. plateau (bias / RMSE / r)

Figure 2 — Hypsometric curve comparison (per reservoir, 1×3):
  (a) SAR   – Gauge      power-law fit (filtered S1A pairs)
  (b) Planet – Gauge      fit (plateau-cleaned gauge)
  (c) Planet – DAHITI(corr) fit
  All three: fitted curve clipped to observed data range; design AEV as reference.
"""

import sys, warnings
import numpy as np
import pandas as pd
import xlrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('analysis/schwatke_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DT = 5   # days tolerance for temporal matching

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIGS = {
    'poma': {
        'title':        'Poma',
        'dahiti_id':    '42134',
        'sar_csv':      'validation_data/morphometric_analysis/shoreline_compactness/area_poma_2014-25.csv',
        'gauge_csv':    'analysis/schwatke_output/gauge_downloads/poma_wl.csv',
        'planet_csv':   'validation_data/statistics/area_statistics/pomaPlanet.csv',
        'aev_xls':      'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Poma.xls',
        'aev_col_h':    2, 'aev_col_A': 4, 'aev_col_V': 5, 'aev_h_min': 100,
        'gauge_min':    170,
        'h0_bound_lo':  155.0,
        'sar_max_year': 2024,   # for hypsometric fit only
        's1c_year':     2025,   # first year dominated by S1C
        'plateau_thresh': 0.10, # m over 14-day window to flag stuck gauge
    },
    'rosamarina': {
        'title':        'Rosamarina',
        'dahiti_id':    '42122',
        'sar_csv':      'validation_data/morphometric_analysis/shoreline_compactness/area_rosamarina_2014-25.csv',
        'gauge_csv':    'analysis/schwatke_output/gauge_downloads/rosamarina_wl.csv',
        'planet_csv':   'validation_data/statistics/area_statistics/rosamarinaPlanet.csv',
        'aev_xls':      'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Rosamarina.xls',
        'aev_col_h':    2, 'aev_col_A': 3, 'aev_col_V': 5, 'aev_h_min': 50,
        'gauge_min':    140,
        'h0_bound_lo':  95.0,
        'sar_max_year': 2023,
        's1c_year':     2025,
        'plateau_thresh': 0.10,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_aev(cfg):
    wb = xlrd.open_workbook(cfg['aev_xls'])
    ws = wb.sheet_by_index(0)
    rows = []
    for i in range(1, ws.nrows):
        r = ws.row_values(i)
        try:
            h = float(r[cfg['aev_col_h']]); A = float(r[cfg['aev_col_A']])
            V = float(r[cfg['aev_col_V']])
            if h > cfg['aev_h_min'] and A >= 0 and V >= 0:
                rows.append({'h': h, 'A_ha': A, 'A_m2': A * 1e4, 'V_Mm3': V})
        except (ValueError, TypeError):
            pass
    return pd.DataFrame(rows).sort_values('h').reset_index(drop=True)


def detect_plateau(gauge_df, window_days=14, range_thresh=0.10):
    """Return boolean array: True where gauge appears stuck (range < range_thresh in window)."""
    vals = gauge_df['wl_m'].values
    n = len(vals)
    flag = np.zeros(n, dtype=bool)
    hw = window_days // 2
    for i in range(n):
        lo = max(0, i - hw); hi = min(n, i + hw + 1)
        rng = float(np.nanmax(vals[lo:hi])) - float(np.nanmin(vals[lo:hi]))
        if rng < range_thresh:
            flag[i] = True
    return flag


def hyps_model(h, a, b, h0):
    return a * np.maximum(h - h0, 0.0) ** b


def fit_hyps(h_vals, A_ha_vals, h0_lo, label=''):
    h = np.asarray(h_vals, dtype=float)
    A = np.asarray(A_ha_vals, dtype=float) * 1e4   # → m²
    if len(h) < 4:
        print(f"  {label}: too few points ({len(h)}), skip")
        return None, np.nan, np.nan
    h0_hi = float(h.min()) - 0.01
    h0_lo = min(h0_lo, h0_hi - 1.0)
    try:
        popt, _ = curve_fit(
            hyps_model, h, A,
            p0=[1e6, 1.5, h0_hi - 5.0],
            bounds=([0, 0.2, h0_lo], [1e9, 6.0, h0_hi]),
            maxfev=30000,
        )
        A_pred = hyps_model(h, *popt)
        r2   = 1.0 - np.sum((A - A_pred)**2) / np.sum((A - A.mean())**2)
        rmse = np.sqrt(np.mean((A - A_pred)**2)) / 1e4
        print(f"  {label}: N={len(h)}, a={popt[0]:.0f}, b={popt[1]:.4f}, h0={popt[2]:.3f} "
              f"→ R²={r2:.4f}, RMSE={rmse:.1f} ha")
        return popt, r2, rmse
    except Exception as e:
        print(f"  {label}: FIT FAILED ({e})")
        return None, np.nan, np.nan


def match_pairs(df_area, df_wl, dt_days=MAX_DT,
                col_area='area_ha', col_date_area='date',
                col_wl='wl_m', col_date_wl='date'):
    """Match area observations to nearest WL within dt_days."""
    pairs = []
    ref = df_wl.copy().sort_values(col_date_wl).reset_index(drop=True)
    for _, row in df_area.iterrows():
        delta = (ref[col_date_wl] - row[col_date_area]).dt.days.abs()
        idx = delta.idxmin()
        if delta[idx] <= dt_days:
            pairs.append({
                'date':    row[col_date_area],
                'area_ha': row[col_area],
                'wl_m':    ref.loc[idx, col_wl],
            })
    return pd.DataFrame(pairs).dropna().reset_index(drop=True)


def scatter_stats(obs, sim):
    if len(obs) < 3:
        return np.nan, np.nan, np.nan, np.nan
    r2   = 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)
    rmse = np.sqrt(np.mean((sim - obs)**2))
    bias = (sim - obs).mean()
    r    = np.corrcoef(obs, sim)[0, 1]
    return r2, rmse, bias, r


def plot_hyps_panel(ax, title, pairs, popt, r2, rmse, aev, color,
                    pairs_plateau=None):
    """Draw one hypsometric panel: design AEV + data pairs + clipped fitted curve."""
    # Design AEV (full range)
    ax.plot(aev['A_ha'], aev['h'], 'k--', lw=1.5, label='Design AEV', zorder=2)

    if len(pairs):
        h_min = pairs['wl_m'].min(); h_max = pairs['wl_m'].max()

        # Shade data coverage range
        ax.axhspan(h_min, h_max, alpha=0.10, color=color, zorder=0,
                   label=f'Data range ({h_min:.1f}–{h_max:.1f} m)')

        # Valid pairs
        ax.scatter(pairs['area_ha'], pairs['wl_m'],
                   s=18, alpha=0.7, color=color, zorder=4,
                   label=f'Valid pairs (N={len(pairs)})')

        # Plateau-flagged pairs (if any)
        if pairs_plateau is not None and len(pairs_plateau):
            ax.scatter(pairs_plateau['area_ha'], pairs_plateau['wl_m'],
                       s=18, alpha=0.5, color='gray', marker='x', zorder=3,
                       label=f'Plateau (N={len(pairs_plateau)})')

        # Fitted curve — clipped to data range only
        if popt is not None:
            margin = max((h_max - h_min) * 0.02, 0.3)
            h_fit = np.linspace(h_min - margin, h_max + margin, 300)
            A_fit = hyps_model(h_fit, *popt) / 1e4
            label_fit = (f'Fit (within range)\n'
                         f'b={popt[1]:.3f}, R²={r2:.3f}\nRMSE={rmse:.1f} ha')
            ax.plot(A_fit, h_fit, '-', lw=2.2, color=color, zorder=5, label=label_fit)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.set_xlabel('Surface area (ha)')
    ax.set_ylabel('Water level (m a.s.l.)')
    ax.set_title(title)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for name, cfg in CONFIGS.items():
    print(f"\n{'='*60}\n  {cfg['title']}\n{'='*60}")

    # ---- Load AEV -------------------------------------------------------
    aev = load_aev(cfg)
    A_des = interp1d(aev['h'], aev['A_ha'],
                     kind='linear', bounds_error=False, fill_value='extrapolate')

    # ---- Load gauge (daily mean) + plateau detection --------------------
    g = pd.read_csv(cfg['gauge_csv'])
    g['time'] = pd.to_datetime(g['time'])
    g['wl_m'] = pd.to_numeric(g['wl_m'], errors='coerce')
    gauge_raw = (g[g['wl_m'] > cfg['gauge_min']]
                 .dropna(subset=['wl_m'])
                 .set_index('time')['wl_m']
                 .resample('D').mean()
                 .reset_index())
    gauge_raw.columns = ['date', 'wl_m']
    gauge_raw = gauge_raw.dropna().sort_values('date').reset_index(drop=True)

    plat_flag = detect_plateau(gauge_raw,
                               window_days=14,
                               range_thresh=cfg['plateau_thresh'])
    gauge_clean = gauge_raw[~plat_flag].reset_index(drop=True)
    n_plat = plat_flag.sum()
    print(f"Gauge: {len(gauge_raw)} daily readings, "
          f"{n_plat} flagged as plateau ({n_plat/len(gauge_raw)*100:.1f}%)")
    print(f"  Raw    WL {gauge_raw.wl_m.min():.2f}–{gauge_raw.wl_m.max():.2f} m")
    print(f"  Clean  WL {gauge_clean.wl_m.min():.2f}–{gauge_clean.wl_m.max():.2f} m")

    # ---- Load DAHITI WL -------------------------------------------------
    dah_raw = pd.read_csv(
        f"validation_data/DAHITI/{cfg['dahiti_id']}_{cfg['title']}_wl.csv")
    dah_raw['date'] = pd.to_datetime(dah_raw['datetime'])
    dahiti = dah_raw[['date', 'wse', 'wse_u']].rename(
        columns={'wse': 'wl_m', 'wse_u': 'wl_u'}).sort_values('date').reset_index(drop=True)
    print(f"DAHITI: n={len(dahiti)}, "
          f"WL {dahiti.wl_m.min():.2f}–{dahiti.wl_m.max():.2f} m "
          f"(u_mean={dahiti.wl_u.mean():.2f} m)")

    # ---- Load SAR areas (full dataset for coherence; filtered for fitting) -
    sar_all = pd.read_csv(cfg['sar_csv'])
    sar_all['date'] = pd.to_datetime(sar_all['date'], dayfirst=False, errors='coerce')
    sar_all = sar_all.dropna(subset=['date'])
    sar_all['year'] = sar_all['date'].dt.year
    sar_all['era']  = np.where(sar_all['year'] < cfg['s1c_year'], 'S1A', 'S1C')
    sar_filt = sar_all[sar_all['year'] <= cfg['sar_max_year']].copy()
    print(f"SAR all:  n={len(sar_all)}")
    print(f"SAR ≤{cfg['sar_max_year']}: n={len(sar_filt)}")

    # ---- Load PlanetScope -----------------------------------------------
    planet = pd.read_csv(cfg['planet_csv'])
    planet['date'] = pd.to_datetime(planet['data'], dayfirst=False, errors='coerce')
    planet = planet.dropna(subset=['date']).rename(columns={'area': 'area_ha'})
    print(f"Planet: n={len(planet)}, "
          f"area {planet.area_ha.min():.1f}–{planet.area_ha.max():.1f} ha  "
          f"{planet.date.min().date()}–{planet.date.max().date()}")

    # ---- Pairs for FIGURE 1 (coherence) — use FULL SAR dataset ----------
    sar_all_df = sar_all.rename(columns={'areaLago': 'area_ha'})
    sd_all = match_pairs(sar_all_df, dahiti, col_date_wl='date')
    sg_all = match_pairs(sar_all_df, gauge_clean, col_date_wl='date')
    # tag era for coloring
    sd_all = sd_all.merge(
        sar_all[['date', 'era']].drop_duplicates('date'), on='date', how='left')
    sg_all = sg_all.merge(
        sar_all[['date', 'era']].drop_duplicates('date'), on='date', how='left')
    print(f"\nFig1 SAR–DAHITI (all years): N={len(sd_all)}")
    print(f"Fig1 SAR–Gauge  (all years, clean): N={len(sg_all)}")

    # ---- DAHITI vs Gauge (clean) for bias analysis ----------------------
    dg = []
    for _, row in dahiti.iterrows():
        delta = (gauge_clean['date'] - row['date']).dt.days.abs()
        if len(delta) == 0:
            continue
        idx = delta.idxmin()
        if delta[idx] <= MAX_DT:
            dg.append({'date': row['date'],
                       'wl_dahiti': row['wl_m'],
                       'wl_u': row['wl_u'],
                       'wl_gauge': gauge_clean.loc[idx, 'wl_m']})
    dg_df = pd.DataFrame(dg).dropna()

    if len(dg_df) >= 3:
        r2_dg, rmse_dg, bias_dg, r_dg = scatter_stats(
            dg_df['wl_gauge'].values, dg_df['wl_dahiti'].values)
        print(f"\nDAHITI vs Gauge-clean ({len(dg_df)} pairs): "
              f"bias={bias_dg:+.2f} m  RMSE={rmse_dg:.2f} m  r={r_dg:.4f}")
        dahiti_corr = dahiti.copy()
        dahiti_corr['wl_m'] = dahiti_corr['wl_m'] - bias_dg
    else:
        bias_dg = 0.0; r_dg = np.nan; rmse_dg = np.nan
        dahiti_corr = dahiti.copy()
        print("  Too few DAHITI–Gauge pairs; no bias correction applied")

    # ---- Pairs for FIGURE 2 (hypsometric fitting) -----------------------
    sar_filt_df = sar_filt.rename(columns={'areaLago': 'area_ha'})

    # SAR-Gauge (filtered SAR, clean gauge)
    sg_fit = match_pairs(sar_filt_df, gauge_clean, col_date_wl='date')

    # Planet-Gauge: clean and plateau versions
    pg_clean = match_pairs(planet, gauge_clean, col_date_wl='date')
    gauge_plat = gauge_raw[plat_flag].reset_index(drop=True)
    pg_plat  = match_pairs(planet, gauge_plat, col_date_wl='date') if len(gauge_plat) else pd.DataFrame()

    # Planet-DAHITI corrected
    pd_corr = match_pairs(planet, dahiti_corr, col_date_wl='date')

    print(f"\nFig2 SAR–Gauge (fit):      N={len(sg_fit)}, "
          + (f"WL {sg_fit.wl_m.min():.2f}–{sg_fit.wl_m.max():.2f} m" if len(sg_fit) else "none"))
    print(f"Fig2 Planet–Gauge (clean): N={len(pg_clean)}, "
          + (f"WL {pg_clean.wl_m.min():.2f}–{pg_clean.wl_m.max():.2f} m" if len(pg_clean) else "none"))
    print(f"Fig2 Planet–Gauge (plat):  N={len(pg_plat)}")
    print(f"Fig2 Planet–DAHITI(corr):  N={len(pd_corr)}, "
          + (f"WL {pd_corr.wl_m.min():.2f}–{pd_corr.wl_m.max():.2f} m" if len(pd_corr) else "none"))

    # ---- Hypsometric fits -----------------------------------------------
    print("\n--- Hypsometric fits ---")
    h0_lo = cfg['h0_bound_lo']
    popt_sg, r2_sg, rmse_sg = fit_hyps(sg_fit['wl_m'], sg_fit['area_ha'],
                                        h0_lo, f'SAR–Gauge({len(sg_fit)})')
    popt_pg, r2_pg, rmse_pg = fit_hyps(pg_clean['wl_m'], pg_clean['area_ha'],
                                        h0_lo, f'Planet–Gauge-clean({len(pg_clean)})')
    popt_pd, r2_pd, rmse_pd = (None, np.nan, np.nan)
    if len(pd_corr) >= 5:
        popt_pd, r2_pd, rmse_pd = fit_hyps(pd_corr['wl_m'], pd_corr['area_ha'],
                                            h0_lo, f'Planet–DAHITI-corr({len(pd_corr)})')

    # ====================================================================
    # FIGURE 1 — Data coherence (2×2)
    # ====================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig1.suptitle(f'{cfg["title"]} — Data source coherence', fontsize=13)

    h_aev = np.linspace(aev['h'].min(), aev['h'].max(), 300)
    era_colors = {'S1A': 'steelblue', 'S1C': 'tomato'}

    # (a) SAR–DAHITI scatter, all years, colored by era
    ax = axes[0, 0]
    for era, grp in sd_all.groupby('era'):
        ax.scatter(grp['area_ha'], grp['wl_m'],
                   s=18, alpha=0.55, color=era_colors.get(era, 'gray'),
                   label=f'{era} (N={len(grp)})')
    ax.plot(A_des(h_aev), h_aev, 'k--', lw=1.5, label='Design AEV')
    ax.set_xlabel('SAR surface area (ha)'); ax.set_ylabel('DAHITI WL (m a.s.l.)')
    ax.set_title('(a) DAHITI WL vs SAR area (all years)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) SAR–Gauge scatter, all years, colored by era
    ax = axes[0, 1]
    for era, grp in sg_all.groupby('era'):
        ax.scatter(grp['area_ha'], grp['wl_m'],
                   s=18, alpha=0.55, color=era_colors.get(era, 'gray'),
                   label=f'{era} (N={len(grp)})')
    ax.plot(A_des(h_aev), h_aev, 'k--', lw=1.5, label='Design AEV')
    ax.set_xlabel('SAR surface area (ha)'); ax.set_ylabel('Gauge WL (m a.s.l.)')
    ax.set_title('(b) Gauge WL vs SAR area (all years, plateau removed)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) DAHITI vs Gauge — time series with plateau flagged
    ax = axes[1, 0]
    # Plateau periods: shade them
    if n_plat > 0:
        plat_dates = gauge_raw['date'][plat_flag]
        # find contiguous blocks
        plat_idx = np.where(plat_flag)[0]
        breaks = np.where(np.diff(plat_idx) > 1)[0]
        starts = np.concatenate([[plat_idx[0]], plat_idx[breaks + 1]])
        ends   = np.concatenate([plat_idx[breaks], [plat_idx[-1]]])
        for s, e in zip(starts, ends):
            ax.axvspan(gauge_raw['date'].iloc[s], gauge_raw['date'].iloc[e],
                       alpha=0.15, color='orange', zorder=0)
    ax.plot(gauge_raw['date'], gauge_raw['wl_m'],
            '-', lw=0.8, color='steelblue', alpha=0.5, label='Gauge (raw)')
    ax.plot(gauge_clean['date'], gauge_clean['wl_m'],
            '-', lw=1.2, color='steelblue', alpha=0.9, label='Gauge (clean)')
    ax.plot(dahiti['date'], dahiti['wl_m'],
            'o', ms=4, color='tomato', alpha=0.7, label='DAHITI (raw)')
    if len(dg_df) > 0 and abs(bias_dg) > 0.01:
        ax.plot(dahiti_corr['date'], dahiti_corr['wl_m'],
                's', ms=3, color='darkorange', alpha=0.8,
                label=f'DAHITI (−{bias_dg:.2f} m)')
    # Orange shade legend entry
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(fc='orange', alpha=0.3, label=f'Plateau (N={n_plat} days)'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Date'); ax.set_ylabel('WL (m a.s.l.)')
    ax.set_title('(c) DAHITI vs Gauge — time series')
    ax.legend(handles=handles, fontsize=7.5); ax.grid(True, alpha=0.3)

    # (d) DAHITI vs Gauge — scatter (clean gauge only)
    ax = axes[1, 1]
    if len(dg_df) >= 3:
        obs = dg_df['wl_gauge'].values
        sim = dg_df['wl_dahiti'].values
        ax.errorbar(obs, sim, yerr=dg_df['wl_u'].values,
                    fmt='o', ms=4, color='tomato', alpha=0.7,
                    ecolor='tomato', elinewidth=0.8, capsize=2,
                    label=f'DAHITI vs Gauge-clean (N={len(dg_df)})')
        lim = (min(obs.min(), sim.min()) - 0.5, max(obs.max(), sim.max()) + 0.5)
        ax.plot(lim, lim, 'k--', lw=1, label='1:1')
        ax.plot(lim, [l + bias_dg for l in lim], 'darkorange', lw=1.2, ls='--',
                label=f'bias line ({bias_dg:+.2f} m)')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.text(0.05, 0.93,
                f'bias={bias_dg:+.2f} m\nRMSE={rmse_dg:.2f} m\nr={r_dg:.4f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    else:
        ax.text(0.5, 0.5, 'No valid pairs\n(all gauge plateau)',
                transform=ax.transAxes, ha='center', va='center', color='gray')
    ax.set_xlabel('Gauge WL (m a.s.l.)')
    ax.set_ylabel('DAHITI WL (m a.s.l.)')
    ax.set_title('(d) DAHITI vs Gauge — scatter (clean)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    p1 = OUT_DIR / f'{name}_altimetry_coherence.png'
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nFigure 1: {p1}")

    # ====================================================================
    # FIGURE 2 — Hypsometric curve comparison (1×3)
    # ====================================================================
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f'{cfg["title"]} — Hypsometric curves by data source', fontsize=13)

    # (a) SAR – Gauge
    plot_hyps_panel(axes[0],
                    '(a) SAR – Gauge\n(S1A, clean gauge)',
                    sg_fit, popt_sg, r2_sg, rmse_sg, aev, 'steelblue')

    # (b) Planet – Gauge
    plot_hyps_panel(axes[1],
                    '(b) Planet – Gauge\n(plateau-cleaned)',
                    pg_clean, popt_pg, r2_pg, rmse_pg, aev, 'seagreen',
                    pairs_plateau=pg_plat if len(pg_plat) else None)

    # (c) Planet – DAHITI (datum-corrected)
    title_c = (f'(c) Planet – DAHITI\n(datum-corrected, −{bias_dg:.2f} m)')
    plot_hyps_panel(axes[2],
                    title_c,
                    pd_corr, popt_pd, r2_pd, rmse_pd, aev, 'tomato')

    fig2.tight_layout()
    p2 = OUT_DIR / f'{name}_hyps_comparison.png'
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Figure 2: {p2}")

print("\nDone.")
