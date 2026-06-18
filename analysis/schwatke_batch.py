"""Schwatke hypsometric reconstruction — Poma and Rosamarina.
Mirrors schwatke_pozzillo.py; configured via RESERVOIRS dict.
"""
import sys, warnings
import numpy as np
import pandas as pd
import xlrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('analysis/schwatke_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DT = 5   # days for SAR–gauge matching

# ---------------------------------------------------------------------------
# Reservoir configurations
# ---------------------------------------------------------------------------
RESERVOIRS = {
    'poma': {
        'area_csv':    'validation_data/morphometric_analysis/shoreline_compactness/area_poma_2014-25.csv',
        'gauge_csv':   'analysis/schwatke_output/gauge_downloads/poma_wl.csv',
        'aev_xls':     'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Poma.xls',
        'aev_col_h':   2,   # column index: elevation (m a.s.l.)
        'aev_col_A':   4,   # column index: area (ha)
        'aev_col_V':   5,   # column index: volume (Mm³)
        'aev_h_min':   100, # threshold to ignore header/zero rows
        'gauge_min':   170, # m a.s.l. — filter zeros/invalid
        'adib_csv':    'validation_data/statistics/volume_statistics/poma_adib.csv',
        'title':       'Poma',
        'h0_bounds':   (155.0, None),
        'max_year':    2024,   # include 2024 for broader WL coverage (S1C launch Dec 5 2024)
    },
    'rosamarina': {
        'area_csv':    'validation_data/morphometric_analysis/shoreline_compactness/area_rosamarina_2014-25.csv',
        'gauge_csv':   'analysis/schwatke_output/gauge_downloads/rosamarina_wl.csv',
        'aev_xls':     'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Rosamarina.xls',
        'aev_col_h':   2,
        'aev_col_A':   3,   # Rosamarina: col 3 = area ha
        'aev_col_V':   5,
        'aev_h_min':   50,
        'gauge_min':   140,
        'adib_csv':    'validation_data/statistics/volume_statistics/rosamarina_adib.csv',
        'title':       'Rosamarina',
        'h0_bounds':   (95.0, None),
        'max_year':    2023,   # clean S1A period sufficient for Rosamarina
    },
}

# ---------------------------------------------------------------------------
def load_aev(cfg):
    wb = xlrd.open_workbook(cfg['aev_xls'])
    ws = wb.sheet_by_index(0)
    ch, cA, cV = cfg['aev_col_h'], cfg['aev_col_A'], cfg['aev_col_V']
    rows = []
    for i in range(1, ws.nrows):
        r = ws.row_values(i)
        try:
            h = float(r[ch]); A = float(r[cA]); V = float(r[cV])
            if h > cfg['aev_h_min'] and A >= 0 and V >= 0:
                rows.append({'h': h, 'A_ha': A, 'A_m2': A * 1e4, 'V_Mm3': V})
        except (ValueError, TypeError):
            pass
    return pd.DataFrame(rows).sort_values('h').reset_index(drop=True)


def hyps_model(h, a, b, h0):
    return a * np.maximum(h - h0, 0.0) ** b


def run_schwatke(name, cfg):
    print(f"\n{'='*60}")
    print(f"  {cfg['title']}")
    print(f"{'='*60}")

    # --- 1. Gauge → daily mean ----------------------------------------
    wl = pd.read_csv(cfg['gauge_csv'])
    wl['time'] = pd.to_datetime(wl['time'])
    wl['wl_m'] = pd.to_numeric(wl['wl_m'], errors='coerce')
    wl = (wl[wl['wl_m'] > cfg['gauge_min']]
            .dropna(subset=['wl_m'])
            .set_index('time')['wl_m']
            .resample('D').mean()
            .reset_index())
    wl.columns = ['date', 'wl_m']
    wl = wl.dropna().sort_values('date').reset_index(drop=True)
    print(f"Gauge: {len(wl)} daily records  "
          f"{wl.date.min().date()}–{wl.date.max().date()}  "
          f"WL {wl.wl_m.min():.2f}–{wl.wl_m.max():.2f} m")

    # --- 2. SAR areas (2014–2025) ------------------------------------
    area = pd.read_csv(cfg['area_csv'])
    area['date'] = pd.to_datetime(area['date'], dayfirst=False, errors='coerce')
    area = area.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    area['year'] = area['date'].dt.year
    print(f"SAR: {len(area)} obs  "
          f"{area.date.min().date()}–{area.date.max().date()}  "
          f"area {area.areaLago.min():.1f}–{area.areaLago.max():.1f} ha")

    # --- 3. Match pairs ----------------------------------------------
    max_year = cfg['max_year']

    def match_pairs(subset):
        pairs = []
        for _, row in subset.iterrows():
            delta = (wl['date'] - row['date']).dt.days.abs()
            idx = delta.idxmin()
            if delta[idx] <= MAX_DT:
                pairs.append({'date': row['date'],
                              'area_ha': row['areaLago'],
                              'wl_m': wl.loc[idx, 'wl_m']})
        return pd.DataFrame(pairs).dropna().reset_index(drop=True)

    all_pairs  = match_pairs(area)
    s1a_pairs  = match_pairs(area[area['year'] <= max_year])
    excl_pairs = match_pairs(area[area['year'] > max_year])
    print(f"Pairs total: {len(all_pairs)}  "
          f"S1A (≤{max_year}): {len(s1a_pairs)}  "
          f"excluded (>{max_year}): {len(excl_pairs)}")

    pairs_df = s1a_pairs.copy()
    print(f"Fit dataset (≤{max_year}): N={len(pairs_df)}  "
          f"area {pairs_df.area_ha.min():.1f}–{pairs_df.area_ha.max():.1f} ha  "
          f"WL {pairs_df.wl_m.min():.2f}–{pairs_df.wl_m.max():.2f} m")

    # --- 4. Design AEV -----------------------------------------------
    aev = load_aev(cfg)
    print(f"Design AEV: {len(aev)} rows  "
          f"h={aev.h.min():.1f}–{aev.h.max():.1f} m  "
          f"V={aev.V_Mm3.min():.2f}–{aev.V_Mm3.max():.2f} Mm³")
    V_design = interp1d(aev['h'], aev['V_Mm3'],
                        kind='linear', bounds_error=False, fill_value='extrapolate')
    A_design = interp1d(aev['h'], aev['A_m2'],
                        kind='linear', bounds_error=False, fill_value='extrapolate')

    # --- 5. Hypsometric fit ------------------------------------------
    h_obs  = pairs_df['wl_m'].values
    A_obs  = pairs_df['area_ha'].values * 1e4   # m²
    h0_lo  = cfg['h0_bounds'][0]
    h0_hi  = float(pairs_df['wl_m'].min()) - 0.01
    h0_lo  = min(h0_lo, h0_hi - 1.0)

    popt, _ = curve_fit(
        hyps_model, h_obs, A_obs,
        p0=[1e6, 1.5, h0_hi - 5.0],
        bounds=([0, 0.2, h0_lo], [1e9, 6.0, h0_hi]),
        maxfev=30000,
    )
    a_fit, b_fit, h0_fit = popt
    A_pred = hyps_model(h_obs, *popt)
    r2   = 1.0 - np.sum((A_obs - A_pred)**2) / np.sum((A_obs - A_obs.mean())**2)
    rmse = np.sqrt(np.mean((A_obs - A_pred)**2)) / 1e4
    print(f"\nHypsometric fit: A = {a_fit:.2f} * (h - {h0_fit:.3f})^{b_fit:.4f}")
    print(f"  R² = {r2:.4f}   RMSE = {rmse:.1f} ha")

    # --- 6. Volume integration (anchor at h_ref) ---------------------
    h_ref  = float(pairs_df['wl_m'].min())
    h_grid = np.arange(h0_fit + 0.01, aev['h'].max() + 1.0, 0.01)
    A_grid = hyps_model(h_grid, *popt)
    dh     = np.diff(h_grid)
    V_int  = np.concatenate([[0.0],
                              np.cumsum(0.5 * (A_grid[:-1] + A_grid[1:]) * dh)])
    idx_r  = int(np.searchsorted(h_grid, h_ref))
    V_int += float(V_design(h_ref)) * 1e6 - V_int[idx_r]
    V_sar  = interp1d(h_grid, V_int / 1e6,
                      kind='linear', bounds_error=False, fill_value=np.nan)

    # --- 7. Level shift and volume summary ---------------------------
    h_des_of_A = interp1d(aev['A_m2'], aev['h'],
                           kind='linear', bounds_error=False, fill_value=np.nan)
    shifts = [row['wl_m'] - float(h_des_of_A(row['area_ha'] * 1e4))
              for _, row in pairs_df.iterrows()
              if np.isfinite(float(h_des_of_A(row['area_ha'] * 1e4)))]
    mean_shift = np.nanmean(shifts)
    print(f"Mean level shift: {mean_shift:+.2f} m")
    h_full = aev['h'].max()
    dV = float(V_sar(h_full)) - float(V_design(h_full))
    print(f"ΔV @ h={h_full:.1f} m: {dV:+.2f} Mm³ "
          f"({dV/float(V_design(h_full))*100:+.1f}%)")

    # --- 8. Volume time series (full gauge period) -------------------
    # Compute V for ALL valid gauge records (not clipped to h_ref)
    wl_full = wl[wl['wl_m'] <= aev['h'].max()].copy()
    wl_full['V_design'] = V_design(wl_full['wl_m'].values)
    wl_full['V_sar']    = V_sar(wl_full['wl_m'].values)

    adib = pd.read_csv(cfg['adib_csv'])
    adib['date'] = pd.to_datetime(adib['date'], dayfirst=False, errors='coerce')
    adib = adib.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    adib = adib.rename(columns={'volume_adib': 'V_adib'})

    wl_monthly = (wl_full.set_index('date')[['V_design','V_sar']]
                  .resample('MS').mean().reset_index())
    merged = pd.merge(wl_monthly, adib[['date','V_adib']], on='date', how='inner')
    print(f"\n=== AdB comparison ({len(merged)} monthly pairs) ===")
    for col, lbl in [('V_design','Design AEV'), ('V_sar','SAR hypsometry')]:
        sub = merged[['V_adib', col]].dropna()
        obs, sim = sub['V_adib'].values, sub[col].values
        r2v  = 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)
        rmse_v = np.sqrt(np.mean((sim - obs)**2))
        bias_v = (sim - obs).mean()
        rp = np.corrcoef(obs, sim)[0, 1]
        print(f"  {lbl:20s}: R²={r2v:.4f}  r={rp:.4f}  "
              f"RMSE={rmse_v:.2f} Mm³  bias={bias_v:+.2f} Mm³")

    # --- 9. Figure (3 panels — hypsometric, V-h, storage) -----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): hypsometric curve
    ax = axes[0]
    h_plot = np.linspace(h0_fit + 0.1, aev['h'].max(), 400)
    ax.scatter(s1a_pairs['area_ha'],  s1a_pairs['wl_m'],
               s=14, alpha=0.55, color='steelblue', zorder=3,
               label=f'S1A pairs used in fit (≤{max_year})')
    if len(excl_pairs):
        ax.scatter(excl_pairs['area_ha'], excl_pairs['wl_m'],
                   s=14, alpha=0.4, color='goldenrod', zorder=3, marker='s',
                   label=f'S1A+C excluded (>{max_year})')
    ax.plot(hyps_model(h_plot, *popt) / 1e4, h_plot,
            'r-', lw=2, label=f'SAR fit (R²={r2:.3f})')
    ax.plot(aev['A_ha'], aev['h'], 'k--', lw=1.5, label='Design AEV')
    ax.set_xlabel('Surface area (ha)')
    ax.set_ylabel('Water level (m a.s.l.)')
    ax.set_title(f'(a) Hypsometric curve — {cfg["title"]}')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel (b): V–h
    ax = axes[1]
    h_rng = np.linspace(h_ref, aev['h'].max(), 400)
    ax.plot(V_sar(h_rng),    h_rng, 'r-',  lw=2,   label='SAR-derived')
    ax.plot(V_design(h_rng), h_rng, 'k--', lw=1.5, label='Design AEV')
    ax.axhline(h_ref, color='gray', lw=0.8, ls=':', label=f'h_ref={h_ref:.1f} m')
    ax.set_xlabel('Volume (Mm³)')
    ax.set_ylabel('Water level (m a.s.l.)')
    ax.set_title('(b) Volume–level curves')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel (c): full-period storage vs AdB
    ax = axes[2]
    ax.plot(adib['date'], adib['V_adib'],
            'o-', ms=3, lw=1.0, color='gray', alpha=0.5, label='AdB official')
    ax.plot(wl_full['date'], wl_full['V_design'],
            'k--', lw=1.2, alpha=0.85, label='Design AEV (gauge)')
    ax.plot(wl_full['date'], wl_full['V_sar'],
            'r-', lw=1.6, label='SAR hypsometry')
    ax.axvspan(wl_full['date'].min(), wl_full['date'].max(),
               alpha=0.06, color='steelblue')
    ax.set_xlabel('Date'); ax.set_ylabel('Volume (Mm³)')
    ax.set_title('(c) Storage time series vs AdB official')
    # avoid label overlap: one tick every 2 years, rotated
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

    fig.suptitle(f'{cfg["title"]} — Schwatke hypsometric reconstruction', fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / f'{name}_schwatke_mvp.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure: {out_path}")

    pairs_df.to_csv(OUT_DIR / f'{name}_hyps_pairs.csv', index=False)
    wl_full[['date','wl_m','V_design','V_sar']].to_csv(
        OUT_DIR / f'{name}_volume_timeseries.csv', index=False)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    for name, cfg in RESERVOIRS.items():
        run_schwatke(name, cfg)
