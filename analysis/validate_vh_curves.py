"""
validate_vh_curves.py

Validates two things:
  1. Direct quota: opendatasicilia daily quota vs AEGIS gauge (Aug 2023 - present)
  2. V->h curves: monthly volume -> official hypsometric curve -> h, vs AEGIS gauge (2022-2025)

Also builds a V->h WL series for the full historical period (2007-2025) for use in
Period A DEM reconstruction.

Outputs (analysis/schwatke_output/vh_validation/):
  - quota_vs_gauge_{res}.png   — scatter + time series: opendatasicilia quota vs AEGIS
  - vh_vs_gauge_{res}.png      — monthly V->h vs monthly-averaged gauge
  - vh_historical_{res}.csv    — full V->h series 2007-2025 for pipeline use
  - validation_summary.csv     — RMSE, bias, R² for all comparisons
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO       = pathlib.Path('.')
ODS_DIR    = REPO / 'raw_data' / 'opendatasicilia'
CURVE_DIR  = pathlib.Path('C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi')
GAUGE_DIR  = REPO / 'analysis' / 'schwatke_output' / 'gauge_downloads'
OUT_DIR    = REPO / 'analysis' / 'schwatke_output' / 'vh_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reservoir config ───────────────────────────────────────────────────────────
RESERVOIRS = {
    'Poma': {
        'cod':       'dig-18',
        'curve_xls': CURVE_DIR / 'Poma.xls',
        'curve_cols': ('quota', 'area_km2', 'area_ha', 'vol_Mm3'),   # cols 2-5
        'gauge_csv': GAUGE_DIR / 'poma_wl.csv',
        'gauge_min': 170.0,
    },
    'Rosamarina': {
        'cod':        'dig-22',
        'curve_xls':  CURVE_DIR / 'Rosamarina.xls',
        'curve_cols': ('quota', 'area_ha', 'area_km2', 'vol_Mm3'),
        'gauge_csv':  GAUGE_DIR / 'rosamarina_wl.csv',
        'gauge_min':  130.0,
    },
    'Pozzillo': {
        'cod':        'dig-19',
        'curve_xls':  CURVE_DIR / 'Pozzillo.xls',
        'curve_cols': ('quota', 'area_km2', 'area_ha', 'vol_Mm3'),
        'gauge_csv':  GAUGE_DIR / 'pozzillo_wl.csv',
        'gauge_min':  330.0,
    },
    'Ancipa': {
        'cod':        'dig-01',
        'curve_xls':  CURVE_DIR / 'Ancipa.xls',
        'curve_cols': ('quota', 'area_km2', 'vol_Mm3'),               # only 3 data cols
        'gauge_csv':  GAUGE_DIR / 'ancipa_livello_secca.csv',
        'gauge_min':  909.0,
    },
    'Garcia': {
        'cod':        'dig-09',
        'curve_xls':  CURVE_DIR / 'Garcia.xls',
        'curve_cols': ('quota', 'area_km2', 'vol_Mm3'),               # only 3 data cols
        'gauge_csv':  GAUGE_DIR / 'garcia_idrometro_radar.csv',
        'gauge_min':  170.0,
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_curve(cfg: dict):
    """Load official hypsometric curve -> (quota_arr, vol_Mm3_arr, interp_V->h)."""
    xls  = cfg['curve_xls']
    cols = cfg['curve_cols']
    n    = len(cols)               # 3 or 4 data columns
    col_idx = list(range(2, 2 + n))
    df = pd.read_excel(xls, sheet_name=0, header=None)
    data = df[col_idx].apply(pd.to_numeric, errors='coerce').dropna()
    data.columns = list(cols)
    data = data.sort_values('quota').reset_index(drop=True)
    v2h = interp1d(data['vol_Mm3'], data['quota'],
                   kind='linear', bounds_error=False, fill_value='extrapolate')
    return data, v2h


def load_gauge(cfg: dict, flat_tol: float = 0.005, flat_min_days: int = 5) -> pd.Series:
    """Load AEGIS gauge; remove flat periods and values below gauge_min."""
    path = cfg['gauge_csv']
    df   = pd.read_csv(path)
    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    dt_col = next((c for c in df.columns if 'dat' in c or 'time' in c), df.columns[0])
    wl_col = next((c for c in df.columns if 'wl' in c or 'level' in c
                   or 'quota' in c or 'livell' in c or 'valore' in c), df.columns[1])
    df['_dt'] = pd.to_datetime(df[dt_col], errors='coerce')
    df['_wl'] = pd.to_numeric(df[wl_col], errors='coerce')
    df = df.dropna(subset=['_dt', '_wl'])
    df = df[df['_wl'] >= cfg['gauge_min']]
    daily = df.set_index('_dt')['_wl'].resample('D').mean().dropna()
    # flat-period filter
    run_len = pd.Series(0, index=daily.index, dtype=int)
    count   = 0
    for i, chg in enumerate(daily.diff().abs()):
        count = count + 1 if (not np.isnan(chg) and chg < flat_tol) else 0
        run_len.iloc[i] = count
    daily = daily[run_len < flat_min_days]
    return daily.rename('gauge_m')


def stats(obs, pred, label=''):
    """Compute bias, RMSE, R² for paired arrays."""
    mask = ~(np.isnan(obs) | np.isnan(pred))
    o, p = obs[mask], pred[mask]
    if len(o) < 3:
        return {'label': label, 'n': len(o), 'bias': np.nan, 'rmse': np.nan, 'r2': np.nan}
    bias = float(np.mean(p - o))
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {'label': label, 'n': int(len(o)), 'bias': round(bias, 3),
            'rmse': round(rmse, 3), 'r2': round(r2, 3)}


# ── Load shared data ───────────────────────────────────────────────────────────
monthly = pd.read_csv(ODS_DIR / 'sicilia_dighe_volumi.csv', parse_dates=['data'])
daily_q = pd.read_csv(ODS_DIR / 'sicilia_dighe_volumi_giornalieri.csv', parse_dates=['data'])

summary_rows = []

for res, cfg in RESERVOIRS.items():
    print(f'\n=== {res} ===')

    # Official hypsometric curve
    curve_df, v2h = load_curve(cfg)

    # AEGIS gauge
    try:
        gauge = load_gauge(cfg)
        print(f'  Gauge: {len(gauge)} valid days | {gauge.index.min().date()} to {gauge.index.max().date()}')
    except Exception as e:
        print(f'  Gauge load failed: {e}')
        gauge = pd.Series(dtype=float, name='gauge_m')

    # ── Monthly V->h series (2007-2025) ────────────────────────────────────────
    cod = cfg['cod']
    mon = monthly[monthly['cod'] == cod][['data', 'volume']].copy()
    mon = mon.sort_values('data').set_index('data')
    mon['h_Vh'] = v2h(mon['volume'].values).round(3)
    mon.index.name = 'date'
    print(f'  Monthly V->h: {len(mon)} obs | {mon.index.min().date()} to {mon.index.max().date()}')
    print(f'  h_Vh range: {mon["h_Vh"].min():.1f}-{mon["h_Vh"].max():.1f} m')

    # Save historical series
    out_csv = OUT_DIR / f'vh_historical_{res}.csv'
    mon[['volume', 'h_Vh']].rename(columns={'volume': 'vol_Mm3'}).to_csv(out_csv)

    # ── Validation: V->h vs gauge (monthly averages) ───────────────────────────
    if len(gauge) > 0:
        gauge_monthly = gauge.resample('MS').mean()
        merged_m = pd.merge(mon[['h_Vh']], gauge_monthly.rename('gauge_m'),
                            left_index=True, right_index=True, how='inner')
        print(f'  V->h vs gauge overlap: {len(merged_m)} months')

        st = stats(merged_m['gauge_m'].values, merged_m['h_Vh'].values,
                   label=f'{res}: V->h_monthly vs gauge')
        print(f'  bias={st["bias"]:+.3f} m | RMSE={st["rmse"]:.3f} m | R²={st["r2"]:.3f}')
        summary_rows.append(st)

        # Plot V->h vs gauge (monthly)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f'{res} — V->h (official curve) vs AEGIS gauge', fontsize=12)

        ax = axes[0]
        ax.scatter(merged_m['gauge_m'], merged_m['h_Vh'], s=20, alpha=0.7, color='steelblue')
        lo = min(merged_m['gauge_m'].min(), merged_m['h_Vh'].min())
        hi = max(merged_m['gauge_m'].max(), merged_m['h_Vh'].max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set_xlabel('Gauge WL (m)')
        ax.set_ylabel('V->h (m)')
        ax.set_title(f'n={st["n"]} | bias={st["bias"]:+.3f} m | RMSE={st["rmse"]:.3f} m | R²={st["r2"]:.3f}')

        ax = axes[1]
        ax.plot(merged_m.index, merged_m['gauge_m'], label='Gauge', color='steelblue', lw=1.2)
        ax.plot(merged_m.index, merged_m['h_Vh'],    label='V->h',   color='firebrick', lw=1.2, ls='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_ylabel('WL (m)')
        ax.legend()
        ax.set_title('Time series (monthly)')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'vh_vs_gauge_{res}.png', dpi=150)
        plt.close()

    # ── Validation: direct quota from opendatasicilia vs AEGIS gauge ──────────
    dq = daily_q[daily_q['cod'] == cod][['data', 'quota']].copy()
    dq = dq.sort_values('data').set_index('data')['quota'].rename('ods_quota')
    dq.index = dq.index.normalize()

    if len(dq) > 0 and len(gauge) > 0:
        merged_d = pd.merge(gauge.rename('gauge_m'), dq,
                            left_index=True, right_index=True, how='inner')
        print(f'  Direct quota vs gauge overlap: {len(merged_d)} days')

        if len(merged_d) >= 5:
            st2 = stats(merged_d['gauge_m'].values, merged_d['ods_quota'].values,
                        label=f'{res}: ODS_quota vs gauge')
            print(f'  bias={st2["bias"]:+.3f} m | RMSE={st2["rmse"]:.3f} m | R²={st2["r2"]:.3f}')
            summary_rows.append(st2)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            fig.suptitle(f'{res} — Opendatasicilia quota vs AEGIS gauge (daily)', fontsize=12)

            ax = axes[0]
            ax.scatter(merged_d['gauge_m'], merged_d['ods_quota'], s=10, alpha=0.5, color='seagreen')
            lo = min(merged_d['gauge_m'].min(), merged_d['ods_quota'].min())
            hi = max(merged_d['gauge_m'].max(), merged_d['ods_quota'].max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
            ax.set_xlabel('AEGIS gauge (m)')
            ax.set_ylabel('ODS quota (m)')
            ax.set_title(f'n={st2["n"]} | bias={st2["bias"]:+.3f} m | RMSE={st2["rmse"]:.3f} m | R²={st2["r2"]:.3f}')

            ax = axes[1]
            ax.plot(merged_d.index, merged_d['gauge_m'],    label='AEGIS gauge', color='steelblue', lw=1.0)
            ax.plot(merged_d.index, merged_d['ods_quota'],  label='ODS quota',   color='seagreen',  lw=1.0, ls='--')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.set_ylabel('WL (m)')
            ax.legend()
            ax.set_title('Time series (daily)')
            plt.tight_layout()
            plt.savefig(OUT_DIR / f'quota_vs_gauge_{res}.png', dpi=150)
            plt.close()
        else:
            print(f'  Insufficient overlap for direct quota comparison ({len(merged_d)} days)')
    elif len(dq) > 0:
        print(f'  ODS daily quota available ({len(dq)} days) but no gauge to compare')

    # ── Print historical V->h for Period A ────────────────────────────────────
    period_a = mon[(mon.index >= '2014-01-01') & (mon.index <= '2017-01-01')]
    if len(period_a) > 0:
        print(f'  Period A (2014-2016): {len(period_a)} monthly obs | h={period_a["h_Vh"].min():.1f}-{period_a["h_Vh"].max():.1f} m')

# ── Save summary ──────────────────────────────────────────────────────────────
if summary_rows:
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / 'validation_summary.csv', index=False)
    print('\n=== Validation summary ===')
    print(df_sum.to_string(index=False))

print(f'\nOutputs saved to {OUT_DIR}')
