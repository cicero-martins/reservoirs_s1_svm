"""
schwatke_extended.py  (extended validation — scalability beyond the core 5)

Scalar hypsometric method (Schwatke-style) on 4 additional Sicilian reservoirs
that have an updated survey curve + AEGIS gauge but no full 3D DEM. For each:
  1. pair the SAR water-area series (exportSicilyExtended.js / VV-Otsu) with the
     daily AEGIS gauge WL (±MAX_DT days);
  2. fit the hypsometric power law  A = a·(h - h0)^b;
  3. VALIDATE the fitted area-elevation curve against the updated official survey
     curve (Arancio 2022, Nicoletti, Olivo 2021) over the observed WL range.

Together with the 5 core reservoirs this extends the validated set to 9 Sicilian
reservoirs spanning A/P ≈ 50–240 m, tying bathymetric reliability to the Paper-1
A/P axis without any global GEE exports.

NOTE: Cimia + Disueri were dropped — their coord-fallback collided onto one JRC
polygon (identical SAR series) and the Disueri gauge is flat (149.8–151.5 m) → both
need a clean re-export (fixed coord / GDW_ID). Castello has an updated curve in an
awkward 256-column layout → fitted here but not yet survey-validated.

Outputs: analysis/schwatke_output/extended/  (schwatke_extended.csv, schwatke_extended.png)
"""

import sys, glob, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

REPO   = pathlib.Path('.')
SAR_DIR = REPO / 'raw_data' / 'exportSicilyExtended' / 'GEE_SicilyExtended_VVotsu'
GAUGE   = REPO / 'analysis' / 'schwatke_output' / 'gauge_downloads'
OUT     = REPO / 'analysis' / 'schwatke_output' / 'extended'
OUT.mkdir(parents=True, exist_ok=True)
NEW     = 'C:/Users/Unipa/Documents/GEE/Data/NewCurves/'
MAX_DT  = 7   # days for SAR↔gauge pairing

# updated curve spec: (glob, sheet, engine, kind, cols)
#   kind 'single' → cols=(quota_col, area_col); 'block4' → cols=[(q,a),...]; area in mq
EXT = {
    'Arancio':   dict(ap=182.2, gauge='arancio_wl.csv',
                      updated=('ARANCIO*', 'BASE', 'openpyxl', 'single', (0, 2))),
    # CASTELLO.xlsx sheet 'Quota_V_S' has ~60 duplicate (Quota/Volume/Superficie)
    # scratch blocks tiled across 256 columns (leftovers from an iterative build),
    # but ONE of them (cols F:H, pandas idx 5:7) is the complete, monotonic master
    # table alone: 3001 rows, quota 267.20-297.20 m, fully populated -- confirmed by
    # scanning every block's non-null extent (all others are ~33-row fragments
    # already covered by this one). No need to merge blocks.
    'Castello':  dict(ap=126.7, gauge='castello_wl.csv',
                      updated=('CASTELLO*', 'Quota_V_S', 'openpyxl', 'single', (5, 7))),
    'Nicoletti': dict(ap=119.7, gauge='nicoletti_wl.csv',
                      updated=('NICOLETTI*', 'Dati Aree-Volumi', 'openpyxl', 'single', (0, 2))),
    'Olivo':     dict(ap=50.7,  gauge='olivo_wl.csv',
                      updated=('OLIVO*', 'Tabella centimetrica 2021', 'openpyxl', 'block4',
                               [(1, 3), (5, 7), (9, 11), (13, 15)])),
}


def load_gauge(fname):
    g = pd.read_csv(GAUGE / fname)
    g.columns = [c.strip().lower() for c in g.columns]
    tc = next(c for c in g.columns if 'time' in c or 'date' in c)
    wc = next(c for c in g.columns if 'wl' in c or 'quota' in c or 'value' in c)
    g = g[[tc, wc]].copy(); g.columns = ['date', 'wl']
    g['date'] = pd.to_datetime(g['date'], errors='coerce')
    g['wl'] = pd.to_numeric(g['wl'], errors='coerce')
    return g.dropna().groupby('date', as_index=False).wl.mean().sort_values('date')


def load_updated(spec):
    """Return interp h->area_ha for the updated survey curve."""
    pat, sheet, eng, kind, cols = spec
    f = [h for h in glob.glob(NEW + pat) if h.lower().endswith(('.xls', '.xlsx'))][0]
    raw = pd.read_excel(f, sheet_name=sheet, header=None, engine=eng)
    if kind == 'single':
        qc, ac = cols
        df = raw[[qc, ac]].apply(pd.to_numeric, errors='coerce').dropna()
        df.columns = ['q', 'a']
    else:  # block4
        parts = []
        for qc, ac in cols:
            p = raw[[qc, ac]].apply(pd.to_numeric, errors='coerce').dropna()
            p.columns = ['q', 'a']; parts.append(p)
        df = pd.concat(parts)
    df = df[(df.q > 50) & (df.q < 1000) & (df.a >= 0)].sort_values('q')
    df = df.groupby('q', as_index=False).a.mean()
    return interp1d(df.q, df.a / 1e4, bounds_error=False, fill_value='extrapolate')  # ha


def power_law(h, a, b, h0):
    return a * np.maximum(h - h0, 1e-6) ** b


rows, fits = [], {}
for name, cfg in EXT.items():
    sar = pd.read_csv(SAR_DIR / f'SAR_area_{name}.csv')
    sar['date'] = pd.to_datetime(sar.date, errors='coerce')
    sar = sar.dropna(subset=['date']).sort_values('date')
    g = load_gauge(cfg['gauge'])

    pair = pd.merge_asof(sar[['date', 'area_ha']], g, on='date',
                         tolerance=pd.Timedelta(f'{MAX_DT}D'), direction='nearest').dropna()
    pair = pair[pair.area_ha > 0]
    if len(pair) < 10:
        print(f'{name}: only {len(pair)} pairs, skipping'); continue

    h, A = pair.wl.values, pair.area_ha.values
    h0_hi = float(h.min()) - 0.01
    try:
        popt, _ = curve_fit(power_law, h, A, p0=[1.0, 1.5, h0_hi - 5],
                            bounds=([0, 0.5, h0_hi - 40], [1e6, 5, h0_hi]), maxfev=20000)
    except Exception as e:
        print(f'{name}: fit failed ({e})'); continue
    A_pred = power_law(h, *popt)
    r2_fit = 1 - np.sum((A - A_pred)**2) / np.sum((A - A.mean())**2)

    # validate fitted curve vs updated survey over the observed WL range
    val = {}
    if cfg['updated']:
        upd = load_updated(cfg['updated'])
        hs = np.linspace(h.min(), h.max(), 100)
        a_fit, a_upd = power_law(hs, *popt), upd(hs)
        m = np.isfinite(a_upd)
        resid = a_fit[m] - a_upd[m]
        val = dict(area_rmse_ha=float(np.sqrt(np.mean(resid**2))),
                   area_bias_ha=float(resid.mean()),
                   area_r2=float(1 - np.sum(resid**2) / np.sum((a_upd[m] - a_upd[m].mean())**2)))
    fits[name] = dict(cfg=cfg, pair=pair, popt=popt, upd=(load_updated(cfg['updated']) if cfg['updated'] else None))

    rows.append(dict(reservoir=name, ap_m=cfg['ap'], n_pairs=len(pair),
                     wl_min=round(float(h.min()), 1), wl_max=round(float(h.max()), 1),
                     area_min_ha=round(float(A.min()), 1), area_max_ha=round(float(A.max()), 1),
                     a=round(popt[0], 4), b=round(popt[1], 3), h0=round(popt[2], 2),
                     r2_fit=round(r2_fit, 3),
                     survey_area_rmse_ha=round(val['area_rmse_ha'], 1) if val else None,
                     survey_area_bias_ha=round(val['area_bias_ha'], 1) if val else None,
                     survey_area_r2=round(val['area_r2'], 3) if val else None))

df = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df.to_csv(OUT / 'schwatke_extended.csv', index=False)
print(df.to_string(index=False))
# Fit quality tracks WL-range width and A/P: wide drawdown + higher A/P → good fit.
wide = df[df.wl_max - df.wl_min >= 8]
print(f"\nFit quality: wide-drawdown reservoirs (≥8 m range) mean fit R² = "
      f"{wide.r2_fit.mean():.3f} (n={len(wide)}); narrow-range are under-constrained.")
val = df.dropna(subset=['survey_area_rmse_ha'])
if len(val):
    best = val.loc[val.wl_max.sub(val.wl_min).idxmax()]
    print(f"Survey validation (area RMSE): " +
          ', '.join(f'{r.reservoir} {r.survey_area_rmse_ha} ha (bias {r.survey_area_bias_ha:+.0f})'
                    for _, r in val.iterrows()))
    print(f"  → best case {best.reservoir} (widest range {best.wl_max-best.wl_min:.0f} m): "
          f"RMSE {best.survey_area_rmse_ha} ha, R²={best.survey_area_r2}.")
    print("  (R² is unreliable over the narrow 2022+ gauge ranges; use RMSE/bias.)")
print(f"\nSaved: {OUT/'schwatke_extended.csv'}")

# ── Figure: A(h) per reservoir ─────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.24)
fig.suptitle('Extended validation — SAR scalar hypsometry vs updated survey curve',
             fontsize=13, fontweight='bold')
for ax_i, (name, F) in enumerate(fits.items()):
    ax = fig.add_subplot(gs[ax_i])
    pair, popt, upd = F['pair'], F['popt'], F['upd']
    hs = np.linspace(pair.wl.min(), pair.wl.max(), 100)
    ax.scatter(pair.area_ha, pair.wl, s=14, c='#90caf9', alpha=0.7, label='SAR–gauge pairs')
    ax.plot(power_law(hs, *popt), hs, 'C0-', lw=2.2, label='SAR power-law fit')
    if upd is not None:
        ax.plot(upd(hs), hs, 'C2-', lw=2, label='Updated survey')
    ax.set_xlabel('Area (ha)'); ax.set_ylabel('Water level (m ASL)')
    ap = F['cfg']['ap']
    row = df[df.reservoir == name].iloc[0]
    sub = f"n={row.n_pairs}, fit R²={row.r2_fit}"
    if pd.notna(row.survey_area_rmse_ha):
        sub += f"  |  vs survey: RMSE {row.survey_area_rmse_ha} ha, bias {row.survey_area_bias_ha:+.0f} ha"
    ax.set_title(f'{name}  (A/P {ap:.0f} m)\n{sub}', fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

fig.subplots_adjust(top=0.9)
fig.savefig(OUT / 'schwatke_extended.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT/'schwatke_extended.png'}")
