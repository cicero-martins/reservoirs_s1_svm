"""Schwatke-style hypsometric reconstruction for Ancipa reservoir.
SAR area (Sentinel-1/SVM) + in-situ gauge → empirical hypsometric curve
→ volume time series, compared with design AEV and GEE-derived volumes.
"""
import sys
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

sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('analysis/schwatke_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DT = 5  # days for SAR–gauge matching

# ---------------------------------------------------------------------------
# 1. Gauge
# ---------------------------------------------------------------------------
WL_PATH = 'C:/Users/Unipa/Documents/GEE/Results/Ancipa Diga R2 - Water Level - 2025-03-03.csv'

df_wl = pd.read_csv(WL_PATH, sep=None, engine='python')
df_wl.columns = ['time', 'raw', 'variation', 'wl_m', 'selective']
df_wl['time'] = pd.to_datetime(df_wl['time'], errors='coerce')
df_wl['wl_m'] = pd.to_numeric(df_wl['wl_m'], errors='coerce')
wl = (df_wl.dropna(subset=['time', 'wl_m'])
           .set_index('time')['wl_m']
           .resample('D').mean()
           .reset_index())
wl.columns = ['date', 'wl_m']
wl = wl.dropna().sort_values('date').reset_index(drop=True)
print(f"Gauge: {len(wl)} daily records  "
      f"{wl.date.min().date()} → {wl.date.max().date()}  "
      f"WL {wl.wl_m.min():.2f}–{wl.wl_m.max():.2f} m")

# ---------------------------------------------------------------------------
# 2. SAR surface areas
# ---------------------------------------------------------------------------
area = pd.read_csv(
    'validation_data/morphometric_analysis/shoreline_compactness/area_ancipa_2014-25.csv')
area['date'] = pd.to_datetime(area['date'])
area['A_m2'] = area['areaLago'] * 1e4          # ha → m²
print(f"SAR: {len(area)} obs  "
      f"{area.date.min().date()} → {area.date.max().date()}  "
      f"area {area.areaLago.min():.1f}–{area.areaLago.max():.1f} ha")

# ---------------------------------------------------------------------------
# 3. Match SAR–gauge pairs (±MAX_DT days)
# ---------------------------------------------------------------------------
pairs = []
for _, row in area.iterrows():
    delta = (wl['date'] - row['date']).dt.days.abs()
    idx = delta.idxmin()
    if delta[idx] <= MAX_DT:
        pairs.append({
            'date':    row['date'],
            'area_ha': row['areaLago'],
            'A_m2':    row['A_m2'],
            'wl_m':    wl.loc[idx, 'wl_m'],
            'dt_days': delta[idx],
        })
pairs_df = pd.DataFrame(pairs).dropna().reset_index(drop=True)
print(f"Pairs: N={len(pairs_df)}  "
      f"area {pairs_df.area_ha.min():.1f}–{pairs_df.area_ha.max():.1f} ha  "
      f"WL {pairs_df.wl_m.min():.2f}–{pairs_df.wl_m.max():.2f} m  "
      f"median |dt|={pairs_df.dt_days.median():.0f} d")

# ---------------------------------------------------------------------------
# 4. Fit hypsometric power-law  A = a * (h - h0)^b
# ---------------------------------------------------------------------------
def hyps_model(h, a, b, h0):
    return a * np.maximum(h - h0, 0.0) ** b

h_obs = pairs_df['wl_m'].values
A_obs = pairs_df['A_m2'].values

h0_upper = float(pairs_df['wl_m'].min()) - 0.01   # strict upper bound for h0
h0_guess = h0_upper - 3.0

popt, pcov = curve_fit(
    hyps_model, h_obs, A_obs,
    p0=[2e5, 1.5, h0_guess],
    bounds=([0, 0.2, 905.0], [1e8, 6.0, h0_upper]),
    maxfev=30000,
)
a_fit, b_fit, h0_fit = popt
A_pred = hyps_model(h_obs, *popt)
ss_res = np.sum((A_obs - A_pred) ** 2)
ss_tot = np.sum((A_obs - A_obs.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot
rmse_ha = np.sqrt(np.mean((A_obs - A_pred) ** 2)) / 1e4
print(f"\nHypsometric fit: A = {a_fit:.2f} * (h - {h0_fit:.3f})^{b_fit:.4f}")
print(f"  R² = {r2:.4f}   RMSE = {rmse_ha:.1f} ha")

# ---------------------------------------------------------------------------
# 5. Design AEV from Ancipa.xls (Foglio1)
#    col 2 = quote (m), col 3 = aree (km²), col 4 = volumi (Mm³)
# ---------------------------------------------------------------------------
AEV_PATH = 'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Ancipa.xls'
wb = xlrd.open_workbook(AEV_PATH)
ws = wb.sheet_by_name('Foglio1')
aev_rows = []
for i in range(1, ws.nrows):
    r = ws.row_values(i)
    try:
        h_val = float(r[2])
        a_km2 = float(r[3])
        v_mm3 = float(r[4])
        if h_val > 100 and a_km2 > 0 and v_mm3 >= 0:
            aev_rows.append({'h': h_val, 'A_km2': a_km2,
                             'A_m2': a_km2 * 1e6, 'V_Mm3': v_mm3})
    except (ValueError, TypeError):
        pass
aev = pd.DataFrame(aev_rows).sort_values('h').reset_index(drop=True)
print(f"\nDesign AEV: {len(aev)} rows  "
      f"h={aev.h.min():.0f}–{aev.h.max():.0f} m  "
      f"V={aev.V_Mm3.min():.2f}–{aev.V_Mm3.max():.2f} Mm³")

V_design = interp1d(aev['h'], aev['V_Mm3'],
                    kind='linear', bounds_error=False, fill_value='extrapolate')
A_design = interp1d(aev['h'], aev['A_m2'],
                    kind='linear', bounds_error=False, fill_value='extrapolate')

# ---------------------------------------------------------------------------
# 6. Integrate SAR-derived volume curve, anchored to design AEV at h_ref
# ---------------------------------------------------------------------------
h_ref = float(pairs_df['wl_m'].min())
h_grid = np.arange(h0_fit + 0.01, aev['h'].max() + 1.0, 0.01)
A_grid = hyps_model(h_grid, *popt)

dh = np.diff(h_grid)
V_int = np.concatenate([[0.0],
                         np.cumsum(0.5 * (A_grid[:-1] + A_grid[1:]) * dh)])

idx_ref = int(np.searchsorted(h_grid, h_ref))
V_ref_design = float(V_design(h_ref)) * 1e6        # m³
V_int += V_ref_design - V_int[idx_ref]              # anchor

V_sar = interp1d(h_grid, V_int / 1e6,
                 kind='linear', bounds_error=False, fill_value=np.nan)

# ---------------------------------------------------------------------------
# 7. GEE volume series (SAR-area-based, for comparison)
# ---------------------------------------------------------------------------
VOL_PATH = 'C:/Users/Unipa/Documents/GEE/Results/ee-chart_Ancipa-volumes.csv'
gee = pd.read_csv(VOL_PATH)
gee.columns = ['date', 'volume_Mm3']
gee['date'] = pd.to_datetime(gee['date'], errors='coerce')
gee = gee.dropna().sort_values('date').reset_index(drop=True)
print(f"\nGEE volumes: {len(gee)} records  "
      f"{gee.date.min().date()} → {gee.date.max().date()}  "
      f"V={gee.volume_Mm3.min():.2f}–{gee.volume_Mm3.max():.2f} Mm³")

# ---------------------------------------------------------------------------
# 8. Volume time series from gauge
# ---------------------------------------------------------------------------
wl_valid = wl[wl['wl_m'].between(h_ref, aev['h'].max())].copy()
wl_valid['V_design_Mm3'] = V_design(wl_valid['wl_m'].values)
wl_valid['V_sar_Mm3']    = V_sar(wl_valid['wl_m'].values)

# Level shift: for same area, what is h_sar - h_design?
# Compute at each observed pair
h_design_interp = interp1d(aev['A_m2'], aev['h'],
                            kind='linear', bounds_error=False, fill_value=np.nan)
shift_vals = []
for _, row in pairs_df.iterrows():
    h_d = float(h_design_interp(row['A_m2']))
    if np.isfinite(h_d):
        shift_vals.append(row['wl_m'] - h_d)
mean_shift = np.nanmean(shift_vals)
print(f"\nMean level shift (observed − design for same area): {mean_shift:+.2f} m")

# Volume loss at full pool (design max = h=950m)
h_full = aev['h'].max()
V_design_full = float(V_design(h_full))
V_sar_full    = float(V_sar(h_full))
dV_full = V_sar_full - V_design_full
pct_full = dV_full / V_design_full * 100
print(f"Volume at h={h_full:.0f} m: design={V_design_full:.2f} Mm³  "
      f"SAR={V_sar_full:.2f} Mm³  ΔV={dV_full:+.2f} Mm³ ({pct_full:+.1f}%)")

# Table
print(f"\n{'Level':>8}  {'V_design':>10}  {'V_SAR':>10}  {'ΔV':>10}  {'ΔV%':>8}")
for ht in [920, 924, 928, 932, 936, 940, 944, 948, 950]:
    if ht < h_ref:
        continue
    vd = float(V_design(ht))
    vs = float(V_sar(ht))
    dv = vs - vd
    pct = dv / vd * 100 if vd > 0 else np.nan
    print(f"{ht:>8.0f}  {vd:>10.3f}  {vs:>10.3f}  {dv:>+10.3f}  {pct:>+7.1f}%")

# ---------------------------------------------------------------------------
# 9. Validation against GEE volumes via gauge-matched comparison
# ---------------------------------------------------------------------------
wl_tmp = wl.copy()
wl_tmp['V_design_Mm3'] = V_design(wl_tmp['wl_m'].values)
wl_tmp['V_sar_Mm3']    = V_sar(wl_tmp['wl_m'].values)
wl_monthly = wl_tmp.set_index('date').resample('MS').mean().reset_index()

# Match GEE volumes to monthly gauge-derived volumes
gee_monthly = gee.set_index('date').resample('MS').mean().reset_index()
merged = pd.merge(wl_monthly[['date', 'V_design_Mm3', 'V_sar_Mm3']],
                  gee_monthly.rename(columns={'volume_Mm3': 'V_gee'}),
                  on='date', how='inner')

if len(merged) >= 3:
    obs = merged['V_gee'].values
    for col, lbl in [('V_design_Mm3', 'Design AEV'), ('V_sar_Mm3', 'SAR hypsometry')]:
        sub = merged[['V_gee', col]].dropna()
        o = sub['V_gee'].values
        s = sub[col].values
        rmse = np.sqrt(np.mean((s - o) ** 2))
        bias = (s - o).mean()
        ratio = (s / o).mean()
        r_pearson = np.corrcoef(o, s)[0, 1]
        print(f"\n{lbl} vs GEE ({len(sub)} monthly pairs):")
        print(f"  RMSE={rmse:.3f} Mm³  bias={bias:+.3f} Mm³  "
              f"ratio={ratio:.2f}  Pearson r={r_pearson:.4f}")

# ---------------------------------------------------------------------------
# 10. Figure — 3 panels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ---- Panel A: hypsometric scatter + fit ----
ax = axes[0]
h_plot = np.linspace(h0_fit + 0.1, aev['h'].max(), 300)
ax.scatter(pairs_df['area_ha'], pairs_df['wl_m'],
           s=18, alpha=0.55, color='steelblue', zorder=3, label='SAR–gauge pairs')
ax.plot(hyps_model(h_plot, *popt) / 1e4, h_plot,
        'r-', lw=2, label=f'SAR fit (R²={r2:.3f})')
ax.plot(aev['A_m2'] / 1e4, aev['h'],
        'k--', lw=1.5, label='Design AEV')
ax.set_xlabel('Surface area (ha)')
ax.set_ylabel('Water level (m a.s.l.)')
ax.set_title('(a) Hypsometric curve — Ancipa')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Panel B: V–h comparison ----
ax = axes[1]
h_range = np.linspace(h_ref, aev['h'].max(), 300)
ax.plot(V_sar(h_range), h_range, 'r-', lw=2, label='SAR-derived')
ax.plot(V_design(h_range), h_range, 'k--', lw=1.5, label='Design AEV')
ax.axhline(h_ref, color='gray', lw=0.8, ls=':', label=f'h_ref={h_ref:.1f} m')
ax.set_xlabel('Volume (Mm³)')
ax.set_ylabel('Water level (m a.s.l.)')
ax.set_title('(b) Volume–level curves')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Panel C: volume time series ----
ax = axes[2]
ax.plot(wl_valid['date'], wl_valid['V_design_Mm3'],
        'k--', lw=1.2, alpha=0.7, label='Design AEV (gauge)')
ax.plot(wl_valid['date'], wl_valid['V_sar_Mm3'],
        'r-', lw=1.5, label='SAR hypsometry (gauge)')
ax.scatter(gee['date'], gee['volume_Mm3'],
           s=10, alpha=0.5, color='steelblue', zorder=3, label='GEE volume (SAR area)')
ax.set_xlabel('Date')
ax.set_ylabel('Volume (Mm³)')
ax.set_title('(c) Storage time series')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle('Ancipa — Schwatke MVP hypsometric reconstruction', fontsize=12)
fig.tight_layout()

out_path = OUT_DIR / 'ancipa_schwatke_mvp.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {out_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Save CSVs
# ---------------------------------------------------------------------------
pairs_df.to_csv(OUT_DIR / 'ancipa_hyps_pairs.csv', index=False)
wl_valid[['date', 'wl_m', 'V_design_Mm3', 'V_sar_Mm3']].to_csv(
    OUT_DIR / 'ancipa_volume_timeseries.csv', index=False)
print("CSVs saved.")
