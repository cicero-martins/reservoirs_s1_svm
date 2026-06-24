#!/usr/bin/env python3
"""
Schwatke-style MVP: empirical hypsometric reconstruction for Poma reservoir (Sicily)

Methodology (after Schwatke et al. 2020, doi: 10.3390/rs12121901):
  1. Daily-average in-situ water level from Protezione Civile R2 gauges
  2. SAR surface area time series from this study (Sentinel-1/SVM, 2014-2025)
  3. Match SAR-gauge pairs within ±MAX_DT days → empirical h-A scatter
  4. Fit power-law hypsometric model: A = a * (h - h0)^b
  5. Derive volume time series by numerical integration of fitted curve
  6. Validate against official monthly storage records (AdB Sicilia)
  7. Compare fitted hypsometry with design-phase AEV curve (sedimentation proxy)

Key advantage over Schwatke (2020): in-situ gauge replaces satellite altimetry
→ sub-daily WL accuracy, tight temporal matching with SAR acquisitions.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import xlrd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
GEE = Path("C:/Users/Unipa/Documents/GEE")
REPO = Path("C:/Users/Unipa/Documents/reservoirs_s1_svm")
OUT  = REPO / "analysis" / "schwatke_output"
OUT.mkdir(exist_ok=True)

WL_FILES = [
    GEE / "Results/Poma Diga R2 - Water Level - 2024-11-11.csv",
    GEE / "Data/protCivile/Poma Diga R2 - Water Level - 2025-03-07.csv",
]
AREA_FILE = REPO / "validation_data/morphometric_analysis/shoreline_compactness/area_poma_2014-25.csv"
AEV_FILE  = GEE / "Data/Curve aree-volumi/Poma.xls"
VOL_FILE  = REPO / "validation_data/statistics/volume_statistics/poma_adib.csv"

MAX_DT = 5  # days: maximum gap allowed between SAR and gauge observations

# ── 1. Water level: load, clean, merge, daily average ────────────────────────
def _load_wl(path):
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = ['time', 'raw', 'variation', 'wl_m', 'selective']
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'wl_m'])
    df['wl_m'] = pd.to_numeric(df['wl_m'], errors='coerce')
    daily = df.set_index('time')['wl_m'].resample('D').mean()
    return daily.rename('wl_m')

parts = [_load_wl(p) for p in WL_FILES]
wl_daily = pd.concat(parts).groupby(level=0).mean().reset_index()
wl_daily.columns = ['date', 'wl_m']
wl_daily = wl_daily.dropna().sort_values('date').reset_index(drop=True)

print(f"[WL]  {len(wl_daily)} daily records  "
      f"{wl_daily.date.min().date()} → {wl_daily.date.max().date()}  "
      f"range {wl_daily.wl_m.min():.2f}–{wl_daily.wl_m.max():.2f} m")

# ── 2. SAR area time series ───────────────────────────────────────────────────
area = pd.read_csv(AREA_FILE)
area['date'] = pd.to_datetime(area['date'], errors='coerce')
area = area.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
# area in ha → convert to m² for internal calculations
area['A_m2'] = area['areaLago'] * 1e4

print(f"[SAR] {len(area)} observations  "
      f"{area.date.min().date()} → {area.date.max().date()}  "
      f"range {area.areaLago.min():.1f}–{area.areaLago.max():.1f} ha")

# ── 3. Match SAR–gauge pairs ──────────────────────────────────────────────────
pairs = []
for _, row in area.iterrows():
    delta = (wl_daily['date'] - row['date']).dt.days.abs()
    idx = delta.idxmin()
    if delta[idx] <= MAX_DT:
        pairs.append({
            'date':    row['date'],
            'area_ha': row['areaLago'],
            'A_m2':    row['A_m2'],
            'wl_m':    wl_daily.loc[idx, 'wl_m'],
            'dt_days': int(delta[idx]),
        })
pairs_df = pd.DataFrame(pairs).dropna().reset_index(drop=True)

print(f"[PAIRS] {len(pairs_df)} matched (±{MAX_DT} days)  "
      f"A={pairs_df.area_ha.min():.0f}–{pairs_df.area_ha.max():.0f} ha  "
      f"WL={pairs_df.wl_m.min():.2f}–{pairs_df.wl_m.max():.2f} m")

# ── 4. Fit hypsometric model: A = a * (h − h0)^b ─────────────────────────────
def hyps_model(h, a, b, h0):
    return a * np.maximum(h - h0, 0.0) ** b

h_obs = pairs_df['wl_m'].values
A_obs = pairs_df['A_m2'].values

h0_upper = float(pairs_df['wl_m'].min()) - 0.01
p0      = [5e5, 1.5, h0_upper - 2.0]
bounds  = ([0, 0.2, 160.0], [1e9, 6.0, h0_upper])

popt, pcov = curve_fit(hyps_model, h_obs, A_obs, p0=p0, bounds=bounds, maxfev=20000)
a_fit, b_fit, h0_fit = popt
perr = np.sqrt(np.diag(pcov))

A_pred    = hyps_model(h_obs, *popt)
residuals = A_obs - A_pred
rmse_ha   = np.sqrt(np.mean(residuals**2)) / 1e4
r2        = 1.0 - np.sum(residuals**2) / np.sum((A_obs - A_obs.mean())**2)

print(f"\n[FIT] A = {a_fit:.3f} · (h − {h0_fit:.3f})^{b_fit:.4f}")
print(f"      R² = {r2:.4f}   RMSE = {rmse_ha:.1f} ha")

# ── 5. Load design-phase AEV ──────────────────────────────────────────────────
wb  = xlrd.open_workbook(str(AEV_FILE))
ws  = wb.sheet_by_name('Foglio1')
aev_rows = []
for i in range(1, ws.nrows):
    r = ws.row_values(i)
    try:
        q = float(r[2]); a_ha = float(r[4]); v_mm3 = float(r[5])
        if q > 100 and a_ha >= 0:
            aev_rows.append({'h': q, 'A_ha': a_ha, 'A_m2': a_ha * 1e4, 'V_Mm3': v_mm3})
    except (ValueError, TypeError):
        continue
aev = pd.DataFrame(aev_rows).sort_values('h').reset_index(drop=True)

# Interpolators over the AEV
h_aev   = aev['h'].values
A_of_h  = interp1d(h_aev, aev['A_m2'].values,  kind='linear', bounds_error=False, fill_value='extrapolate')
V_of_h  = interp1d(h_aev, aev['V_Mm3'].values, kind='linear', bounds_error=False, fill_value='extrapolate')

print(f"\n[AEV] design-phase: {len(aev)} rows  "
      f"h={aev.h.min():.0f}–{aev.h.max():.1f} m  "
      f"V={aev.V_Mm3.min():.1f}–{aev.V_Mm3.max():.1f} Mm³")

# ── 6. Volume time series from SAR-fitted hypsometry ─────────────────────────
# Integrate the fitted A(h) curve from h0 upward (trapezoidal), then anchor
# to the design-phase AEV at the minimum observed gauge level.
h_grid   = np.arange(h0_fit + 0.01, aev['h'].max() + 1.0, 0.01)
A_grid   = hyps_model(h_grid, *popt)
dh       = np.diff(h_grid)
V_int    = np.concatenate([[0.0], np.cumsum(0.5 * (A_grid[:-1] + A_grid[1:]) * dh)])

# Anchor: match to AEV volume at the reference level (lowest paired WL)
h_ref    = float(pairs_df['wl_m'].min())
V_ref_design = float(V_of_h(h_ref)) * 1e6   # Mm³ → m³
idx_ref  = int(np.searchsorted(h_grid, h_ref))
V_offset = V_ref_design - V_int[idx_ref]
V_int   += V_offset                           # m³

V_of_h_sar = interp1d(h_grid, V_int / 1e6, kind='linear',   # → Mm³
                       bounds_error=False, fill_value=np.nan)

# Apply to the complete gauge record
wl_plot = wl_daily.copy()
wl_plot['V_design_Mm3'] = V_of_h(wl_plot['wl_m'])
wl_plot['V_sar_Mm3']    = V_of_h_sar(wl_plot['wl_m'])

# ── 7. Official monthly volumes (AdB) ────────────────────────────────────────
vol = pd.read_csv(VOL_FILE)
vol['date'] = pd.to_datetime(vol['date'], errors='coerce')
vol = vol.dropna(subset=['date', 'volume_adib']).sort_values('date').reset_index(drop=True)

print(f"\n[VOL] official AdB: {len(vol)} monthly records  "
      f"{vol.date.min().date()} → {vol.date.max().date()}  "
      f"{vol.volume_adib.min():.1f}–{vol.volume_adib.max():.1f} Mm³")

# ── 8. Hypsometric shift: SAR-fit vs design AEV ──────────────────────────────
# For each observed area value, compare water level implied by each curve
A_ha_test = np.linspace(pairs_df.area_ha.min(), pairs_df.area_ha.max(), 200)
A_m2_test = A_ha_test * 1e4

# SAR-fit: solve h from A = a*(h-h0)^b → h = h0 + (A/a)^(1/b)
h_sar_pred  = h0_fit + (A_m2_test / a_fit) ** (1.0 / b_fit)

# AEV inverse: interpolate h from A (using the design table)
h_of_A_design = interp1d(aev['A_m2'].values, aev['h'].values, kind='linear',
                          bounds_error=False, fill_value=np.nan)
h_design_pred = h_of_A_design(A_m2_test)

dh_shift = h_sar_pred - h_design_pred
print(f"\n[SHIFT] SAR-fit − design AEV water level at matched areas:")
print(f"  mean = {np.nanmean(dh_shift):+.2f} m   "
      f"std = {np.nanstd(dh_shift):.2f} m   "
      f"range {np.nanmin(dh_shift):+.2f} to {np.nanmax(dh_shift):+.2f} m")
print("  Positive = same area found at HIGHER level → reservoir has LESS volume")
print("  (consistent with sedimentation reducing bathymetric capacity)")

# ── 9. Figures ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# ── Panel A: hypsometric scatter + fitted curve + design AEV ─────────────────
ax_A = fig.add_subplot(gs[0, :2])
sc = ax_A.scatter(pairs_df['area_ha'], pairs_df['wl_m'],
                  c=pairs_df['date'].astype(np.int64) / 1e18,
                  cmap='plasma', s=30, alpha=0.85, zorder=3, label='SAR–gauge pairs')
plt.colorbar(sc, ax=ax_A, label='')

h_curve = np.linspace(h0_fit + 0.01, aev['h'].max(), 300)
A_curve = hyps_model(h_curve, *popt) / 1e4
ax_A.plot(A_curve, h_curve, 'b-', lw=2.0, label=f'SAR fit (R²={r2:.3f})')
ax_A.plot(aev['A_ha'], aev['h'], 'r--', lw=2.0, label='Design AEV')

ax_A.set_xlabel('Water surface area (ha)')
ax_A.set_ylabel('Water level (m a.s.l.)')
ax_A.set_title('Poma — Hypsometric curve: SAR fit vs design AEV')
ax_A.legend(fontsize=9)
ax_A.grid(True, alpha=0.3)

# ── Panel B: level-shift between curves ──────────────────────────────────────
ax_B = fig.add_subplot(gs[0, 2])
ax_B.plot(A_ha_test, dh_shift, 'k-', lw=1.5)
ax_B.axhline(0, color='gray', lw=0.8, ls='--')
ax_B.fill_between(A_ha_test, 0, dh_shift,
                  where=dh_shift > 0, alpha=0.3, color='firebrick', label='SAR level > design')
ax_B.fill_between(A_ha_test, 0, dh_shift,
                  where=dh_shift < 0, alpha=0.3, color='steelblue', label='SAR level < design')
ax_B.set_xlabel('Water surface area (ha)')
ax_B.set_ylabel('Δh SAR − design (m)')
ax_B.set_title('Level shift (same area)')
ax_B.legend(fontsize=8)
ax_B.grid(True, alpha=0.3)

# ── Panel C: volume time series ───────────────────────────────────────────────
ax_C = fig.add_subplot(gs[1, :])
mask = wl_plot['wl_m'].between(aev['h'].min(), aev['h'].max())
ax_C.plot(wl_plot.loc[mask, 'date'], wl_plot.loc[mask, 'V_design_Mm3'],
          'r-', lw=0.9, alpha=0.75, label='Gauge + design AEV')
ax_C.plot(wl_plot.loc[mask, 'date'], wl_plot.loc[mask, 'V_sar_Mm3'],
          'b-', lw=0.9, alpha=0.75, label='Gauge + SAR hypsometry')
ax_C.scatter(vol['date'], vol['volume_adib'], s=18, c='black',
             zorder=4, label='Official AdB (monthly)', alpha=0.9)
ax_C.set_xlabel('Date')
ax_C.set_ylabel('Storage volume (Mm³)')
ax_C.set_title('Poma — Storage time series')
ax_C.legend(fontsize=9)
ax_C.grid(True, alpha=0.3)

# colour-bar date label
cbar_ax = fig.axes[1]   # the colorbar axes added by plt.colorbar
years = pd.date_range(pairs_df['date'].min(), pairs_df['date'].max(), freq='YS')
ticks = [d.value / 1e18 for d in years]
cbar_ax.set_yticks(ticks)
cbar_ax.set_yticklabels([str(d.year) for d in years], fontsize=7)
cbar_ax.set_ylabel('Year', fontsize=8)

plt.suptitle('Schwatke-style MVP — Poma Reservoir (Sicily)', fontsize=13, fontweight='bold')
fig_path = OUT / 'poma_schwatke_mvp.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n[DONE] Figure saved → {fig_path}")

# ── 10. Save results tables ───────────────────────────────────────────────────
pairs_df.to_csv(OUT / 'poma_hyps_pairs.csv', index=False)

wl_out = wl_plot.loc[mask, ['date','wl_m','V_design_Mm3','V_sar_Mm3']].copy()
wl_out.to_csv(OUT / 'poma_volume_timeseries.csv', index=False)

shift_out = pd.DataFrame({'area_ha': A_ha_test, 'h_sar_m': h_sar_pred,
                           'h_design_m': h_design_pred, 'dh_m': dh_shift})
shift_out.to_csv(OUT / 'poma_hyps_shift.csv', index=False)

print("[DONE] CSV tables saved to analysis/schwatke_output/")
