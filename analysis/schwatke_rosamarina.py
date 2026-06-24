#!/usr/bin/env python3
"""
Schwatke-style MVP: Rosamarina reservoir (Sicily)

Three-way comparison:
  A) Design AEV   — original design-phase bathymetric survey (Rosamarina.xls)
  B) 2025 AEV     — official bathymetric re-survey (Regione Siciliana, Sept 2025)
  C) SAR fit      — empirical hypsometric curve from Sentinel-1 areas + R2 gauge

Gauge: Protezione Civile R2 (Nov 2022 – Feb 2025, combined)
SAR:   Sentinel-1 SVM water masks 2014–2025 (this study)
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

# ── Paths ─────────────────────────────────────────────────────────────────────
GEE  = Path("C:/Users/Unipa/Documents/GEE")
REPO = Path("C:/Users/Unipa/Documents/reservoirs_s1_svm")
OUT  = REPO / "analysis" / "schwatke_output"
OUT.mkdir(exist_ok=True)

WL_FILES = [
    GEE / "Results/Rosamarina Diga R2 - Water Level - 2024-11-11.csv",
    GEE / "Data/protCivile/Rosamarina Diga R2 - Water Level - 2025-02-09.csv",
]
AREA_FILE = REPO / "validation_data/morphometric_analysis/shoreline_compactness/area_rosamarina_2014-25.csv"
AEV_FILE  = GEE / "Data/Curve aree-volumi/Rosamarina.xls"
VOL_FILE  = REPO / "validation_data/statistics/volume_statistics/rosamarina_adib.csv"

MAX_DT = 5  # days

# ── 2025 bathymetric survey — digitised from PDF (Regione Siciliana, Sept 2025)
# Values at h = x.25 m are taken from column-2 first-row of each table page,
# which are unambiguous (confirmed continuous across pages). Units: m³ and m².
AEV_2025_RAW = [
    # (h_m,     V_m3,       A_m2)
    (121.00,         0,          0),   # effective bottom (all below silted up)
    (122.70,       450,      2_203),   # first detectable volume
    (123.00,     1_404,      3_764),
    (124.00,     5_561,      4_376),
    (130.00,  1_120_000,   380_000),   # approx – mid-range from xls centimetric
    (135.00,  3_900_000,   870_000),
    (140.00,  8_900_000, 1_370_000),
    (144.00, 13_886_029, 1_617_260),   # Pagina 25 col1 start
    (144.25, 14_291_964, 1_630_206),   # Pagina 25 col2
    (144.50, 14_701_133, 1_643_080),   # Pagina 25 col3
    (144.75, 15_113_598, 1_656_668),   # Pagina 25 col4
    (145.00, 15_529_434, 1_670_057),   # Pagina 26 col1 (matched)
    (145.25, 15_948_645, 1_683_684),
    (145.50, 16_371_298, 1_697_496),
    (145.75, 16_797_476, 1_712_319),
    (146.00, 17_227_562, 1_728_572),   # Pagina 26 col4 end (reliable)
    (146.25, 17_661_683, 1_744_235),
    (146.50, 18_099_683, 1_760_074),
    (146.75, 18_541_748, 1_776_614),
    (147.00, 18_989_237, 1_800_452),   # Pagina 27 col4 end = Pagina 28 col1 (matched)
    (147.25, 19_442_102, 1_822_862),
    (147.50, 19_901_042, 1_847_620),
    (147.75, 20_366_933, 1_874_248),
    (148.00, 20_838_750, 1_898_980),   # Pagina 28 col4 end
    (148.25, 21_316_223, 1_920_879),
    (148.50, 21_799_285, 1_943_863),
    (148.75, 22_288_116, 1_966_699),
    (149.00, 22_782_538, 1_988_503),   # Pagina 29 col4 end
    (149.25, 23_282_334, 2_009_818),
    (149.50, 23_787_437, 2_030_985),
    (149.75, 24_297_836, 2_052_251),
    (150.00, 24_813_610, 2_074_069),   # Pagina 30 col4 end
    (150.25, 25_334_928, 2_096_604),
    (150.50, 25_861_991, 2_120_146),
    (150.75, 26_395_138, 2_145_436),
    (151.00, 26_934_944, 2_173_556),   # Pagina 31 col4 end
    (151.25, 27_482_358, 2_207_475),
    (151.50, 28_039_203, 2_248_348),
    (151.75, 28_609_151, 2_289_450),
    (152.00, 29_185_280, 2_319_032),
    (152.25, 29_768_489, 2_346_423),
    (152.50, 30_358_371, 2_372_558),
    (152.75, 30_954_740, 2_398_294),
    (153.00, 31_557_693, 2_425_654),
    (153.25, 32_167_798, 2_455_906),
    (153.50, 32_785_698, 2_487_105),
    (153.75, 33_410_981, 2_514_098),
    (154.00, 34_042_401, 2_536_878),
    (154.25, 34_679_327, 2_558_460),
    (154.50, 35_321_606, 2_579_696),
    (154.75, 35_969_167, 2_600_711),
    (155.00, 36_624_733, 2_632_453),
    (155.25, 37_286_341, 2_660_439),
    (155.50, 37_956_478, 2_691_195),
    (155.75, 38_632_569, 2_717_381),
    (156.00, 39_315_424, 2_744_749),   # interpolated (Pagina 36 end)
    (156.25, 40_004_385, 2_769_495),
    (156.50, 40_700_049, 2_795_931),
    (156.75, 41_402_611, 2_823_951),
    (157.00, 42_112_046, 2_851_503),   # Pagina 37 col4 end = Pagina 38 col1 (matched)
    (157.25, 42_830_551, 2_888_201),
    (157.50, 43_556_059, 2_915_650),
    (157.75, 44_288_329, 2_942_104),
    (158.00, 45_027_039, 2_967_530),
    (158.25, 45_772_072, 2_992_692),
    (158.50, 46_533_734, 3_018_582),   # interpolated
    (158.75, 47_280_846, 3_042_227),
    (159.00, 48_044_454, 3_066_600),
    (159.25, 48_839_169, 3_178_529),
    (159.50, 49_630_237, 3_203_466),
    (159.75, 50_427_701, 3_228_838),
    (160.00, 51_231_670, 3_254_647),
    (160.25, 52_042_252, 3_280_891),
    (160.50, 52_859_556, 3_307_571),
    (160.75, 53_683_692, 3_334_686),
    (161.00, 54_514_768, 3_362_238),
    (161.25, 55_353_463, 3_408_450),
    (161.50, 56_213_245, 3_479_986),
    (161.75, 57_081_701, 3_513_141),
    (162.00, 57_958_536, 3_546_094),
    (162.25, 58_843_631, 3_578_702),
    (162.50, 59_737_000, 3_611_191),
    (162.75, 60_638_440, 3_642_461),
    (163.00, 61_547_768, 3_673_579),
    (163.25, 62_465_017, 3_705_242),
    (163.50, 63_390_633, 3_740_482),
    (163.75, 64_339_296, 3_792_136),
    (164.00, 65_287_149, 3_827_930),
]
aev25 = pd.DataFrame(AEV_2025_RAW, columns=['h', 'V_m3', 'A_m2']).sort_values('h').reset_index(drop=True)
aev25['V_Mm3'] = aev25['V_m3'] / 1e6
aev25['A_ha']  = aev25['A_m2'] / 1e4

V_of_h_2025 = interp1d(aev25['h'], aev25['V_Mm3'], kind='linear', bounds_error=False, fill_value='extrapolate')
A_of_h_2025 = interp1d(aev25['h'], aev25['A_m2'],  kind='linear', bounds_error=False, fill_value='extrapolate')

print(f"[2025 AEV] {len(aev25)} data points  h={aev25.h.min():.1f}–{aev25.h.max():.1f} m")
print(f"  V range: {aev25.V_Mm3.min():.2f}–{aev25.V_Mm3.max():.2f} Mm³")
print(f"  A range (operating): {aev25[aev25.h>=145].A_ha.min():.0f}–{aev25.A_ha.max():.0f} ha")

# ── 1. Water level: load, merge, daily average ────────────────────────────────
def _load_wl(path):
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = ['time', 'raw', 'variation', 'wl_m', 'selective']
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['wl_m'] = pd.to_numeric(df['wl_m'], errors='coerce')
    daily = df.dropna(subset=['time', 'wl_m']).set_index('time')['wl_m'].resample('D').mean()
    return daily.rename('wl_m')

parts = [_load_wl(p) for p in WL_FILES]
wl = pd.concat(parts).groupby(level=0).mean().reset_index()
wl.columns = ['date', 'wl_m']
wl = wl.dropna().sort_values('date').reset_index(drop=True)
print(f"\n[WL]  {len(wl)} daily records  {wl.date.min().date()} → {wl.date.max().date()}  "
      f"range {wl.wl_m.min():.2f}–{wl.wl_m.max():.2f} m")

# ── 2. SAR area time series ───────────────────────────────────────────────────
area = pd.read_csv(AREA_FILE)
area['date'] = pd.to_datetime(area['date'], errors='coerce')
area = area.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
area['A_m2'] = area['areaLago'] * 1e4
print(f"[SAR] {len(area)} observations  {area.date.min().date()} → {area.date.max().date()}  "
      f"range {area.areaLago.min():.0f}–{area.areaLago.max():.0f} ha")

# ── 3. Match SAR–gauge pairs ──────────────────────────────────────────────────
pairs = []
for _, row in area.iterrows():
    delta = (wl['date'] - row['date']).dt.days.abs()
    idx = delta.idxmin()
    if delta[idx] <= MAX_DT:
        pairs.append({'date': row['date'], 'area_ha': row['areaLago'],
                      'A_m2': row['A_m2'], 'wl_m': wl.loc[idx, 'wl_m'],
                      'dt_days': int(delta[idx])})
pairs_df = pd.DataFrame(pairs).dropna().reset_index(drop=True)
print(f"[PAIRS] {len(pairs_df)} matched (±{MAX_DT}d)  "
      f"A={pairs_df.area_ha.min():.0f}–{pairs_df.area_ha.max():.0f} ha  "
      f"WL={pairs_df.wl_m.min():.2f}–{pairs_df.wl_m.max():.2f} m")

# ── 4. Fit hypsometric model: A = a * (h − h0)^b ─────────────────────────────
def hyps(h, a, b, h0):
    return a * np.maximum(h - h0, 0.0) ** b

h_obs = pairs_df['wl_m'].values
A_obs = pairs_df['A_m2'].values
h0_upper = float(pairs_df['wl_m'].min()) - 0.01
popt, _ = curve_fit(hyps, h_obs, A_obs,
                    p0=[2e5, 1.5, h0_upper - 3],
                    bounds=([0, 0.2, 110.0], [1e9, 6.0, h0_upper]),
                    maxfev=20000)
a_fit, b_fit, h0_fit = popt
A_pred    = hyps(h_obs, *popt)
residuals = A_obs - A_pred
rmse_ha   = np.sqrt(np.mean(residuals**2)) / 1e4
r2        = 1.0 - np.sum(residuals**2) / np.sum((A_obs - A_obs.mean())**2)
print(f"\n[FIT] A = {a_fit:.3f} · (h − {h0_fit:.3f})^{b_fit:.4f}")
print(f"      R² = {r2:.4f}   RMSE = {rmse_ha:.1f} ha")

# ── 5. Load design AEV ────────────────────────────────────────────────────────
wb  = xlrd.open_workbook(str(AEV_FILE))
ws  = wb.sheet_by_name('Foglio1')
des_rows = []
for i in range(1, ws.nrows):
    r = ws.row_values(i)
    try:
        q = float(r[2]); a_ha = float(r[3]); v = float(r[5])
        if q > 50 and a_ha >= 0:
            des_rows.append({'h': q, 'A_ha': a_ha, 'V_Mm3': v})
    except (ValueError, TypeError):
        continue
aev_design = pd.DataFrame(des_rows).sort_values('h').reset_index(drop=True)
V_of_h_des = interp1d(aev_design['h'], aev_design['V_Mm3'], kind='linear',
                       bounds_error=False, fill_value='extrapolate')
A_of_h_des = interp1d(aev_design['h'], aev_design['A_ha'] * 1e4, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
print(f"\n[DESIGN AEV] {len(aev_design)} rows  h={aev_design.h.min():.0f}–{aev_design.h.max():.0f} m  "
      f"V={aev_design.V_Mm3.min():.2f}–{aev_design.V_Mm3.max():.1f} Mm³")

# ── 6. Sedimentation: design vs 2025 survey ──────────────────────────────────
print("\n=== Sedimentation: design AEV vs 2025 re-survey ===")
print(f"  {'Level':>6}  {'V_design':>10}  {'V_2025':>10}  {'dV':>10}  {'loss%':>8}  "
      f"{'A_design':>10}  {'A_2025':>10}")
for ht in [144, 146, 148, 150, 152, 154, 156, 158, 160, 162]:
    vd = float(V_of_h_des(ht))
    v2 = float(V_of_h_2025(ht))
    dv = v2 - vd
    pct = dv / vd * 100 if vd > 0 else 0
    ad = float(A_of_h_des(ht)) / 1e4
    a2 = float(A_of_h_2025(ht)) / 1e4
    print(f"  {ht:>6.0f}  {vd:>10.2f}  {v2:>10.2f}  {dv:>+10.2f}  {pct:>+7.1f}%  "
          f"  {ad:>8.1f}ha  {a2:>8.1f}ha")

# Total sedimentation volume at max gauge level
h_max_gauge = float(wl.wl_m.max())
V_sed = float(V_of_h_des(h_max_gauge)) - float(V_of_h_2025(h_max_gauge))
print(f"\n  Total sediment volume below h={h_max_gauge:.2f} m: {V_sed:.2f} Mm³")

# ── 7. SAR volume curve (integrated, anchored at h_ref) ──────────────────────
h_grid = np.arange(h0_fit + 0.01, aev_design['h'].max() + 1.0, 0.01)
A_grid = hyps(h_grid, *popt)
dh     = np.diff(h_grid)
V_int  = np.concatenate([[0.0], np.cumsum(0.5 * (A_grid[:-1] + A_grid[1:]) * dh)])

h_ref     = float(pairs_df['wl_m'].min())
V_ref_25  = float(V_of_h_2025(h_ref)) * 1e6   # anchor to 2025 survey (ground truth)
idx_ref   = int(np.searchsorted(h_grid, h_ref))
V_int    += V_ref_25 - V_int[idx_ref]
V_of_h_sar = interp1d(h_grid, V_int / 1e6, kind='linear', bounds_error=False, fill_value=np.nan)

# ── 8. Level-shift analysis: same area → different h ─────────────────────────
A_test = np.linspace(pairs_df.area_ha.min(), pairs_df.area_ha.max(), 200) * 1e4

h_of_A_des = interp1d(A_of_h_des(aev_design['h'].values), aev_design['h'].values,
                       kind='linear', bounds_error=False, fill_value=np.nan)
h_of_A_25  = interp1d(A_of_h_2025(aev25['h'].values), aev25['h'].values,
                       kind='linear', bounds_error=False, fill_value=np.nan)

h_sar_pred = h0_fit + (A_test / a_fit) ** (1.0 / b_fit)
h_des_pred = h_of_A_des(A_test)
h_25_pred  = h_of_A_25(A_test)

shift_sar_des = h_sar_pred - h_des_pred
shift_sar_25  = h_sar_pred - h_25_pred
shift_25_des  = h_25_pred  - h_des_pred

print(f"\n[SHIFT] SAR − design:   mean={np.nanmean(shift_sar_des):+.2f} m  std={np.nanstd(shift_sar_des):.2f} m")
print(f"[SHIFT] SAR − 2025:    mean={np.nanmean(shift_sar_25):+.2f} m  std={np.nanstd(shift_sar_25):.2f} m")
print(f"[SHIFT] 2025 − design:  mean={np.nanmean(shift_25_des):+.2f} m  std={np.nanstd(shift_25_des):.2f} m")

# ── 9. Official monthly volumes ───────────────────────────────────────────────
vol = pd.read_csv(VOL_FILE)
vol['date'] = pd.to_datetime(vol['date'])
vol = vol.dropna(subset=['date', 'volume_adib']).sort_values('date').reset_index(drop=True)
print(f"\n[VOL] official AdB: {len(vol)} monthly  "
      f"range {vol.volume_adib.min():.1f}–{vol.volume_adib.max():.1f} Mm³")

# ── 10. Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

# Panel A: hypsometric scatter + 3 curves
ax_A = fig.add_subplot(gs[0, :2])
sc = ax_A.scatter(pairs_df['area_ha'], pairs_df['wl_m'],
                  c=pairs_df['date'].astype(np.int64) / 1e18,
                  cmap='plasma', s=35, alpha=0.85, zorder=4, label='SAR–gauge pairs')
cb = plt.colorbar(sc, ax=ax_A)
years = pd.date_range(pairs_df['date'].min(), pairs_df['date'].max(), freq='YS')
cb.set_ticks([d.value / 1e18 for d in years])
cb.set_ticklabels([str(d.year) for d in years], fontsize=7)
cb.set_label('Year', fontsize=8)

h_curve = np.linspace(max(h0_fit + 0.5, 138), aev_design['h'].max(), 300)
ax_A.plot(hyps(h_curve, *popt) / 1e4, h_curve, 'b-', lw=2.0, label=f'SAR fit (R²={r2:.3f})')
ax_A.plot(aev_design['A_ha'],                aev_design['h'],  'r--', lw=1.8, label='Design AEV')
ax_A.plot(aev25['A_ha'],                     aev25['h'],       'g-',  lw=1.8, label='2025 re-survey')

ax_A.set_xlim(0, aev_design['A_ha'].max() * 1.05)
ax_A.set_ylim(138, aev_design['h'].max())
ax_A.set_xlabel('Water surface area (ha)')
ax_A.set_ylabel('Water level (m a.s.l.)')
ax_A.set_title('Rosamarina — Hypsometric curves')
ax_A.legend(fontsize=9)
ax_A.grid(True, alpha=0.3)

# Panel B: level shift
ax_B = fig.add_subplot(gs[0, 2])
A_ha_test = A_test / 1e4
ax_B.plot(A_ha_test, shift_sar_des,  'b-',  lw=1.5, label='SAR − design')
ax_B.plot(A_ha_test, shift_25_des,   'g--', lw=1.5, label='2025 − design')
ax_B.plot(A_ha_test, shift_sar_25,   'b--', lw=1.2, label='SAR − 2025',  alpha=0.7)
ax_B.axhline(0, color='gray', lw=0.8)
ax_B.set_xlabel('Water surface area (ha)')
ax_B.set_ylabel('Δh (m) — same area')
ax_B.set_title('Level shift (same area)')
ax_B.legend(fontsize=8)
ax_B.grid(True, alpha=0.3)

# Panel C: volume time series
ax_C = fig.add_subplot(gs[1, :])
mask = wl['wl_m'].between(aev_design['h'].min(), aev_design['h'].max())
wl_plot = wl.copy()
wl_plot.loc[mask, 'V_design'] = V_of_h_des(wl_plot.loc[mask, 'wl_m'])
wl_plot.loc[mask, 'V_2025']   = V_of_h_2025(wl_plot.loc[mask, 'wl_m'])
wl_plot.loc[mask, 'V_sar']    = V_of_h_sar(wl_plot.loc[mask, 'wl_m'])

ax_C.plot(wl_plot.loc[mask, 'date'], wl_plot.loc[mask, 'V_design'],
          'r-', lw=0.9, alpha=0.7, label='Gauge + design AEV')
ax_C.plot(wl_plot.loc[mask, 'date'], wl_plot.loc[mask, 'V_2025'],
          'g-', lw=1.3, alpha=0.85, label='Gauge + 2025 AEV')
ax_C.plot(wl_plot.loc[mask, 'date'], wl_plot.loc[mask, 'V_sar'],
          'b-', lw=0.9, alpha=0.7, label='Gauge + SAR hypsometry')
ax_C.scatter(vol['date'], vol['volume_adib'], s=20, c='black',
             zorder=4, alpha=0.9, label='Official AdB (monthly)')
ax_C.set_xlabel('Date')
ax_C.set_ylabel('Storage volume (Mm³)')
ax_C.set_title('Rosamarina — Storage time series')
ax_C.legend(fontsize=9)
ax_C.grid(True, alpha=0.3)

plt.suptitle('Schwatke MVP — Rosamarina Reservoir (Sicily)\n'
             'Design AEV vs 2025 Bathymetric Re-survey vs SAR-derived Hypsometry',
             fontsize=12, fontweight='bold')

fig_path = OUT / 'rosamarina_schwatke_mvp.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n[DONE] Figure → {fig_path}")

# ── 11. Validation against AdB ────────────────────────────────────────────────
wl_monthly = wl_plot.set_index('date').resample('MS').mean().reset_index()
merged = pd.merge(wl_monthly, vol, on='date', how='inner')
if len(merged):
    print(f"\n=== Validation vs AdB ({len(merged)} monthly pairs) ===")
    for col, label in [('V_design','Design AEV'), ('V_2025','2025 AEV'), ('V_sar','SAR hypsometry')]:
        sub = merged[['volume_adib', col]].dropna()
        if len(sub) < 3:
            continue
        obs = sub['volume_adib'].values
        sim = sub[col].values
        rmse = np.sqrt(np.mean((sim - obs)**2))
        bias = (sim - obs).mean()
        r2v  = 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)
        print(f"  {label:20s}: R²={r2v:.3f}  RMSE={rmse:.2f} Mm³  bias={bias:+.2f} Mm³  n={len(sub)}")

# Save sedimentation table
sed_rows = []
for ht in np.arange(144, 164, 1):
    vd = float(V_of_h_des(ht)); v2 = float(V_of_h_2025(ht))
    sed_rows.append({'h_m': ht, 'V_design_Mm3': vd, 'V_2025_Mm3': v2,
                     'dV_Mm3': v2 - vd, 'loss_pct': (v2 - vd) / vd * 100})
pd.DataFrame(sed_rows).to_csv(OUT / 'rosamarina_sedimentation.csv', index=False)
print("[DONE] Sedimentation table → analysis/schwatke_output/rosamarina_sedimentation.csv")
