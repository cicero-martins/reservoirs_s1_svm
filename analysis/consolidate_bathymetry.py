"""
consolidate_bathymetry.py  (E1/E5 — Paper 2 core-5 consolidation)

One standardized table + summary figure across the 5 core Sicilian reservoirs
(Poma, Rosamarina, Pozzillo, Ancipa, Garcia) from their Period-B satellite DEMs.

Key distinction (see below): the SAR DEM only reconstructs the drawdown-EXPOSED
band [floor_B, max_B], so capacity change is measured TWO ways:
  • BAND-relative (rel. to the DEM floor)  → the SAR-observable change, and the
    fair apples-to-apples comparison against independent ground truth.
  • TOTAL (absolute, full design curve)    → the whole capacity loss, INCLUDING
    the deep zone below floor_B that the SAR cannot see. Available only where a
    full updated survey curve exists (Poma, Rosamarina).
The gap between them is the deep-zone sedimentation the SAR misses → the method
gives a validated LOWER BOUND on total capacity loss.

Sign convention: change_pct < 0  ==>  less storage than the design.

Independent references (from the E4 outputs): Garcia echosounder survey,
Poma updated centimetric curve, Rosamarina 2025 survey curve.

Outputs: analysis/schwatke_output/ bathymetry_consolidated.csv, bathymetry_change_summary.png
"""

import pathlib, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from scipy.interpolate import interp1d

sys.stdout.reconfigure(encoding='utf-8')

REPO = pathlib.Path('.')
OUT  = REPO / 'analysis' / 'schwatke_output'
CURVE_DIR = 'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi'
PIXEL_HA = 0.01

# design=(quota_col, area_col, vol_col, area_unit); ap from app SICILY_AP_TABLE (JRC poly);
# updated = independent full survey curve source (None if unavailable).
RES = {
    'Ancipa':     dict(design=(2, 3, 4, 'km2'), ap=90.5,  updated=None),
    'Garcia':     dict(design=(2, 3, 4, 'km2'), ap=167.7, updated='garcia_2026'),
    'Rosamarina': dict(design=(2, 3, 5, 'ha'),  ap=187.4, updated='rosamarina_2025'),
    'Poma':       dict(design=(2, 4, 5, 'ha'),  ap=190.1, updated='poma_new'),
    'Pozzillo':   dict(design=(2, 4, 5, 'ha'),  ap=240.5, updated=None),
    # --- Fase 3 extended set (2026-07): design cols read straight off each
    # reservoir's own design-curve xls (verified against schwatke_bathymetry_3d.py's
    # boletin_cfg, same files); 'updated' points at a real official survey curve
    # (NewCurves/, the same files schwatke_extended.py validates the scalar
    # hypsometry against) that -- unlike Poma/Rosamarina's -- carries its own
    # volume column directly (no area->volume integration needed).
    'Arancio':    dict(design=(2, 3, 4, 'km2'), ap=182.2, updated='arancio_2022'),
    'Castello':   dict(design=(2, 3, 4, 'km2'), ap=126.7, updated='castello_updated'),
    'Olivo':      dict(design=(2, 4, 5, 'ha'),  ap=50.7,  updated='olivo_2021'),
    'Nicoletti':  dict(design=(2, 4, 5, 'ha'),  ap=119.7, updated='nicoletti_updated'),
}


def aev_from_dem(elev, mask, levels):
    areas = np.array([np.sum((elev < h) & mask) * PIXEL_HA for h in levels])
    vols = np.zeros_like(areas)
    for i in range(1, len(levels)):
        vols[i] = vols[i-1] + (areas[i] + areas[i-1]) / 2 * (levels[i] - levels[i-1]) * 0.01
    return areas, vols


def vol_exact(elev, mask, floor, top):
    """Volume above floor (Mm3), exact per-pixel water-column sum -- not an
    approximation. Found 2026-07-24: aev_from_dem's 0.5 m-step trapezoidal
    integration systematically UNDERESTIMATES volume by 1.7-11.7% depending on
    the reservoir (worst for the sparsest/steppiest DEMs -- Nicoletti, Pozzillo,
    Arancio, Garcia), because a fixed 0.5 m level grid poorly resolves the
    area(h) step function these ~10-mask level-slice DEMs produce, especially
    after the windowed drought-refill masks made the steps coarser. This exact
    sum has no such discretization error: by the layer-cake identity,
    integral_floor^top A(h) dh = sum_pixels pixelArea*(top-elev_pixel) exactly."""
    pixel_m2 = PIXEL_HA * 1e4  # 0.01 ha = 100 m2
    col = np.clip(top - elev, 0.0, top - floor)
    return float(np.sum(col[mask]) * pixel_m2 / 1e6)  # m -> m3 -> Mm3


def load_design_vol(name, cfg):
    qc, ac, vc, unit = cfg['design']
    df = pd.read_excel(f'{CURVE_DIR}/{name}.xls', sheet_name=0, header=None, engine='xlrd')[[qc, vc]]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df.columns = ['quota', 'vol_Mm3']
    df = df[df.quota > 80].sort_values('quota').reset_index(drop=True)
    return interp1d(df.quota, df.vol_Mm3, bounds_error=False, fill_value='extrapolate')


NEW_CURVES = 'C:/Users/Unipa/Documents/GEE/Data/NewCurves/'

# Fase 3 extended reservoirs: each official survey curve (from the same NewCurves/
# files schwatke_extended.py validates the scalar hypsometry against) carries its
# own volume column directly -- (pattern, sheet, quota_col, vol_col) per block, no
# area->volume integration needed. Olivo has 4 duplicate quota/vol/area blocks
# tiled across the sheet; concatenated below.
EXT_CURVE_SPEC = {
    'arancio_2022':     ('ARANCIO*', 'BASE', [(0, 1)]),
    'castello_updated': ('CASTELLO*', 'Quota_V_S', [(5, 6)]),
    'nicoletti_updated':('NICOLETTI*', 'Dati Aree-Volumi', [(0, 1)]),
    'olivo_2021':       ('OLIVO*', 'Tabella centimetrica 2021', [(1, 2), (5, 6), (9, 10), (13, 14)]),
}


def load_updated_vol(kind):
    """Return interp h->absolute volume (Mm3) for the updated survey curve, or None."""
    if kind == 'poma_new':
        f = sorted(glob.glob(f'C:/Users/Unipa/Documents/GEE/Data/NewCurves/POMA*.XLS'), key=len)[0]
        u = pd.read_excel(f, sheet_name='foglio1', header=None, engine='xlrd')[[0, 1]]
        u.columns = ['quota', 'vol_m3']
        u = u.apply(pd.to_numeric, errors='coerce').dropna().sort_values('quota')
        return interp1d(u.quota, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate')
    if kind == 'rosamarina_2025':
        u = pd.read_csv(REPO / 'validation_data' / 'updated_curves' / 'rosamarina_2025.csv')
        return interp1d(u.quota_m, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate')
    if kind == 'garcia_2026':
        # Curated official quota-area-volume table, NOT our own re-gridded
        # echosounder raster -- see tool/bathymetry.py's updated_curve() for why
        # (the raster's shore/terrain interpolation undercounts area/volume above
        # the survey's own waterline, worse at higher elevations).
        u = pd.read_csv(REPO / 'validation_data' / 'updated_curves' / 'garcia_2026.csv')
        return interp1d(u.quota_m, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate')
    if kind in EXT_CURVE_SPEC:
        pat, sheet, blocks = EXT_CURVE_SPEC[kind]
        f = [h for h in glob.glob(NEW_CURVES + pat) if h.lower().endswith(('.xls', '.xlsx'))][0]
        raw = pd.read_excel(f, sheet_name=sheet, header=None, engine='openpyxl')
        parts = []
        for qc, vc in blocks:
            p = raw[[qc, vc]].apply(pd.to_numeric, errors='coerce').dropna()
            p.columns = ['quota', 'vol_m3']
            parts.append(p)
        u = pd.concat(parts)
        u = u[(u.quota > 50) & (u.quota < 1000)].sort_values('quota')
        return interp1d(u.quota, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate')
    return None


def dem_capacity_metrics(dem_path, name, cfg):
    """Band-observable/capacity-change metrics for one DEM variant. Returns
    None if the file is missing; otherwise a dict with floor/top/band metrics
    plus (where an independent curve exists) the truth-band/total comparison.
    Factored out of the per-reservoir loop so it can run once for the
    production (gauge+SWOT-fallback) DEM and once for the FRS (SWOT-only) DEM
    with identical logic -- Fase C, Paper 2 sec:res_change restructuring."""
    if not dem_path.exists():
        return None
    with rasterio.open(dem_path) as s:
        dem = s.read(1).astype(np.float64)
    mask = ~np.isnan(dem)
    floor, top = float(np.nanmin(dem[mask])), float(np.nanmax(dem[mask]))

    des_vol = load_design_vol(name, cfg)
    vdes_floor_abs = float(des_vol(floor))
    vdes_abs_max = float(des_vol(top))
    vdes_rel_max = vdes_abs_max - vdes_floor_abs
    vdem_rel_max = vol_exact(dem, mask, floor, top)

    # Observability envelope: what fraction of the design curve's TOTAL volume (down to
    # its own lowest tabulated point, which for every core reservoir is at or near true
    # zero storage, i.e. the pre-impoundment valley floor) falls inside the SAR-observable
    # band [floor, top] versus the invisible deep zone below floor. Ancipa's curve bottoms
    # out at a small residual area (17.8 ha, not exactly 0) rather than a true zero point;
    # its lowest tabulated quota also sits ABOVE the SAR-observed floor after the corrected
    # AOI/mask fix, so the raw ratio can exceed 100% (extrapolation below the curve's valid
    # domain, not a real deep zone). Clip to [0, 100] and flag it.
    band_observable_pct_raw = 100 * vdes_rel_max / vdes_abs_max if vdes_abs_max else np.nan
    extrapolated_below_curve = bool(vdes_abs_max) and band_observable_pct_raw > 100
    band_observable_pct = min(band_observable_pct_raw, 100.0) if vdes_abs_max else np.nan
    deepzone_pct = 100 - band_observable_pct if vdes_abs_max else np.nan

    # SAR-detected change vs design, BAND-relative (the SAR-observable capacity change)
    sar_band = (vdem_rel_max - vdes_rel_max) / vdes_rel_max * 100 if vdes_rel_max else np.nan

    truth_band = np.nan; truth_total = np.nan; ref_lbl = '—'; field_rmse = np.nan
    if cfg['updated'] in ('poma_new', 'rosamarina_2025', 'garcia_2026') or cfg['updated'] in EXT_CURVE_SPEC:
        ref_lbl = {'rosamarina_2025': '2025 survey', 'garcia_2026': 'echosounder'}.get(cfg['updated'], 'updated curve')
        upd_vol = load_updated_vol(cfg['updated'])
        vupd_rel_max = float(upd_vol(top) - upd_vol(floor))
        vupd_abs_max = float(upd_vol(top))
        truth_band  = (vupd_rel_max - vdes_rel_max) / vdes_rel_max * 100 if vdes_rel_max else np.nan
        truth_total = (vupd_abs_max - vdes_abs_max) / vdes_abs_max * 100 if vdes_abs_max else np.nan
    # Garcia's field RMSE (pixel-level DEM-vs-survey-raster comparison, a
    # genuinely 2D need the 1D curated curve above can't replace) is independent
    # of which curve is used as the AEV truth reference.
    if name == 'Garcia':
        st = OUT / 'garcia_survey' / 'garcia_comparison_stats.csv'
        if st.exists():
            s2 = pd.read_csv(st)
            r = s2.loc[s2['analysis'] == 'shallow_pixel', 'rmse_m']
            if len(r): field_rmse = float(r.iloc[0])

    return {
        'floor_m': round(floor, 2), 'max_m': round(top, 2), 'obs_range_m': round(top - floor, 1),
        'vol_dem_rel_Mm3': round(vdem_rel_max, 2), 'vol_design_rel_Mm3': round(vdes_rel_max, 2),
        'band_observable_pct': round(band_observable_pct, 1), 'deepzone_pct': round(deepzone_pct, 1),
        'band_pct_capped': extrapolated_below_curve,
        'sar_change_band_pct':   round(sar_band, 1),
        'truth_change_band_pct': None if np.isnan(truth_band)  else round(truth_band, 1),
        'truth_change_total_pct':None if np.isnan(truth_total) else round(truth_total, 1),
        'indep_ref': ref_lbl,
        'field_rmse_m': None if np.isnan(field_rmse) else round(field_rmse, 2),
    }


rows = []
for name, cfg in RES.items():
    prod = dem_capacity_metrics(OUT / f'dem_{name}_B.tif', name, cfg)
    if prod is None:
        print(f'  ! missing dem_{name}_B.tif, skipping'); continue
    frs = dem_capacity_metrics(OUT / f'dem_{name}_B_swotonly.tif', name, cfg)

    row = {'reservoir': name, 'ap_m': cfg['ap'], **prod}
    if frs is not None:
        row.update({f'{k}_frs': v for k, v in frs.items()})
    rows.append(row)

df = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df.to_csv(OUT / 'bathymetry_consolidated.csv', index=False)
print(df.to_string(index=False))
print(f'\nSaved: {OUT / "bathymetry_consolidated.csv"}')

# ── Summary figure ────────────────────────────────────────────────────────────
# Primary: SAR (band) vs independent truth (band) — the fair validation.
# Overlay: total capacity loss (full curve) as an open marker where available,
# showing the extra deep-zone loss the SAR cannot observe.
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(df)); w = 0.38
ax.axhline(0, color='gray', lw=0.8)
ax.bar(x - w/2, df.sar_change_band_pct, w, color='C0', label='SAR DEM vs design (observable band)')
tb = df.truth_change_band_pct.astype(float)
ax.bar(x + w/2, tb, w, color='C2', label='Independent survey vs design (observable band)')
tt = df.truth_change_total_pct.astype(float)
for i, r in df.iterrows():
    if pd.notna(r.truth_change_total_pct):
        ax.plot(i + w/2, r.truth_change_total_pct, 'D', color='#8a2d04', ms=8, zorder=5)
        ax.annotate('', xy=(i + w/2, r.truth_change_total_pct), xytext=(i + w/2, r.truth_change_band_pct),
                    arrowprops=dict(arrowstyle='->', color='#8a2d04', lw=1.2))
    if pd.notna(r.truth_change_band_pct):
        ax.text(i + w/2, min(r.truth_change_band_pct, r.get('truth_change_total_pct', 0) or 0) - 1.6,
                r.indep_ref, ha='center', fontsize=7, color='#2e7d32')
    if pd.notna(r.field_rmse_m):
        ax.text(i - w/2, r.sar_change_band_pct - 1.6, f'RMSE {r.field_rmse_m} m',
                ha='center', fontsize=7, color='#555')
ax.plot([], [], 'D', color='#8a2d04', ms=8, label='Total loss incl. deep zone (full curve)')
ax.set_xticks(x)
ax.set_xticklabels([f'{r.reservoir}\nA/P {r.ap_m:.0f}' for _, r in df.iterrows()], fontsize=9)
ax.set_ylabel('Capacity change vs design (%)   (negative = loss)')
ax.set_title('Paper 2 — SAR-detected reservoir capacity loss and independent validation\n'
             'Nine Sicilian reservoirs, Period B (2022–2026), ordered by A/P')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT / 'bathymetry_change_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT / "bathymetry_change_summary.png"}')
