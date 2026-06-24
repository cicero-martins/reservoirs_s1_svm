"""
garcia_survey_comparison.py

Compares the satellite-derived bathymetric DEM (Period B, 2022-2026) with the
December 2025 echosounder/GPS survey for Lago Garcia.

Two analyses:
  1. AEV curve comparison  (area-elevation-volume, full design range 176-190 m)
     -- avoids floor-assignment and hillside artifacts
  2. Shallow-zone pixel comparison  (dem_b > floor_B+1 m AND survey < 200 m)
     -- restricts to pixels where both sources have real contour measurements

Notes on data:
  Survey (dicembre25.gpkg): ~194k XYZ points, EPSG:32633, elevation +1.86 m
  already corrected. Range 132-289 m (sonar below WL + terrain above WL in Dec 2025).
  In Dec 2025, WL ~ 175.8 m, so all DEM_B pixels (176-189 m) were exposed.

Outputs: analysis/schwatke_output/garcia_survey/
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
import rasterio
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

# Pixels farther than this from any survey point are masked (removes Delaunay
# interpolation artifacts between sparse transects).
MAX_DIST_M = 50

sys.stdout.reconfigure(encoding='utf-8')

REPO     = pathlib.Path('.')
BAT_PATH = REPO / 'raw_data'  / 'garcia_bat' / 'dicembre25.gpkg'
DEM_B    = REPO / 'analysis'  / 'schwatke_output' / 'dem_Garcia_B.tif'
OUT_DIR  = REPO / 'analysis'  / 'schwatke_output' / 'garcia_survey'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_HA = 0.01   # 10 m x 10 m pixel = 0.01 ha

# ── 1. Load satellite DEM_B ────────────────────────────────────────────────────
with rasterio.open(DEM_B) as src:
    dem_b   = src.read(1).astype(np.float64)
    meta    = src.meta.copy()
    tf      = src.transform
    bounds  = src.bounds
    nrows, ncols = src.shape

floor_B   = float(np.nanmin(dem_b[~np.isnan(dem_b)]))
max_B     = float(np.nanmax(dem_b[~np.isnan(dem_b)]))
lake_mask = ~np.isnan(dem_b)   # SAR-derived reservoir boundary (used to clip survey grids)
print(f'DEM_B: {nrows}x{ncols} px  elev {floor_B:.2f}-{max_B:.2f} m  '
      f'({np.sum(lake_mask)} valid px)')

# ── 2. Load and grid survey ────────────────────────────────────────────────────
print('Loading survey points...')
gdf = gpd.read_file(BAT_PATH)
x   = gdf['field_1'].values.astype(np.float64)
y   = gdf['field_2'].values.astype(np.float64)
z   = gdf['field_3'].values.astype(np.float64)

# Keep only points inside DEM_B extent (+ small buffer)
buf = 50
inside = ((x >= bounds.left - buf) & (x <= bounds.right + buf) &
          (y >= bounds.bottom - buf) & (y <= bounds.top + buf))
x, y, z = x[inside], y[inside], z[inside]
print(f'Survey points in extent: {len(x):,}  elev {z.min():.1f}-{z.max():.1f} m')

# Pixel-centre coordinate grids
cols_c = np.arange(ncols) * tf.a + tf.c + tf.a / 2
rows_c = np.arange(nrows) * tf.e + tf.f + tf.e / 2
gc, gr = np.meshgrid(cols_c, rows_c)

# Grid ALL survey points (sonar + terrain)
cached_tif = OUT_DIR / 'survey_dem_Garcia.tif'
print('Interpolating full survey to 10 m grid (~30 s)...')
pts_all  = np.column_stack([x, y])
srv_grid_raw = griddata(
    pts_all, z,
    np.column_stack([gc.ravel(), gr.ravel()]),
    method='linear',
).reshape(nrows, ncols)
tree_all = cKDTree(pts_all)
dist_all, _ = tree_all.query(np.column_stack([gc.ravel(), gr.ravel()]), workers=-1)
dist_all  = dist_all.reshape(nrows, ncols)
srv_grid  = np.where((dist_all <= MAX_DIST_M) & lake_mask, srv_grid_raw, np.nan)
srv_meta  = meta.copy()
srv_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
with rasterio.open(cached_tif, 'w', **srv_meta) as dst:
    dst.write(srv_grid.astype(np.float32)[np.newaxis])
print(f'Survey grid saved: {cached_tif.name}')

# ── 3. AEV curve comparison ────────────────────────────────────────────────────
# In Dec 2025 the gauge was ~175.8 m. Survey terrain points (>175.8m) were above
# waterline and correctly represent the former lake floor/shoreline at those elevations.
# Sonar points (<175.8m) represent the deep underwater zone DEM_B cannot see.
WL_SURVEY = 175.8   # approximate WL during survey

# Grid of ONLY terrain points (exposed shore, elev > WL_SURVEY) — for pixel compare
terr_mask = z > WL_SURVEY
print(f'Survey terrain points (>{WL_SURVEY} m): {terr_mask.sum():,} / {len(z):,}')
cached_terr = OUT_DIR / 'survey_terrain_Garcia.tif'
# Always regenerate (cache invalidated when MAX_DIST_M or filtering logic changes)
print(f'Gridding terrain survey points (MAX_DIST={MAX_DIST_M} m)...')
pts_terr = np.column_stack([x[terr_mask], y[terr_mask]])
srv_terr_raw = griddata(
    pts_terr, z[terr_mask],
    np.column_stack([gc.ravel(), gr.ravel()]),
    method='linear',
).reshape(nrows, ncols)
# Mask pixels farther than MAX_DIST_M from any terrain survey point
print('Computing distance mask...')
tree_terr = cKDTree(pts_terr)
dist_terr, _ = tree_terr.query(np.column_stack([gc.ravel(), gr.ravel()]), workers=-1)
dist_terr = dist_terr.reshape(nrows, ncols)
srv_terr = np.where((dist_terr <= MAX_DIST_M) & lake_mask, srv_terr_raw, np.nan)
removed = int(np.sum(~np.isnan(srv_terr_raw)) - np.sum(~np.isnan(srv_terr)))
print(f'  Distance filter removed {removed:,} px (>{MAX_DIST_M} m from survey point)')
srv_meta = meta.copy()
srv_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
with rasterio.open(cached_terr, 'w', **srv_meta) as dst:
    dst.write(srv_terr.astype(np.float32)[np.newaxis])
print(f'Terrain grid saved: {cached_terr.name}')

# AEV — use full survey grid inside lake mask (sonar provides deep zone info for AEV)
# V(h) = trapezoidal integral of A(z) dz from h_ref to h (all relative)
wl_levels    = np.arange(floor_B, max_B + 0.5, 0.5)
srv_lake     = np.where(lake_mask, srv_grid, np.nan)

def compute_aev(elev_grid, mask, levels, pixel_ha):
    """area(h) = pixels with elevation < h; volume via trapezoidal rule."""
    areas = np.array([np.sum((elev_grid < h) & mask) * pixel_ha for h in levels])
    # trapezoidal volume relative to first level
    vols  = np.zeros_like(areas)
    for i in range(1, len(levels)):
        vols[i] = vols[i-1] + (areas[i] + areas[i-1]) / 2 * (levels[i] - levels[i-1]) * 0.01
    return areas, vols

aev_dem_area, aev_dem_vol = compute_aev(dem_b,    lake_mask, wl_levels, PIXEL_HA)
aev_srv_area, aev_srv_vol = compute_aev(srv_lake, lake_mask, wl_levels, PIXEL_HA)

# Design curve (absolute volume; make relative to floor_B)
from scipy.interpolate import interp1d
curve = pd.read_excel(
    'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Garcia.xls',
    sheet_name=0, header=None
)[[2,3,4]].apply(pd.to_numeric, errors='coerce').dropna()
curve.columns = ['quota', 'area_km2', 'vol_Mm3']
curve = curve.sort_values('quota').reset_index(drop=True)
h2a   = interp1d(curve['quota'], curve['area_km2'] * 100,
                 kind='linear', bounds_error=False, fill_value='extrapolate')
h2v   = interp1d(curve['quota'], curve['vol_Mm3'],
                 kind='linear', bounds_error=False, fill_value='extrapolate')
aev_design_area = h2a(wl_levels)
v_ref = float(h2v(floor_B))
aev_design_vol  = h2v(wl_levels) - v_ref   # relative to floor_B

# ── 4. Pixel comparison — SAR-observable range only ───────────────────────────
# Only compare within [floor_B+1, max_B]:
#   - dem_b > floor_B+1  : excludes floor-assigned always-wet pixels
#   - srv_terr <= max_B  : excludes hillside above the highest SAR WL (never seen as wet)
#   - srv_terr >= floor_B: excludes deep zone below lowest SAR WL (floor assignment)
# Using terrain survey grid (points gridded from survey elev > WL_SURVEY ≈ 175.8 m)
# so both DEM_B and survey represent exposed-shoreline elevations in the same range.
shallow = (lake_mask &
           ~np.isnan(srv_terr) &
           (dem_b   > floor_B + 1.0) &
           (srv_terr >= floor_B)       &
           (srv_terr <= max_B))

print(f'\nSAR-observable range: {floor_B+1.0:.1f} m (floor+1) to {max_B:.1f} m (max WL)')

d_sh = dem_b[shallow]
s_sh = srv_terr[shallow]
diff_sh = d_sh - s_sh
bias_sh = float(np.mean(diff_sh))
rmse_sh = float(np.sqrt(np.mean(diff_sh ** 2)))
mae_sh  = float(np.mean(np.abs(diff_sh)))
ss_res  = np.sum((s_sh - d_sh) ** 2)
ss_tot  = np.sum((s_sh - s_sh.mean()) ** 2)
r2_sh   = float(1 - ss_res / ss_tot)

print(f'\n--- Pixel comparison ({floor_B+1.0:.1f}-{max_B:.1f} m, SAR-observable range) ---')
print(f'n pixels : {shallow.sum():,}')
print(f'DEM_B    : {d_sh.min():.1f} - {d_sh.max():.1f} m')
print(f'Survey   : {s_sh.min():.1f} - {s_sh.max():.1f} m')
print(f'bias     : {bias_sh:+.3f} m  (satellite - survey)')
print(f'RMSE     : {rmse_sh:.3f} m')
print(f'MAE      : {mae_sh:.3f} m')
print(f'R2       : {r2_sh:.3f}')

# AEV area at 190 m (max WL in Period B)
idx_190 = np.argmin(np.abs(wl_levels - 190.0))
print(f'\n--- AEV at 190 m ---')
print(f'Design curve : {aev_design_area[idx_190]:.1f} ha  /  {aev_design_vol[idx_190]:.2f} Mm3')
print(f'Survey       : {aev_srv_area[idx_190]:.1f} ha  /  {aev_srv_vol[idx_190]:.4f} Mm3')
print(f'DEM_B        : {aev_dem_area[idx_190]:.1f} ha  /  {aev_dem_vol[idx_190]:.4f} Mm3')

# ── 5. Save stats ──────────────────────────────────────────────────────────────
rows = [
    {'analysis': 'shallow_pixel', 'n': shallow.sum(),
     'bias_m': round(bias_sh,3), 'rmse_m': round(rmse_sh,3),
     'mae_m': round(mae_sh,3), 'r2': round(r2_sh,3)},
    {'analysis': 'AEV_area_at_190m_ha',
     'design': round(float(aev_design_area[idx_190]),1),
     'survey': round(float(aev_srv_area[idx_190]),1),
     'dem_b':  round(float(aev_dem_area[idx_190]),1)},
]
pd.DataFrame(rows).to_csv(OUT_DIR / 'garcia_comparison_stats.csv', index=False)

# ── 6. Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 6))
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
fig.suptitle('Garcia: Satellite DEM_B (2022-2026) vs Echo-sounder Survey (Dec 2025)',
             fontsize=12, fontweight='bold')

# Panel A: AEV area curve
ax0 = fig.add_subplot(gs[0])
ax0.plot(aev_design_area, wl_levels, 'k--', lw=1.5, label='Design curve (1960s)')
ax0.plot(aev_srv_area,    wl_levels, 'C2-',  lw=2.0, label='Survey Dec 2025')
ax0.plot(aev_dem_area,    wl_levels, 'C0-',  lw=2.0, label='Satellite DEM_B')
ax0.set_xlabel('Area (ha)')
ax0.set_ylabel('Water level (m ASL)')
ax0.set_title('Area-elevation curve')
ax0.legend(fontsize=7)
ax0.grid(True, alpha=0.3)
ax0.set_xlim(left=0)

# Panel B: Volume curve
ax1 = fig.add_subplot(gs[1])
ax1.plot(aev_design_vol, wl_levels, 'k--', lw=1.5, label='Design curve')
ax1.plot(aev_srv_vol,    wl_levels, 'C2-',  lw=2.0, label='Survey')
ax1.plot(aev_dem_vol,    wl_levels, 'C0-',  lw=2.0, label='Satellite DEM_B')
ax1.set_xlabel('Volume (Mm3)')
ax1.set_ylabel('Water level (m ASL)')
ax1.set_title('Volume-elevation curve')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)

# Panel C: Scatter (shallow zone only)
ax2 = fig.add_subplot(gs[2])
lo = min(d_sh.min(), s_sh.min()); hi = max(d_sh.max(), s_sh.max())
ax2.hexbin(s_sh, d_sh, gridsize=50, cmap='Blues', mincnt=1)
ax2.plot([lo, hi], [lo, hi], 'k--', lw=1.0)
ax2.set_xlabel('Survey elevation (m ASL)')
ax2.set_ylabel('Satellite DEM_B (m ASL)')
ax2.set_title(f'SAR range {floor_B+1.0:.0f}-{max_B:.0f} m\n'
              f'n={shallow.sum():,}  bias={bias_sh:+.2f} m  RMSE={rmse_sh:.2f} m  R2={r2_sh:.2f}')
ax2.grid(True, alpha=0.3)

# Panel D: Difference map (SAR-observable range)
ax3 = fig.add_subplot(gs[3])
diff_map = np.where(shallow, dem_b - srv_terr, np.nan)
vmax = max(np.nanpercentile(np.abs(diff_map[shallow]), 95), 1.0)
im = ax3.imshow(diff_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
plt.colorbar(im, ax=ax3, label='Sat - Survey (m)', fraction=0.046)
ax3.set_title(f'Difference map (shallow zone)\n'
              f'(+) = satellite higher than survey')
ax3.axis('off')

fig.tight_layout()
out_fig = OUT_DIR / 'garcia_dem_vs_survey.png'
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nFigure saved: {out_fig}')
