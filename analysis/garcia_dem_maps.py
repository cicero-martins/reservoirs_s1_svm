"""Generate separate bathymetry map images for satellite DEM_B and survey terrain grid.
Reads the already-filtered survey_terrain_Garcia.tif produced by garcia_survey_comparison.py."""
import pathlib, sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio

sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = pathlib.Path('analysis/schwatke_output/garcia_survey')
DEM_B_P = pathlib.Path('analysis/schwatke_output/dem_Garcia_B.tif')
SRV_P   = OUT_DIR / 'survey_terrain_Garcia.tif'

with rasterio.open(DEM_B_P) as src:
    dem_b  = src.read(1).astype(np.float64)
    nd     = src.nodata
    bounds = src.bounds
    ext    = [bounds.left, bounds.right, bounds.bottom, bounds.top]

with rasterio.open(SRV_P) as src:
    srv = src.read(1).astype(np.float64)

# clean nodata
if nd is not None:
    dem_b[dem_b == nd] = np.nan
srv[srv < -1e10] = np.nan

floor_B = float(np.nanmin(dem_b))
max_B   = float(np.nanmax(dem_b))

# lake_mask from DEM_B defines the reservoir boundary (SAR max extent)
lake_mask = ~np.isnan(dem_b)

# clip both to SAR-observable range [floor_B, max_B] AND reservoir boundary
dem_plot = np.where((dem_b >= floor_B) & (dem_b <= max_B), dem_b, np.nan)
srv_plot = np.where((srv  >= floor_B) & (srv  <= max_B) & lake_mask, srv, np.nan)

print(f'DEM_B  valid: {int(np.sum(~np.isnan(dem_plot))):,} px  {np.nanmin(dem_plot):.2f}-{np.nanmax(dem_plot):.2f} m')
print(f'Survey valid: {int(np.sum(~np.isnan(srv_plot))):,} px  {np.nanmin(srv_plot):.2f}-{np.nanmax(srv_plot):.2f} m')

vmin, vmax = floor_B, max_B
cmap = 'terrain_r'

datasets = [
    (dem_plot, f'Satellite DEM₂ (2022–2026)\n{floor_B:.1f}–{max_B:.1f} m ASL', 'garcia_dem_b_map.png'),
    (srv_plot, f'Echosounder survey (Dec 2025)\n{floor_B:.1f}–{max_B:.1f} m ASL',       'garcia_survey_map.png'),
]

for data, title, fname in datasets:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, extent=ext, origin='upper',
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal',
                   interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label('Elevation (m ASL)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m, UTM 33N)', fontsize=9)
    ax.set_ylabel('Northing (m, UTM 33N)', fontsize=9)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
