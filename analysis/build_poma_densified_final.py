"""
build_poma_densified_final.py (2026-07-29)

Final densified-and-corrected Poma Period-B bathymetry: the 24 masks that pass
the Paper-1-style area-outlier check (10 production dates, now itself fixed to
exclude 2026-02-12/2026-03-26, plus the 14 remaining densification dates that
were not promoted into the official 10). Saves dem_Poma_B_densified.tif plus a
2D depth map and an interactive 3D plotly view, in the same visual style as
schwatke_bathymetry_3d.py::phase3().
"""
import pathlib, sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

RES = 'Poma'
MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')
BAD_DATES = {'2026-02-12', '2026-03-26'}

df = pd.read_csv(OUT_DIR / 'poma_densify_prototype_pairs.csv', parse_dates=['date'])
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
df = df[~df['date_str'].isin(BAD_DATES)].dropna(subset=['wl_m'])
print(f'Building densified DEM from {len(df)} clean masks '
      f'(WL {df["wl_m"].min():.1f}-{df["wl_m"].max():.1f} m)')

raw_arrays, wls, meta = [], [], None
for _, row in df.iterrows():
    with rasterio.open(MASK_DIR / f'mask_{RES}_{row["date_str"]}.tif') as src:
        arr = src.read(1).astype(np.float32)
        if meta is None:
            meta = src.meta.copy()
    raw_arrays.append(arr)
    wls.append(row['wl_m'])

dem = m.build_dem_from_arrays(raw_arrays, wls)

out_meta = meta.copy()
out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
out_tif = OUT_DIR / f'dem_{RES}_B_densified.tif'
with rasterio.open(out_tif, 'w', **out_meta) as dst:
    dst.write(dem[np.newaxis, :, :])
print(f'Saved {out_tif}  WL range {min(wls):.1f}-{max(wls):.1f} m  '
      f'depth range {np.nanmin(dem):.1f}-{max(wls):.1f} m  n_masks={len(wls)}')

# ── 2D depth map ──────────────────────────────────────────────────────────
wl_max = np.nanmax(dem)
depth = dem - wl_max
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(depth, cmap='Blues_r', vmin=np.nanmin(depth), vmax=0, origin='upper')
plt.colorbar(im, ax=ax, label='Depth (m)')
ax.set_title(f'{RES} — densified+corrected (n={len(wls)}) bathymetric depth map', fontsize=11)
ax.set_xlabel('Column (px)'); ax.set_ylabel('Row (px)')
fig.tight_layout()
out_png = OUT_DIR / f'bathymetry_{RES}_B_densified_2D.png'
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f'Saved {out_png}')

# ── 3D interactive (plotly) ───────────────────────────────────────────────
with rasterio.open(out_tif) as src:
    cols = np.arange(src.width); rows_ = np.arange(src.height)
    xs = src.transform.c + cols * src.transform.a
    ys = src.transform.f + rows_ * src.transform.e

step = max(1, max(dem.shape) // 200)
dem_s = dem[::step, ::step]; xs_s = xs[::step]; ys_s = ys[::step]

EXAG = 20
x_range = float(abs(xs_s[-1] - xs_s[0]))
y_range = float(abs(ys_s[0] - ys_s[-1]))
z_range = float(np.nanmax(dem_s) - np.nanmin(dem_s))
horiz = max(x_range, y_range, 1.0)
z_ratio = max((z_range * EXAG) / horiz, 0.01)
y_ratio = y_range / horiz

fig_3d = go.Figure(go.Surface(
    z=dem_s, x=xs_s, y=ys_s, colorscale='Blues_r', showscale=True,
    colorbar=dict(title='WL (m a.s.l.)'),
))
fig_3d.update_layout(
    title=f'{RES} — densified+corrected bathymetry (3D, n={len(wls)} masks)',
    scene=dict(
        xaxis_title='Easting (m)', yaxis_title='Northing (m)',
        zaxis_title='Elevation (m a.s.l.)',
        aspectmode='manual', aspectratio=dict(x=1.0, y=y_ratio, z=z_ratio),
    ),
)
out_html = OUT_DIR / f'bathymetry_{RES}_B_densified_3D.html'
fig_3d.write_html(str(out_html))
print(f'Saved {out_html}')
