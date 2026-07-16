"""
fetch_terrain.py — pre-fetch surrounding topography for the 3D topo-bathymetry view.

For each core reservoir, download the Copernicus GLO-30 DEM (asset GLO30_2024_1) over
the reservoir's extent plus a buffer, resample onto the reservoir's own 10 m /
EPSG:32633 grid, and save `analysis/schwatke_output/terrain/terrain_<Res>.tif`.

These bundled tiles let the Streamlit tool render the reservoir seated in its real
valley (bathymetry below the max shoreline, real terrain above) fully offline — the
deployed app has no Earth-Engine credentials. Run once (needs EE auth):

    python analysis/fetch_terrain.py
"""
import pathlib, urllib.request
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

REPO = pathlib.Path(__file__).resolve().parent.parent
DEM_DIR = REPO / 'analysis' / 'schwatke_output'
OUT_DIR = DEM_DIR / 'terrain'
RESERVOIRS = ['Ancipa', 'Garcia', 'Rosamarina', 'Poma', 'Pozzillo']
BUFFER_M = 600.0          # surrounding context shown around the reservoir
PIX = 10.0                # match the SAR DEM grid
GLO30 = 'COPERNICUS/DEM/GLO30_2024_1'   # current asset (GLO30 is deprecated)


def _init_ee():
    try:
        import truststore; truststore.inject_into_ssl()   # UniPa TLS proxy
    except Exception:
        pass
    import ee
    ee.Initialize(project='ee-ciceromartinsjr')
    return ee


def _union_bounds(name):
    """Union bounding box of all of this reservoir's DEM tiles (Periods A/B/variants)."""
    fps = list(DEM_DIR.glob(f'dem_{name}_*.tif'))
    L = B = R = T = None
    for fp in fps:
        with rasterio.open(fp) as s:
            b = s.bounds
        L = b.left if L is None else min(L, b.left)
        B = b.bottom if B is None else min(B, b.bottom)
        R = b.right if R is None else max(R, b.right)
        T = b.top if T is None else max(T, b.top)
    return L, B, R, T


def fetch(name, ee):
    L, B, R, T = _union_bounds(name)
    # buffer + snap to the 10 m grid
    xmin = np.floor((L - BUFFER_M) / PIX) * PIX
    ymin = np.floor((B - BUFFER_M) / PIX) * PIX
    xmax = np.ceil((R + BUFFER_M) / PIX) * PIX
    ymax = np.ceil((T + BUFFER_M) / PIX) * PIX
    W = int(round((xmax - xmin) / PIX)); H = int(round((ymax - ymin) / PIX))
    dst_tf = from_origin(xmin, ymax, PIX, PIX)

    dem = ee.ImageCollection(GLO30).select('DEM').mosaic()
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj='EPSG:32633', geodesic=False)
    url = dem.getDownloadURL({'region': region, 'scale': 30,
                              'crs': 'EPSG:32633', 'format': 'GEO_TIFF'})
    data = urllib.request.urlopen(url, timeout=180).read()

    with rasterio.MemoryFile(data) as mf, mf.open() as src:
        Tsrc = src.read(1).astype(np.float32)
        src_tf, src_crs = src.transform, src.crs

    Tg = np.full((H, W), np.nan, np.float32)
    reproject(Tsrc, Tg, src_transform=src_tf, src_crs=src_crs,
              dst_transform=dst_tf, dst_crs='EPSG:32633', resampling=Resampling.bilinear)
    # fill any residual edge gaps with nearest so the tile is gap-free
    if not np.isfinite(Tg).all():
        from scipy.ndimage import distance_transform_edt
        fin = np.isfinite(Tg)
        _, idx = distance_transform_edt(~fin, return_indices=True)
        Tg = Tg[tuple(idx)]

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f'terrain_{name}.tif'
    with rasterio.open(out, 'w', driver='GTiff', height=H, width=W, count=1,
                       dtype='float32', crs='EPSG:32633', transform=dst_tf,
                       compress='deflate', nodata=np.nan) as dst:
        dst.write(Tg, 1)
    print(f'{name:11s} {W}x{H}  terrain {np.nanmin(Tg):.0f}-{np.nanmax(Tg):.0f} m  -> {out.name}')


if __name__ == '__main__':
    ee = _init_ee()
    for r in RESERVOIRS:
        try:
            fetch(r, ee)
        except Exception as e:
            print(f'{r}: FAILED {e!r}')
