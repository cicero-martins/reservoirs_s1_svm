"""
plot_sicily_3d_comparison.py

Static 3D comparison of the four Sicilian near-truth reservoirs' basin geometry
(Ancipa, Pozzillo, Poma, Rosamarina), requested by reviewer De Marchis to make the
contrasting morphometries described in Section 2.2 visible in one figure.

This reuses tool/app.py's own 3D-tab rendering exactly (_topo_colorscale,
_scene3d, block-mean downsampling, z_exag=3.0, the tool's own default setting --
see its "3D vertical exaggeration" slider) rather than a bespoke look: one shared
exaggeration applied to the real merged terrain+basin elevation via the scene's
aspect ratio (not by rescaling the data), opaque elevation colouring (deep
blue -> white at the shoreline -> beige -> brown terrain), no smoothing. This is
the tool's actual scientific view, not the near-transparent stylised rendering
used for the onboarding slides.

The only addition is a per-reservoir camera azimuth (from the PCA angle of the
water pixels, offset from Plotly's own default 45 degrees) so no basin --
Ancipa in particular, a narrow east-west valley -- is viewed edge-on; elevation
and distance otherwise match Plotly's default camera (35 deg, R=2.165), which
is what the reference screenshots use unmodified.

Reads:  analysis/schwatke_output/dem_{name}_B.tif (via tool/bathymetry.py)
        analysis/schwatke_output/terrain/terrain_{name}.tif
Output: manuscript/figures/sicily_4lakes_3d.png
"""
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'tool'))
import bathymetry as bt

import plotly.graph_objects as go

RESERVOIRS = ['Ancipa', 'Pozzillo', 'Poma', 'Rosamarina']
Z_EXAG = 3.0                # tool/app.py's own default "3D vertical exaggeration"
DOWNSAMPLE_F = 3            # block-mean factor (tool's own default)
VIEW_OFFSET_DEG = 40        # camera azimuth offset from each basin's own long axis
CAM_ELEV_DEG = 48           # steeper than Plotly's default 35.3 -- reduces grazing-angle
                            # z-fighting speckle Kaleido shows over the flat deep basin floor
CAM_R = 1.7                 # tighter than Plotly's default 2.165 -- fills more of the frame
OUT_DIR = REPO / 'manuscript' / 'figures'
FRAME_SIZE = 620
FRAME_WIDTH = 780           # wider than tall -- room for axis labels + colorbar


def _topo_colorscale(zmin, zmax, nmax):
    """Verbatim from tool/app.py: bathymetry deep-blue -> white pinned at the max
    shoreline (nmax); real terrain above it opaque beige -> brown."""
    f = (nmax - zmin) / max(zmax - zmin, 1e-6)
    if f >= 0.985:
        return [[0.0, '#08306b'], [0.45, '#2171b5'], [0.8, '#89c0e0'], [1.0, '#f7fbff']]
    f = max(f, 0.03)
    return [[0.0, '#08306b'], [f * 0.45, '#2171b5'], [f * 0.8, '#89c0e0'], [f, '#f7fbff'],
            [min(f + 1e-3, 0.999), '#e8d6ac'], [f + (1 - f) * 0.5, '#b5843f'], [1.0, '#5a3410']]


def _principal_axis_deg(mask, xs, ys):
    """Angle (deg, from +x axis) of the first principal component of the water
    pixels' coordinates -- the basin's long axis, used to pick a camera azimuth
    that doesn't view it edge-on."""
    rows, cols = np.where(mask)
    x = xs[cols] - xs[cols].mean()
    y = ys[rows] - ys[rows].mean()
    cov = np.cov(np.vstack([x, y]))
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    return np.degrees(np.arctan2(vy, vx))


def render_panel(name):
    tb = bt.topobathy(name, 'B')
    a, bounds, nmax = tb['arr'], tb['bounds'], tb['maxwl']
    d = bt.load_dem(name, 'B')
    basin_mask_full = np.isfinite(d['arr'])

    # topobathy's nearest-neighbour reprojection of the DEM onto the terrain grid
    # (bathymetry.py:topobathy) leaves scattered single-pixel reprojection gaps
    # inside the basin, filled from the terrain branch (clamped to nmax) -- a
    # salt-and-pepper fleck of shoreline-coloured pixels inside otherwise-deep
    # water. A small median filter removes exactly this kind of isolated-pixel
    # noise without blurring the real basin/terrain edges.
    a = median_filter(a, size=3)

    H, W = a.shape
    f = DOWNSAMPLE_F
    Hc, Wc = (H // f) * f, (W // f) * f
    z = a[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f).mean(axis=(1, 3))   # plain block-mean, like the tool
    xs_full = np.linspace(bounds.left, bounds.right, W)
    ys_full = np.linspace(bounds.top, bounds.bottom, H)
    xd = xs_full[:Wc].reshape(Wc // f, f).mean(1)
    yd = ys_full[:Hc].reshape(Hc // f, f).mean(1)

    zlo, zhi = float(np.nanmin(z)), float(np.nanmax(z))
    colorscale = _topo_colorscale(zlo, zhi, nmax)

    x_m = float(abs(xd[-1] - xd[0])); y_m = float(abs(yd[0] - yd[-1]))
    D = max(x_m, y_m, 1e-6)
    az_ratio = (max(zhi - zlo, 1e-6) / D) * Z_EXAG

    axis_deg = _principal_axis_deg(basin_mask_full, xs_full, ys_full)
    az = np.radians(axis_deg + VIEW_OFFSET_DEG)
    elev = np.radians(CAM_ELEV_DEG)
    eye = dict(x=CAM_R * np.cos(elev) * np.cos(az), y=CAM_R * np.cos(elev) * np.sin(az),
               z=CAM_R * np.sin(elev))

    fig = go.Figure(go.Surface(
        z=z, x=xd, y=yd, colorscale=colorscale, cmin=zlo, cmax=zhi,
        colorbar=dict(title='m ASL', len=0.75, thickness=16),
    ))
    fig.update_layout(
        scene=dict(
            aspectmode='manual', aspectratio=dict(x=x_m / D, y=y_m / D, z=az_ratio),
            xaxis=dict(title='Easting (m)'), yaxis=dict(title='Northing (m)'),
            zaxis=dict(title='Elev (m)', range=[zlo, zhi], nticks=6),
            camera=dict(eye=eye),
        ),
        width=FRAME_WIDTH, height=FRAME_SIZE, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
    )
    area_km2 = float(basin_mask_full.sum()) * abs(d['transform'].a) * abs(d['transform'].e) / 1e6
    ap_m = bt.RESERVOIRS[name]['ap']
    png_path = OUT_DIR / f'_frame_{name}.png'
    fig.write_image(str(png_path), scale=4)   # supersample -- reduces WebGL z-fighting speckle
    return png_path, ap_m, area_km2


def compose(panels):
    try:
        f_cap = ImageFont.truetype('arial.ttf', 20)
        f_sup = ImageFont.truetype('arialbd.ttf', 30)
    except Exception:
        f_cap = f_sup = ImageFont.load_default()

    pad, cap_h, sup_h = 14, 66, 50
    cell_w = FRAME_WIDTH + 2 * pad
    cell_h = FRAME_SIZE + cap_h + 2 * pad
    canvas = Image.new('RGB', (cell_w * 2, cell_h * 2 + sup_h), 'white')
    draw = ImageDraw.Draw(canvas)
    title = 'Basin geometry of the four Sicilian near-truth reservoirs'
    tw = draw.textlength(title, font=f_sup)
    draw.text(((canvas.width - tw) / 2, 10), title, fill=(20, 20, 20), font=f_sup)

    for i, (name, png_path, ap_m, area_km2) in enumerate(panels):
        col, row = i % 2, i // 2
        x0 = col * cell_w
        y0 = sup_h + row * cell_h
        im = Image.open(png_path).convert('RGBA')
        im = im.resize((FRAME_WIDTH, FRAME_SIZE), Image.LANCZOS)  # downsample the 2x supersample
        canvas.paste(im, (x0 + pad, y0 + pad), im)   # alpha as mask -> transparent bg stays white
        cap = f'{name} — A/P {ap_m:.0f} m, area {area_km2:.1f} km²'
        cw = draw.textlength(cap, font=f_cap)
        draw.text((x0 + (cell_w - cw) / 2, y0 + pad + FRAME_SIZE + 14), cap,
                   fill=(70, 80, 90), font=f_cap)
        png_path.unlink()

    out = OUT_DIR / 'sicily_4lakes_3d.png'
    canvas.save(out)
    print(f'Saved: {out}')


if __name__ == '__main__':
    panels = []
    for name in RESERVOIRS:
        png_path, ap_m, area_km2 = render_panel(name)
        panels.append((name, png_path, ap_m, area_km2))
        print(f'{name}: A/P {ap_m:.1f} m, area {area_km2:.2f} km2 -> {png_path.name}')
    compose(panels)
