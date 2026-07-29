"""
plot_method_flowchart.py (2026-07-29)

Graphical-abstract-style method flowchart for the manuscript (Methods section):
four panels, illustrated with real Rosamarina data in the same visual language as
the Streamlit tool's onboarding slides (tool/generate_intro_assets.py), plus a
fourth panel (not in the tool's intro) showing the reconstructed AEV curve
against the design and updated reference curves.

  A. Sentinel-1 SAR water masks, stacked in 3D (grayscale land + blue water).
  B. Waterlines nested by level, + schematic remote water-level source (SWOT).
  C. Reconstructed bathymetric DEM (2D depth map).
  D. Reconstructed AEV curve vs. design and updated reference curves.

Output: manuscript_paper2/figures/method_flowchart.png
"""
import json, pathlib, sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter

REPO = pathlib.Path('.')
MASK_DIR = REPO / 'raw_data' / 'GEE_SicilyMasks'
OUT_FIG = REPO / 'manuscript_paper2' / 'figures' / 'method_flowchart.png'

sys.path.insert(0, str(REPO / 'tool'))
import bathymetry as bt
sys.path.insert(0, str(REPO / 'analysis'))
from plot_sicily_3d_comparison import render_panel as render_3d_panel

RES = 'Rosamarina'
# 7 real dates spanning dry -> full within the current (post-2026-07 windowed
# reselection) Period-B reconstruction window, evenly spaced by water level;
# driest-first once re-sorted by area below.
DATES = ['2026-01-08', '2025-09-22', '2026-02-07', '2026-02-13',
         '2026-02-19', '2026-03-21', '2026-05-03']

BLUE = '#1565c0'
DARK = '#0d3b66'
GOLD = '#b5843f'


def date_areas():
    df = pd.read_csv(REPO / 'analysis' / 'schwatke_output' /
                      'rosamarina_densify_prototype_pairs.csv')
    df['date'] = df['date'].astype(str)
    both = dict(zip(df['date'], df['area_ha']))
    return {d_: both[d_] for d_ in DATES}


def load_masks():
    arrs = []
    for d in DATES:
        fp = MASK_DIR / f'mask_{RES}_{d}.tif'
        with rasterio.open(fp) as src:
            arrs.append(src.read(1).astype(float))
    return arrs


def crop_to_content(arrs, pad=6):
    stacked = np.stack(arrs)
    any_water = (stacked > 0).any(axis=0)
    ys, xs = np.where(any_water)
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad, stacked.shape[1])
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad, stacked.shape[2])
    return [a[r0:r1, c0:c1] for a in arrs]


def pad_to_square(arrs):
    side = max(max(a.shape) for a in arrs)
    out = []
    for a in arrs:
        h, w = a.shape
        canvas = np.zeros((side, side))
        r0, c0 = (side - h) // 2, (side - w) // 2
        canvas[r0:r0 + h, c0:c0 + w] = a
        out.append(canvas)
    return out


def order_by_area(dates, arrs):
    order = sorted(range(len(arrs)), key=lambda i: (arrs[i] > 0).sum())
    return [dates[i] for i in order], [arrs[i] for i in order]


def land_texture_field(mask, seed):
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.normal(size=mask.shape), sigma=3)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    return np.where(mask > 0, -1.0, tex)


LAND_LEVELS = [-1.5, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.001]
LAND_COLORS = [BLUE, '#8a94a0', '#9aa4af', '#aab3bd', '#bac2cb', '#cad1d8', '#dae0e5']


def draw_satellite_2d(ax, cx, cy, scale=1.0, color=DARK):
    body = Rectangle((cx - 0.05 * scale, cy - 0.035 * scale), 0.10 * scale, 0.07 * scale,
                      facecolor=color, edgecolor='none', zorder=5, transform=ax.transAxes)
    ax.add_patch(body)
    for sign in (-1, 1):
        panel = Rectangle((cx + sign * 0.06 * scale - (0.09 * scale if sign < 0 else 0),
                           cy - 0.025 * scale), 0.09 * scale, 0.05 * scale,
                          facecolor='#8ec6e6', edgecolor=color, linewidth=0.8, zorder=4,
                          transform=ax.transAxes)
        ax.add_patch(panel)


def panel_a(fig, rect, dates, arrs, areas):
    ax = fig.add_axes(rect, projection='3d')
    n = len(arrs)
    h, w = arrs[0].shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    zstep = 11.0                          # wide gaps: each scene reads as its own sheet
    for i in range(n):
        field = land_texture_field(arrs[i], seed=i)
        ax.contourf(X, Y, field, levels=LAND_LEVELS, colors=LAND_COLORS,
                   alpha=0.42, zdir='z', offset=i * zstep)   # translucent: overlap shows through
        fx = [0, w, w, 0, 0]; fy = [0, 0, h, h, 0]
        ax.plot(fx, fy, [i * zstep] * 5, color='#c9d4e0', lw=1.0)
    ax.set_box_aspect((w, h, (n - 1) * zstep * 1.35))
    ax.view_init(elev=16, azim=-60)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(-zstep * 0.5, (n - 0.5) * zstep)
    ax.axis('off')
    return ax


def panel_b(fig, rect, dates, arrs, levels_m):
    ax = fig.add_axes(rect, projection='3d')
    n = len(arrs)
    h, w = arrs[0].shape
    cmap = plt.get_cmap('Blues')
    zstep = 2.2
    tmp_fig, tmp_ax = plt.subplots()
    segs = [tmp_ax.contour(a, levels=[0.5]).allsegs[0] for a in arrs]
    plt.close(tmp_fig)
    for i in range(n):
        color = cmap(0.35 + 0.55 * i / max(n - 1, 1))
        for seg in segs[i]:
            ax.plot(seg[:, 0], seg[:, 1], zs=i * zstep, color=color, lw=2.2)
    ax.set_box_aspect((w, h, (n - 1) * zstep * 3.2))
    ax.view_init(elev=26, azim=-60)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(-zstep * 0.5, (n - 0.5) * zstep)
    ax.axis('off')

    # Satellite icon on a separate 2D overlay axes (a 3D Axes cannot host plain
    # 2D patches -- it tries to do_3d_projection() every child artist).
    sat_ax = fig.add_axes(rect, zorder=5)
    sat_ax.set_xlim(0, 1); sat_ax.set_ylim(0, 1); sat_ax.axis('off')
    sat_ax.patch.set_alpha(0)
    sx, sy = 0.80, 0.86
    draw_satellite_2d(sat_ax, sx, sy, scale=0.60)
    for tx, ty in ((0.64, 0.60), (0.72, 0.54), (0.84, 0.54), (0.92, 0.60)):
        sat_ax.plot([sx, tx], [sy - 0.015, ty], color=GOLD, lw=1.1,
                   ls=(0, (3, 2)), alpha=0.9, zorder=3, transform=sat_ax.transAxes)
    sat_ax.text(sx, sy + 0.07, 'SWOT / gauge', ha='center', fontsize=9.5,
               color=DARK, weight='bold', transform=sat_ax.transAxes)
    return ax


def panel_c(fig, rect, name):
    """Real merged terrain+basin 3D render, reusing plot_sicily_3d_comparison.py's
    own renderer verbatim (same colorscale/camera logic as the Paper-1 figure and
    the tool's own 3D tab) -- rendered once via Plotly/Kaleido to a PNG, then
    embedded as an image so it composites cleanly with the matplotlib panels."""
    from PIL import Image, ImageChops
    png_path, ap_m, area_km2 = render_3d_panel(name)
    im = Image.open(png_path).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        pad = 8
        bbox = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height))
        im = im.crop(bbox)
    img = np.asarray(im)
    png_path.unlink()
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.axis('off')
    return ax


def panel_d(fig, rect, name, dem):
    ax = fig.add_axes(rect)
    levels = np.arange(dem['floor'], dem['top'] + 1e-6, 0.5)
    a_dem, _ = bt.aev(dem['arr'], dem['mask'], levels, dem['pixel_ha'])
    ax.plot(a_dem, levels, color=BLUE, lw=2.6, label='Reconstructed (SAR)', zorder=3)
    dc = bt.design_curve(name)
    if dc is not None:
        ax.plot(dc[0](levels), levels, color='0.25', ls='--', lw=1.6, label='Design curve')
    uc = bt.updated_curve(name)
    if uc is not None and uc[0] is not None:
        ax.plot(uc[0](levels), levels, color='#2e7d32', lw=1.8, label='Updated survey')
    ax.set_xlabel('Area (ha)', fontsize=9.5)
    ax.set_ylabel('Water level (m ASL)', fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc='lower right', frameon=False)
    ax.grid(alpha=0.25)
    return ax


def arrow(fig, xy_from, xy_to):
    a = FancyArrowPatch(xy_from, xy_to, transform=fig.transFigure,
                        arrowstyle='-|>', mutation_scale=22,
                        color=DARK, lw=2.0, zorder=10, shrinkA=2, shrinkB=2)
    fig.patches.append(a)


def main():
    arrs0 = pad_to_square(crop_to_content(load_masks()))
    dates_sorted, arrs = order_by_area(DATES, arrs0)
    areas = date_areas()

    dem = bt.load_dem(RES, 'B')
    aev_levels = np.arange(dem['floor'], dem['top'] + 1e-6, 0.25)
    area_curve, _ = bt.aev(dem['arr'], dem['mask'], aev_levels, dem['pixel_ha'])
    levels_m = {d_: float(np.interp(areas[d_], area_curve, aev_levels)) for d_ in DATES}

    fig = plt.figure(figsize=(15.5, 4.6))
    w0 = 0.005
    pw = 0.225
    gap = 0.028
    y0, ph = 0.13, 0.72

    rectA = [w0, y0, pw, ph]
    rectB = [w0 + pw + gap, y0, pw, ph]
    rectC = [w0 + 2 * (pw + gap), y0, pw, ph]
    rectD = [w0 + 3 * (pw + gap) + 0.01, y0, pw - 0.005, ph]

    panel_a(fig, rectA, dates_sorted, arrs, areas)
    panel_b(fig, rectB, dates_sorted, arrs, levels_m)
    panel_c(fig, rectC, RES)
    panel_d(fig, rectD, RES, dem)

    titles = ['A · SAR water masks', 'B · Waterlines + remote level',
              'C · Reconstructed DEM', 'D · Updated AEV curve']
    for rect, t in zip([rectA, rectB, rectC, rectD], titles):
        fig.text(rect[0] + rect[2] / 2, y0 + ph + 0.075, t, ha='center',
                 fontsize=11.5, color=DARK, weight='bold')

    subtitles = ['stacked, one per\nacquisition date', 'level source: gauge\nor SWOT altimetry',
                 'level-slice stack,\nexposed band only', 'replaces the design\ncurve, band-relative']
    for rect, s in zip([rectA, rectB, rectC, rectD], subtitles):
        fig.text(rect[0] + rect[2] / 2, y0 - 0.09, s, ha='center', va='top',
                 fontsize=8, color='#5a6b7b')

    yarrow = y0 + ph / 2
    arrow(fig, (rectA[0] + rectA[2] + 0.003, yarrow), (rectB[0] - 0.003, yarrow))
    arrow(fig, (rectB[0] + rectB[2] + 0.003, yarrow), (rectC[0] - 0.003, yarrow))
    arrow(fig, (rectC[0] + rectC[2] + 0.003, yarrow), (rectD[0] - 0.003, yarrow))

    fig.text(0.5, 0.99, f'{RES}: Sentinel-1 waterline stacking to updated bathymetry and AEV curve',
             ha='center', va='top', fontsize=13, weight='bold', color=DARK)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220, facecolor='white', bbox_inches='tight', pad_inches=0.15)
    print(f'Saved {OUT_FIG}')


if __name__ == '__main__':
    main()
