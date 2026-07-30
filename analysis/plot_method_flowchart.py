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
    """Area-elevation-volume curve built by simple linear interpolation between the
    real observed (area, level) mask pairs themselves -- not from the reconstructed
    DEM's pixel count (that mixes in the per-pixel optimal-threshold and Gaussian
    smoothing used for the *spatial* DEM, which forces area to 0 at the exact floor
    elevation, a smoothing artefact with no physical meaning: the driest observed
    mask already covers a large, nonzero area). The design curve's own deep,
    unobserved branch is added below the lowest observed pair. Area is pointwise
    and plotted on its natural scale, directly comparable to the design curve's;
    volume is cumulative and has no natural absolute scale (integration only sums
    volume above the lowest observed pair), so it is anchored to the design
    curve's own volume there -- the same estimate already used for the
    deep-zone/band-capacity split in Section 3.4 -- so it connects continuously
    into the deep-zone branch below."""
    ax = fig.add_axes(rect)
    p = pd.read_csv(REPO / 'analysis' / 'schwatke_output' / f'mask_wl_pairs_{name}.csv')
    p = p[p.period == 'B'].sort_values('wl_m')
    levels = np.linspace(p.wl_m.min(), p.wl_m.max(), 200)
    a_obs = np.interp(levels, p.wl_m, p.area_ha)
    v_rel = np.zeros_like(a_obs)
    for i in range(1, len(levels)):
        v_rel[i] = v_rel[i - 1] + (a_obs[i] + a_obs[i - 1]) / 2 * (levels[i] - levels[i - 1]) * 0.01

    VOLCOL, AREACOL = BLUE, '#2e7d32'
    dc = bt.design_curve(name)
    deep_levels = None
    if dc is not None:
        v_obs = v_rel + float(dc[1](levels[0]))
        deep_min = float(dc[1].x.min())
        if deep_min < levels[0]:
            deep_levels = np.linspace(deep_min, levels[0], 40)
    else:
        v_obs = v_rel

    l1, = ax.plot(v_obs, levels, color=VOLCOL, lw=2.6, label='Volume (SAR-reconstructed)')
    if deep_levels is not None:
        ax.plot(dc[1](deep_levels), deep_levels, color=VOLCOL, lw=1.6, ls=':', alpha=0.75)
    ax.set_xlabel('Volume (Mm$^3$)', fontsize=9.5, color=VOLCOL)
    ax.tick_params(axis='x', labelsize=8, labelcolor=VOLCOL)
    ax.set_ylabel('Water level (m ASL)', fontsize=9.5)
    ax.tick_params(axis='y', labelsize=8)

    ax_top = ax.twiny()
    l2, = ax_top.plot(a_obs, levels, color=AREACOL, lw=2.2, ls='--', label='Area (SAR-reconstructed)')
    if deep_levels is not None:
        ax_top.plot(dc[0](deep_levels), deep_levels, color=AREACOL, lw=1.6, ls=':', alpha=0.75)
    ax_top.scatter(p.area_ha, p.wl_m, s=16, color=AREACOL, zorder=5,
                   edgecolors='white', linewidths=0.5, label='Observed mask/level pairs')
    ax_top.set_xlabel('Area (ha)', fontsize=9.5, color=AREACOL)
    ax_top.tick_params(axis='x', labelsize=8, labelcolor=AREACOL)

    ax.legend(handles=[l1, l2], fontsize=7.5, loc='lower right', frameon=False)
    if deep_levels is not None:
        ax.text(0.03, 0.03, 'dotted: design curve,\nunobserved deep zone',
                transform=ax.transAxes, fontsize=6.5, color='#5a6b7b', va='bottom')
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

    # 2x2 grid, laid out as a Z: A(top-left) -> B(top-right) -> C(bottom-right)
    # -> D(bottom-left), so every connecting arrow is a plain horizontal/vertical
    # segment (no diagonal routing across the title/subtitle text band).
    fig = plt.figure(figsize=(11.6, 12.6))
    x0, colw, gapx = 0.05, 0.44, 0.055
    x1 = x0 + colw + gapx
    rowh, gapy = 0.32, 0.22
    y_bot = 0.055
    y_top = y_bot + rowh + gapy

    rectA = [x0, y_top, colw, rowh]
    rectB = [x1, y_top, colw, rowh]
    rectC = [x1, y_bot, colw, rowh]
    rectD = [x0, y_bot, colw, rowh]

    panel_a(fig, rectA, dates_sorted, arrs, areas)
    panel_b(fig, rectB, dates_sorted, arrs, levels_m)
    panel_c(fig, rectC, RES)
    panel_d(fig, rectD, RES, dem)

    panels = [
        (rectA, 'A · SAR water masks', 'stacked, one per\nacquisition date'),
        (rectB, 'B · Waterlines + remote level', 'level source: gauge\nor SWOT altimetry'),
        (rectC, 'C · Reconstructed DEM', 'level-slice stack,\nexposed band only'),
        (rectD, 'D · Updated AEV curve', 'observed band + design\ncurve deep zone'),
    ]
    for rect, title, sub in panels:
        fig.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + 0.055, title, ha='center',
                 va='bottom', fontsize=13, color=DARK, weight='bold')
        fig.text(rect[0] + rect[2] / 2, rect[1] - 0.055, sub, ha='center', va='top',
                 fontsize=10.5, color='#5a6b7b', linespacing=1.3)

    ymid_top = y_top + rowh / 2
    ymid_bot = y_bot + rowh / 2
    xvert = x1 + colw - 0.03           # near column 2's right edge, clear of the
                                        # centred title/subtitle text below it
    arrow(fig, (rectA[0] + rectA[2] + 0.008, ymid_top), (rectB[0] - 0.008, ymid_top))
    arrow(fig, (xvert, rectB[1] - 0.008), (xvert, rectC[1] + rectC[3] + 0.008))
    arrow(fig, (rectC[0] - 0.008, ymid_bot), (rectD[0] + rectD[2] + 0.008, ymid_bot))

    fig.text(0.5, y_top + rowh + 0.16, f'{RES}: Sentinel-1 waterline stacking to updated bathymetry and AEV curve',
             ha='center', va='bottom', fontsize=15, weight='bold', color=DARK)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220, facecolor='white', bbox_inches='tight', pad_inches=0.15)
    print(f'Saved {OUT_FIG}')


if __name__ == '__main__':
    main()
