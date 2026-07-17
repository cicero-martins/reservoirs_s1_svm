"""
generate_intro_assets.py — builds the 3 onboarding-slide illustrations for the
Streamlit tool, from REAL pipeline data (Rosamarina), in 3D perspective, all as
animated GIFs building up one acquisition/shoreline at a time:
  1. sar_stack.gif       — ~10 Sentinel-1 SAR scenes (grayscale backscatter + blue
     water mask) stacked directly above each other (no x/y shift), revealed one at
     a time from a fairly dry date to a nearly full one, each frame captioned with
     that date's observed water area.
  2. waterlines_swot.gif — the same shorelines nested by level (largest on top),
     revealed one at a time and captioned with that shoreline's water level, + a
     schematic SWOT satellite.
  3. dem_result.gif      — the resulting DEM merged into its (near-transparent)
     surrounding terrain, orbiting the lake (same renderer/colorscale as the
     tool's own 3D tab).

Run once (or whenever the illustration reservoir/dates change):
    python tool/generate_intro_assets.py
Output: tool/intro_assets/{sar_stack,waterlines_swot,dem_result}.gif
"""
import json
import pathlib
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = pathlib.Path(__file__).resolve().parent.parent
MASK_DIR = REPO / 'raw_data' / 'GEE_SicilyMasks'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'intro_assets'
OUT_DIR.mkdir(exist_ok=True)

RES = 'Rosamarina'
# 10 real selected SAR-mask dates spanning a fairly dry reservoir to a nearly full
# one, sorted by observed water area (see analysis/selected_mask_dates.json) --
# NOT simply chronological, since the reservoir fills and drains non-monotonically.
DATES = ['2024-11-02', '2025-08-17', '2025-06-12', '2016-12-08', '2023-02-17',
         '2023-06-17', '2022-05-29', '2015-11-26', '2015-06-23', '2016-03-13']

BLUE = '#1565c0'
DARK = '#0d3b66'
GOLD = '#b5843f'


def date_areas():
    """Look up the (already-computed, authoritative) observed water area for each
    DATES entry from the same JSON the export/reconstruction pipeline uses."""
    d = json.load(open(REPO / 'analysis' / 'selected_mask_dates.json'))
    both = {e['date']: e['area_ha'] for e in d[RES]['A'] + d[RES]['B']}
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
    """Sort (date, mask) pairs by actual water-pixel count, ascending (driest
    first). Calendar order is not reliable for this -- the reservoir fills and
    drains non-monotonically -- and getting this wrong is exactly what made the
    stacked shorelines render smallest-on-top instead of largest-on-top."""
    order = sorted(range(len(arrs)), key=lambda i: (arrs[i] > 0).sum())
    return [dates[i] for i in order], [arrs[i] for i in order]


def save_gif(frame_paths, out_name, durations):
    frames = [Image.open(p).convert('RGB') for p in frame_paths]
    W = max(f.width for f in frames); H = max(f.height for f in frames)
    padded = []
    for f in frames:
        canvas = Image.new('RGB', (W, H), 'white')
        canvas.paste(f, ((W - f.width) // 2, (H - f.height) // 2))
        padded.append(canvas)
    padded[0].save(OUT_DIR / out_name, save_all=True, append_images=padded[1:],
                   duration=durations, loop=0)
    for p in frame_paths:
        p.unlink()


def land_texture_field(mask, seed):
    """A smooth mottled brightness field for the land pixels (SAR speckle averages
    out visually at a glance to soft patches, not literal per-pixel noise) plus a
    sentinel value for water, so a single contourf call can render both the
    grayscale 'image' and the blue mask as one flat plane at a given Z."""
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.normal(size=mask.shape), sigma=3)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)        # 0..1
    return np.where(mask > 0, -1.0, tex)                  # -1 = water sentinel


LAND_LEVELS = [-1.5, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.001]
LAND_COLORS = [BLUE, '#8a94a0', '#9aa4af', '#aab3bd', '#bac2cb', '#cad1d8', '#dae0e5']


def slide1_sar_stack(dates, arrs, areas):
    """~10 SAR scenes stacked directly above each other (no x/y offset) and
    revealed one at a time as an animated GIF, each frame captioned with that
    date's observed water area."""
    n = len(arrs)
    h, w = arrs[0].shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    zstep = 4.2                                            # exaggerated inter-scene gap

    frame_paths = []
    for k in range(1, n + 1):
        fig = plt.figure(figsize=(6.4, 6.4))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(k):
            field = land_texture_field(arrs[i], seed=i)
            ax.contourf(X, Y, field, levels=LAND_LEVELS, colors=LAND_COLORS,
                       alpha=0.62, zdir='z', offset=i * zstep)
            fx = [0, w, w, 0, 0]; fy = [0, 0, h, h, 0]
            ax.plot(fx, fy, [i * zstep] * 5, color='#c9d4e0', lw=1.0)
        ax.set_box_aspect((w, h, (n - 1) * zstep * 1.05))
        ax.view_init(elev=20, azim=-60)
        ax.set_xlim(0, w); ax.set_ylim(0, h)
        ax.set_zlim(-zstep * 0.5, (n - 0.5) * zstep)
        ax.axis('off')
        fig.text(0.5, 0.94, 'Sentinel-1 SAR', ha='center', fontsize=17,
                 color=DARK, weight='bold')
        cur_date = dates[k - 1]
        fig.text(0.5, 0.08, f'{cur_date}  ·  water area ≈ {areas[cur_date]:.0f} ha',
                 ha='center', fontsize=12, color=DARK, weight='bold')
        fig.text(0.5, 0.04, f'{k} of {n} acquisitions, dry → full',
                 ha='center', fontsize=10, color='#5a6b7b')
        fp = OUT_DIR / f'_frame_{k:02d}.png'
        fig.savefig(fp, dpi=130, facecolor='white', bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        frame_paths.append(fp)

    durations = [750] * (n - 1) + [2400]
    save_gif(frame_paths, 'sar_stack.gif', durations)


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


def slide2_waterlines_swot(dates, arrs, levels_m):
    """Waterline contour rings nested by level, largest on top / smallest on the
    bottom, revealed one at a time as an animated GIF, each frame captioned with
    that shoreline's water level, + a schematic SWOT satellite."""
    n = len(arrs)
    h, w = arrs[0].shape
    cmap = plt.get_cmap('Blues')
    zstep = 2.2

    # pre-compute every shoreline's contour segments once (2D scratch axes)
    tmp_fig, tmp_ax = plt.subplots()
    segs = [tmp_ax.contour(a, levels=[0.5]).allsegs[0] for a in arrs]
    plt.close(tmp_fig)

    frame_paths = []
    for k in range(1, n + 1):
        fig = plt.figure(figsize=(6.4, 6.4))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(k):
            color = cmap(0.35 + 0.55 * i / max(n - 1, 1))   # smallest=light, largest=dark
            for seg in segs[i]:
                ax.plot(seg[:, 0], seg[:, 1], zs=i * zstep, color=color, lw=2.0)
        ax.set_box_aspect((w, h, (n - 1) * zstep * 3.2))
        ax.view_init(elev=26, azim=-60)
        ax.set_xlim(0, w); ax.set_ylim(0, h)
        ax.set_zlim(-zstep * 0.5, (n - 0.5) * zstep)
        ax.axis('off')

        sat_ax = fig.add_axes([0, 0, 1, 1]); sat_ax.axis('off')
        sat_ax.set_xlim(0, 1); sat_ax.set_ylim(0, 1); sat_ax.patch.set_alpha(0)
        sx, sy = 0.78, 0.82
        draw_satellite_2d(sat_ax, sx, sy, scale=0.62)
        for tx, ty in ((0.62, 0.58), (0.70, 0.52), (0.82, 0.52), (0.90, 0.58)):
            sat_ax.plot([sx, tx], [sy - 0.015, ty], color=GOLD, lw=1.1,
                        ls=(0, (3, 2)), alpha=0.85, zorder=3, transform=sat_ax.transAxes)
        sat_ax.text(sx, sy + 0.055, 'SWOT altimetry', ha='center', fontsize=10,
                    color=DARK, weight='bold', transform=sat_ax.transAxes)

        fig.text(0.5, 0.94, 'Waterlines by level', ha='center', fontsize=17,
                 color=DARK, weight='bold')
        cur_date = dates[k - 1]
        fig.text(0.5, 0.08, f'{cur_date}  ·  level ≈ {levels_m[cur_date]:.1f} m ASL',
                 ha='center', fontsize=12, color=DARK, weight='bold')
        fig.text(0.5, 0.04, 'Shoreline at each water level + remote water-level source',
                 ha='center', fontsize=10, color='#5a6b7b')
        fp = OUT_DIR / f'_wl_frame_{k:02d}.png'
        fig.savefig(fp, dpi=130, facecolor='white', bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        frame_paths.append(fp)

    durations = [750] * (n - 1) + [2400]
    save_gif(frame_paths, 'waterlines_swot.gif', durations)


def _topo_colorscale_transparent(zmin, zmax, nmax):
    """Bathymetry deep->shallow as opaque dark-blue -> white pinned at the max
    shoreline; real surrounding terrain above it faded to near-transparent (alpha
    ~0.10) so it reads as context without hiding the basin depth cue."""
    f = (nmax - zmin) / max(zmax - zmin, 1e-6)
    f = max(f, 0.03)
    return [[0.0, 'rgba(8,48,107,1.0)'], [f * 0.45, 'rgba(33,113,181,1.0)'],
            [f * 0.8, 'rgba(137,192,224,1.0)'], [f, 'rgba(247,251,255,1.0)'],
            [min(f + 1e-3, 0.999), 'rgba(232,214,172,0.10)'],
            [f + (1 - f) * 0.5, 'rgba(181,132,63,0.10)'],
            [1.0, 'rgba(90,52,16,0.10)']]


def slide3_dem():
    """The resulting bathymetric DEM merged into its real surrounding terrain (same
    merge as the tool's own 3D tab), terrain faded near-transparent and left at its
    true relative relief, the reconstructed basin alone exaggerated in Z, camera
    orbiting the lake (dipping below the horizon too, to sell the 3D) as a slow
    animated GIF."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bathymetry as bt
    import plotly.graph_objects as go

    tb = bt.topobathy(RES, 'B')
    a, nmax = tb['arr'], tb['maxwl']

    f = 2
    H2, W2 = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    a = a[:H2, :W2].reshape(H2 // f, f, W2 // f, f).mean(axis=(1, 3))

    # Exaggerate the reconstructed basin's depth below the max shoreline much more
    # than the surrounding terrain's true relief above it (a single shared
    # aspectratio.z can't do this on its own -- it scales the whole surface
    # uniformly, and the terrain's true relief here vastly exceeds the basin's, so
    # left alone it swamps the basin and reads as "tall" on its own account).
    TERRAIN_EXAG = 0.12   # compress the true ~500 m surrounding relief to a subdued backdrop
    BASIN_EXAG = 3.0      # the lake bed should read as deep, the hills as a flat-ish backdrop
    z = np.where(a <= nmax, nmax - (nmax - a) * BASIN_EXAG, nmax + (a - nmax) * TERRAIN_EXAG)
    zlo_disp, zhi_disp = float(np.nanmin(z)), float(np.nanmax(z))
    # the transform is continuous at a==nmax (both branches evaluate to nmax there),
    # so the shoreline boundary in the NEW z-space is still exactly nmax itself --
    # passing the (deep, exaggerated) basin floor here instead was the earlier bug,
    # it told the colorscale the blue/tan split was at the very bottom of the range.
    colorscale = _topo_colorscale_transparent(zlo_disp, zhi_disp, nmax)

    n_frames = 48
    frame_paths = []
    for k in range(n_frames):
        az = 2 * np.pi * k / n_frames
        elev = np.radians(35) * np.sin(2 * np.pi * k / n_frames)   # dips below the horizon once per loop
        R = 1.35
        eye = dict(x=R * np.cos(elev) * np.cos(az), y=R * np.cos(elev) * np.sin(az),
                  z=R * np.sin(elev))
        fig = go.Figure(go.Surface(z=z, colorscale=colorscale, cmin=zlo_disp, cmax=zhi_disp,
                                   showscale=False))
        fig.update_layout(
            scene=dict(aspectratio=dict(x=1, y=1, z=0.55),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                      camera=dict(eye=eye)),
            width=640, height=640, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )
        fp = OUT_DIR / f'_dem_frame_{k:02d}.png'
        fig.write_image(str(fp))
        frame_paths.append(fp)

    from PIL import ImageDraw, ImageFont
    try:
        f_title = ImageFont.truetype('arialbd.ttf', 30)
        f_cap = ImageFont.truetype('arial.ttf', 19)
    except Exception:
        f_title = f_cap = ImageFont.load_default()
    composited = []
    for fp in frame_paths:
        im = Image.open(fp).convert('RGBA')
        canvas = Image.new('RGBA', (im.width, im.height + 100), (255, 255, 255, 255))
        canvas.paste(im, (0, 82), im)
        draw = ImageDraw.Draw(canvas)
        title = 'Reconstructed bathymetry'
        tw = draw.textlength(title, font=f_title)
        draw.text(((canvas.width - tw) / 2, 12), title, fill=DARK, font=f_title)
        cap = f'{RES} — DEM + surrounding terrain'
        cw = draw.textlength(cap, font=f_cap)
        draw.text(((canvas.width - cw) / 2, canvas.height - 34), cap, fill='#5a6b7b', font=f_cap)
        cfp = OUT_DIR / f'_dem_composited_{fp.stem}.png'
        canvas.convert('RGB').save(cfp)
        composited.append(cfp)
        fp.unlink()

    save_gif(composited, 'dem_result.gif', [110] * n_frames)


if __name__ == '__main__':
    dates_sorted, arrs = order_by_area(DATES, pad_to_square(crop_to_content(load_masks())))
    areas = date_areas()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bathymetry as bt
    dem = bt.load_dem(RES, 'B')
    aev_levels = np.arange(dem['floor'], dem['top'] + 1e-6, 0.25)
    area_curve, _ = bt.aev(dem['arr'], dem['mask'], aev_levels, dem['pixel_ha'])
    levels_m = {d_: float(np.interp(areas[d_], area_curve, aev_levels)) for d_ in DATES}

    slide1_sar_stack(dates_sorted, arrs, areas)
    slide2_waterlines_swot(dates_sorted, arrs, levels_m)
    slide3_dem()
    print(f'Saved 3 illustrations (GIFs) to {OUT_DIR}')
