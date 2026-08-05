"""
Reservoir SAR Bathymetry Explorer — Paper-2 tool (Streamlit MVP)

Interactive explorer over the SAR-waterline reconstructed bathymetry of all 9
validated Sicilian reservoirs: pick a reservoir, view the 2D depth map, 3D surface,
area-elevation-volume (AEV) curves against the design & updated-survey references,
and download the DEM GeoTIFF.

Run:  streamlit run tool/app.py
Scope (MVP): explorer over the already-reconstructed DEMs. Live Earth-Engine
reconstruction of an arbitrary global reservoir is future work (see project plan).
"""

import io
import pathlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import distance_transform_edt, gaussian_filter

import bathymetry as bt

INTRO_DIR = pathlib.Path(__file__).resolve().parent / 'intro_assets'


def _closed_surface(a):
    """Close the NaN exterior of a masked DEM for the 3D view only.

    The reservoir's steep walls meet a transparent NaN exterior along a 1-px ragged
    boundary; plotly then draws every protruding edge pixel as a vertical spike, which
    reads as a 'sawtooth' top margin (the elevation data itself is smooth — the rim is
    uniform). Filling the outside with the nearest rim value makes the surface
    continuous, so the basin renders as a smooth depression in a plane at the
    max-shoreline level. Display only: the DEM, 2D map, AEV curves and GeoTIFF download
    keep the true NaN mask and are unchanged."""
    fin = np.isfinite(a)
    if not fin.any():
        return a
    _, idx = distance_transform_edt(~fin, return_indices=True)
    return gaussian_filter(a[tuple(idx)], 1.2)   # nearest-fill + light de-facet smooth


def _topo_colorscale(zmin, zmax, nmax):
    """Bathymetry deep->shallow as dark-blue -> WHITE pinned exactly at the maximum
    shoreline (NMax), so the waterline is demarcated; real terrain above it as
    beige -> brown. When nothing is shown above NMax (basin-only view) the scale is a
    clean deep-blue -> white."""
    f = (nmax - zmin) / max(zmax - zmin, 1e-6)
    if f >= 0.985:                                       # basin only: deep blue -> surface white
        return [[0.0, '#08306b'], [0.45, '#2171b5'], [0.8, '#89c0e0'], [1.0, '#f7fbff']]
    f = max(f, 0.03)
    return [[0.0, '#08306b'], [f * 0.45, '#2171b5'], [f * 0.8, '#89c0e0'], [f, '#f7fbff'],
            [min(f + 1e-3, 0.999), '#e8d6ac'], [f + (1 - f) * 0.5, '#b5843f'], [1.0, '#5a3410']]


def _scene3d(x, y, zlo, zhi, z_exag):
    """A plotly 3D scene with a true horizontal aspect and the z-axis fixed to the common
    [zlo, zhi] window (so both water-level modes share one scale). The vertical relief is
    the real terrain scale multiplied by z_exag — no hidden per-view exaggeration."""
    x_m = float(abs(x[-1] - x[0])); y_m = float(abs(y[-1] - y[0])); D = max(x_m, y_m, 1e-6)
    az = (max(zhi - zlo, 1e-6) / D) * z_exag
    return dict(aspectmode='manual', aspectratio=dict(x=x_m / D, y=y_m / D, z=az),
                zaxis=dict(title='Elev (m)', range=[zlo, zhi]))

# Default plotly Surface lighting has specular highlights that dither into a
# salt-and-pepper speckle over large perfectly-flat regions (found 2026-07-31
# on Arancio/Garcia's deepest exposed area, a big flat plateau since it's all
# assigned one calibration mask's single water level) -- the surface normal is
# degenerate there and specular reflection amplifies floating-point noise into
# visible dots. Dropping specular to 0 removes it; ambient/diffuse are kept at
# plotly's own defaults (0.8 each) rather than a dimmer 0.6 -- an earlier version
# of this fix cut ambient too, making the whole terrain visibly darker than the
# unlit default for no reason (specular was the only offending term).
_FLAT_LIGHTING = dict(specular=0, diffuse=0.8, ambient=0.8, roughness=1.0)
_FLAT_LIGHTPOS = dict(x=0, y=0, z=100000)

st.set_page_config(page_title='Reservoir SAR Bathymetry Explorer', layout='wide')

AP_COLORS = {'low': '#f88f4d', 'med': '#d64a02', 'high': '#8a2d04'}
def ap_band(ap):
    return 'low' if ap < 120 else ('med' if ap < 250 else 'high')


@st.cache_data(show_spinner=False)
def get_dem(name, period):
    return bt.load_dem(name, period)

@st.cache_data(show_spinner=False)
def get_capacity(name, period='B'):
    return bt.capacity_change(name, period)

@st.cache_data(show_spinner=False)
def get_topobathy(name, period):
    return bt.topobathy(name, period)

@st.cache_data(show_spinner=False)
def get_vrange(name):
    return bt.vertical_range(name)


# ── Onboarding (3-slide intro, shown once per session) ──────────────────────────
INTRO_SLIDES = [
    dict(
        title='Reservoir bathymetry from SAR and SWOT',
        subtitle='Companion tool',
        image=INTRO_DIR / 'sar_stack.gif',
        body=[
            "This application accompanies Martins Jr. et al. (in preparation), "
            "*\"Fully remote-sensing bathymetry and storage of reservoirs from Sentinel-1 "
            "waterlines and SWOT altimetry\"* — a method for reconstructing reservoir "
            "**bathymetry** from satellite observations alone, without echo-sounder survey "
            "or field access.",
            "The input combines Sentinel-1 SAR acquisitions per reservoir, sampled across "
            "the observed range between drought and flood periods, with SWOT satellite "
            "altimetry for water level. All-weather, day/night SAR acquisition keeps the "
            "record dense even in persistently cloudy regions.",
        ],
    ),
    dict(
        title='Method',
        image=INTRO_DIR / 'waterlines_swot.gif',
        body=[
            "Each SAR scene is classified into water and non-water classes, delineating the "
            "reservoir's instantaneous **shoreline**.",
            "As the reservoir fills and drains, successive shorelines expose different "
            "elevation bands of the submerged slope. Stacking these observations, adapting "
            "Schwatke et al. (2020), reconstructs a digital elevation model of the exposed "
            "basin.",
            "Each shoreline requires a co-located water-level estimate. Satellite altimetry "
            "— **SWOT** and the DAHITI database — serves as the primary elevation source; "
            "in-situ gauge records, where available and reliable, provide independent "
            "validation.",
        ],
    ),
    dict(
        title='Results',
        image=INTRO_DIR / 'dem_result.gif',
        body=[
            "Across 9 Sicilian reservoirs, the reconstruction reproduces a December-2025 "
            "echo-sounder survey to **2.7 m RMSE**, and recovers **94%** of the storage loss "
            "measured by that survey.",
            "Because the satellite record only ever observes the drawdown-exposed band, each "
            "reconstruction covers a **fraction of total design volume**: 39–92% across the "
            "9 reservoirs (mean ≈ 71%), depending on drawdown amplitude and basin slope.",
            "The residual deep pool below the lowest observed waterline is extrapolated from "
            "the design curve's low-elevation branch and displayed as a distinct, dashed "
            "**estimate** — not a measurement.",
        ],
    ),
]


def _intro_wizard():
    """Renders the 3-slide onboarding sequence; returns True while it is still showing
    (caller should st.stop() after it) and False once the user has entered the tool."""
    if st.session_state.get('intro_done'):
        return False
    step = st.session_state.get('intro_step', 0)
    n = len(INTRO_SLIDES)
    s = INTRO_SLIDES[step]

    _, skip_col = st.columns([5, 1])
    with skip_col:
        if st.button('Skip intro ›', key='intro_skip'):
            st.session_state.intro_done = True
            st.rerun()

    img_col, txt_col = st.columns([1, 1], gap='large')
    with img_col:
        st.image(str(s['image']), use_container_width=True)
    with txt_col:
        st.markdown(f"## {s['title']}")
        if s.get('subtitle'):
            st.caption(s['subtitle'])
        for p in s['body']:
            st.markdown(p)
        st.progress((step + 1) / n, text=f'Step {step + 1} of {n}')
        st.write('')
        nav_l, nav_r = st.columns([1, 1])
        with nav_l:
            if step > 0 and st.button('‹ Back', key='intro_back'):
                st.session_state.intro_step = step - 1
                st.rerun()
        with nav_r:
            label = 'Enter the tool →' if step == n - 1 else 'Continue →'
            if st.button(label, key='intro_next', type='primary'):
                if step == n - 1:
                    st.session_state.intro_done = True
                else:
                    st.session_state.intro_step = step + 1
                st.rerun()
    return True


if _intro_wizard():
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title('SAR Bathymetry Explorer')
st.sidebar.caption('SAR-waterline reservoir bathymetry · Sicily')
name = st.sidebar.selectbox('Reservoir', list(bt.RESERVOIRS.keys()), index=3)
cfg = bt.RESERVOIRS[name]

# Gauge+SWOT-fallback vs full-remote-sensing (FRS, SWOT-only, no gauge anywhere in the
# chain), available wherever build_frs_dem.py has produced a dem_{name}_B_swotonly.tif.
dem_period = 'B'
if bt.has_period(name, 'B_swotonly'):
    method = st.sidebar.radio(
        'Water-level source', ['Gauge + SWOT-fallback', 'Full remote sensing (SWOT-only)'],
        horizontal=True,
        help='Gauge + SWOT-fallback is the reconstruction used elsewhere in the paper '
             '(gauge primary, SWOT substituted only in documented malfunction windows). '
             'Full remote sensing uses SWOT altimetry alone, with no gauge anywhere in the '
             'chain, to test how much of the reconstruction survives without any in-situ input.')
    if method.startswith('Full'):
        dem_period = 'B_swotonly'
downsample = st.sidebar.slider('3D detail (downsample factor)', 1, 6, 3,
                               help='Higher = coarser/faster 3D surface')
show_terrain = st.sidebar.toggle('Surrounding terrain (3D)', value=True,
                                 help='On: reservoir seated in its real GLO-30 valley '
                                      '(white marks the max shoreline). Off: only the '
                                      'basin — terrain above the waterline is clipped away.')
z_exag = st.sidebar.slider('3D vertical exaggeration', 1.0, 20.0, 3.0, 0.5,
                           help='Same base scale as the surrounding terrain; increase '
                                'to amplify the vertical relief (e.g. to inspect the basin).')
st.sidebar.markdown(f"**A/P** = {cfg['ap']:.0f} m &nbsp; "
                    f"<span style='background:{AP_COLORS[ap_band(cfg['ap'])]};color:white;"
                    f"padding:2px 8px;border-radius:4px'>{ap_band(cfg['ap']).upper()}</span>",
                    unsafe_allow_html=True)
st.sidebar.caption(cfg['notes'])

dem = get_dem(name, dem_period)
dem_label = 'SAR DEM'
title_suffix = ' (full remote sensing, SWOT-only)' if dem_period == 'B_swotonly' else ''
st.title(f'{name} — 2022–2026{title_suffix}')
if dem_period == 'B_swotonly':
    st.caption('Full-remote-sensing reconstruction: water levels from SWOT altimetry alone, '
               'no gauge anywhere in the chain — an independent test of the gauge+SWOT-fallback '
               'reconstruction shown in the other mode.')

if dem is None:
    st.warning(f'No reconstructed DEM found for {name}. '
               f'Expected {bt.dem_file(name, dem_period)}.')
    st.stop()

# ── Metrics row ─────────────────────────────────────────────────────────────────
cap = get_capacity(name, dem_period)
c1, c2, c3, c4 = st.columns(4)
c1.metric('Observable floor', f"{dem['floor']:.1f} m")
c2.metric('Max reconstructed WL', f"{dem['top']:.1f} m")
c3.metric('Exposed band', f"{dem['top'] - dem['floor']:.1f} m")
if cap and 'sar_band_pct' in cap:
    delta_txt = None
    if 'truth_band_pct' in cap:
        delta_txt = f"survey {cap['truth_band_pct']:+.1f}%"
    c4.metric('SAR capacity change vs design (band)',
              f"{cap['sar_band_pct']:+.1f}%", delta=delta_txt, delta_color='off')
else:
    c4.metric('Reconstructed pixels', f"{int(dem['mask'].sum()):,}")

if cap and cap.get('truth_total_pct') is not None:
    st.caption(f"Independent survey shows a **total** capacity change of "
               f"**{cap['truth_total_pct']:+.1f}%** vs design (incl. the deep zone below the "
               f"SAR-observable floor). The SAR band estimate is a validated lower bound.")

# ── Layout: maps left, curves right ─────────────────────────────────────────────
b = dem['bounds']
xs = np.linspace(b.left, b.right, dem['arr'].shape[1])
ys = np.linspace(b.top, b.bottom, dem['arr'].shape[0])

tab2d, tab3d, tabaev = st.tabs(['2D depth map', '3D surface', 'AEV curves'])

with tab2d:
    fig = go.Figure(go.Heatmap(
        z=dem['arr'], x=xs, y=ys, colorscale='Blues_r',
        colorbar=dict(title='Elev (m ASL)'), hovertemplate='%{z:.1f} m<extra></extra>'))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(scaleanchor='x', scaleratio=1))
    st.plotly_chart(fig, width='stretch')

with tab3d:
    f = downsample
    tb = get_topobathy(name, dem_period) if bt.has_terrain(name) else None
    if tb is not None:
        # Continuous topo-bathymetry grid (bathymetry below the shoreline, real GLO-30
        # terrain above); gap-free, so no ragged edge / sawtooth in either mode.
        a, tbounds, nmax = tb['arr'], tb['bounds'], tb['maxwl']
        x3 = np.linspace(tbounds.left, tbounds.right, a.shape[1])
        y3 = np.linspace(tbounds.top, tbounds.bottom, a.shape[0])
    else:
        a, x3, y3, nmax = _closed_surface(dem['arr']), xs, ys, dem['top']
    # one common elevation window across both water-level modes, so they are comparable
    vr = get_vrange(name)
    zlo, basin_hi, terr_hi = vr if vr else (float(np.nanmin(a)), nmax, float(np.nanmax(a)))
    zhi = terr_hi if (show_terrain and tb is not None) else basin_hi   # basin view clips terrain
    H2, W2 = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    z = a[:H2, :W2].reshape(H2 // f, f, W2 // f, f).mean(axis=(1, 3))   # plain block-mean (no NaN)
    xd = x3[:W2].reshape(W2 // f, f).mean(1); yd = y3[:H2].reshape(H2 // f, f).mean(1)
    # plotly drops surface vertices that fall outside zaxis.range, and anything sitting
    # exactly ON the bound goes with them, rendering as a transparent hole rather than a
    # clipped edge. zlo is the floor shared by both water-level modes, so in whichever
    # mode owns the lower floor the deepest pixels land exactly on it (measured delta is
    # 0.0 for one mode of every one of the nine reservoirs) and the lake bottom drops
    # out. That is why it struck some 3D views and not others. Pad the floor to keep the
    # deepest water drawn. The ceiling is lifted only when terrain is shown, because
    # terr_hi is read off the raw tile before the shoreline offset is applied and
    # Ancipa's peaks end up above it; in the basin-only view the ceiling stays where it
    # is, since clipping the terrain away is the point of that view. Colour bounds remain
    # (zlo, zhi) so the two modes stay directly comparable.
    span = max(zhi - zlo, 1e-6)
    zax_lo = min(zlo, float(np.nanmin(z))) - 0.005 * span
    zax_hi = max(zhi, float(np.nanmax(z))) + 0.005 * span if (show_terrain and tb is not None) else zhi
    fig = go.Figure(go.Surface(
        z=z, x=xd, y=yd, colorscale=_topo_colorscale(zlo, zhi, nmax),
        cmin=zlo, cmax=zhi, colorbar=dict(title='m ASL'),
        lighting=_FLAT_LIGHTING, lightposition=_FLAT_LIGHTPOS))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                      scene=_scene3d(xd, yd, zax_lo, zax_hi, z_exag))
    st.plotly_chart(fig, width='stretch')

with tabaev:
    levels = np.arange(dem['floor'], dem['top'] + 1e-6, 0.5)
    a_dem, v_dem = bt.aev(dem['arr'], dem['mask'], levels, dem['pixel_ha'])
    dc = bt.design_curve(name)
    uc = bt.updated_curve(name)
    dz = bt.deepzone_split(name, dem_period)

    def _drop_zero_floor(x, y):
        """bt.aev()'s area(h)=pixels-below-h is 0 by construction at h=floor (nothing is
        strictly below the reconstruction's own minimum), so the SAR-DEM/echo-sounder
        curve's leading point is a level-slicing artifact, not an observed area -- it
        reads as the line free-falling to zero right at the bottom. Drop any leading
        points at or below this near-zero floor value; the curve starts at the first
        genuinely observed (level, area) pair instead."""
        x = np.asarray(x); y = np.asarray(y)
        nz = np.flatnonzero(x > 1e-6)
        if len(nz) == 0:
            return x, y
        return x[nz[0]:], y[nz[0]:]

    colA, colV = st.columns(2)
    figA = go.Figure()
    xa, ya = _drop_zero_floor(a_dem, levels)
    figA.add_scatter(x=xa, y=ya, name=dem_label, line=dict(color='#1565c0', width=3))
    if dc:
        figA.add_scatter(x=dc[0](levels), y=levels, name='Design curve',
                         line=dict(color='black', dash='dash'))
        if dz and dz['deep_min'] < dem['floor']:
            # Symbolic deep-zone extension: the design curve's OWN low-elevation branch
            # (not a separate terrain source, not a historical minimum level) down to its
            # lowest surveyed point -- an extrapolation, clearly dashed/greyed, not a
            # SAR observation.
            deep_levels = np.linspace(dz['deep_min'], dem['floor'], 40)
            figA.add_scatter(x=dc[0](deep_levels), y=deep_levels,
                             name='Design curve (extrapolated, unobserved deep zone)',
                             line=dict(color='#b5843f', dash='dot', width=2))
    if uc and cfg['updated'] in ('poma_new', 'rosamarina_2025', 'garcia_survey'):
        uc_label = 'Echo-sounder survey' if cfg['updated'] == 'garcia_survey' else 'Updated survey'
        xu, yu = _drop_zero_floor(uc[0](levels), levels)
        figA.add_scatter(x=xu, y=yu, name=uc_label,
                         line=dict(color='#2e7d32', width=2))
    figA.update_layout(title='Area–elevation', xaxis_title='Area (ha)',
                       yaxis_title='Water level (m ASL)', height=460, margin=dict(t=40))
    colA.plotly_chart(figA, width='stretch')

    figV = go.Figure()
    figV.add_scatter(x=v_dem, y=levels, name=dem_label, line=dict(color='#1565c0', width=3))
    if dc:
        floor_vol = float(dc[1](dem['floor']))
        v_des_rel = dc[1](levels) - floor_vol
        figV.add_scatter(x=v_des_rel, y=levels, name='Design curve (rel.)',
                         line=dict(color='black', dash='dash'))
        if dz and dz['deep_min'] < dem['floor']:
            deep_levels = np.linspace(dz['deep_min'], dem['floor'], 40)
            v_deep_rel = dc[1](deep_levels) - floor_vol   # negative: volume below the floor
            figV.add_scatter(x=v_deep_rel, y=deep_levels,
                             name='Design curve (extrapolated, unobserved deep zone)',
                             line=dict(color='#b5843f', dash='dot', width=2))
    if uc and cfg['updated'] in ('poma_new', 'rosamarina_2025', 'garcia_survey'):
        uc_label = 'Echo-sounder survey (rel.)' if cfg['updated'] == 'garcia_survey' else 'Updated survey (rel.)'
        v_upd_rel = uc[1](levels) - float(uc[1](dem['floor']))
        figV.add_scatter(x=v_upd_rel, y=levels, name=uc_label,
                         line=dict(color='#2e7d32', width=2))
    figV.update_layout(title='Volume above floor', xaxis_title='Volume (Mm³)',
                       yaxis_title='Water level (m ASL)', height=460, margin=dict(t=40))
    colV.plotly_chart(figV, width='stretch')
    if dc is None:
        st.info('Design/updated curves are external files not found on this machine — '
                'showing the SAR DEM only. (Bundle the curves for deployment.)')
    elif dz and dz.get('capped'):
        st.caption("Observable band ≈ **100%** of this reservoir's total design-curve "
                   "volume. The design curve's lowest tabulated level sits above the "
                   "reconstruction's floor, so the share cannot be resolved any further — "
                   "a limitation of the design curve, not a measured deep zone.")
    elif dz:
        st.caption(f"Observable band ≈ **{dz['band_pct']:.0f}%** of this reservoir's total "
                   f"design-curve volume; the dotted extension below the DEM floor "
                   f"({dz['deepzone_pct']:.0f}%) is an estimate from the design curve's own "
                   f"low-elevation geometry, not a SAR observation.")

# ── Download ────────────────────────────────────────────────────────────────────
with open(bt.dem_file(name, dem_period), 'rb') as fh:
    st.sidebar.download_button('⬇ Download DEM GeoTIFF', fh.read(),
                               file_name=f'dem_{name}_{dem_period}.tif', mime='image/tiff')
