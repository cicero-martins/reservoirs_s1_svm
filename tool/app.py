"""
Reservoir SAR Bathymetry Explorer — Paper-2 tool (Streamlit MVP)

Interactive explorer over the SAR-waterline reconstructed bathymetry of the 5 core
Sicilian reservoirs: pick a reservoir + period, view the 2D depth map, 3D surface,
area-elevation-volume (AEV) curves against the design & updated-survey references,
the A-vs-B sedimentation change map, and download the DEM GeoTIFF.

Run:  streamlit run tool/app.py
Scope (MVP): explorer over the already-reconstructed DEMs. Live Earth-Engine
reconstruction of an arbitrary global reservoir is future work (see project plan).
"""

import io
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import distance_transform_edt, gaussian_filter

import bathymetry as bt


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

st.set_page_config(page_title='Reservoir SAR Bathymetry Explorer', layout='wide')

AP_COLORS = {'low': '#f88f4d', 'med': '#d64a02', 'high': '#8a2d04'}
def ap_band(ap):
    return 'low' if ap < 120 else ('med' if ap < 250 else 'high')


@st.cache_data(show_spinner=False)
def get_dem(name, period):
    return bt.load_dem(name, period)

@st.cache_data(show_spinner=False)
def get_change(name):
    return bt.change_map(name)

@st.cache_data(show_spinner=False)
def get_capacity(name):
    return bt.capacity_change(name)

@st.cache_data(show_spinner=False)
def get_topobathy(name, period):
    return bt.topobathy(name, period)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title('SAR Bathymetry Explorer')
st.sidebar.caption('SAR-waterline reservoir bathymetry · Sicily')
name = st.sidebar.selectbox('Reservoir', list(bt.RESERVOIRS.keys()), index=3)
cfg = bt.RESERVOIRS[name]
PLABEL = {'B': 'B — 2022–2026', 'A': 'A — 2014–2016', 'Planet': 'PlanetScope (optical, 3 m)'}
periods = ['B', 'A'] + (['Planet'] if bt.has_period(name, 'Planet') else [])
period = st.sidebar.radio('Reconstruction', periods,
                          format_func=lambda p: PLABEL[p], horizontal=True)
downsample = st.sidebar.slider('3D detail (downsample factor)', 1, 6, 3,
                               help='Higher = coarser/faster 3D surface')
show_terrain = st.sidebar.toggle('Surrounding terrain (3D)', value=True,
                                 help='On: reservoir seated in its real GLO-30 valley '
                                      '(white marks the max shoreline). Off: only the '
                                      'basin — terrain above the waterline is clipped away.')
st.sidebar.markdown(f"**A/P** = {cfg['ap']:.0f} m &nbsp; "
                    f"<span style='background:{AP_COLORS[ap_band(cfg['ap'])]};color:white;"
                    f"padding:2px 8px;border-radius:4px'>{ap_band(cfg['ap']).upper()}</span>",
                    unsafe_allow_html=True)
st.sidebar.caption(cfg['notes'])

dem = get_dem(name, period)
dem_label = 'PlanetScope DEM' if period == 'Planet' else 'SAR DEM'
st.title(f'{name} — {PLABEL[period]}')
if period == 'Planet':
    st.caption('Optical reconstruction (PlanetScope 3 m NDWI waterlines) — an independent '
               'cross-check of the SAR (Period B) reconstruction.')

if dem is None:
    st.warning(f'No reconstructed DEM found for {name} (Period {period}). '
               f'Expected {bt.dem_file(name, period)}.')
    st.stop()

# ── Metrics row ─────────────────────────────────────────────────────────────────
cap = get_capacity(name) if period == 'B' else None
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

tab2d, tab3d, tabaev, tabchg = st.tabs(['2D depth map', '3D surface', 'AEV curves', 'A-vs-B change'])

with tab2d:
    fig = go.Figure(go.Heatmap(
        z=dem['arr'], x=xs, y=ys, colorscale='Blues_r',
        colorbar=dict(title='Elev (m ASL)'), hovertemplate='%{z:.1f} m<extra></extra>'))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(scaleanchor='x', scaleratio=1))
    st.plotly_chart(fig, width='stretch')

with tab3d:
    f = downsample
    tb = get_topobathy(name, period) if bt.has_terrain(name) else None
    if tb is not None:
        # Continuous topo-bathymetry grid (bathymetry below the shoreline, real GLO-30
        # terrain above); gap-free, so no ragged edge / sawtooth in either mode.
        a, tbounds, nmax = tb['arr'], tb['bounds'], tb['maxwl']
        x3 = np.linspace(tbounds.left, tbounds.right, a.shape[1])
        y3 = np.linspace(tbounds.top, tbounds.bottom, a.shape[0])
    else:
        a, x3, y3, nmax = _closed_surface(dem['arr']), xs, ys, dem['top']
    floor, zmax = float(np.nanmin(a)), float(np.nanmax(a))
    H2, W2 = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    # no NaN left, so a plain block-mean downsample (no nanmean boundary aliasing)
    z = a[:H2, :W2].reshape(H2 // f, f, W2 // f, f).mean(axis=(1, 3))
    if show_terrain and tb is not None:
        cmin, cmax, zrange, zasp = floor, zmax, None, 0.2       # full valley: blue->white->brown
    else:
        # Basin detail: clip the z-axis at the max shoreline so terrain above the
        # waterline is cut away (no sawtooth — the surface stays continuous below).
        cmin, cmax, zrange, zasp = floor, nmax, [floor, nmax], 0.25
    fig = go.Figure(go.Surface(
        z=z, x=x3[:W2].reshape(W2 // f, f).mean(1), y=y3[:H2].reshape(H2 // f, f).mean(1),
        colorscale=_topo_colorscale(cmin, cmax, nmax), cmin=cmin, cmax=cmax,
        colorbar=dict(title='m ASL')))
    scene = dict(aspectratio=dict(x=1, y=1, z=zasp), zaxis=dict(title='Elev (m)'))
    if zrange is not None:
        scene['zaxis']['range'] = zrange
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), scene=scene)
    st.plotly_chart(fig, width='stretch')

with tabaev:
    levels = np.arange(dem['floor'], dem['top'] + 1e-6, 0.5)
    a_dem, v_dem = bt.aev(dem['arr'], dem['mask'], levels, dem['pixel_ha'])
    dc = bt.design_curve(name)
    uc = bt.updated_curve(name)

    colA, colV = st.columns(2)
    figA = go.Figure()
    figA.add_scatter(x=a_dem, y=levels, name=dem_label, line=dict(color='#1565c0', width=3))
    if dc:
        figA.add_scatter(x=dc[0](levels), y=levels, name='Design (1960s)',
                         line=dict(color='black', dash='dash'))
    if uc and cfg['updated'] in ('poma_new', 'rosamarina_2025'):
        figA.add_scatter(x=uc[0](levels), y=levels, name='Updated survey',
                         line=dict(color='#2e7d32', width=2))
    figA.update_layout(title='Area–elevation', xaxis_title='Area (ha)',
                       yaxis_title='Water level (m ASL)', height=460, margin=dict(t=40))
    colA.plotly_chart(figA, width='stretch')

    figV = go.Figure()
    figV.add_scatter(x=v_dem, y=levels, name=dem_label, line=dict(color='#1565c0', width=3))
    if dc:
        v_des_rel = dc[1](levels) - float(dc[1](dem['floor']))
        figV.add_scatter(x=v_des_rel, y=levels, name='Design (rel.)',
                         line=dict(color='black', dash='dash'))
    if uc and cfg['updated'] in ('poma_new', 'rosamarina_2025'):
        v_upd_rel = uc[1](levels) - float(uc[1](dem['floor']))
        figV.add_scatter(x=v_upd_rel, y=levels, name='Updated survey (rel.)',
                         line=dict(color='#2e7d32', width=2))
    figV.update_layout(title='Volume above floor', xaxis_title='Volume (Mm³)',
                       yaxis_title='Water level (m ASL)', height=460, margin=dict(t=40))
    colV.plotly_chart(figV, width='stretch')
    if dc is None:
        st.info('Design/updated curves are external files not found on this machine — '
                'showing the SAR DEM only. (Bundle the curves for deployment.)')

with tabchg:
    chg = get_change(name)
    if chg is None:
        st.info(f'A-vs-B change map needs both Period-A and Period-B DEMs for {name}.')
    else:
        vmax = float(np.nanpercentile(np.abs(chg['diff']), 95)) or 1.0
        fig = go.Figure(go.Heatmap(
            z=chg['diff'], x=xs, y=ys, colorscale='RdBu', zmid=0, zmin=-vmax, zmax=vmax,
            colorbar=dict(title='B−A (m)'), hovertemplate='%{z:+.2f} m<extra></extra>'))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(scaleanchor='x', scaleratio=1))
        st.plotly_chart(fig, width='stretch')
        st.caption('Elevation difference B − A over the co-observed range '
                   f"(≥ {chg['lo']:.1f} m). Positive (red) = higher lakebed in Period B "
                   '= net deposition / sedimentation proxy.')

# ── Download ────────────────────────────────────────────────────────────────────
with open(bt.dem_file(name, period), 'rb') as fh:
    st.sidebar.download_button('⬇ Download DEM GeoTIFF', fh.read(),
                               file_name=f'dem_{name}_{period}.tif', mime='image/tiff')
