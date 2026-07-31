"""
validate_area_volume_timeseries.py (2026-07-31)

Item 3 of the restructured sec:res_change sequence: apply the full-remote-
sensing (FRS, SWOT-only, no gauge anywhere in the chain) AEV curve, and
separately the best available independent reference curve (updated survey
where one exists, else the design curve -- same convention as
hypsometry_comparison.py's reference_curve()), to each reservoir's full
multi-year Sentinel-1 area time series, converting area -> volume. Then
compare both resulting volume time series against the water authority's own
officially registered volume records (raw_data/opendatasicilia/) -- not a
model of ours, an independent operational record.

Originally this compared the production gauge+SWOT-fallback curve ("new")
against the design curve ("old"); superseded 2026-07-31 once the FRS DEMs
existed for all 9 reservoirs (build_frs_dem.py) -- the point of this pass is
to test how well a fully remote reconstruction, with no in-situ input at all,
tracks real reservoir management, against the best independent reference
rather than only the (often outdated) original design curve.

Covers the 5 reservoirs with a full multi-year SAR area series already
available (Ancipa, Poma, Pozzillo, Rosamarina: 2014-2025; Garcia: 2022-2026).
The other 4 (Arancio, Castello, Olivo, Nicoletti) only have the ~10-24 mask
dates used to calibrate the reconstruction itself, not a standalone area
time series -- out of scope here (a future export, not attempted).

Two independent official-record comparisons per reservoir:
  - monthly (sicilia_dighe_volumi.csv, 2007-2025, all 5, tolerance 15 days)
  - daily (sicilia_dighe_volumi_giornalieri.csv, 2023-08 to 2026-02, all but
    Pozzillo -- not in this file at all -- tolerance 2 days); this is the
    operationally relevant window, since it overlaps the reconstruction's own
    2022-2026 period.

Output: analysis/schwatke_output/area_volume_timeseries/
  area_volume_{name}.csv    -- per-date area, vol_frs, vol_current
  area_volume_summary.csv   -- per-reservoir/record RMSE, bias, r for both curves
  area_volume_timeseries.png
"""
import pathlib, sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path('.')
sys.path.insert(0, str(REPO / 'tool'))
sys.path.insert(0, str(REPO / 'analysis'))
import bathymetry as bt
import schwatke_bathymetry_3d as sb   # noqa: E402 (power_law, fit_hyps_model)
from build_frs_dem import fit_swot_curve   # noqa: E402

OUT = REPO / 'analysis' / 'schwatke_output' / 'area_volume_timeseries'
OUT.mkdir(parents=True, exist_ok=True)

# reservoir -> (area-series file, date col, area col, opendatasicilia cod)
AREA_SERIES = {
    'Ancipa':     ('validation_data/morphometric_analysis/shoreline_compactness/area_ancipa_2014-25.csv', 'date', 'value', 'dig-01'),
    'Poma':       ('validation_data/morphometric_analysis/shoreline_compactness/area_poma_2014-25.csv', 'date', 'value', 'dig-18'),
    'Pozzillo':   ('validation_data/morphometric_analysis/shoreline_compactness/area_pozzillo_2014-25.csv', 'date', 'value', 'dig-19'),
    'Rosamarina': ('validation_data/morphometric_analysis/shoreline_compactness/area_rosamarina_2014-25.csv', 'date', 'value', 'dig-22'),
    'Garcia':     ('validation_data/statistics/area_statistics/ee-chart_garcia2022-26.csv', 'data', 'areaLago_smoothed', 'dig-09'),
}

MONTHLY_FP = REPO / 'raw_data' / 'opendatasicilia' / 'sicilia_dighe_volumi.csv'
DAILY_FP = REPO / 'raw_data' / 'opendatasicilia' / 'sicilia_dighe_volumi_giornalieri.csv'


def frs_curve(name):
    """Area(ha) -> volume(Mm3), from the full-remote-sensing (FRS) reconstruction:
    the power-law hypsometric fit A=a(h-h0)^b on genuine SAR-area/SWOT-WL
    coincident pairs (build_frs_dem.fit_swot_curve, no gauge anywhere in the
    chain), NOT from the pixel-count DEM raster (bt.aev()). With only a
    handful of calibration masks, the level-slice DEM has real "shelf and
    cliff" steps in area(h) between calibration dates (confirmed by direct
    inspection, not a rendering artefact) -- invisible in the manuscript's own
    volume-elevation figure (Fig.~aevgrid) because integrating area into
    volume smooths it out, but a serious problem here, where area is queried
    pointwise at hundreds of real time-series values. The smooth power-law fit
    avoids this.

    Also returns the curve's own observed area domain (area_min, area_max) --
    the reconstruction only covers the drawdown-exposed band, so applying it to
    an area outside that range is extrapolation beyond what was ever observed,
    not a genuine estimate (same "observability envelope" limitation as the
    rest of the paper, e.g. Section~res_change's band-observable percentages).

    The fit gives area(h), not volume, so volume is built by numerically
    integrating area(h) from the FRS DEM's own floor and then anchored exactly
    as Figure 2 panel D does: add the design curve's own volume at that floor
    (the same deep-zone estimate already used throughout Section~sec:uncertainty,
    kept as the anchor for both curves so the two remain on a common absolute
    scale)."""
    dem = bt.load_dem(name, 'B_swotonly')
    floor, top = dem['floor'], dem['top']
    fit, _ = fit_swot_curve(name)
    a, h0, b = fit
    h = np.linspace(floor, top, 400)
    areas = sb.power_law(h, a, h0, b)
    vols_rel = np.concatenate([[0.0], np.cumsum(
        (areas[1:] + areas[:-1]) / 2 * np.diff(h) * 0.01)])   # ha*m -> Mm3
    dc = bt.design_curve(name)
    vols = vols_rel + (float(dc[1](floor)) if dc is not None else 0.0)
    f = interp1d(areas, vols, bounds_error=False, fill_value='extrapolate')
    return f, float(areas.min()), float(areas.max())


def current_curve_area_to_vol(name):
    """Area(ha) -> volume(Mm3) from the best available independent reference:
    the updated survey curve where one exists, else the original design curve
    -- same convention as hypsometry_comparison.py's reference_curve(). Both
    updated_curve() and design_curve() give quota->area and quota->volume, so
    compose the two on a common quota grid spanning the curve's own tabulated
    domain.

    Also returns this curve's own observed area domain (area_min, area_max),
    which can be much narrower than the FRS curve's (e.g. Garcia's echosounder
    survey tops out at 327 ha, well short of the 101-381 ha the full SAR area
    record spans) -- scoring outside it means extrapolating a hypsometric
    curve near a near-zero local slope, which blew up to +-inf until this was
    tracked and scored separately, same 'observability envelope' discipline
    used for the FRS curve's own domain.

    area_i(h) can also SATURATE (go flat) at BOTH ends of its own domain --
    Garcia's echosounder raster has area~0 for its lowest ~30% of tabulated
    levels (below the survey's own useful floor) and area~ceiling for its
    top ~50% (levels beyond where any new pixels flood in), with only the
    middle ~20% actually informative. This isn't a sharp cliff -- checked and
    rejected two attempts before this one: (1) trimming only where area is
    within 0.1% of its exact max still left area~324-327 in, where dV/dA
    already reached 4-80 (a still-visible kink in the final figure); (2)
    flagging |dV/dA| > 10x its own median blew up instead, because roughly
    half the domain sits in the saturated plateau, dragging the 'typical'
    estimate up to include the very anomaly it was meant to detect. Both
    failed because dV/dA divides by a near-zero area DIFFERENCE right where
    the problem is worst -- exactly the numerically unstable quantity.
    dArea/dh has no such instability (h is evenly gridded, never near-zero
    spacing), so use IT to find the genuinely-rising middle segment directly
    -- keep from the first to the last point where dA/dh exceeds 2% of its
    own peak, and restrict the area/volume arrays to that span before
    building the inverse interpolator. Poma's genuinely stepped survey (no
    direct area column, see tool/bathymetry.py's updated_curve) has brief
    flat treads throughout, but its first and last points both sit well
    inside its true rising range, so this doesn't truncate it."""
    area_i, vol_i = bt.updated_curve(name) or (None, None)
    if area_i is None or vol_i is None:
        area_i, vol_i = bt.design_curve(name)
    if area_i is None:
        return None, np.nan, np.nan
    h = np.linspace(float(area_i.x.min()), float(area_i.x.max()), 400)
    areas = area_i(h)
    vols = vol_i(h)
    dadh = np.gradient(areas, h)
    significant = np.flatnonzero(dadh > 0.02 * dadh.max())
    if len(significant):
        lo_i, hi_i = significant[0], significant[-1]
        areas, vols = areas[lo_i:hi_i + 1], vols[lo_i:hi_i + 1]
    f = interp1d(areas, vols, bounds_error=False, fill_value='extrapolate')
    return f, float(areas.min()), float(areas.max())


def load_area_series(name):
    fp, dcol, acol, _ = AREA_SERIES[name]
    df = pd.read_csv(REPO / fp)
    df['date'] = pd.to_datetime(df[dcol])
    df = df[['date', acol]].rename(columns={acol: 'area_ha'}).dropna().sort_values('date')
    return df.reset_index(drop=True)


def load_official(cod, kind):
    fp = MONTHLY_FP if kind == 'monthly' else DAILY_FP
    df = pd.read_csv(fp, encoding='utf-8')
    df = df[df.cod == cod].copy()
    if df.empty:
        return df.assign(date=pd.NaT, vol_official=np.nan)[['date', 'vol_official']]
    df['date'] = pd.to_datetime(df['data'])
    df['vol_official'] = df['volume'] if kind == 'monthly' else df['volume'] / 1e6
    return df[['date', 'vol_official']].dropna().sort_values('date').reset_index(drop=True)


def score(a, b):
    if len(a) < 4:
        return dict(n=len(a), rmse=np.nan, bias=np.nan, r=np.nan)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    bias = float(np.mean(a - b))
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return dict(n=len(a), rmse=round(rmse, 2), bias=round(bias, 2), r=round(r, 2))


rows = []
merged_monthly = {}
fig, axes = plt.subplots(len(AREA_SERIES), 1, figsize=(9, 2.6 * len(AREA_SERIES)), sharex=False)

for ax, (name, (_, _, _, cod)) in zip(axes, AREA_SERIES.items()):
    area_df = load_area_series(name)
    f_frs, area_min, area_max = frs_curve(name)
    f_cur, cur_area_min, cur_area_max = current_curve_area_to_vol(name)
    area_df['vol_frs'] = f_frs(area_df.area_ha)
    area_df['vol_current'] = f_cur(area_df.area_ha) if f_cur is not None else np.nan
    area_df['in_domain'] = area_df.area_ha.between(area_min, area_max)
    # Scored separately from the FRS domain: the independent reference curve's
    # own observed range can be materially narrower (Garcia's echosounder survey
    # tops out at 327 ha vs 381 ha in the full SAR record) -- extrapolating past
    # it blows up rather than giving a genuine estimate.
    area_df['in_domain_current'] = area_df.area_ha.between(cur_area_min, cur_area_max) \
        if f_cur is not None else False
    coverage_pct = round(100 * area_df.in_domain.mean(), 1)

    for kind, tol_days in (('monthly', 15), ('daily', 2)):
        off = load_official(cod, kind)
        if off.empty:
            continue
        merged = pd.merge_asof(area_df.sort_values('date'), off, on='date',
                                tolerance=pd.Timedelta(days=tol_days),
                                direction='nearest').dropna(subset=['vol_official'])
        in_dom = merged[merged.in_domain]
        s_frs = score(in_dom.vol_frs.values, in_dom.vol_official.values)
        in_dom_cur = merged[merged.in_domain_current]
        s_cur = score(in_dom_cur.vol_current.values, in_dom_cur.vol_official.values) if f_cur is not None \
            else dict(n=0, rmse=np.nan, bias=np.nan, r=np.nan)
        rows.append(dict(reservoir=name, record=kind, coverage_pct=coverage_pct,
                          **{f'frs_{k}': v for k, v in s_frs.items()},
                          **{f'current_{k}': v for k, v in s_cur.items()}))
        if kind == 'monthly':
            merged_monthly[name] = merged

    area_df.to_csv(OUT / f'area_volume_{name}.csv', index=False)

    # In-domain (observed band) plotted solid; out-of-domain (extrapolated
    # beyond what the reconstruction ever saw) plotted faint/dotted, same
    # convention as the unobserved-deep-zone lines elsewhere in the paper.
    in_seg = area_df.where(area_df.in_domain)
    out_seg = area_df.where(~area_df.in_domain)
    ax.plot(area_df.date, out_seg.vol_frs, color='#8e24aa', lw=1.0, ls=':', alpha=0.5)
    ax.plot(area_df.date, in_seg.vol_frs, color='#8e24aa', lw=1.6, label='Volume via FRS (SWOT-only) curve')
    if f_cur is not None:
        cur_in_seg = area_df.where(area_df.in_domain_current)
        cur_out_seg = area_df.where(~area_df.in_domain_current)
        ax.plot(area_df.date, cur_out_seg.vol_current, color='#2e7d32', lw=0.8, ls=':', alpha=0.5)
        ax.plot(area_df.date, cur_in_seg.vol_current, color='#2e7d32', lw=1.2, ls='--',
                 label='Volume via current-best curve')
    # Pre-2014 official monthly records predate every curve/area series plotted here
    # (SAR area starts 2014-2015 at the earliest) and only add empty-looking years of
    # dots the plotted lines never reach -- drop them rather than pad the x-axis.
    off_m = load_official(cod, 'monthly')
    off_m = off_m[off_m.date >= '2014-01-01']
    ax.scatter(off_m.date, off_m.vol_official, s=4, color='#333333', zorder=5, label='Official record (monthly)')
    ax.set_xlim(left=pd.Timestamp('2014-01-01'))
    ax.set_title(f'{name}  (FRS-curve coverage {coverage_pct:.0f}% of dates)', fontsize=10, loc='left')
    ax.set_ylabel('Volume (Mm$^3$)', fontsize=8)
    ax.tick_params(labelsize=7)
    # 'best' (rather than a fixed corner) lets matplotlib place the legend where it
    # overlaps the plotted lines least, panel by panel -- a fixed corner clipped
    # through the curves on at least one reservoir every time this was regenerated.
    ax.legend(fontsize=6.5, loc='best', framealpha=0.85)

fig.tight_layout()
fig.savefig(OUT / 'area_volume_timeseries.png', dpi=180)
plt.close(fig)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'area_volume_summary.csv', index=False)
pd.set_option('display.width', 160)
print(summary.to_string(index=False))
print(f'\nSaved to {OUT}')

# ── Diagnostic: AEV-space comparison -- area (x) vs volume (y), both curves,
# plus the real (area, official-volume) pairs -- shows directly whether the
# FRS curve diverges from real data by a shift or by a different slope/shape,
# rather than inferring it indirectly from a time series RMSE number.
fig2, axes2 = plt.subplots(1, len(AREA_SERIES), figsize=(4 * len(AREA_SERIES), 4.2))
for ax2, (name, (_, _, _, cod)) in zip(axes2, AREA_SERIES.items()):
    f_frs, area_min, area_max = frs_curve(name)
    f_cur, cur_area_min, cur_area_max = current_curve_area_to_vol(name)
    a_grid = np.linspace(area_min, area_max, 200)
    ax2.plot(a_grid, f_frs(a_grid), color='#8e24aa', lw=2, label='FRS (SWOT-only) curve')
    if f_cur is not None:
        a_grid_cur = np.linspace(cur_area_min, cur_area_max, 200)
        ax2.plot(a_grid_cur, f_cur(a_grid_cur), color='#2e7d32', lw=1.6, ls='--', label='Current-best curve')
    m = merged_monthly.get(name)
    if m is not None and len(m):
        in_m = m[m.in_domain]
        out_m = m[~m.in_domain]
        ax2.scatter(out_m.area_ha, out_m.vol_official, s=10, color='#999999', alpha=0.5,
                    label='Official record (out of FRS-curve range)')
        ax2.scatter(in_m.area_ha, in_m.vol_official, s=14, color='#333333',
                    label='Official record (in FRS-curve range)')
    ax2.set_title(name, fontsize=10)
    ax2.set_xlabel('Area (ha)', fontsize=8)
    ax2.set_ylabel('Volume (Mm$^3$)', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=6, loc='upper left')

fig2.tight_layout()
fig2.savefig(OUT / 'area_volume_curves_diag.png', dpi=180)
plt.close(fig2)
print(f'Saved diagnostic AEV-space figure to {OUT / "area_volume_curves_diag.png"}')
