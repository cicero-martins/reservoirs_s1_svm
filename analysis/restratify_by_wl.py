"""
restratify_by_wl.py  (diagnostic / date-selection preview -- no GEE export)

Fixes the "staircase" artifact from area-percentile date selection (see
2026-07-21 audit, diag_wl_source_gap.py): for the 8 reservoirs with real
candidate-image slack (2.6x-5.7x more images available than the 10 used),
recomputes the optimal 10 Period-B dates by WATER-LEVEL percentile instead
of area percentile, using the same gauge/SWOT source-priority as phase1().
Ancipa is excluded (revisit-limited: candidates == dates already used, no
selection can help).

Prints old vs new date lists with inter-level WL gaps, for review BEFORE
submitting any new GEE export tasks.
"""
import sys, pathlib
import numpy as np
import pandas as pd

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
import ee

try:
    ee.Initialize(project='ee-ciceromartinsjr')
except Exception:
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='ee-ciceromartinsjr')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m
from export_windowed_masks import get_lake_poly, select_best_orbit, S1_GRD

sys.stdout.reconfigure(encoding='utf-8')

# (name, lat, lon, hylak_id, win_start, win_end, known_orbit_or_None) -- identical
# to export_windowed_masks.py's RESERVOIRS so the orbit/pass this script resolves
# (via the SAME select_best_orbit angle-based logic) matches what generated the
# already-downloaded masks exactly, rather than guessing from a raw image count.
RESERVOIRS = [
    ('Poma',       38.011037, 13.056135, 173610, '2025-12-15', '2026-05-10', None),
    ('Rosamarina', 37.960336, 13.654665, 173633, '2025-09-15', '2026-05-15', None),
    ('Castello',   37.582494, 13.420304, None,   '2025-09-15', '2026-04-25', None),
    ('Olivo',      37.405048, 14.286604, None,   '2025-09-25', '2026-04-05', None),
    ('Arancio',    37.634491, 13.065184, None,   '2025-09-01', '2026-04-15', None),
    ('Nicoletti',  37.604822, 14.346314, None,   '2025-10-15', '2026-03-20', None),
    ('Pozzillo',   37.674037, 14.610613, 173729, '2025-10-01', '2026-03-25', 117),
    ('Garcia',     37.799,    13.119,    None,   '2025-08-06', '2026-05-31', None),
]
N_DATES = 10


def candidate_dates(lat, lon, hylak_id, win_start, win_end, known_orbit):
    lake_poly = get_lake_poly(lat, lon, hylak_id)
    aoi = lake_poly.buffer(100)
    s1_raw = (S1_GRD.filterBounds(aoi).filterDate(win_start, win_end)
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.eq('resolution_meters', 10))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    if known_orbit is not None:
        orbit, pass_dir = known_orbit, 'ASCENDING'
    else:
        orbit, pass_dir = select_best_orbit(s1_raw, aoi)
    col = (s1_raw.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit))
           .filter(ee.Filter.eq('orbitProperties_pass', pass_dir)))
    ts = col.aggregate_array('system:time_start').getInfo()
    dates = sorted(set(pd.Timestamp(t, unit='ms').strftime('%Y-%m-%d') for t in ts))
    print(f'    (resolved orbit={orbit} {pass_dir})')
    return dates


def wl_for_date(res, cfg, dt, gauge, swot, bad_windows):
    in_bad = any(lo <= dt <= hi for lo, hi in bad_windows)
    if len(gauge) > 0 and not in_bad:
        val = m.interp_wl(gauge, dt, m.MAX_DT)
        if not np.isnan(val):
            return val, 'gauge'
    if len(swot) > 0:
        val = m.interp_wl(swot, dt, m.MAX_DT)
        if not np.isnan(val):
            return val, 'swot'
    if len(gauge) > 0:
        val = m.interp_wl(gauge, dt, m.MAX_DT)
        if not np.isnan(val):
            return val, 'gauge(bad-window fallback)'
    return np.nan, 'none'


def stratify_by_wl(cand, n=N_DATES):
    """cand: list of (date_str, wl, source). Pick n dates evenly spaced in WL
    VALUE (not in percentile-of-the-candidate-distribution): targets are a
    linspace over the [5th,95th] percentile WL range, so a target's spacing
    directly controls the resulting DEM step size regardless of how densely
    candidate images happen to cluster in any sub-range (percentile-of-
    distribution targets would instead reproduce that clustering -- e.g.
    oversampling a slow/flat stretch and leaving the fast transition just as
    under-sampled as area-percentile did)."""
    valid = [c for c in cand if not np.isnan(c[1])]
    if len(valid) == 0:
        return []
    wls = np.array([c[1] for c in valid])
    wl_lo, wl_hi = np.percentile(wls, [5, 95])
    targets = np.linspace(wl_lo, wl_hi, min(n, len(valid)))
    used = set(); picked = []
    for target in targets:
        best_i, best_d = None, np.inf
        for i, (d, w, s) in enumerate(valid):
            if d in used:
                continue
            diff = abs(w - target)
            if diff < best_d:
                best_d, best_i = diff, i
        if best_i is not None:
            used.add(valid[best_i][0])
            picked.append(valid[best_i])
    return sorted(picked, key=lambda x: x[1])


def old_wl_pairs(res):
    f = m.OUT_DIR / f'mask_wl_pairs_{res}.csv'
    df = pd.read_csv(f, parse_dates=['date'])
    sub = df[df.period == 'B'].dropna(subset=['wl_m']).sort_values('wl_m')
    return list(zip(sub['date'].dt.strftime('%Y-%m-%d'), sub['wl_m'], sub['wl_source']))


def report_gaps(label, entries):
    if len(entries) < 2:
        print(f'  {label}: <2 valid entries'); return None
    wls = sorted(w for _, w, _ in entries)
    gaps = np.diff(wls)
    print(f'  {label}: n={len(entries)}  range={wls[0]:.2f}-{wls[-1]:.2f} m  '
          f'max_gap={gaps.max():.2f} m  mean_gap={gaps.mean():.2f} m')
    return gaps.max()


if __name__ == '__main__':
    summary = []
    for name, lat, lon, hylak_id, ws, we, known_orbit in RESERVOIRS:
        print(f'\n=== {name} ===')
        cfg = m.CONFIGS[name]
        try:
            gauge = m.load_gauge(cfg)
        except Exception:
            gauge = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        swot_f = pathlib.Path('validation_data/SWOT') / f'{name}_swot.csv'
        swot = m.load_swot(swot_f) if swot_f.exists() else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        gauge_bad = cfg.get('gauge_bad_window')
        bad_windows = []
        if gauge_bad:
            raw = gauge_bad if isinstance(gauge_bad[0], (tuple, list)) else [gauge_bad]
            bad_windows = [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in raw]

        dates = candidate_dates(lat, lon, hylak_id, ws, we, known_orbit)
        cand = []
        for d in dates:
            dt = pd.Timestamp(d)
            wl, src = wl_for_date(name, cfg, dt, gauge, swot, bad_windows)
            cand.append((d, wl, src))
        n_valid = sum(1 for _, w, _ in cand if not np.isnan(w))
        print(f'  candidates: {len(dates)} images, {n_valid} with a valid WL')

        new_sel = stratify_by_wl(cand, N_DATES)
        old_sel = old_wl_pairs(name)

        print('  OLD (area-percentile):')
        old_max_gap = report_gaps('old', old_sel)
        for d, w, s in old_sel:
            print(f'    {d}  wl={w:.2f}  src={s}')

        print('  NEW (WL-percentile):')
        new_max_gap = report_gaps('new', new_sel)
        for d, w, s in new_sel:
            print(f'    {d}  wl={w:.2f}  src={s}')

        new_dates_set = set(d for d, _, _ in new_sel)
        old_dates_set = set(d for d, _, _ in old_sel)
        added = new_dates_set - old_dates_set
        print(f'  New exports needed: {len(added)}  ({sorted(added)})')

        summary.append(dict(reservoir=name, n_candidates=len(dates), n_valid_wl=n_valid,
                            old_max_gap=old_max_gap, new_max_gap=new_max_gap,
                            n_new_exports=len(added)))

    print('\n=== Summary ===')
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
