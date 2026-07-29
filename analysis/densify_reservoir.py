"""
densify_reservoir.py (2026-07-29)

Generalized version of densify_poma_prototype.py / densify_rosamarina_prototype.py:
for a given reservoir, pairs WL for the FULL candidate mask pool (production B dates
+ any additional masks now on disk within the same reconstruction window), using the
same source-priority chain as schwatke_bathymetry_3d.phase1() (gauge, unless a known
bad window applies, then SWOT, then curve-inversion as a last resort). Also runs both
QA checks (area-deviation-from-continuous-series, area-vs-level monotonicity) on the
resulting pool so bad dates are flagged before any DEM gets built.

Run:
    python analysis/densify_reservoir.py Pozzillo
"""
import argparse, json, pathlib, re, sys
import numpy as np
import pandas as pd
import rasterio

DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m
from export_windowed_masks import RESERVOIRS as WINDOWED_RESERVOIRS

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')
BY_NAME = {r[0]: r for r in WINDOWED_RESERVOIRS}


def main(res):
    cfg = m.CONFIGS[res]
    _, lat, lon, hylak_id, win_start, win_end, known_orbit = BY_NAME[res]

    dates_json = json.loads(m.DATES_JSON.read_text())
    orig_entries = dates_json[res]['B']
    orig_dates = [e['date'] for e in orig_entries]
    orig_area = {e['date']: e['area_ha'] for e in orig_entries}

    # All mask files on disk within the reconstruction window (production B dates +
    # anything else the densification export added), rather than a hardcoded list.
    win_lo, win_hi = pd.Timestamp(win_start), pd.Timestamp(win_end)
    candidate_stems = (p.stem.replace(f'mask_{res}_', '') for p in MASK_DIR.glob(f'mask_{res}_*.tif'))
    all_dates = sorted({
        d for d in candidate_stems
        if DATE_RE.match(d) and win_lo <= pd.Timestamp(d) <= win_hi
    })
    new_dates = [d for d in all_dates if d not in orig_dates]
    print(f'{res}: {len(orig_dates)} original + {len(new_dates)} new = {len(all_dates)} total dates '
          f'in window {win_start}..{win_end}')

    sar = pd.read_csv(cfg['sar_csv'], parse_dates=['date'])
    sar = sar[['date', 'area_ha']].dropna().set_index('date').sort_index()['area_ha']
    sar = sar.groupby(sar.index).mean()

    gauge = m.load_gauge(cfg)
    swot = pd.Series(dtype=float)
    if 'swot_csv' in cfg and cfg['swot_csv'].exists():
        swot = m.load_swot_corrected(cfg, cfg['swot_csv'], res)

    gauge_bad = cfg.get('gauge_bad_window')
    bad_windows = []
    if gauge_bad:
        raw = gauge_bad if isinstance(gauge_bad[0], (tuple, list)) else [gauge_bad]
        bad_windows = [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in raw]
    swot_first = cfg.get('swot_priority', False)

    pairs_orig = m.match_sar_gauge(sar, gauge, orig_entries)
    model = m.fit_hyps_model(pairs_orig, cfg['h0_bound_lo'])
    if model is not None:
        a, h0, b = model
        print(f'Model fit on {len(pairs_orig.dropna())} ORIGINAL pairs: '
              f'A = {a:.3f}*(h-{h0:.2f})^{b:.3f}')
    else:
        a, h0, b = None, None, None
        print(f'Model fit FAILED on {len(pairs_orig.dropna())} ORIGINAL pairs -- '
              f'model fallback disabled, relying on gauge+SWOT coverage only')

    rows = []
    for date_str in all_dates:
        dt = pd.Timestamp(date_str)
        tif_path = MASK_DIR / f'mask_{res}_{date_str}.tif'
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            mask_ha = (arr == 1).sum() * src.res[0] * src.res[1] / 10000

        if date_str in orig_area and not np.isnan(orig_area[date_str]):
            area_ha = orig_area[date_str]
        else:
            near = sar[(sar.index >= dt - pd.Timedelta(days=2)) & (sar.index <= dt + pd.Timedelta(days=2))]
            area_ha = float(near.iloc[(near.index - dt).to_series().abs().values.argmin()]) if len(near) else np.nan
            if np.isnan(area_ha):
                # Continuous series doesn't reach this date (Arancio/Castello/Olivo/
                # Nicoletti's archive currently stops Dec-2025) -- fall back to the
                # mask's own pixel-derived area rather than leave it unusable.
                area_ha = mask_ha

        cont_near = sar[(sar.index >= dt - pd.Timedelta(days=3)) & (sar.index <= dt + pd.Timedelta(days=3))]
        cont_ha = float(cont_near.iloc[(cont_near.index - dt).to_series().abs().values.argmin()]) if len(cont_near) else np.nan
        dev_pct = abs(mask_ha - cont_ha) / cont_ha * 100 if cont_ha else np.nan
        outlier = bool(not np.isnan(dev_pct) and dev_pct > 60)

        in_bad = any(lo <= dt <= hi for lo, hi in bad_windows)
        wl_m, source = np.nan, 'none'
        if swot_first and len(swot) > 0:
            val = m.interp_wl(swot, dt, m.MAX_DT)
            if not np.isnan(val):
                wl_m, source = val, 'swot'
        if np.isnan(wl_m) and not in_bad:
            val = m.interp_wl(gauge, dt, m.MAX_DT)
            if not np.isnan(val):
                wl_m, source = val, 'gauge'
        if np.isnan(wl_m) and len(swot) > 0:
            val = m.interp_wl(swot, dt, m.MAX_DT)
            if not np.isnan(val):
                wl_m, source = val, 'swot'
        if np.isnan(wl_m) and a is not None and not np.isnan(area_ha):
            wl_m, source = m.invert_power_law(area_ha, a, h0, b), 'model'

        rows.append({'date': date_str, 'mask_ha': round(mask_ha, 1), 'area_ha': area_ha,
                      'continuous_ha': round(cont_ha, 1) if not np.isnan(cont_ha) else None,
                      'dev_pct': round(dev_pct, 1) if not np.isnan(dev_pct) else None,
                      'area_outlier': outlier, 'wl_m': wl_m, 'source': source,
                      'is_new': date_str in new_dates})

    df = pd.DataFrame(rows)

    # Monotonicity check (level up, area down), restricted to temporally-close
    # (<=45d) WL-adjacent pairs -- see audit_area_level_monotonicity.py.
    valid = df.dropna(subset=['wl_m']).sort_values('wl_m').reset_index(drop=True)
    valid['date_dt'] = pd.to_datetime(valid['date'])
    valid['area_prev'] = valid['area_ha'].shift(1)
    valid['wl_prev'] = valid['wl_m'].shift(1)
    valid['date_prev'] = valid['date'].shift(1)
    valid['day_gap'] = (valid['date_dt'] - pd.to_datetime(valid['date_prev'])).abs().dt.days
    valid['area_drop_pct'] = (valid['area_prev'] - valid['area_ha']) / valid['area_prev'] * 100
    mono_flag = valid[(valid['wl_m'] > valid['wl_prev']) & (valid['area_drop_pct'] > 0)
                       & (valid['day_gap'] <= 45)]

    print(f'\nSource breakdown (new dates only):\n{df[df["is_new"]]["source"].value_counts().to_string()}')
    print(f'\nArea-outlier flags (>60% dev from continuous series): '
          f'{df[df["area_outlier"]]["date"].tolist()}')
    print(f'\nMonotonicity flags (level up, area down, <=45d gap):')
    if len(mono_flag):
        print(mono_flag[['date', 'date_prev', 'day_gap', 'area_ha', 'area_prev',
                          'wl_m', 'wl_prev', 'area_drop_pct', 'source']].to_string(index=False))
    else:
        print('  none')

    out_csv = OUT_DIR / f'{res.lower()}_densify_prototype_pairs.csv'
    df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f'\nSaved {out_csv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reservoir')
    args = ap.parse_args()
    main(args.reservoir)
