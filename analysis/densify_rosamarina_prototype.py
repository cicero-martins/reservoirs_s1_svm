"""
densify_rosamarina_prototype.py — bathymetry-densification prototype,
reservoir 2 (2026-07-29). Mirrors densify_poma_prototype.py; see that file
for the general rationale. Rosamarina's declared gauge_bad_window
(2025-07-24 to 2026-01-29) overlaps most of this window, so unlike Poma,
many dates here get their level from SWOT rather than the gauge -- a real
external source either way, just the mixed-source stress test.

Also runs the Paper-1-style area-outlier cross-check (>60% deviation from
the continuous SAR series) up front on ALL 41 candidates, since that check
was added AFTER Poma's initial prototype and found 2 bad production masks
there -- applying it before building anything this time.
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

RES = 'Rosamarina'
MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')
cfg = m.CONFIGS[RES]

dates_json = json.loads(m.DATES_JSON.read_text())
orig_entries = dates_json[RES]['B']
orig_dates = [e['date'] for e in orig_entries]
orig_area = {e['date']: e['area_ha'] for e in orig_entries}
new_dates = ['2025-09-16', '2025-09-28', '2025-10-16', '2025-10-22', '2025-10-28',
             '2025-11-03', '2025-11-09', '2025-11-15', '2025-12-03', '2025-12-09',
             '2025-12-21', '2025-12-27', '2026-01-02', '2026-01-08', '2026-01-14',
             '2026-01-20', '2026-01-26', '2026-02-07', '2026-02-25', '2026-03-03',
             '2026-03-09', '2026-03-21', '2026-03-27', '2026-04-08', '2026-04-14',
             '2026-04-20', '2026-04-21', '2026-04-26', '2026-05-03', '2026-05-08',
             '2026-05-14']
all_dates = sorted(set(orig_dates) | set(new_dates))
print(f'{RES}: {len(orig_dates)} original + {len(new_dates)} new = {len(all_dates)} total dates')

sar = pd.read_csv(cfg['sar_csv'], parse_dates=['date'])
sar = sar[['date', 'area_ha']].dropna().set_index('date').sort_index()['area_ha']
sar = sar.groupby(sar.index).mean()

gauge = m.load_gauge(cfg)
swot = pd.Series(dtype=float)
if 'swot_csv' in cfg and cfg['swot_csv'].exists():
    swot = m.load_swot_corrected(cfg, cfg['swot_csv'], RES)

gauge_bad = cfg.get('gauge_bad_window')
bad_windows = []
if gauge_bad:
    raw = gauge_bad if isinstance(gauge_bad[0], (tuple, list)) else [gauge_bad]
    bad_windows = [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in raw]

pairs_orig = m.match_sar_gauge(sar, gauge, orig_entries)
model = m.fit_hyps_model(pairs_orig, cfg['h0_bound_lo'])
if model is not None:
    a, h0, b = model
    print(f'Model fit on {len(pairs_orig.dropna())} ORIGINAL pairs: '
          f'A = {a:.3f}*(h-{h0:.2f})^{b:.3f}')
else:
    a, h0, b = None, None, None
    print(f'Model fit FAILED on {len(pairs_orig.dropna())} ORIGINAL pairs '
          f'(expected: most fall inside the gauge_bad_window) -- model fallback disabled, '
          f'relying on gauge+SWOT coverage only')

rows = []
for date_str in all_dates:
    dt = pd.Timestamp(date_str)
    tif_path = MASK_DIR / f'mask_{RES}_{date_str}.tif'
    if not tif_path.exists():
        continue
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        mask_ha = (arr == 1).sum() * src.res[0] * src.res[1] / 10000

    if date_str in orig_area:
        area_ha = orig_area[date_str]
    else:
        near = sar[(sar.index >= dt - pd.Timedelta(days=2)) & (sar.index <= dt + pd.Timedelta(days=2))]
        area_ha = float(near.iloc[(near.index - dt).to_series().abs().values.argmin()]) if len(near) else np.nan

    cont_near = sar[(sar.index >= dt - pd.Timedelta(days=3)) & (sar.index <= dt + pd.Timedelta(days=3))]
    cont_ha = float(cont_near.iloc[(cont_near.index - dt).to_series().abs().values.argmin()]) if len(cont_near) else np.nan
    dev_pct = abs(mask_ha - cont_ha) / cont_ha * 100 if cont_ha and not np.isnan(cont_ha) else np.nan
    outlier = bool(dev_pct is not None and not np.isnan(dev_pct) and dev_pct > 60)

    in_bad = any(lo <= dt <= hi for lo, hi in bad_windows)
    wl_m, source = np.nan, 'none'
    if not in_bad:
        val = m.interp_wl(gauge, dt, m.MAX_DT)
        if not np.isnan(val):
            wl_m, source = val, 'gauge'
    if np.isnan(wl_m) and len(swot) > 0:
        val = m.interp_wl(swot, dt, m.MAX_DT)
        if not np.isnan(val):
            wl_m, source = val, 'swot'
    if np.isnan(wl_m) and in_bad:
        val = m.interp_wl(gauge, dt, m.MAX_DT)
        if not np.isnan(val):
            wl_m, source = val, 'swot'
    if np.isnan(wl_m) and a is not None and not np.isnan(area_ha):
        wl_m, source = m.invert_power_law(area_ha, a, h0, b), 'model'

    rows.append({'date': date_str, 'mask_ha': round(mask_ha, 1), 'area_ha': area_ha,
                  'continuous_ha': round(cont_ha, 1) if not np.isnan(cont_ha) else None,
                  'dev_pct': round(dev_pct, 1) if dev_pct is not None and not np.isnan(dev_pct) else None,
                  'outlier': outlier, 'wl_m': wl_m, 'source': source,
                  'is_new': date_str in new_dates})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f'\nSource breakdown (new dates only):\n{df[df["is_new"]]["source"].value_counts().to_string()}')
print(f'\nOutliers flagged: {df[df["outlier"]]["date"].tolist()}')

df.to_csv(OUT_DIR / 'rosamarina_densify_prototype_pairs.csv', index=False, float_format='%.4f')
print(f'\nSaved {OUT_DIR / "rosamarina_densify_prototype_pairs.csv"}')
