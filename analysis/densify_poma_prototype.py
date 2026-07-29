"""
densify_poma_prototype.py — bathymetry-densification prototype (2026-07-29)

Rebuilds Poma's Period-B DEM using ALL 26 available Sentinel-1 masks in the
window (10 originally selected + 16 exported by export_poma_densify.py),
instead of the production 10. Water level for each date uses the SAME
source-priority chain as schwatke_bathymetry_3d.phase1() (gauge first, SWOT
fallback, curve-inversion last resort). The hypsometric curve used for that
fallback is fit ONLY on the 10 ORIGINAL gauge-paired points -- the densified
dates must never feed back into the curve used to infer them, to avoid
circularity with the AEV-curve validation.

Writes to SEPARATE outputs (dem_Poma_B_densified.tif,
poma_densify_prototype_pairs.csv) -- does not touch selected_mask_dates.json,
mask_wl_pairs_Poma.csv, or dem_Poma_B.tif, so the manuscript's Poma numbers
are unaffected.
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

RES = 'Poma'
MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')
cfg = m.CONFIGS[RES]

dates_json = json.loads(m.DATES_JSON.read_text())
orig_entries = dates_json[RES]['B']
orig_dates = [e['date'] for e in orig_entries]
orig_area = {e['date']: e['area_ha'] for e in orig_entries}
new_dates = ['2025-12-20', '2025-12-26', '2026-01-07', '2026-01-13', '2026-01-19',
             '2026-03-02', '2026-03-08', '2026-03-14', '2026-03-20', '2026-04-01',
             '2026-04-13', '2026-04-19', '2026-04-20', '2026-04-25', '2026-05-01',
             '2026-05-07']
all_dates = sorted(set(orig_dates) | set(new_dates))
print(f'{RES}: {len(orig_dates)} original + {len(new_dates)} new = {len(all_dates)} total dates')

sar = pd.read_csv(cfg['sar_csv'], parse_dates=['date'])
sar = sar[['date', 'area_ha']].dropna().set_index('date').sort_index()['area_ha']

gauge = m.load_gauge(cfg)
swot = pd.Series(dtype=float)
if 'swot_csv' in cfg and cfg['swot_csv'].exists():
    swot = m.load_swot_corrected(cfg, cfg['swot_csv'], RES)

pairs_orig = m.match_sar_gauge(sar, gauge, orig_entries)
a, h0, b = m.fit_hyps_model(pairs_orig, cfg['h0_bound_lo'])
print(f'Model fit on {len(pairs_orig.dropna())} ORIGINAL pairs: '
      f'A = {a:.3f}*(h-{h0:.2f})^{b:.3f}')

rows = []
for date_str in all_dates:
    dt = pd.Timestamp(date_str)
    if date_str in orig_area:
        area_ha = orig_area[date_str]
    else:
        near = sar[(sar.index >= dt - pd.Timedelta(days=2)) & (sar.index <= dt + pd.Timedelta(days=2))]
        area_ha = float(near.iloc[(near.index - dt).to_series().abs().values.argmin()]) if len(near) else np.nan

    wl_m, source = np.nan, 'none'
    val = m.interp_wl(gauge, dt, m.MAX_DT)
    if not np.isnan(val):
        wl_m, source = val, 'gauge'
    if np.isnan(wl_m) and len(swot) > 0:
        val = m.interp_wl(swot, dt, m.MAX_DT)
        if not np.isnan(val):
            wl_m, source = val, 'swot'
    if np.isnan(wl_m) and a is not None and not np.isnan(area_ha):
        wl_m, source = m.invert_power_law(area_ha, a, h0, b), 'model'

    rows.append({'date': date_str, 'area_ha': area_ha, 'wl_m': wl_m, 'source': source,
                 'is_new': date_str in new_dates})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nSource breakdown (new dates only):\n"
      f"{df[df['is_new']]['source'].value_counts().to_string()}")

raw_arrays, wls, meta = [], [], None
for _, row in df.dropna(subset=['wl_m']).iterrows():
    tif_path = MASK_DIR / f'mask_{RES}_{row["date"]}.tif'
    if not tif_path.exists():
        print(f'  MISSING: {tif_path.name}'); continue
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        if meta is None:
            meta = src.meta.copy()
    raw_arrays.append(arr)
    wls.append(row['wl_m'])

print(f'\n{len(raw_arrays)} masks with valid WL -> building densified DEM...')
dem = m.build_dem_from_arrays(raw_arrays, wls)

out_meta = meta.copy()
out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
out_path = OUT_DIR / f'dem_{RES}_B_densified.tif'
with rasterio.open(out_path, 'w', **out_meta) as dst:
    dst.write(dem[np.newaxis, :, :])
print(f'Saved {out_path}  WL range {min(wls):.1f}-{max(wls):.1f} m  '
      f'depth range {np.nanmin(dem):.1f}-{max(wls):.1f} m  n_masks={len(wls)}')

df.to_csv(OUT_DIR / 'poma_densify_prototype_pairs.csv', index=False, float_format='%.4f')
print(f'Saved {OUT_DIR / "poma_densify_prototype_pairs.csv"}')
