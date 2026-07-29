"""
audit_mask_area_outliers.py (2026-07-29)

Cross-checks every currently-selected mask (selected_mask_dates.json, all 9
reservoirs x periods A/B) against its own reservoir's trusted continuous SAR
area series (Paper 1's pipeline output, CONFIGS[res]['sar_csv']), using the
SAME >60% deviation threshold already used to clean that continuous series
(select_mask_dates.py::remove_outliers' spike filter).

Found via the Poma densification prototype: two of Poma's ORIGINAL 10 B-period
masks (2026-02-12, 2026-03-26) collapse to ~73 ha in the exported GeoTIFF while
the continuous series and gauge-implied trend both show ~226/311 ha that same
date -- a windowed-export classification failure on a specific image, not a
real physical draw-down. This script checks whether the same failure mode
affects any of the other 8 reservoirs.
"""
import json, pathlib
import numpy as np
import pandas as pd
import rasterio

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
DATES_JSON = pathlib.Path('analysis/selected_mask_dates.json')
THRESH = 0.60

dates_json = json.loads(DATES_JSON.read_text())

all_rows = []
for res, cfg in m.CONFIGS.items():
    cont = pd.read_csv(cfg['sar_csv'], parse_dates=['date']).sort_values('date')
    cont = cont.groupby('date')['area_ha'].mean()

    for period in ('A', 'B'):
        for entry in dates_json.get(res, {}).get(period, []):
            date_str = entry['date']
            dt = pd.Timestamp(date_str)
            tif_path = MASK_DIR / f'mask_{res}_{date_str}.tif'
            if not tif_path.exists():
                all_rows.append({'reservoir': res, 'period': period, 'date': date_str,
                                  'mask_ha': None, 'continuous_ha': None, 'dev_pct': None,
                                  'flag': 'MISSING_TIF'})
                continue
            with rasterio.open(tif_path) as src:
                arr = src.read(1)
                mask_ha = (arr == 1).sum() * src.res[0] * src.res[1] / 10000

            cont_val = cont.reindex([dt], method='nearest', tolerance=pd.Timedelta(days=3))
            cont_ha = float(cont_val.iloc[0]) if len(cont_val.dropna()) else np.nan
            if np.isnan(cont_ha):
                all_rows.append({'reservoir': res, 'period': period, 'date': date_str,
                                  'mask_ha': round(mask_ha, 1), 'continuous_ha': None,
                                  'dev_pct': None, 'flag': 'NO_CONTINUOUS_MATCH'})
                continue
            dev = abs(mask_ha - cont_ha) / cont_ha
            all_rows.append({
                'reservoir': res, 'period': period, 'date': date_str,
                'mask_ha': round(mask_ha, 1), 'continuous_ha': round(cont_ha, 1),
                'dev_pct': round(dev * 100, 1),
                'flag': 'OUTLIER' if dev > THRESH else 'ok',
            })

out = pd.DataFrame(all_rows)
out.to_csv('analysis/schwatke_output/mask_area_outlier_audit.csv', index=False)

print(f'Audited {len(out)} mask/date entries across {out["reservoir"].nunique()} reservoirs.\n')
bad = out[out['flag'] != 'ok']
if len(bad):
    print(f'{len(bad)} flagged entries:')
    print(bad.to_string(index=False))
else:
    print('No flagged entries.')
print(f'\nSaved full audit to analysis/schwatke_output/mask_area_outlier_audit.csv')
