"""
check_jrc_candidates.py

Vet the GDW candidate shortlist for JRC time-series availability BEFORE committing them
to the export batch. For each candidate (by GDW_ID → GDW polygon) it queries, via the
Earth Engine Python API, whether a usable JRC reference series exists over 2015-2021:

  water_max_ha : JRC max_extent water area inside the polygon (is there a reservoir at all?)
  n_valid_mon  : # MonthlyHistory months with valid_frac ≥ 0.80 (observed, not cloud/gap)
  jrc_mean_ha  : mean monthly JRC water area over the valid months
  jrc_cv       : coeff. of variation of that area (KGE needs the obs to VARY; a flat
                 series gives a meaningless/degenerate KGE)
  verdict      : OK if water_max ≥ 150 ha AND n_valid ≥ 24 AND jrc_cv ≥ 0.03

Mirrors the export's polygon source (GDW asset, GDW_ID) and JRC product (GSW 1_4).

Reads:  analysis/gdw_proposal.csv   (name, GDW_ID, …)
Output: analysis/gdw_proposal_jrc.csv  (+ printed table)
"""

import sys
import pandas as pd

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import ee
sys.stdout.reconfigure(encoding='utf-8')
EE_PROJECT = 'ee-ciceromartinsjr'
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

GDW     = ee.FeatureCollection('projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0')
JRC_GSW = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
MH      = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory').filterDate('2015-01-01', '2021-12-31')
PIXA    = ee.Image.pixelArea()

IN_CSV = sys.argv[1] if len(sys.argv) > 1 else 'analysis/gdw_proposal.csv'
prop = pd.read_csv(IN_CSV)
rows = []
for _, r in prop.iterrows():
    gid, name = int(r['GDW_ID']), r['name']
    try:
        poly = GDW.filter(ee.Filter.eq('GDW_ID', gid)).first().geometry()
        poly_ha = poly.area(1).divide(1e4)

        # JRC max-extent water inside the polygon
        wmax = (JRC_GSW.select('max_extent').eq(1).multiply(PIXA)
                .reduceRegion(ee.Reducer.sum(), poly, 30, maxPixels=1e9, bestEffort=True)
                .getNumber('max_extent').divide(1e4))

        # per-month: water area (value==2) and valid fraction (value>=1 = observed)
        def per_month(img):
            water = img.eq(2)
            obs   = img.gte(1)
            wha = (water.multiply(PIXA)
                   .reduceRegion(ee.Reducer.sum(), poly, 30, maxPixels=1e9, bestEffort=True)
                   .getNumber('water'))
            ofrac = (obs.multiply(PIXA)
                     .reduceRegion(ee.Reducer.sum(), poly, 30, maxPixels=1e9, bestEffort=True)
                     .getNumber('water')).divide(poly.area(1))
            return ee.Feature(None, {'wha': ee.Number(wha).divide(1e4),
                                     'vfrac': ofrac})
        fc = ee.FeatureCollection(MH.map(per_month))
        valid = fc.filter(ee.Filter.gte('vfrac', 0.80))
        n_valid = valid.size()
        stats = valid.aggregate_stats('wha') if False else None
        wmean = valid.aggregate_mean('wha')
        wstd  = valid.aggregate_total_sd('wha')

        info = ee.Dictionary({
            'poly_ha': poly_ha, 'wmax': wmax, 'n_valid': n_valid,
            'wmean': wmean, 'wstd': wstd,
        }).getInfo()

        wmean_v = info['wmean'] or 0.0
        wstd_v  = info['wstd'] or 0.0
        cv = (wstd_v / wmean_v) if wmean_v else 0.0
        ok = (info['wmax'] or 0) >= 150 and (info['n_valid'] or 0) >= 24 and cv >= 0.03
        rows.append({'name': name, 'GDW_ID': gid, 'region': r['region'],
                     'poly_ha': round(info['poly_ha'] or 0, 0),
                     'water_max_ha': round(info['wmax'] or 0, 0),
                     'n_valid_mon': int(info['n_valid'] or 0),
                     'jrc_mean_ha': round(wmean_v, 0),
                     'jrc_cv': round(cv, 3),
                     'verdict': 'OK' if ok else 'CHECK'})
        print(f"{name:<22} wmax={info['wmax'] or 0:>6.0f}ha  n_valid={int(info['n_valid'] or 0):>2}  "
              f"mean={wmean_v:>6.0f}ha  cv={cv:>.2f}  -> {'OK' if ok else 'CHECK'}")
    except Exception as e:
        rows.append({'name': name, 'GDW_ID': gid, 'region': r['region'],
                     'verdict': f'ERROR: {str(e)[:40]}'})
        print(f'{name:<22} ERROR: {str(e)[:60]}')

out = pd.DataFrame(rows)
out.to_csv('analysis/gdw_proposal_jrc.csv', index=False)
n_ok = (out['verdict'] == 'OK').sum()
print(f'\n{n_ok}/{len(out)} candidates have a usable JRC series (OK). '
      f'Saved -> analysis/gdw_proposal_jrc.csv')
