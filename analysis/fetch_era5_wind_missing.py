"""Fetch ERA5 hourly wind at S1 overpass times for the reservoirs in the
current 4-way global set (pilot_kge_4way.csv) that lack a wind file, so the
wind analysis can be redone on the FULL current set instead of the old N=28.
Ports exportEra5Wind.js to EE Python (evaluated locally, no Drive export).
"""
import os, sys
import pandas as pd
import truststore
truststore.inject_into_ssl()
import ee

ee.Initialize(project='ee-ciceromartinsjr')

HERE = os.path.dirname(os.path.abspath(__file__))
WINDDIR = os.path.join(HERE, '..', 'raw_data', 'GEE_GlobalPilotV4b', 'GEE_Era5Wind')
os.makedirs(WINDDIR, exist_ok=True)

fw = pd.read_csv(os.path.join(HERE, 'pilot_kge_4way.csv'))
existing = {f[len('Era5Wind_'):-4] for f in os.listdir(WINDDIR) if f.startswith('Era5Wind_')}
missing = sorted(set(fw.name) - existing)
print(f"{len(missing)} reservoirs missing wind data:", missing)

cand = pd.read_csv(os.path.join(HERE, 'global_pilot_v4_candidates.csv')).set_index('name')
coord = {n: (r.lat, r.lon) for n, r in cand.iterrows()}

ERA5 = ee.ImageCollection('ECMWF/ERA5/HOURLY')
U, V = 'u_component_of_wind_10m', 'v_component_of_wind_10m'
S1 = ee.ImageCollection('COPERNICUS/S1_GRD')


def wind_at_overpass(img, point):
    t = ee.Date(img.get('system:time_start'))
    era = ee.Image(ERA5.filterDate(t.advance(-1, 'hour'), t.advance(1, 'hour')).first())
    uv = era.select([U, V]).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=27830, bestEffort=True)
    u, v = ee.Number(uv.get(U)), ee.Number(uv.get(V))
    spd = u.pow(2).add(v.pow(2)).sqrt()
    return ee.Feature(None, {
        'date': t.format('YYYY-MM-dd'),
        'datetime_utc': t.format('YYYY-MM-dd HH:mm'),
        'wind_ms': spd,
    })


done, failed = [], []
for name in missing:
    if name not in coord:
        print(f"  [skip] {name}: no coordinates in candidates CSV")
        failed.append(name)
        continue
    lat, lon = coord[name]
    point = ee.Geometry.Point([lon, lat])
    s1 = (S1.filterBounds(point).filterDate('2014-10-01', '2021-12-31')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.eq('resolution_meters', 10))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    fc = ee.FeatureCollection(s1.map(lambda img: wind_at_overpass(img, point))) \
           .filter(ee.Filter.notNull(['wind_ms']))
    try:
        info = fc.getInfo()
        rows = [f['properties'] for f in info['features']]
        if not rows:
            print(f"  [warn] {name}: 0 acquisitions returned")
            failed.append(name)
            continue
        out = pd.DataFrame(rows)[['date', 'datetime_utc', 'wind_ms']]
        out.to_csv(os.path.join(WINDDIR, f'Era5Wind_{name}.csv'), index=False)
        print(f"  [ok] {name}: {len(out)} rows")
        done.append(name)
    except Exception as e:
        print(f"  [fail] {name}: {repr(e)[:150]}")
        failed.append(name)

print(f"\nDone: {len(done)}  Failed: {len(failed)}")
if failed:
    print("Failed list:", failed)
