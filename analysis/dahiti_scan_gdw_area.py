"""
Cross-reference DAHITI scan results with GDW to get reservoir area.

For each DAHITI target coordinate, finds the GDW polygon that:
  (a) intersects the point, OR
  (b) has centroid within 5 km if no intersection

Adds columns: area_km2_gee, area_ha_gee, gdw_id, gdw_name, gdw_country

Outputs:
  validation_data/DAHITI/dahiti_reservoir_scan_results.csv  (updated in-place)
  validation_data/DAHITI/dahiti_scan_medium.csv  (5 <= area_km2 <= 100, i.e. 500-10,000 ha)

Run from project root:
  python analysis/dahiti_scan_gdw_area.py
"""

import csv, math, sys, time
import ee
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# ── GEE initialise ──────────────────────────────────────────────────────────
import requests, urllib3
urllib3.disable_warnings()
_orig_send = requests.Session.send
def _no_verify_send(self, *args, **kwargs):
    kwargs['verify'] = False
    return _orig_send(self, *args, **kwargs)
requests.Session.send = _no_verify_send

ee.Initialize(project='ee-ciceromartinsjr')

GDW = ee.FeatureCollection(
    'projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0'
)

IN_CSV  = Path('validation_data/DAHITI/dahiti_reservoir_scan_results.csv')
OUT_CSV = Path('validation_data/DAHITI/dahiti_reservoir_scan_results.csv')
MED_CSV = Path('validation_data/DAHITI/dahiti_scan_medium.csv')

AREA_MIN_KM2 = 5.0    # 500 ha
AREA_MAX_KM2 = 100.0  # 10,000 ha

# ── Load CSV ────────────────────────────────────────────────────────────────
with open(IN_CSV, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
fields = list(rows[0].keys()) if rows else []
for field in ['area_km2_gee', 'area_ha_gee', 'gdw_id', 'gdw_name', 'gdw_country']:
    if field not in fields:
        fields.append(field)

print(f"Loaded {len(rows)} targets. Looking up GDW areas ...\n")

# ── GEE lookup — batch in groups of 20 to stay under eval limits ────────────
BATCH = 20

def lookup_batch(batch_rows):
    """Returns list of dicts with gdw_id, area_km2, etc. Same order as input."""
    pts = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
            {'idx': i, 'dahiti_id': r['dahiti_id']}
        )
        for i, r in enumerate(batch_rows)
    ])

    # Spatial join: each point gets the intersecting GDW polygon (if any)
    join = ee.Join.saveFirst(matchKey='gdw_match', outer=True)
    dist_filter = ee.Filter.withinDistance(distance=5000, leftField='.geo', rightField='.geo')
    joined = join.apply(pts, GDW, dist_filter)

    def extract(f):
        g = f.get('gdw_match')
        area = ee.Algorithms.If(
            ee.Algorithms.IsEqual(g, None),
            ee.Number(-1),
            ee.Feature(g).geometry().area(maxError=500).divide(1e6)  # km²
        )
        gdw_id = ee.Algorithms.If(
            ee.Algorithms.IsEqual(g, None), '',
            ee.Feature(g).get('GDW_ID')
        )
        gdw_name = ee.Algorithms.If(
            ee.Algorithms.IsEqual(g, None), '',
            ee.Feature(g).get('Res_name')
        )
        gdw_country = ee.Algorithms.If(
            ee.Algorithms.IsEqual(g, None), '',
            ee.Feature(g).get('Country')
        )
        return f.set({'_area_km2': area, '_gdw_id': gdw_id,
                      '_gdw_name': gdw_name, '_gdw_country': gdw_country})

    result = joined.map(extract).getInfo()
    out = {}
    for feat in result['features']:
        p = feat['properties']
        out[p['idx']] = {
            'area_km2': p.get('_area_km2', -1),
            'gdw_id':   p.get('_gdw_id', ''),
            'gdw_name': p.get('_gdw_name', ''),
            'gdw_country': p.get('_gdw_country', ''),
        }
    return out

# ── Process all rows ─────────────────────────────────────────────────────────
n_found = 0
n_medium = 0

for start in range(0, len(rows), BATCH):
    batch = rows[start:start + BATCH]
    try:
        results = lookup_batch(batch)
    except Exception as e:
        print(f"  WARN batch {start}–{start+len(batch)}: {e}")
        results = {}

    for i, row in enumerate(batch):
        info = results.get(i, {})
        area_km2 = float(info.get('area_km2', -1))
        area_ha  = round(area_km2 * 100, 0) if area_km2 > 0 else -1
        row['area_km2_gee']  = round(area_km2, 3) if area_km2 > 0 else ''
        row['area_ha_gee']   = int(area_ha)        if area_km2 > 0 else ''
        row['gdw_id']        = info.get('gdw_id', '')
        row['gdw_name']      = info.get('gdw_name', '')
        row['gdw_country']   = info.get('gdw_country', '')
        if area_km2 > 0:
            n_found += 1
            if AREA_MIN_KM2 <= area_km2 <= AREA_MAX_KM2:
                n_medium += 1

    done = min(start + BATCH, len(rows))
    print(f"  Processed {done:4d}/{len(rows)}  (GDW matches so far: {n_found})")
    time.sleep(0.5)

# ── Save updated CSV ─────────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")

# ── Save medium-only CSV ─────────────────────────────────────────────────────
medium = [r for r in rows
          if r.get('area_km2_gee') != '' and
          AREA_MIN_KM2 <= float(r['area_km2_gee']) <= AREA_MAX_KM2]
medium.sort(key=lambda r: float(r['area_km2_gee']))

with open(MED_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(medium)
print(f"Saved: {MED_CSV}  ({len(medium)} medium reservoirs 500–10,000 ha)")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'DAHITI-id':>8s}  {'Name':35s}  {'WL2014':>6s}  {'Range(m)':>8s}  "
      f"{'Area(km2)':>9s}  {'Area(ha)':>8s}  {'GDW Name':25s}")
print('-' * 110)
for r in medium:
    print(f"  {r['dahiti_id']:>7s}  {r['name']:35s}  {r.get('wl_2014',''):>6s}  "
          f"{r.get('wl_range_m',''):>8s}  {r.get('area_km2_gee',''):>9s}  "
          f"{r.get('area_ha_gee',''):>8s}  {r.get('gdw_name',''):25s}")

print(f"\n{'='*70}")
print(f"Reservoirs 500–10,000 ha with DAHITI WL 2014+ >= 10 obs: {len(medium)}")
