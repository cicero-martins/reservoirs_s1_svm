"""
_lookup_gdw_ids.py  —  Find GDW_IDs for flagged v4 reservoirs via Earth Engine Python API.

Run from project root:
    python analysis/_lookup_gdw_ids.py
"""

import ee
ee.Initialize()

GDW = ee.FeatureCollection('projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0')

FLAGGED = [
    # Grupo A — wrong large polygon matched
    ('Aguilar',       42.793,  -4.270),
    ('Almus',         40.378,  36.908),
    ('Boegoeberg',   -29.026,  22.155),
    ('Cardinia',     -37.935, 145.510),
    ('Chelmsford',   -27.967,  29.852),
    ('Cruz_del_Eje', -30.728, -64.804),
    ('Ebro_Embalse',  42.960,  -4.060),
    ('Eleven_Mile',   38.930,-105.534),
    ('Occhito',       41.534,  14.913),
    ('Pineview',      41.273,-111.839),
    ('Plastiras',     39.233,  21.776),
    ('Riano',         42.993,  -5.017),
    # Grupo B — wrong small / not found
    ('Abdelmoumen',   30.373,  -9.545),
    ('Bleiloch',      50.637,  11.697),
    ('Blue_Rock',    -38.320, 146.180),
    ('Cecita',        39.333,  16.620),
    ('Demirkopru',    38.794,  28.621),
    ('Guajaraz',      39.675,  -4.107),
    ('Katse',        -29.365,  28.521),
    ('La_Vina',      -31.533, -64.503),
    ('Mohale',       -29.550,  28.143),
    ('Nagle',        -29.597,  30.784),
    ('Oued_Makhazine',35.167,  -5.533),
    ('Shaharchay',    37.640,  45.009),
    ('Siurana',       41.197,   0.914),
    ('Suat_Ugurlu',   41.117,  36.050),
    ('Triouzoune',    45.520,   2.265),
    ('Wadi_Dayqah',   22.724,  57.863),
]

print(f'\n{"Name":<20}  {"GDW_ID":>7}  {"DAM_NAME":<30}  {"RES_NAME":<25}  {"km²":>6}  {"dist_km":>7}')
print('─' * 105)

results = {}
for name, lat, lon in FLAGGED:
    pt = ee.Geometry.Point([lon, lat])
    candidates = (GDW
        .filterBounds(pt.buffer(25000))
        .map(lambda f: f.set('_dist_m', f.geometry().centroid(1).distance(pt, 1)))
        .sort('_dist_m'))

    n    = candidates.size().getInfo()
    best = candidates.first().getInfo() if n > 0 else None

    if best:
        p        = best['properties']
        gdw_id   = int(p.get('GDW_ID', -1))
        dam_name = str(p.get('DAM_NAME', ''))[:29]
        res_name = str(p.get('RES_NAME', ''))[:24]
        area_skm = float(p.get('AREA_SKM', 0))
        dist_km  = float(p.get('_dist_m', 0)) / 1000
        print(f'{name:<20}  {gdw_id:>7}  {dam_name:<30}  {res_name:<25}  {area_skm:>6.2f}  {dist_km:>7.1f}')
        results[name] = gdw_id
    else:
        print(f'{name:<20}  {"NOT FOUND":>7}  {"(no GDW feature within 25 km)"}')
        results[name] = None

print('─' * 105)
print(f'\nFound: {sum(v is not None for v in results.values())} / {len(FLAGGED)}')

# Print ready-to-paste JS snippet for updating exportGlobalPilotV4.js
print('\n── JS snippet (paste into PILOT_RESERVOIRS gdw_id fields) ──')
for name, gdw_id in results.items():
    if gdw_id is not None:
        print(f"  // {name}: gdw_id = {gdw_id}")
