"""
DAHITI coverage screening for global pilot revision.

Queries DAHITI API v2 for ~80 candidate reservoirs across 6 continents,
covering different biomes, area ranges, and the existing 24-reservoir pilot
(for continuity comparison).

For each candidate:
  1. list-targets (bbox) → find closest DAHITI match + check data_access
  2. download-water-level (if public) → count obs, compute WL range, date range
  3. download-surface-area (if public, as supplement) → count obs

Outputs:
  validation_data/DAHITI/dahiti_longlist_screening.csv   — full results
  validation_data/DAHITI/dahiti_longlist_wl/             — WL CSVs for matched targets

Run from project root:
  python analysis/dahiti_longlist_screening.py
"""

import json, time, math, csv, sys
import requests, urllib3
from pathlib import Path

urllib3.disable_warnings()
sys.stdout.reconfigure(encoding='utf-8')

API_KEY  = '510183B30DFFC0A80C004524BABA85C6EFE0ECD67998D7F9E85EFA06930AA375'
BASE_URL = 'https://dahiti.dgfi.tum.de/api/v2'
OUT_DIR  = Path('validation_data/DAHITI')
WL_DIR   = OUT_DIR / 'dahiti_longlist_wl'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WL_DIR.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.verify = False

# ── API helpers ────────────────────────────────────────────────────────────
def api_post(endpoint, params):
    p = dict(params)
    p['api_key'] = API_KEY
    if endpoint.startswith('download-'):
        p.setdefault('format', 'json')
    try:
        r = S.post(f'{BASE_URL}/{endpoint}/', data=p, timeout=40)
        return r.json()
    except Exception as e:
        return {'code': -1, 'message': str(e), 'data': None}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Candidate list ────────────────────────────────────────────────────────
# Fields: name, lat, lon, country, continent, biome, area_ha_est, in_pilot
# continent codes: AF=Africa, ME=Middle East/Caucasus, CA=Central Asia,
#                  AS=South Asia, EA=East/SE Asia, EU=Europe, NA=North America,
#                  SA=South America, OC=Oceania
# in_pilot: True = already in the existing 24-reservoir pilot set

CANDIDATES = [

    # ═══════════════════════════════════════════════════════
    # EXISTING PILOT (24 reservoirs) — kept for comparison
    # Coordinates from pilot_summary.csv
    # ═══════════════════════════════════════════════════════
    # India
    {'name': 'Manikdoh',            'lat': 19.248, 'lon':  73.772, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-monsoon',   'area_ha_est':  2900, 'in_pilot': True},
    {'name': 'Panam',               'lat': 23.023, 'lon':  73.769, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-dry',        'area_ha_est':  4000, 'in_pilot': True},
    {'name': 'Periyar',             'lat':  9.535, 'lon':  77.192, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-wet',        'area_ha_est':  2400, 'in_pilot': True},
    {'name': 'Kakki',               'lat':  9.305, 'lon':  77.168, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-wet',        'area_ha_est':  1600, 'in_pilot': True},
    {'name': 'Pench_Totladoh',      'lat': 21.713, 'lon':  79.230, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-dry',        'area_ha_est':  7000, 'in_pilot': True},
    {'name': 'Thein_Ranjit_Sagar',  'lat': 32.484, 'lon':  75.765, 'country': 'IN', 'continent': 'AS', 'biome': 'Subtropical',         'area_ha_est': 28000, 'in_pilot': True},
    # Europe — Spain / Belgium
    {'name': 'San_Esteban',         'lat': 42.416, 'lon':  -7.649, 'country': 'ES', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':  1300, 'in_pilot': True},
    {'name': 'Ry_de_Rome',          'lat': 50.024, 'lon':   4.537, 'country': 'BE', 'continent': 'EU', 'biome': 'Oceanic-temperate',   'area_ha_est':   100, 'in_pilot': True},
    {'name': 'Borbollon',           'lat': 40.127, 'lon':  -6.576, 'country': 'ES', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':  1200, 'in_pilot': True},
    {'name': 'Minilla',             'lat': 37.668, 'lon':  -6.184, 'country': 'ES', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':   165, 'in_pilot': True},
    {'name': 'Gabriel_y_Galan',     'lat': 40.223, 'lon':  -6.132, 'country': 'ES', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':  7500, 'in_pilot': True},
    {'name': 'Ricobayo',            'lat': 41.529, 'lon':  -5.984, 'country': 'ES', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':  2200, 'in_pilot': True},
    # USA / Mexico
    {'name': 'Pine_River',          'lat': 46.668, 'lon': -94.112, 'country': 'US', 'continent': 'NA', 'biome': 'Temperate-continental','area_ha_est':  3200, 'in_pilot': True},
    {'name': 'Benito_Juarez',       'lat': 16.446, 'lon': -95.398, 'country': 'MX', 'continent': 'NA', 'biome': 'Tropical-dry',        'area_ha_est':  4500, 'in_pilot': True},
    {'name': 'Pokegama',            'lat': 47.248, 'lon': -93.587, 'country': 'US', 'continent': 'NA', 'biome': 'Temperate-continental','area_ha_est':  1900, 'in_pilot': True},
    {'name': 'Winnibigoshish',      'lat': 47.430, 'lon': -94.050, 'country': 'US', 'continent': 'NA', 'biome': 'Boreal-transition',   'area_ha_est': 16300, 'in_pilot': True},
    {'name': 'Leech',               'lat': 47.248, 'lon': -94.222, 'country': 'US', 'continent': 'NA', 'biome': 'Boreal-transition',   'area_ha_est': 17300, 'in_pilot': True},
    {'name': 'Elephant_Butte',      'lat': 33.154, 'lon':-107.192, 'country': 'US', 'continent': 'NA', 'biome': 'Arid',                'area_ha_est':  7700, 'in_pilot': True},
    # Australia
    {'name': 'Maroondah',           'lat':-37.643, 'lon': 145.555, 'country': 'AU', 'continent': 'OC', 'biome': 'Mediterranean-AU',   'area_ha_est':  1700, 'in_pilot': True},
    {'name': 'OShannassy',          'lat':-37.675, 'lon': 145.806, 'country': 'AU', 'continent': 'OC', 'biome': 'Mediterranean-AU',   'area_ha_est':   100, 'in_pilot': True},
    {'name': 'Silvan',              'lat':-37.837, 'lon': 145.417, 'country': 'AU', 'continent': 'OC', 'biome': 'Mediterranean-AU',   'area_ha_est':   188, 'in_pilot': True},
    {'name': 'Upper_Coliban',       'lat':-37.289, 'lon': 144.394, 'country': 'AU', 'continent': 'OC', 'biome': 'Mediterranean-AU',   'area_ha_est':  2000, 'in_pilot': True},
    {'name': 'Chichester',          'lat':-32.238, 'lon': 151.690, 'country': 'AU', 'continent': 'OC', 'biome': 'Subtropical',         'area_ha_est':   135, 'in_pilot': True},
    {'name': 'Yarra',               'lat':-37.674, 'lon': 145.898, 'country': 'AU', 'continent': 'OC', 'biome': 'Mediterranean-AU',   'area_ha_est':  1900, 'in_pilot': True},
    # Sicily anchors
    {'name': 'Ancipa',              'lat': 37.830, 'lon':  14.573, 'country': 'IT', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':   870, 'in_pilot': True},
    {'name': 'Pozzillo',            'lat': 37.700, 'lon':  14.530, 'country': 'IT', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':   930, 'in_pilot': True},
    {'name': 'Poma',                'lat': 37.994, 'lon':  13.090, 'country': 'IT', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':   620, 'in_pilot': True},
    {'name': 'Rosamarina',          'lat': 37.944, 'lon':  13.640, 'country': 'IT', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est':   440, 'in_pilot': True},
    # New regions (added to pilot later)
    {'name': 'Passauna',            'lat':-25.520, 'lon': -49.390, 'country': 'BR', 'continent': 'SA', 'biome': 'Subtropical',         'area_ha_est':   800, 'in_pilot': True},
    {'name': 'Tres_Marias',         'lat':-18.210, 'lon': -45.270, 'country': 'BR', 'continent': 'SA', 'biome': 'Cerrado',             'area_ha_est':109000, 'in_pilot': True},
    {'name': 'Cerros_Colorados',    'lat':-38.660, 'lon': -68.730, 'country': 'AR', 'continent': 'SA', 'biome': 'Arid-Patagonia',      'area_ha_est': 29000, 'in_pilot': True},
    {'name': 'Calima',              'lat':  3.930, 'lon': -76.520, 'country': 'CO', 'continent': 'SA', 'biome': 'Tropical',            'area_ha_est':  1800, 'in_pilot': True},
    {'name': 'Theewaterskloof',     'lat':-34.040, 'lon':  19.180, 'country': 'ZA', 'continent': 'AF', 'biome': 'Mediterranean',       'area_ha_est': 10000, 'in_pilot': True},
    {'name': 'Roseires',            'lat': 11.790, 'lon':  34.380, 'country': 'SD', 'continent': 'AF', 'biome': 'Sahel',               'area_ha_est':115000, 'in_pilot': True},

    # ═══════════════════════════════════════════════════════
    # NEW GEOGRAPHIC CANDIDATES
    # ═══════════════════════════════════════════════════════

    # ── AFRICA — East ──────────────────────────────────────
    {'name': 'Masinga',          'lat': -0.520, 'lon':  37.570, 'country': 'KE', 'continent': 'AF', 'biome': 'Tropical-equatorial', 'area_ha_est': 11000, 'in_pilot': False},
    {'name': 'Gitaru',           'lat': -0.680, 'lon':  37.040, 'country': 'KE', 'continent': 'AF', 'biome': 'Tropical-equatorial', 'area_ha_est':  2100, 'in_pilot': False},
    {'name': 'Gilgel_Gibe_III',  'lat':  7.120, 'lon':  37.530, 'country': 'ET', 'continent': 'AF', 'biome': 'Tropical-highland',   'area_ha_est': 20000, 'in_pilot': False},
    {'name': 'Tekeze',           'lat': 13.450, 'lon':  38.420, 'country': 'ET', 'continent': 'AF', 'biome': 'Tropical-highland',   'area_ha_est': 18000, 'in_pilot': False},

    # ── AFRICA — Southern ──────────────────────────────────
    {'name': 'Cahora_Bassa',     'lat':-15.620, 'lon':  32.710, 'country': 'MZ', 'continent': 'AF', 'biome': 'Tropical-savanna',    'area_ha_est':267000, 'in_pilot': False},
    {'name': 'Kariba',           'lat':-16.500, 'lon':  28.850, 'country': 'ZW', 'continent': 'AF', 'biome': 'Tropical-savanna',    'area_ha_est':520000, 'in_pilot': False},
    {'name': 'Gariep',           'lat':-30.630, 'lon':  25.520, 'country': 'ZA', 'continent': 'AF', 'biome': 'Arid',                'area_ha_est':361000, 'in_pilot': False},
    {'name': 'Vanderkloof',      'lat':-29.660, 'lon':  24.730, 'country': 'ZA', 'continent': 'AF', 'biome': 'Arid',                'area_ha_est':  7600, 'in_pilot': False},

    # ── AFRICA — West ──────────────────────────────────────
    {'name': 'Akosombo',         'lat':  6.500, 'lon':  -0.070, 'country': 'GH', 'continent': 'AF', 'biome': 'Tropical-wet',        'area_ha_est':850000, 'in_pilot': False},
    {'name': 'Kossou',           'lat':  6.970, 'lon':  -5.370, 'country': 'CI', 'continent': 'AF', 'biome': 'Tropical-wet',        'area_ha_est':165000, 'in_pilot': False},
    {'name': 'Kainji',           'lat': 10.400, 'lon':   4.630, 'country': 'NG', 'continent': 'AF', 'biome': 'Tropical-savanna',    'area_ha_est':125000, 'in_pilot': False},
    {'name': 'Mape',             'lat':  5.880, 'lon':  12.870, 'country': 'CM', 'continent': 'AF', 'biome': 'Tropical-wet',        'area_ha_est':  3800, 'in_pilot': False},

    # ── AFRICA — North ─────────────────────────────────────
    {'name': 'Lake_Nasser',      'lat': 23.970, 'lon':  32.880, 'country': 'EG', 'continent': 'AF', 'biome': 'Hyper-arid',          'area_ha_est':168000, 'in_pilot': False},
    {'name': 'Al_Massira',       'lat': 32.530, 'lon':  -8.190, 'country': 'MA', 'continent': 'AF', 'biome': 'Semi-arid',           'area_ha_est':  6300, 'in_pilot': False},

    # ── MIDDLE EAST / CAUCASUS ─────────────────────────────
    {'name': 'Ataturk',          'lat': 37.490, 'lon':  38.350, 'country': 'TR', 'continent': 'ME', 'biome': 'Semi-arid',           'area_ha_est': 81700, 'in_pilot': False},
    {'name': 'Keban',            'lat': 38.780, 'lon':  38.740, 'country': 'TR', 'continent': 'ME', 'biome': 'Continental',         'area_ha_est': 67500, 'in_pilot': False},
    {'name': 'Karakaya',         'lat': 38.220, 'lon':  38.950, 'country': 'TR', 'continent': 'ME', 'biome': 'Continental',         'area_ha_est': 29800, 'in_pilot': False},
    {'name': 'Dez',              'lat': 32.440, 'lon':  48.850, 'country': 'IR', 'continent': 'ME', 'biome': 'Semi-arid',           'area_ha_est':  6400, 'in_pilot': False},
    {'name': 'Karkheh',          'lat': 31.580, 'lon':  48.240, 'country': 'IR', 'continent': 'ME', 'biome': 'Semi-arid',           'area_ha_est': 16200, 'in_pilot': False},
    {'name': 'Mingachevir',      'lat': 40.770, 'lon':  47.060, 'country': 'AZ', 'continent': 'ME', 'biome': 'Semi-arid',           'area_ha_est': 62500, 'in_pilot': False},

    # ── CENTRAL ASIA ───────────────────────────────────────
    {'name': 'Toktogul',         'lat': 41.800, 'lon':  73.050, 'country': 'KG', 'continent': 'CA', 'biome': 'Mountainous-arid',    'area_ha_est': 28400, 'in_pilot': False},
    {'name': 'Nurek',            'lat': 37.900, 'lon':  69.460, 'country': 'TJ', 'continent': 'CA', 'biome': 'Mountainous-arid',    'area_ha_est':  9800, 'in_pilot': False},
    {'name': 'Chardara',         'lat': 41.260, 'lon':  67.990, 'country': 'KZ', 'continent': 'CA', 'biome': 'Arid',                'area_ha_est':  9000, 'in_pilot': False},

    # ── SOUTH ASIA ─────────────────────────────────────────
    {'name': 'Tarbela',          'lat': 34.120, 'lon':  72.720, 'country': 'PK', 'continent': 'AS', 'biome': 'Semi-arid',           'area_ha_est': 24000, 'in_pilot': False},
    {'name': 'Mangla',           'lat': 33.150, 'lon':  73.630, 'country': 'PK', 'continent': 'AS', 'biome': 'Subtropical',         'area_ha_est': 25900, 'in_pilot': False},
    {'name': 'Kaptai',           'lat': 22.490, 'lon':  92.320, 'country': 'BD', 'continent': 'AS', 'biome': 'Tropical-monsoon',    'area_ha_est': 68800, 'in_pilot': False},
    {'name': 'Nagarjuna_Sagar',  'lat': 16.570, 'lon':  79.320, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-dry',        'area_ha_est': 28500, 'in_pilot': False},
    {'name': 'Srisailam',        'lat': 16.080, 'lon':  78.900, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-dry',        'area_ha_est': 61600, 'in_pilot': False},
    {'name': 'Tehri',            'lat': 30.380, 'lon':  78.480, 'country': 'IN', 'continent': 'AS', 'biome': 'Alpine',              'area_ha_est':  4500, 'in_pilot': False},
    {'name': 'Idukki',           'lat':  9.850, 'lon':  76.980, 'country': 'IN', 'continent': 'AS', 'biome': 'Tropical-wet',        'area_ha_est':  6000, 'in_pilot': False},
    {'name': 'Kotmale',          'lat':  7.070, 'lon':  80.620, 'country': 'LK', 'continent': 'AS', 'biome': 'Tropical',            'area_ha_est':  1740, 'in_pilot': False},

    # ── EAST / SE ASIA ─────────────────────────────────────
    {'name': 'Miyun',            'lat': 40.520, 'lon': 116.970, 'country': 'CN', 'continent': 'EA', 'biome': 'Temperate-semi-arid', 'area_ha_est': 18800, 'in_pilot': False},
    {'name': 'Danjiangkou',      'lat': 32.650, 'lon': 111.470, 'country': 'CN', 'continent': 'EA', 'biome': 'Subtropical',         'area_ha_est': 74500, 'in_pilot': False},
    {'name': 'Ertan',            'lat': 26.550, 'lon': 101.880, 'country': 'CN', 'continent': 'EA', 'biome': 'Subtropical',         'area_ha_est':  8700, 'in_pilot': False},
    {'name': 'Bhumibol',         'lat': 17.240, 'lon':  99.040, 'country': 'TH', 'continent': 'EA', 'biome': 'Tropical-monsoon',    'area_ha_est': 30000, 'in_pilot': False},
    {'name': 'Nam_Ngum',         'lat': 18.520, 'lon': 102.560, 'country': 'LA', 'continent': 'EA', 'biome': 'Tropical-wet',        'area_ha_est': 37000, 'in_pilot': False},
    {'name': 'Kenyir',           'lat':  5.000, 'lon': 102.700, 'country': 'MY', 'continent': 'EA', 'biome': 'Tropical-rainforest', 'area_ha_est': 36900, 'in_pilot': False},
    {'name': 'Cirata',           'lat': -6.720, 'lon': 107.350, 'country': 'ID', 'continent': 'EA', 'biome': 'Tropical-rainforest', 'area_ha_est':  6220, 'in_pilot': False},

    # ── EUROPE — beyond Mediterranean ──────────────────────
    {'name': 'Alqueva',          'lat': 38.200, 'lon':  -7.490, 'country': 'PT', 'continent': 'EU', 'biome': 'Mediterranean',       'area_ha_est': 25000, 'in_pilot': False},
    {'name': 'Edertalsperre',    'lat': 51.180, 'lon':   8.850, 'country': 'DE', 'continent': 'EU', 'biome': 'Oceanic-temperate',   'area_ha_est':  1200, 'in_pilot': False},
    {'name': 'Bleiloch',         'lat': 50.490, 'lon':  11.720, 'country': 'DE', 'continent': 'EU', 'biome': 'Continental',         'area_ha_est':  1080, 'in_pilot': False},
    {'name': 'Kardzhali',        'lat': 41.630, 'lon':  25.250, 'country': 'BG', 'continent': 'EU', 'biome': 'Continental',         'area_ha_est':  2850, 'in_pilot': False},
    {'name': 'Kremenchuk',       'lat': 49.080, 'lon':  33.200, 'country': 'UA', 'continent': 'EU', 'biome': 'Continental',         'area_ha_est':232500, 'in_pilot': False},
    {'name': 'Rybinsk',          'lat': 58.500, 'lon':  37.720, 'country': 'RU', 'continent': 'EU', 'biome': 'Boreal',              'area_ha_est':455000, 'in_pilot': False},
    {'name': 'Pieve_di_Cadore',  'lat': 46.420, 'lon':  12.380, 'country': 'IT', 'continent': 'EU', 'biome': 'Alpine',              'area_ha_est':   250, 'in_pilot': False},

    # ── NORTH AMERICA — beyond current pilot ───────────────
    {'name': 'Lake_Mead',        'lat': 36.130, 'lon':-114.460, 'country': 'US', 'continent': 'NA', 'biome': 'Hyper-arid',          'area_ha_est': 63200, 'in_pilot': False},
    {'name': 'Lake_Powell',      'lat': 37.050, 'lon':-111.270, 'country': 'US', 'continent': 'NA', 'biome': 'Arid',                'area_ha_est': 65280, 'in_pilot': False},
    {'name': 'Oroville',         'lat': 39.540, 'lon':-121.490, 'country': 'US', 'continent': 'NA', 'biome': 'Mediterranean-CA',    'area_ha_est':  3840, 'in_pilot': False},
    {'name': 'Folsom',           'lat': 38.710, 'lon':-121.170, 'country': 'US', 'continent': 'NA', 'biome': 'Mediterranean-CA',    'area_ha_est':  4430, 'in_pilot': False},
    {'name': 'Hartwell',         'lat': 34.370, 'lon': -82.890, 'country': 'US', 'continent': 'NA', 'biome': 'Subtropical-humid',   'area_ha_est': 22600, 'in_pilot': False},
    {'name': 'Sam_Rayburn',      'lat': 31.070, 'lon': -94.120, 'country': 'US', 'continent': 'NA', 'biome': 'Subtropical-humid',   'area_ha_est': 46000, 'in_pilot': False},
    {'name': 'W_A_C_Bennett',    'lat': 56.000, 'lon':-122.200, 'country': 'CA', 'continent': 'NA', 'biome': 'Boreal',              'area_ha_est':167000, 'in_pilot': False},
    {'name': 'LG3_La_Grande',    'lat': 53.760, 'lon': -76.480, 'country': 'CA', 'continent': 'NA', 'biome': 'Boreal',              'area_ha_est': 26000, 'in_pilot': False},

    # ── SOUTH AMERICA ──────────────────────────────────────
    {'name': 'Sobradinho',       'lat': -9.450, 'lon': -42.800, 'country': 'BR', 'continent': 'SA', 'biome': 'Semi-arid',           'area_ha_est':421400, 'in_pilot': False},
    {'name': 'Serra_da_Mesa',    'lat':-13.830, 'lon': -48.300, 'country': 'BR', 'continent': 'SA', 'biome': 'Cerrado',             'area_ha_est':178400, 'in_pilot': False},
    {'name': 'Furnas',           'lat':-20.730, 'lon': -46.320, 'country': 'BR', 'continent': 'SA', 'biome': 'Subtropical-humid',   'area_ha_est':138400, 'in_pilot': False},
    {'name': 'Itaipu',           'lat':-25.410, 'lon': -54.590, 'country': 'BR', 'continent': 'SA', 'biome': 'Subtropical-humid',   'area_ha_est':135000, 'in_pilot': False},
    {'name': 'Guri',             'lat':  7.770, 'lon': -62.990, 'country': 'VE', 'continent': 'SA', 'biome': 'Tropical',            'area_ha_est':430000, 'in_pilot': False},
    {'name': 'Rapel',            'lat':-34.070, 'lon': -71.460, 'country': 'CL', 'continent': 'SA', 'biome': 'Mediterranean',       'area_ha_est':  8900, 'in_pilot': False},
    {'name': 'Betania',          'lat':  2.890, 'lon': -75.420, 'country': 'CO', 'continent': 'SA', 'biome': 'Tropical',            'area_ha_est':  7400, 'in_pilot': False},
    {'name': 'Chixoy',           'lat': 15.310, 'lon': -90.450, 'country': 'GT', 'continent': 'SA', 'biome': 'Tropical',            'area_ha_est':  1400, 'in_pilot': False},

    # ── OCEANIA — replacements / additions ─────────────────
    {'name': 'Thomson',          'lat':-37.830, 'lon': 146.110, 'country': 'AU', 'continent': 'OC', 'biome': 'Mediterranean-AU',    'area_ha_est': 23000, 'in_pilot': False},
    {'name': 'Hume',             'lat':-36.090, 'lon': 147.030, 'country': 'AU', 'continent': 'OC', 'biome': 'Semi-arid',           'area_ha_est': 20200, 'in_pilot': False},
    {'name': 'Wivenhoe',         'lat':-27.390, 'lon': 152.610, 'country': 'AU', 'continent': 'OC', 'biome': 'Subtropical',         'area_ha_est': 10700, 'in_pilot': False},
    {'name': 'Clyde_NZ',         'lat':-45.200, 'lon': 169.310, 'country': 'NZ', 'continent': 'OC', 'biome': 'Temperate-oceanic',   'area_ha_est':  7500, 'in_pilot': False},
]

# ── Query DAHITI ───────────────────────────────────────────────────────────
BBOX_PILOT = 0.20   # ±° for existing pilot (precise coords)
BBOX_NEW   = 0.35   # ±° for new candidates (approximate coords)
DIST_MAX   = 30.0   # km — maximum allowed distance to DAHITI match

print(f"\n{'Name':28s}  {'Ctry':4s}  {'Cont':5s}  {'Biome':22s}  "
      f"{'Area(ha)':>8s}  {'DAHITI':>7s}  {'Type':10s}  "
      f"{'Dist':>5s}  {'WL-tot':>6s}  {'WL-2014':>7s}  "
      f"{'WL-range(m)':>12s}  {'Period':>22s}  {'In pilot':>8s}")
print('─' * 160)

results = []

for res in CANDIDATES:
    name      = res['name']
    lat, lon  = res['lat'], res['lon']
    in_pilot  = res['in_pilot']
    bbox      = BBOX_PILOT if in_pilot else BBOX_NEW

    # Step 1: list-targets in bounding box
    r1 = api_post('list-targets', {
        'min_lat': lat - bbox, 'max_lat': lat + bbox,
        'min_lon': lon - bbox, 'max_lon': lon + bbox,
    })
    targets = r1.get('data') or []

    best = None
    best_dist = 9999.0
    for t in targets:
        d = haversine(lat, lon, float(t['latitude']), float(t['longitude']))
        if d < best_dist:
            best_dist = d
            best = t

    base_row = {
        'name': name, 'lat': lat, 'lon': lon,
        'country': res['country'], 'continent': res['continent'],
        'biome': res['biome'], 'area_ha_est': res['area_ha_est'],
        'in_pilot': in_pilot,
        'dahiti_id': '', 'dahiti_name': '', 'dahiti_type': '',
        'dist_km': round(best_dist, 1),
        'wl_total': 0, 'wl_2014': 0,
        'sa_total': 0, 'sa_2014': 0,
        'wl_min': '', 'wl_max': '', 'wl_range_m': '',
        'wl_date_start': '', 'wl_date_end': '',
    }

    if best is None or best_dist > DIST_MAX:
        n_box = len(targets)
        print(f"{name:28s}  {res['country']:4s}  {res['continent']:5s}  "
              f"{res['biome']:22s}  {res['area_ha_est']:8d}  "
              f"{'—':>7s}  {'no match':10s}  {best_dist if best else 0.0:5.1f}  "
              f"{'—':>6s}  {'—':>7s}  {'—':>12s}  {'—':>22s}  {str(in_pilot):>8s}  "
              f"({n_box} in box)")
        results.append(base_row)
        time.sleep(0.15)
        continue

    dahiti_id   = str(best['dahiti_id'])
    dahiti_name = best.get('target_name', '')
    dahiti_type = best.get('type', '?')
    da          = best.get('data_access') or {}

    base_row.update({
        'dahiti_id':   dahiti_id,
        'dahiti_name': dahiti_name,
        'dahiti_type': dahiti_type,
        'dist_km':     round(best_dist, 1),
    })

    # Step 2: download water level if public
    wl_total = wl_2014 = 0
    wl_min = wl_max = wl_range_m = ''
    wl_date_start = wl_date_end = ''

    if da.get('water_level_altimetry') == 'public':
        r2 = api_post('download-water-level', {'dahiti_id': dahiti_id})
        data_wl = r2.get('data') or []
        if data_wl:
            wl_total = len(data_wl)
            # DAHITI v2: 'datetime' (YYYY-MM-DD HH:MM:SS), 'wse', 'wse_u'
            d2014 = [d for d in data_wl if str(d.get('datetime', ''))[:10] >= '2014-01-01']
            wl_2014 = len(d2014)
            wl_vals = [float(d['wse']) for d in data_wl if d.get('wse') is not None]
            if wl_vals:
                wl_min = round(min(wl_vals), 2)
                wl_max = round(max(wl_vals), 2)
                wl_range_m = round(wl_max - wl_min, 2)
            all_dates = sorted(str(d.get('datetime', ''))[:10] for d in data_wl
                               if d.get('datetime'))
            if all_dates:
                wl_date_start = all_dates[0]
                wl_date_end   = all_dates[-1]
            # Save WL CSV
            safe = name.replace(' ', '_').replace('/', '_')
            out = WL_DIR / f"{dahiti_id}_{safe}_wl.csv"
            with open(out, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(data_wl[0].keys()))
                w.writeheader(); w.writerows(data_wl)
        time.sleep(0.2)

    # Step 3: surface area (count only)
    sa_total = sa_2014 = 0
    if da.get('surface_area') == 'public':
        r3 = api_post('download-surface-area', {'dahiti_id': dahiti_id})
        data_sa = r3.get('data') or []
        sa_total = len(data_sa)
        sa_2014  = len([d for d in data_sa if str(d.get('date', '')) >= '2014-01-01'])
        time.sleep(0.2)

    base_row.update({
        'wl_total': wl_total, 'wl_2014': wl_2014,
        'sa_total': sa_total, 'sa_2014': sa_2014,
        'wl_min': wl_min, 'wl_max': wl_max, 'wl_range_m': wl_range_m,
        'wl_date_start': wl_date_start, 'wl_date_end': wl_date_end,
    })
    results.append(base_row)

    period_str = (f"{wl_date_start[:7]} → {wl_date_end[:7]}"
                  if wl_date_start else '—')
    range_str  = f"{wl_range_m} m" if wl_range_m != '' else '—'

    print(f"{name:28s}  {res['country']:4s}  {res['continent']:5s}  "
          f"{res['biome']:22s}  {res['area_ha_est']:8d}  "
          f"{dahiti_id:>7s}  {dahiti_type:10s}  {best_dist:5.1f}  "
          f"{wl_total:6d}  {wl_2014:7d}  {range_str:>12s}  "
          f"{period_str:>22s}  {str(in_pilot):>8s}")

    time.sleep(0.15)

# ── Save results ───────────────────────────────────────────────────────────
out_csv = OUT_DIR / 'dahiti_longlist_screening.csv'
fields = ['name', 'country', 'continent', 'biome', 'area_ha_est', 'in_pilot',
          'lat', 'lon', 'dahiti_id', 'dahiti_name', 'dahiti_type', 'dist_km',
          'wl_total', 'wl_2014', 'sa_total', 'sa_2014',
          'wl_min', 'wl_max', 'wl_range_m', 'wl_date_start', 'wl_date_end']
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(results)

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"Total candidates queried : {len(results)}")
matched  = [r for r in results if r['dahiti_id']]
wl_ok    = [r for r in results if r.get('wl_2014', 0) >= 10]
wl_ok_new= [r for r in wl_ok   if not r['in_pilot']]

print(f"DAHITI match found       : {len(matched)}")
print(f"WL 2014+ obs ≥ 10        : {len(wl_ok)}  ({len(wl_ok_new)} new)")

print(f"\n{'Continent':8s}  {'Total':>5s}  {'Matched':>7s}  {'WL≥10':>6s}  {'(new)':>6s}")
print('─'*40)
for cont in sorted(set(r['continent'] for r in results)):
    sub   = [r for r in results if r['continent'] == cont]
    m     = [r for r in sub if r['dahiti_id']]
    ok    = [r for r in sub if r.get('wl_2014', 0) >= 10]
    ok_n  = [r for r in ok  if not r['in_pilot']]
    print(f"{cont:8s}  {len(sub):5d}  {len(m):7d}  {len(ok):6d}  {len(ok_n):6d}")

print(f"\nOutput : {out_csv}")
print(f"WL CSVs: {WL_DIR}  ({len(list(WL_DIR.glob('*_wl.csv')))} files)")
