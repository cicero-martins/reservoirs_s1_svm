/**
 * _lookup_gdw_ids.js
 *
 * For each flagged v4 reservoir, finds the nearest GDW feature within 25 km
 * and prints: name | GDW_ID | DAM_NAME | RES_NAME | AREA_SKM | dist_km
 *
 * Run via EE Tasks (EE Tasks: run GEE script) — output appears in the
 * VS Code OUTPUT panel (note: print() is async, order may vary).
 * Use the GDW_ID values to fill in PILOT_RESERVOIRS in exportGlobalPilotV4.js.
 */

var GDW = ee.FeatureCollection('projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0');

var FLAGGED = [
  // Grupo A — wrong large polygon
  ['Aguilar',      42.793,  -4.270],
  ['Almus',        40.378,  36.908],
  ['Boegoeberg',  -29.026,  22.155],
  ['Cardinia',    -37.935, 145.510],
  ['Chelmsford',  -27.967,  29.852],
  ['Cruz_del_Eje',-30.728, -64.804],
  ['Ebro_Embalse', 42.960,  -4.060],
  ['Eleven_Mile',  38.930,-105.534],
  ['Occhito',      41.534,  14.913],
  ['Pineview',     41.273,-111.839],
  ['Plastiras',    39.233,  21.776],
  ['Riano',        42.993,  -5.017],
  // Grupo B — wrong small / not found
  ['Abdelmoumen',  30.373,  -9.545],
  ['Bleiloch',     50.637,  11.697],
  ['Blue_Rock',   -38.320, 146.180],
  ['Cecita',       39.333,  16.620],
  ['Demirkopru',   38.794,  28.621],
  ['Guajaraz',     39.675,  -4.107],
  ['Katse',       -29.365,  28.521],
  ['La_Vina',     -31.533, -64.503],
  ['Mohale',      -29.550,  28.143],
  ['Nagle',       -29.597,  30.784],
  ['Oued_Makhazine',35.167, -5.533],
  ['Shaharchay',   37.640,  45.009],
  ['Siurana',      41.197,   0.914],
  ['Suat_Ugurlu',  41.117,  36.050],
  ['Triouzoune',   45.520,   2.265],
  ['Wadi_Dayqah',  22.724,  57.863],
];

print('name | GDW_ID | DAM_NAME | RES_NAME | AREA_SKM | dist_km');
print('─────────────────────────────────────────────────────────');

FLAGGED.forEach(function(r) {
  var name = r[0];
  var pt   = ee.Geometry.Point([r[2], r[1]]);

  var candidates = GDW.filterBounds(pt.buffer(25000))
    .map(function(f) {
      return f.set('_dist_m', f.geometry().centroid(1).distance(pt, 1));
    }).sort('_dist_m');

  var best = candidates.first();

  // Format output as pipe-separated line for easy reading
  var line = ee.String(name).cat(' | ')
    .cat(ee.Number(best.get('GDW_ID')).int().format())
    .cat(' | ')
    .cat(ee.String(best.get('DAM_NAME')))
    .cat(' | ')
    .cat(ee.String(best.get('RES_NAME')))
    .cat(' | ')
    .cat(ee.Number(best.get('AREA_SKM')).format('%.2f'))
    .cat(' km² | ')
    .cat(ee.Number(best.get('_dist_m')).divide(1000).format('%.1f'))
    .cat(' km');

  print(line);

  // Also print count so you know if no match was found
  print(ee.String('  (').cat(candidates.size().format()).cat(' candidates within 25 km)'));
});
