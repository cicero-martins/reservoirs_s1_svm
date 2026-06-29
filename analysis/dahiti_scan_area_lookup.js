/**
 * dahiti_scan_area_lookup.js
 *
 * Paste into GEE Code Editor.
 * For each of the 79 DAHITI Reservoir targets (from dahiti_reservoir_scan_results.csv),
 * finds the nearest GDW polygon within 5 km and prints its area_km2.
 * Output in Console: copy-paste into a spreadsheet to fill area_km2_gee column.
 */

var GDW = ee.FeatureCollection('projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0');

// ── DAHITI scan results (79 rows from dahiti_reservoir_scan_results.csv) ──────
// Format: [dahiti_id, name, lat, lon, wl_2014, wl_range_m]
var TARGETS = [
  [11364, 'Kakhovka',          47.4764, 34.0704, 576,  13.42],
  [10246, 'Sam Rayburn',       31.1348,-94.2137, 556,   7.48],
  [10146, 'Ray Roberts',       33.3615,-97.0557, 555,   5.84],
  [10247, 'Toledo Bend',       31.5527,-93.7793, 531,   4.26],
  [10351, 'Nova Ponte',       -19.1291,-47.3831, 527,  34.13],
  [12974, 'Elwell',            48.3489,-111.3282,505,  10.49],
  [11370, 'Guntersville',      34.4708,-86.1955, 504,   5.20],
  [10348, 'Tabatinga',         -5.9471,-35.4229, 479,   9.85],
  [11473, 'Agua Vermelha',    -19.9524,-49.8783, 467,  10.82],
  [106,   'Tsimlyansk',        47.9090, 42.8741, 436,   4.88],
  [11286, 'Syrdarya',          41.1564, 68.0376, 430,  10.46],
  [11303, 'Smallwood',         54.0255,-64.7099, 415,   7.77],
  [12973, 'Canyon Ferry',      46.5532,-111.6185,393,   7.78],
  [10272, 'Hubbard Creek',     32.7909,-98.9992, 392,   9.60],
  [10271, 'OH Ivie',           31.5436,-99.6825, 381,  11.94],
  [10276, 'Hugo Lake',         34.0592,-95.4137, 373,  12.43],
  [118,   'Mosul Dam',         36.7279, 42.7730, 369,  39.81],
  [10297, 'Yesa',              42.6057,  -1.1147,368,  33.65],
  [112,   'Ataturk',           37.5794, 38.5648, 349,  11.81],
  [11148, 'Eder',              51.1950,  9.0443, 348,  26.75],
  [10274, 'Stamford',          33.0633,-99.6004, 345,   6.88],
  [10304, 'Puente Nuevo',      38.1266,  -4.9769,340,  23.01],
  [11360, 'Passo Fundo',      -27.6353,-52.7460, 336,   9.38],
  [117,   'Kuybyshev',         55.2760, 49.5950, 332,   5.76],
  [11353, 'Falcon',            26.7979,-99.2489, 326,  16.76],
  [10337, 'Unknown BR',        -7.7370,-37.6063, 308,  12.45],
  [10216, 'Tarbela',           34.1067, 72.7435, 303,  62.12],
  [10310, 'Alcantara',         39.7639,  -6.6882,293,  33.95],
  [11331, 'Gouin',             48.5713,-74.6674, 282,   5.73],
  [12971, 'Allegheny',         41.9105,-78.9387, 274,  20.24],
  [11410, 'Paraibuna',        -23.3697,-45.6543, 268,  20.47],
  [11409, 'Peixoto',          -20.4125,-46.8460, 268,  11.55],
  [10307, 'Encoro de Salas',   41.9220,  -7.9328,265,  13.41],
  [11391, 'Paranaiba',        -18.3467,-48.6667, 264,  23.87],
  [10302, 'Caia',              39.0405,  -7.2022,246,  11.30],
  [111,   'Assad',             35.9580, 38.2571, 244,   8.21],
  [10479, 'Vani Vilasa',       13.8366, 76.4365, 244,  19.18],
  [10134, 'Xekaman 1',         14.9714,107.1716, 242,  95.24],
  [115,   'Kremenchuk',        49.2965, 32.5690, 237,   5.02],
  [11266, 'Grupiara',         -18.4443,-47.8403, 224,  38.48],
  [1065,  'Tucurui',           -4.1903,-49.6246, 213,  18.73],
  [10987, 'Truman',            38.2717,-93.4136, 191,   9.73],
  [10147, 'Benbrook',          32.6199,-97.4679, 189,   8.08],
  [11576, 'Gilgel Gibe III',    6.8701, 37.4160, 184,  66.67],
  [10298, 'Mequinenza',        41.3856,  0.2478, 181,  15.33],
  [10341, 'Forggen',           47.6316, 10.7432, 177,  13.45],
  [10311, 'Giribaile',         38.1050,  -3.4701,165,  12.55],
  [11136, 'Eufaula',           35.3095,-95.4200, 162,   4.63],
  [10346, 'Frunas',           -20.6870,-46.1944, 148,  16.61],
  [1007,  'Umbuluzi',         -26.1098, 32.2224, 148,  24.85],
  [10319, 'Walker Lake',       38.7040,-118.7064, 146, 15.10],
  [12968, 'Norfolk Lake',      36.4052,-92.2861, 138,  11.58],
  [11255, 'Ujani',             18.1445, 75.0854, 132,  13.57],
  [10475, 'Broken Bow',        34.1830,-94.6899, 131,   8.72],
  [11108, 'Harlan County',     40.0568,-99.2649, 130,   7.01],
  [10344, 'Capivara',         -22.7734,-51.0553, 128,  13.17],
  [10360, 'Moultrie',          33.3159,-80.0504, 128,   1.24],
  [10301, 'Zujar',             38.9295,  -5.2318,126,  35.23],
  [11372, 'Ozarks',            38.1455,-92.8424, 120,   2.31],
  [10495, 'Murvaul',           32.0368,-94.4350, 117,   2.42],
  [10434, 'Itaipu',           -25.3820,-54.5498, 112,   4.00],
  [10349, 'Serra de Mesa',    -14.1034,-48.3010, 107,  33.29],
  [10562, 'Yguazu',           -25.2594,-55.2200,  99,   5.92],
  [10305, 'Vegus',             38.0686,  -4.2432, 95,  35.33],
  [11393, 'Sterkfontein',     -28.4108, 29.0078,  93,   4.60],
  [10594, 'Acude Oros',        -6.2443,-39.0179,  89,  19.56],
  [10592, 'Contas',           -13.8454,-40.3294,  69,  35.88],
  [10493, 'Ouachita',          34.5972,-93.3560,  68,   4.93],
  [11234, 'Ray Hubbard',       32.8879,-96.5101,  46,   1.51],
  [10501, 'De Gray',           34.2371,-93.1901,  42,   4.40],
  [10279, 'Kickapoo',          33.6439,-98.8256,  27,   3.96],
  [11219, 'Tres Marias',      -18.6404,-45.2486,  20,  22.27],
  [10321, 'Alvaro Obregon',    27.8790,-109.8455, 19,  28.62],
  [11239, 'Clear Lake',        41.8568,-121.1730, 17,   6.25],
  [12967, 'Seminoe',           42.0692,-106.8672, 17,  21.81],
  [10595, 'Jaguaribe',         -5.5470,-38.4446,  13,  36.27],
  [10345, 'Jacarei',          -22.9680,-46.3516,  12,  28.26],
  [10347, 'Itutinga',         -21.3433,-44.6041,  11,  13.06],
  [11252, 'Promissao',        -21.3719,-49.6252,  11,   5.22],
];

// ── Lookup: nearest GDW polygon ───────────────────────────────────────────────
var pts = ee.FeatureCollection(TARGETS.map(function(t) {
  return ee.Feature(
    ee.Geometry.Point([t[3], t[2]]),
    {dahiti_id: t[0], name: t[1], wl_2014: t[4], wl_range_m: t[5]}
  );
}));

var join = ee.Join.saveFirst(matchKey: 'gdw_match', outer: true);
var distFilter = ee.Filter.withinDistance({
  distance: 10000, leftField: '.geo', rightField: '.geo', maxError: 100
});
var joined = join.apply(pts, GDW, distFilter);

var result = joined.map(function(f) {
  var g = f.get('gdw_match');
  var area_km2 = ee.Algorithms.If(
    ee.Algorithms.IsEqual(g, null),
    ee.Number(-1),
    ee.Feature(g).geometry().area(500).divide(1e6)
  );
  var gdwName = ee.Algorithms.If(
    ee.Algorithms.IsEqual(g, null), 'NO_MATCH',
    ee.Feature(g).getString('Res_name')
  );
  return f.set({area_km2: area_km2, gdw_name: gdwName});
});

// Print sorted by area
result.sort('area_km2').evaluate(function(fc) {
  print('dahiti_id | name                     | wl_2014 | wl_range_m | area_km2 | area_ha  | gdw_name');
  fc.features.forEach(function(f) {
    var p = f.properties;
    var ha = p.area_km2 > 0 ? Math.round(p.area_km2 * 100) : -1;
    var inRange = p.area_km2 >= 5 && p.area_km2 <= 100 ? ' ← MEDIUM' : '';
    print([p.dahiti_id, p.name, p.wl_2014, p.wl_range_m, p.area_km2.toFixed(1), ha, p.gdw_name].join(' | ') + inRange);
  });
});
