// ============================================================
// GLOBAL PILOT: A/P × SAR Quality for GROWL Reservoirs
// Sentinel-1 SVM + JRC auto-training → area time series
// AOI: HydroLAKES polygons (users/ee-ciceromartinsjr/HydroLAKES_pilot24)
// Export to Drive: SAR_area_* + JRC_area_* per reservoir
// ============================================================

// ── Pilot reservoir list ──────────────────────────────────────
// Format: [RES_ID, Hylak_ID, name]
var PILOT = [
  // Asia — India
  [3345, 15647,    'Manikdoh'          ],
  [3344, 15553,    'Panam'             ],
  [3439, 15840,    'Periyar'           ],
  [3438, 179882,   'Kakki'             ],
  [3533, 15600,    'Pench_Totladoh'    ],
  [3402, 15105,    'Thein_Ranjit_Sagar'],
  // Europe — Spain / Belgium
  [2332, 172148,   'San_Esteban_ES'   ],
  [2760, 1338571,  'Ry_de_Rome_BE'    ],
  [2376, 172818,   'Borbollon_ES'     ],
  [2405, 173730,   'Minilla_ES'       ],
  [2408, 14552,    'Gabriel_y_Galan_ES'],
  [2418, 14457,    'Ricobayo_ES'      ],
  // N. America — USA / Mexico
  [1240, 8630,     'Pine_River_US'    ],
  [1164, 9794,     'Benito_Juarez_MX' ],
  [1268, 105914,   'Pokegama_US'      ],
  [1244, 735,      'Winnibigoshish_US'],
  [1233, 738,      'Leech_US'         ],
  [ 550, 831,      'Elephant_Butte_US'],
  // Oceania — Australia
  [3938, 184942,   'Maroondah_AU'     ],
  [3941, 1426822,  'OShannassy_AU'    ],
  [3936, 184949,   'Silvan_AU'        ],
  [3911, 184923,   'Upper_Coliban_AU' ],
  [4089, 184368,   'Chichester_AU'    ],
  [3942, 184943,   'Yarra_AU'         ],
];

// ── HydroLAKES (uploaded asset — 24 pilot polygons) ──────────
var hydroLakes = ee.FeatureCollection(
  'projects/ee-ciceromartinsjr/assets/HydroLAKES_pilot24'
);

// ── Study period ──────────────────────────────────────────────
var START = '2014-01-01';
var END   = '2025-06-01';

// ── JRC auto-training ─────────────────────────────────────────
function autoTrainingSamples(aoi) {
  var occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
  // Water: ≥90% occurrence inside AOI; fallback to ≥50%
  var water90 = occ.gte(90).selfMask().sample({
    region: aoi, scale: 30, numPixels: 500, seed: 42, geometries: true
  }).map(function(f) { return f.set('landcover', 1); });
  var water50 = occ.gte(50).selfMask().sample({
    region: aoi, scale: 30, numPixels: 500, seed: 42, geometries: true
  }).map(function(f) { return f.set('landcover', 1); });
  var waterSamples = ee.FeatureCollection(
    ee.Algorithms.If(water90.size().gt(10), water90, water50)
  );
  // Land: ESA WorldCover — independent of JRC water history
  // Exclude permanent water (80), wetland (90), mangrove (95) and JRC max extent
  var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
  var landMask = worldcover.neq(80)
    .and(worldcover.neq(90))
    .and(worldcover.neq(95))
    .and(occ.unmask(0).eq(0))  // also exclude anything JRC ever saw as water
    .selfMask();
  var landSamples = landMask.sample({
    region: aoi.buffer(2000), scale: 10, numPixels: 500, seed: 42, geometries: true
  }).map(function(f) { return f.set('landcover', 2); });
  return waterSamples.merge(landSamples);
}

// ── Best Sentinel-1 orbit ─────────────────────────────────────
function selectBestOrbit(col, aoi, callback) {
  var withAngle = col.map(function(img) {
    var mean = img.select('angle').reduceRegion({
      reducer: ee.Reducer.mean(), geometry: aoi,
      scale: 100, maxPixels: 1e6
    }).get('angle');
    return img.set('mean_angle', mean);
  });
  // Drop images where AOI had no actual coverage (angle reduceRegion → null)
  var validAngle = withAngle.filter(ee.Filter.notNull(['mean_angle']));
  var descending = validAngle.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
  // Fall back to all passes if no descending images exist for this AOI
  var useCol = ee.ImageCollection(
    ee.Algorithms.If(descending.size().gt(0), descending, validAngle)
  );
  var best        = useCol.sort('mean_angle').first();
  var targetAngle = best.getNumber('mean_angle');
  var filtered    = useCol.filter(ee.Filter.and(
    ee.Filter.gte('mean_angle', targetAngle.subtract(3)),
    ee.Filter.lte('mean_angle', targetAngle.add(3))
  ));
  callback(filtered);
}

// ── Main export function ──────────────────────────────────────
function exportReservoir(resId, hylakId, name) {

  // AOI from HydroLAKES polygon
  var feat    = hydroLakes.filter(ee.Filter.eq('Hylak_id', hylakId)).first();
  var aoi     = feat.geometry();
  var area_m2 = aoi.area(1);
  var perim_m = aoi.perimeter(1);
  var ap_m    = area_m2.divide(perim_m);

  // ── 1. SAR SVM classification ─────────────────────────────
  var s1raw = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterDate(START, END)
    .filterBounds(aoi)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  selectBestOrbit(s1raw, aoi, function(s1) {
    s1 = s1.select('VV', 'VH');

    // S1 composite to extract VV/VH values at JRC sample locations
    var s1composite  = s1.median();
    var samplePoints = autoTrainingSamples(aoi);
    var training     = s1composite.sampleRegions({
      collection: samplePoints,
      properties: ['landcover'],
      scale: 10,
      tileScale: 4
    });
    var classifier = ee.Classifier.libsvm({
      kernelType: 'RBF', cost: 1, gamma: 0.01, decisionProcedure: 'Margin'
    }).train({ features: training, classProperty: 'landcover',
               inputProperties: ['VV', 'VH'] });

    var classified = s1.map(function(img) {
      var smooth  = img.focal_mean(30, 'square', 'meters');
      var water   = smooth.classify(classifier).eq(1);
      var area_ha = water.multiply(ee.Image.pixelArea())
        .reduceRegion({ reducer: ee.Reducer.sum(), geometry: aoi,
                        scale: 10, maxPixels: 1e9 })
        .get('classification');
      return ee.Feature(null, {
        date:         img.date().format('YYYY-MM-dd'),
        area_ha:      ee.Number(area_ha).divide(10000),
        ap_m:         ap_m,
        area_aoi_ha:  area_m2.divide(10000),
        perim_aoi_m:  perim_m,
        n_water_pts:  training.filter(ee.Filter.eq('landcover', 1)).size(),
        n_land_pts:   training.filter(ee.Filter.eq('landcover', 2)).size(),
        hylak_id:     hylakId,
        growl_res_id: resId,
        reservoir:    name
      });
    });

    Export.table.toDrive({
      collection:  classified,
      description: 'SAR_area_' + name + '_' + resId,
      folder:      'GROWL_SAR_pilot',
      fileFormat:  'CSV'
    });
  });

  // ── 2. JRC monthly optical area (independent reference) ───
  // waterClass: 0 = no obs, 1 = land, 2 = water
  var jrcMonthly = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
    .filterDate(START, END)
    .filterBounds(aoi)
    .map(function(img) {
      var wc = img.select('water');
      var water_ha = wc.eq(2).multiply(ee.Image.pixelArea())
        .reduceRegion({ reducer: ee.Reducer.sum(), geometry: aoi,
                        scale: 30, maxPixels: 1e9 })
        .get('water');
      var valid_frac = wc.gt(0).reduceRegion({
        reducer: ee.Reducer.mean(), geometry: aoi,
        scale: 30, maxPixels: 1e9
      }).get('water');
      return ee.Feature(null, {
        date:         img.date().format('YYYY-MM-dd'),
        jrc_area_ha:  ee.Number(water_ha).divide(10000),
        valid_frac:   valid_frac,
        ap_m:         ap_m,
        area_aoi_ha:  area_m2.divide(10000),
        hylak_id:     hylakId,
        growl_res_id: resId,
        reservoir:    name
      });
    });

  Export.table.toDrive({
    collection:  jrcMonthly,
    description: 'JRC_area_' + name + '_' + resId,
    folder:      'GROWL_SAR_pilot',
    fileFormat:  'CSV'
  });
}

// ── Launch all exports ────────────────────────────────────────
// Each reservoir → 2 tasks: SAR_area_* + JRC_area_*
PILOT.forEach(function(p) {
  exportReservoir(p[0], p[1], p[2]);
});

print('Iniciando ' + PILOT.length + ' reservatórios → ' +
      (PILOT.length * 2) + ' tasks em GROWL_SAR_pilot/');
