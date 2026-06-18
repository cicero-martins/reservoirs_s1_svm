// ============================================================
// New regions pilot — end-to-end automatic pipeline
// South America (BR, AR, CO) + Africa (ZA, SD)
//
// Same methodology as exportSicilianJRC.js:
//   AOI = JRC Global Surface Water max_extent polygon
//         (largest water body within 2km of HydroLAKES centroid)
//   HydroLAKES = search anchor only
//   Training = WorldCover auto (land ring 100m–2km from AOI)
//   Reference = JRC monthly optical area
//
// Upload HydroLAKES_newregions6.zip as GEE asset before running:
//   Asset path: projects/ee-ciceromartinsjr/assets/HydroLAKES_newregions6
//
// Exports (→ Google Drive / GROWL_SAR_pilot/):
//   SAR_auto_{name}.csv   — SAR with WorldCover auto-training
//   JRC_area_{name}.csv   — JRC monthly optical area
// ============================================================

var START = '2014-01-01';
var END   = '2025-06-01';

var hydroLakes = ee.FeatureCollection(
  'projects/ee-ciceromartinsjr/assets/HydroLAKES_newregions6'
);

var RESERVOIRS = [
  // Brazil
  {name: 'Passauna_BR',           hylak_id: 120969, country: 'Brazil',    region: 'S. America'},
  {name: 'Tres_Marias_BR',        hylak_id: 928,    country: 'Brazil',    region: 'S. America'},
  // Argentina
  {name: 'Cerros_Colorados_AR',   hylak_id: 979,    country: 'Argentina', region: 'S. America'},
  // Colombia
  {name: 'Calima_CO',             hylak_id: 116772, country: 'Colombia',  region: 'S. America'},
  // Africa
  {name: 'Theewaterskloof_ZA',    hylak_id: 16612,  country: 'S. Africa', region: 'Africa'},
  {name: 'Roseires_SD',           hylak_id: 1543,   country: 'Sudan',     region: 'Africa'},
];

// ── JRC max_extent polygon ────────────────────────────────────
// Largest contiguous water body within searchBuffer of HydroLAKES centroid.
// Tres_Marias (800 km²) and Roseires (225 km²) need a larger search radius.
function jrcMaxExtentPoly(hydroPoly, searchBuffer) {
  var searchArea = hydroPoly.buffer(searchBuffer);
  var jrcMax = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    .select('max_extent').eq(1).selfMask();
  var waterVecs = jrcMax.reduceToVectors({
    geometry:       searchArea,
    scale:          30,
    maxPixels:      1e10,
    bestEffort:     true,
    geometryType:   'polygon',
    eightConnected: true,
  });
  return waterVecs.sort('count', false).first().geometry();
}

// ── Auto-training samples ─────────────────────────────────────
// lakePoly : JRC max_extent polygon (water samples + A/P)
// aoi      : lakePoly.buffer(100) (detection zone)
// Land ring: aoi → 2km (unambiguously land)
function autoTrainingSamples(lakePoly, aoi) {
  var occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');

  var water90 = occ.gte(90).selfMask().sample({
    region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true
  }).map(function(f) { return f.set('landcover', 1); });

  var water50 = occ.gte(50).selfMask().sample({
    region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true
  }).map(function(f) { return f.set('landcover', 1); });

  var waterSamples = ee.FeatureCollection(
    ee.Algorithms.If(water90.size().gt(10), water90, water50)
  );

  var landRing = lakePoly.buffer(2000).difference(aoi);
  var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
  var landMask = wc.neq(80).and(wc.neq(90)).and(wc.neq(95))
    .and(occ.unmask(0).eq(0)).selfMask();
  var landSamples = landMask.sample({
    region: landRing, scale: 10, numPixels: 500, seed: 42, geometries: true
  }).map(function(f) { return f.set('landcover', 2); });

  return waterSamples.merge(landSamples);
}

// ── Best orbit ────────────────────────────────────────────────
function selectBestOrbit(col, aoi, callback) {
  var withAngle = col.map(function(img) {
    var mean = img.select('angle').reduceRegion({
      reducer: ee.Reducer.mean(), geometry: aoi, scale: 100, maxPixels: 1e6
    }).get('angle');
    return img.set('mean_angle', mean);
  });
  var valid      = withAngle.filter(ee.Filter.notNull(['mean_angle']));
  var descending = valid.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
  var useCol     = ee.ImageCollection(
    ee.Algorithms.If(descending.size().gt(0), descending, valid)
  );
  var best        = useCol.sort('mean_angle').first();
  var targetAngle = best.getNumber('mean_angle');
  callback(useCol.filter(ee.Filter.and(
    ee.Filter.gte('mean_angle', targetAngle.subtract(3)),
    ee.Filter.lte('mean_angle', targetAngle.add(3))
  )));
}

// ── Export per reservoir ──────────────────────────────────────
RESERVOIRS.forEach(function(r) {
  var hydroPoly = hydroLakes
    .filter(ee.Filter.eq('Hylak_id', r.hylak_id))
    .first().geometry();

  // Large reservoirs need wider search to capture full JRC extent
  var searchBuffer = (r.hylak_id === 928 || r.hylak_id === 1543) ? 5000 : 2000;

  var lakePoly = jrcMaxExtentPoly(hydroPoly, searchBuffer);
  var aoi      = lakePoly.buffer(100);

  var lake_area_m2      = lakePoly.area(1);
  var lake_perim_m      = lakePoly.perimeter(1);
  var ap_m              = lake_area_m2.divide(lake_perim_m);
  var aoi_area_m2       = aoi.area(1);
  var hydrolakes_area_m2 = hydroPoly.area(1);

  // ── 1. SAR with WorldCover auto-training ─────────────────
  var s1raw = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterDate(START, END)
    .filterBounds(aoi)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  selectBestOrbit(s1raw, aoi, function(s1) {
    s1 = s1.select('VV', 'VH');
    var s1composite  = s1.median();
    var samplePoints = autoTrainingSamples(lakePoly, aoi);
    var training     = s1composite.sampleRegions({
      collection: samplePoints, properties: ['landcover'],
      scale: 10, tileScale: 4
    });
    var classifier = ee.Classifier.libsvm({
      kernelType: 'RBF', cost: 1, gamma: 0.01, decisionProcedure: 'Margin'
    }).train({features: training, classProperty: 'landcover',
              inputProperties: ['VV', 'VH']});

    var classified = s1.map(function(img) {
      var smooth  = img.focal_mean(30, 'square', 'meters');
      var water   = smooth.classify(classifier).eq(1);
      var area_ha = water.multiply(ee.Image.pixelArea())
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                       scale: 10, maxPixels: 1e10})
        .get('classification');
      return ee.Feature(null, {
        date:               img.date().format('YYYY-MM-dd'),
        area_ha:            ee.Number(area_ha).divide(10000),
        ap_m:               ap_m,
        area_aoi_ha:        aoi_area_m2.divide(10000),
        jrc_max_area_ha:    lake_area_m2.divide(10000),
        hydrolakes_area_ha: hydrolakes_area_m2.divide(10000),
        n_water_pts:        training.filter(ee.Filter.eq('landcover', 1)).size(),
        n_land_pts:         training.filter(ee.Filter.eq('landcover', 2)).size(),
        hylak_id:           r.hylak_id,
        reservoir:          r.name,
        country:            r.country,
        region:             r.region,
      });
    });

    Export.table.toDrive({
      collection:  classified,
      description: 'SAR_auto_' + r.name,
      folder:      'GROWL_SAR_pilot',
      fileFormat:  'CSV'
    });
  });

  // ── 2. JRC monthly optical area ──────────────────────────
  var jrcMonthly = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
    .filterDate(START, END)
    .filterBounds(aoi)
    .map(function(img) {
      var wc = img.select('water');
      var water_ha = wc.eq(2).multiply(ee.Image.pixelArea())
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                       scale: 30, maxPixels: 1e10})
        .get('water');
      var valid_frac = wc.gt(0).reduceRegion({
        reducer: ee.Reducer.mean(), geometry: aoi,
        scale: 30, maxPixels: 1e10
      }).get('water');
      return ee.Feature(null, {
        date:               img.date().format('YYYY-MM-dd'),
        jrc_area_ha:        ee.Number(water_ha).divide(10000),
        valid_frac:         valid_frac,
        ap_m:               ap_m,
        area_aoi_ha:        aoi_area_m2.divide(10000),
        jrc_max_area_ha:    lake_area_m2.divide(10000),
        hydrolakes_area_ha: hydrolakes_area_m2.divide(10000),
        hylak_id:           r.hylak_id,
        reservoir:          r.name,
        country:            r.country,
        region:             r.region,
      });
    });

  Export.table.toDrive({
    collection:  jrcMonthly,
    description: 'JRC_area_' + r.name,
    folder:      'GROWL_SAR_pilot',
    fileFormat:  'CSV'
  });
});

print('12 tasks → GROWL_SAR_pilot/');
print('  SAR_auto_{name}.csv  (6 tasks)');
print('  JRC_area_{name}.csv  (6 tasks)');
print('AOI: JRC max_extent | HydroLAKES = search anchor only');
print('Note: Tres_Marias and Roseires use 5km search buffer (large reservoirs)');
