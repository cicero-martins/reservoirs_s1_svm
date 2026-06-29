/**
 * _diag_orbit_scoring.js
 *
 * Diagnostic: exports orbit scoring table for the 4 Sicilian reservoirs.
 * For each reservoir, produces a CSV with one row per orbit:
 *   reservoir, orbit, pass, n_images, mean_cov_px, selected
 *
 * Used to diagnose selectBestOrbit returning 1 result for Pozzillo / Rosamarina.
 * Paste into GEE Code Editor and run. Exports to Drive folder GEE_GlobalPilotV2.
 */

var JRC_GSW        = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var S1_GRD         = ee.ImageCollection('COPERNICUS/S1_GRD');
var HYDROLAKES_SIC = ee.FeatureCollection('projects/ee-ciceromartinsjr/assets/HydroLAKES_sicily4');
var BANDS          = ['VV', 'VH'];
var S1_START       = '2014-10-01';
var S1_END         = '2024-12-31';
var DRIVE_FOLDER   = 'GEE_GlobalPilotV2';

// Sicily reservoirs: [name, hylak_id]
var SICILY = [
  ['Ancipa',     1369046],
  ['Poma',        173610],
  ['Pozzillo',    173729],
  ['Rosamarina',  173633],
];

function getLakePoly(hylakId) {
  var hydroGeom = HYDROLAKES_SIC
    .filter(ee.Filter.eq('Hylak_id', hylakId)).first().geometry();
  var searchArea = hydroGeom.buffer(2000);
  var jrcMax     = JRC_GSW.select('max_extent').eq(1).selfMask();
  var waterVecs  = jrcMax.reduceToVectors({
    geometry: searchArea, scale: 30, maxPixels: 1e9,
    bestEffort: true, geometryType: 'polygon',
    eightConnected: true, tileScale: 4,
  });
  return waterVecs.sort('count', false).first().geometry();
}

SICILY.forEach(function(res) {
  var rName   = res[0];
  var hylakId = res[1];

  var lakePoly = getLakePoly(hylakId);
  var aoi      = lakePoly.buffer(100);

  // ── S1 collection ────────────────────────────────────────────────────────
  var s1col = S1_GRD
    .filterBounds(aoi)
    .filterDate(S1_START, S1_END)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(BANDS)
    .map(function(img) {
      return img.focal_mean(30, 'circle', 'meters')
        .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                              'relativeOrbitNumber_start', 'angle']);
    });

  // Print total image count to console
  s1col.size().evaluate(function(n) {
    print(rName + '  total S1 images:', n);
  });

  // Print AOI area to console (sanity check on geometry)
  aoi.area(1).evaluate(function(a) {
    print(rName + '  AOI area (ha):', (a / 1e4).toFixed(1));
  });

  // ── Per-orbit scoring ────────────────────────────────────────────────────
  var withOrbit = s1col.map(function(img) {
    return img.set({
      '_relOrbit': img.getNumber('relativeOrbitNumber_start'),
      '_pass':     img.getString('orbitProperties_pass'),
    });
  });

  var orbits = withOrbit.aggregate_array('_relOrbit').distinct();

  var scored = ee.FeatureCollection(orbits.map(function(o) {
    var sub  = withOrbit.filter(ee.Filter.eq('_relOrbit', o));
    var n    = sub.size();
    var pass = ee.String(sub.first().get('_pass'));
    var covPerImg = sub.map(function(img) {
      var count = img.reduceRegion({
        reducer: ee.Reducer.count(), geometry: aoi,
        scale: 100, maxPixels: 1e7, bestEffort: true,
      }).getNumber('VV');
      return img.set('_cov', count);
    });
    var meanCov = covPerImg.aggregate_mean('_cov');
    return ee.Feature(null, {
      'reservoir':    rName,
      'orbit':        o,
      'pass':         pass,
      'n_images':     n,
      'mean_cov_px':  meanCov,
    });
  }));

  // Mark which orbit would be selected (max mean_cov_px)
  var best      = scored.sort('mean_cov_px', false).first();
  var bestOrbit = best.getNumber('orbit');
  var scoredWithFlag = scored.map(function(f) {
    return f.set('selected', f.getNumber('orbit').eq(bestOrbit));
  });

  Export.table.toDrive({
    collection:     scoredWithFlag.sort('mean_cov_px', false),
    description:    'diag_orbits_' + rName,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: 'diag_orbits_' + rName,
    fileFormat:     'CSV',
  });

  print('Orbit diagnostic queued: ' + rName);
});
