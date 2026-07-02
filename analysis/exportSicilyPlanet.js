/**
 * exportSicilyPlanet.js
 *
 * Re-export of SAR water-area for the 4 PlanetScope-validated Sicilian reservoirs
 * over the PlanetScope window (2024–2025), using the IDENTICAL pipeline as
 * exportGlobalPilotV4.js (same AOI / orbit selection / Otsu / SVM / post-processing).
 *
 * WHY: the pre-existing validation_data/GEEvalidation-extracted series were produced
 * by a different (older) extraction — visibly noisier (no clean+smooth, possibly
 * different AOI/orbit) → NOT comparable to the 28-reservoir global pilot. This script
 * regenerates Otsu (Tier 1) and SVM dual (Tier 3) for Sicily under the same pipeline so
 * the PlanetScope-anchored comparison is apples-to-apples with the global ΔKGE.
 *
 * Run per CLASSIFIER: 'SVM' → GEE_SicilyPlanet, 'VV_OTSU' → GEE_SicilyPlanet_VVotsu,
 * 'SVM_ADAPTIVE' (per-scene training) → GEE_SicilyPlanet_SVMadapt.
 * Only 4 reservoirs → single batch [0,4] fits one Code Editor session.
 * JRC is OFF: GSW Monthly History ends ~2021, so PlanetScope (not JRC) is the reference.
 */

// ── Configuration ─────────────────────────────────────────────────────────────
var CFG = {
  s1_start:               '2024-01-01',   // PlanetScope window is 2024-05 → 2025-05
  s1_end:                 '2025-12-31',
  jrc_occ_thresh:         95,
  jrc_occ_fallback:       80,
  train_year:             2023,           // fixed annual mosaic — same as global pilot
  clean_scale_m:          30,
  max_pixels:             1e9,
  keep_largest_only:      false,
  land_ring_inner_m:      500,
  land_ring_outer_m:      2000,
  drive_folder:           'GEE_SicilyPlanet',
  drive_folder_jrc:       'GEE_SicilyPlanet_JRC',
  composite_window_days:  6,
  coverage_strict_pct:    90,
  min_coverage_pct:       50,
};

var JRC_ONLY = false;

// ── Classifier selection (run per mode) ───────────────────────────────────────
// 'SVM'          : dual-pol VV+VH RBF SVM trained once on the fixed 2023 mosaic (Tier 3).
// 'SVM_ADAPTIVE' : dual-pol VV+VH SVM RE-TRAINED PER SCENE — fixed JRC water/land sample
//                  points, but backscatter sampled from each scene → boundary adapts per
//                  scene like Otsu (removes the arbitrary 2023 baseline). Folder ..._SVMadapt.
// 'VV_OTSU'      : single-pol VV per-scene Otsu, same post-processing — Tier 1 (..._VVotsu).
var CLASSIFIER = 'VV_OTSU_FAST';  // 'SVM' | 'SVM_ADAPTIVE' | 'VV_OTSU' | 'VV_OTSU_FAST'

var USE_OTSU = (CLASSIFIER === 'VV_OTSU' || CLASSIFIER === 'VV_OTSU_FAST');
var FAST     = (CLASSIFIER === 'VV_OTSU_FAST');
var ADAPTIVE = (CLASSIFIER === 'SVM_ADAPTIVE');
var USE_SVM  = (CLASSIFIER === 'SVM' || CLASSIFIER === 'SVM_ADAPTIVE');

var OTSU = { band: 'VV', hist_buffer_m: 500, hist_buckets: 256 };

// PlanetScope is the reference here — JRC has no data for 2024-2025. Never export JRC.
var EXPORT_JRC = false;

var MODE_SUFFIX = (CLASSIFIER === 'VV_OTSU')      ? '_VVotsu'
                : (CLASSIFIER === 'VV_OTSU_FAST') ? '_VVfast'
                : (CLASSIFIER === 'SVM_ADAPTIVE') ? '_SVMadapt'
                : '';
var SAR_FOLDER  = CFG.drive_folder + MODE_SUFFIX;

// ── Sicilian reservoir list ───────────────────────────────────────────────────
// Format: [name, lat, lon, gdw_id, dahiti_id, area_ha_approx, hylak_id]
// gdw_id = null → coordinate fallback (largest JRC polygon within 10 km), same as
// most global-pilot entries. Coords from exportGlobalPilotV2.js / exportSicilyMasks.js.
var PILOT_RESERVOIRS = [
  ['Ancipa',     37.887, 14.565, null, null,    330, null],  // Nebrodi; narrow
  ['Pozzillo',   37.700, 14.530, null, null,    930, null],  // largest Sicilian — coord FIXED:
  // 37.783,14.635 resolved to ANCIPA's polygon (108 ha, ap_m=90) via the 20 km
  // fallback (no JRC water within 10 km of that point). 37.700,14.530 resolves to
  // the true Pozzillo body (JRC max_extent 646 ha, ap_m=240). Verified vs JRC, 30 Jun.
  ['Poma',       37.994, 13.090, null, 42134,   620, null],
  ['Rosamarina', 37.944, 13.640, null, 42122,   440, null],
];

// ── Datasets ──────────────────────────────────────────────────────────────────
var BANDS   = ['VV', 'VH'];
var S1_GRD  = ee.ImageCollection('COPERNICUS/S1_GRD');
var JRC_GSW = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var JRC_OCC = JRC_GSW.select('occurrence');
var WC      = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var GDW     = ee.FeatureCollection('projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0');

// ── Helper: get JRC max_extent polygon ────────────────────────────────────────
function getLakePoly(lat, lon, gdwId, hylakId) {
  var jrcMax = JRC_GSW.select('max_extent').eq(1).selfMask();

  function jrcMaxExtentPoly(hydroGeom) {
    var searchArea = hydroGeom.buffer(2000);
    var waterVecs  = jrcMax.reduceToVectors({
      geometry: searchArea, scale: 30, maxPixels: 1e9,
      bestEffort: true, geometryType: 'polygon',
      eightConnected: true, tileScale: 4,
    });
    return waterVecs.sort('count', false).first().geometry();
  }

  if (gdwId !== null) {
    var hydroGeom = GDW.filter(ee.Filter.eq('GDW_ID', gdwId)).first().geometry();
    return jrcMaxExtentPoly(hydroGeom);
  }

  var pt   = ee.Geometry.Point([lon, lat]);
  var vecs = jrcMax.reduceToVectors({
    geometry: pt.buffer(20000), scale: 30, maxPixels: 1e8,
    bestEffort: true, reducer: ee.Reducer.countEvery(),
    geometryType: 'polygon', eightConnected: true, tileScale: 4,
  });
  var withDist = vecs.map(function(f) {
    return f.set('_dist', f.geometry().centroid(1).distance(pt, 1));
  });
  var nearby = withDist.filter(ee.Filter.lte('_dist', 10000));
  return ee.Geometry(ee.Algorithms.If(
    nearby.size().gt(0),
    nearby.sort('count', false).first().geometry(),
    withDist.sort('count', false).first().geometry()
  ));
}

// ── Helper: select best orbit (highest incidence angle + AOI coverage gate) ────
function selectBestOrbit(s1Raw, aoi) {
  var aoiArea = aoi.area(1);
  var withStats = s1Raw.map(function(img) {
    var meanAngle = img.select('angle').reduceRegion({
      reducer: ee.Reducer.mean(), geometry: aoi,
      scale: 100, maxPixels: 1e7, bestEffort: true,
    }).getNumber('angle');
    var pct = img.select('VV').mask().multiply(ee.Image.pixelArea())
      .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                     scale: 100, maxPixels: 1e7, bestEffort: true})
      .getNumber('VV').divide(aoiArea).multiply(100);
    return img.set({
      '_relOrbit':      img.getNumber('relativeOrbitNumber_start'),
      '_pass':          img.getString('orbitProperties_pass'),
      'angle_mean':     meanAngle,
      'percentCovered': pct,
    });
  }).filter(ee.Filter.notNull(['angle_mean']));

  var orbits = withStats.aggregate_array('_relOrbit').distinct();
  var scored = ee.FeatureCollection(orbits.map(function(o) {
    var sub     = withStats.filter(ee.Filter.eq('_relOrbit', o));
    var covered = sub.filter(ee.Filter.gte('percentCovered', CFG.coverage_strict_pct));
    return ee.Feature(null, {
      'orbit':      o,
      'pass':       ee.String(sub.first().get('_pass')),
      'angle_mean': sub.aggregate_mean('angle_mean'),
      'n':          sub.size(),
      'n_covered':  covered.size(),
    });
  }));

  var qualified  = scored.filter(ee.Filter.gte('n', 20)).filter(ee.Filter.gt('n_covered', 0));
  var candidates = ee.FeatureCollection(ee.Algorithms.If(
    qualified.size().gt(0), qualified,
    scored.filter(ee.Filter.gt('n_covered', 0))
  ));
  candidates = ee.FeatureCollection(ee.Algorithms.If(
    candidates.size().gt(0), candidates, scored
  ));
  var best = ee.Feature(candidates.sort('angle_mean', false).first());

  var bestCol = withStats.filter(ee.Filter.and(
    ee.Filter.eq('_relOrbit', best.getNumber('orbit')),
    ee.Filter.eq('_pass',     best.getString('pass'))
  ));
  var strict = bestCol.filter(ee.Filter.gte('percentCovered', CFG.coverage_strict_pct));
  return ee.ImageCollection(ee.Algorithms.If(
    strict.size().gt(0), strict,
    bestCol.filter(ee.Filter.gte('percentCovered', CFG.min_coverage_pct))
  ));
}

// ── Helper: fill partial swath-edge coverage gaps ─────────────────────────────
function fillCoverageGaps(s1Col, windowDays) {
  if (!windowDays) return s1Col;
  var dates = s1Col.aggregate_array('date').distinct().sort();
  return ee.ImageCollection(dates.map(function(d) {
    var t      = ee.Date(d);
    var window = s1Col.filterDate(
      t.advance(ee.Number(windowDays).multiply(-1), 'day'),
      t.advance(ee.Number(windowDays).add(1),       'day')
    );
    return window.sort('system:time_start', false).mosaic()
      .set('system:time_start', t.millis())
      .set('date', d);
  }));
}

// ── Otsu threshold (between-class-variance) ───────────────────────────────────
function otsu(histogram) {
  histogram = ee.Dictionary(histogram);
  var counts  = ee.Array(histogram.get('histogram'));
  var means   = ee.Array(histogram.get('bucketMeans'));
  var size    = means.length().get([0]);
  var total   = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum     = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean    = sum.divide(total);
  var indices = ee.List.sequence(1, size);
  var bss = indices.map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount  = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans  = means.slice(0, 0, i);
    var aMean   = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
    var bCount  = total.subtract(aCount);
    var bMean   = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2))
       .add(bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  return means.sort(ee.Array(bss)).get([-1]);
}

function computeOtsuWater(img, lakePoly) {
  var histRegion = lakePoly.buffer(OTSU.hist_buffer_m);
  var hist = img.select(OTSU.band).reduceRegion({
    reducer:  ee.Reducer.histogram(OTSU.hist_buckets),
    geometry: histRegion, scale: 10, maxPixels: 1e9, bestEffort: true, tileScale: 4,
  }).get(OTSU.band);
  var threshold = ee.Number(otsu(hist));
  return img.select(OTSU.band).lt(threshold).rename('Water').set('_otsu_db', threshold);
}

// ── Helper: classify single image and compute water area ──────────────────────
// samplePoints: JRC water(1)/land(2) sample geometries (null unless USE_SVM).
//   SVM (fixed) uses pre-trained `svm`; SVM_ADAPTIVE re-trains per scene from samplePoints.
function classifyImage(img, svm, lakePoly, samplePoints) {
  var aoi = lakePoly.buffer(100);
  var water;
  if (USE_OTSU) {
    water = computeOtsuWater(img, lakePoly).clip(aoi);
  } else if (ADAPTIVE) {
    var perScene = img.select(BANDS).sampleRegions({
      collection: samplePoints, properties: ['landcover'], scale: 30,
    }).filter(ee.Filter.inList('landcover', ee.List([1, 2])))
      .filter(ee.Filter.notNull(BANDS));
    var svmScene = ee.Classifier.libsvm({kernelType: 'RBF', cost: 1, gamma: 0.01})
      .train({features: perScene, classProperty: 'landcover', inputProperties: BANDS});
    water = img.select(BANDS).classify(svmScene).eq(1).clip(aoi).rename('Water');
  } else {
    water = img.select(BANDS).classify(svm).eq(1).clip(aoi).rename('Water');
  }

  if (FAST) {
    var areaFast_m2 = water.rename('Water').multiply(ee.Image.pixelArea())
      .reduceRegion({reducer: ee.Reducer.sum(), geometry: lakePoly,
                     scale: 10, maxPixels: 1e8, bestEffort: true}).get('Water');
    return img.addBands(water.rename('Water'))
      .set('_area_m2',      areaFast_m2)
      .set('_area_ha',      ee.Number(areaFast_m2).divide(1e4))
      .set('_ap_m_dynamic', -1);
  }
  var mask   = water.unmask(0).clip(aoi);
  var dist   = mask.fastDistanceTransform(30).clip(aoi);
  var filled = dist.lte(0.5).updateMask(dist.lte(0.5)).where(mask, 1).rename('WaterFilled');

  var polys = filled.reduceToVectors({
    geometryType: 'polygon', reducer: ee.Reducer.countEvery(),
    scale: CFG.clean_scale_m, maxPixels: CFG.max_pixels,
    bestEffort: true, tileScale: 4,
  });
  var withFlag = polys.map(function(f) {
    var inside = lakePoly.contains(f.geometry().centroid({maxError: 1}), ee.ErrorMargin(1));
    return f.set('_inside', inside);
  });
  var insidePolys = withFlag.filter(ee.Filter.eq('_inside', 1));
  var keptPolys   = ee.FeatureCollection(ee.Algorithms.If(
    insidePolys.size().gt(0),
    insidePolys,
    ee.FeatureCollection([polys.map(function(f) {
      return f.set('_area', f.geometry().area({maxError: 1}));
    }).sort('_area', false).first()])
  ));

  var keptMask = ee.Image().paint({featureCollection: keptPolys, color: 1}).rename('KeptRegionMask');
  var cleaned  = filled.updateMask(keptMask).rename('WaterCleaned');
  var area_m2  = cleaned.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, scale: 10, maxPixels: 1e8, bestEffort: true,
  }).get('WaterCleaned');

  var mergedGeom = keptPolys.union(1).geometry();
  var dynPerim_m = mergedGeom.perimeter(1);
  var dynAP_m    = ee.Number(area_m2).divide(ee.Number(dynPerim_m).max(1));

  return img.addBands([water, filled, cleaned])
    .set('_area_m2', area_m2)
    .set('_area_ha', ee.Number(area_m2).divide(1e4))
    .set('_ap_m_dynamic', dynAP_m);
}

// ── Main export loop ───────────────────────────────────────────────────────────
var BATCH_SLICE = [0, 4];   // 4 reservoirs fit one session

PILOT_RESERVOIRS.slice(BATCH_SLICE[0], BATCH_SLICE[1]).forEach(function(res) {
  var rName     = res[0];
  var lat       = res[1];
  var lon       = res[2];
  var gdwId     = res[3];
  var areaHaEst = res[5];
  var hylakId   = res[6];

  var lakePoly  = getLakePoly(lat, lon, gdwId, hylakId);
  var aoi       = lakePoly.buffer(100);
  var trainClip = aoi.buffer(CFG.land_ring_outer_m);

  var lakeArea_m2 = lakePoly.area(1);
  var lakePerim_m = lakePoly.perimeter(1);
  var ap_m        = ee.Number(lakeArea_m2).divide(lakePerim_m);

  if (!JRC_ONLY) {
    var svm = null;
    var samplePoints = null;
    if (USE_SVM) {
    // JRC water/land sample POINTS (static; backscatter sampled later — fixed 2023 or per-scene).
    var waterStrict = JRC_OCC.gte(CFG.jrc_occ_thresh).selfMask().sample({
      region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true,
    }).map(function(f) { return f.set('landcover', 1); });
    var waterFallback = JRC_OCC.gte(CFG.jrc_occ_fallback).selfMask().sample({
      region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true,
    }).map(function(f) { return f.set('landcover', 1); });
    var waterSamples = ee.FeatureCollection(
      ee.Algorithms.If(waterStrict.size().gt(10), waterStrict, waterFallback)
    );

    var landRing = lakePoly.buffer(CFG.land_ring_outer_m)
                    .difference(lakePoly.buffer(CFG.land_ring_inner_m));
    var landMask = WC.neq(80).and(WC.neq(90)).and(WC.neq(95))
      .and(JRC_OCC.unmask(0).eq(0)).selfMask();
    var landSamples = landMask.sample({
      region: landRing, scale: 30, numPixels: 500, seed: 42, geometries: true,
    }).map(function(f) { return f.set('landcover', 2); });

    samplePoints = waterSamples.merge(landSamples);

    if (CLASSIFIER === 'SVM') {
      var s1Composite = S1_GRD
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filterBounds(trainClip)
        .filter(ee.Filter.calendarRange(CFG.train_year, CFG.train_year, 'year'))
        .select(BANDS).mosaic()
        .focal_mean(30, 'circle', 'meters').clip(trainClip);

      var trainedSamples = s1Composite.select(BANDS).sampleRegions({
        collection: samplePoints, properties: ['landcover'], scale: 30,
      }).filter(ee.Filter.inList('landcover', ee.List([1, 2])))
        .filter(ee.Filter.notNull(BANDS));

      svm = ee.Classifier.libsvm({kernelType: 'RBF', cost: 1, gamma: 0.01})
        .train({features: trainedSamples, classProperty: 'landcover', inputProperties: BANDS});
    }  // SVM_ADAPTIVE trains per-scene inside classifyImage
    }  // end if (USE_SVM)

    var s1Raw = S1_GRD
      .filterBounds(aoi).filterDate(CFG.s1_start, CFG.s1_end)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.eq('resolution_meters', 10))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

    var bestCol = selectBestOrbit(s1Raw, aoi);
    var bestOrbitNum  = ee.Number(bestCol.first().get('relativeOrbitNumber_start'));
    var bestOrbitPass = ee.String(bestCol.first().get('orbitProperties_pass'));

    var s1Proc = bestCol.map(function(img) {
      return img.select(BANDS).focal_mean(30, 'circle', 'meters')
        .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                              'relativeOrbitNumber_start'])
        .set('date', img.date().format('YYYY-MM-dd'));
    });

    var filledCol = fillCoverageGaps(s1Proc, CFG.composite_window_days)
      .map(function(img) {
        return img.set('relativeOrbitNumber_start', bestOrbitNum)
                  .set('orbitProperties_pass',      bestOrbitPass);
      });

    var areaSeries = filledCol.map(function(img) {
      return classifyImage(img, svm, lakePoly, samplePoints);
    }).map(function(img) {
      return ee.Feature(null, {
        'date':          ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
        'area_m2':       img.get('_area_m2'),
        'area_ha':       img.get('_area_ha'),
        'relOrbit':      img.get('relativeOrbitNumber_start'),
        'passDirection': img.get('orbitProperties_pass'),
        'ap_m':          ap_m,
        'ap_m_dynamic':  img.get('_ap_m_dynamic'),
      });
    }).filter(ee.Filter.gt('area_ha', 0));

    Export.table.toDrive({
      collection:     areaSeries,
      description:    'SAR_area_' + rName + MODE_SUFFIX,
      folder:         SAR_FOLDER,
      fileNamePrefix: 'SAR_area_' + rName,
      fileFormat:     'CSV',
    });
  }

  print('Export queued: ' + rName + '  (~' + areaHaEst + ' ha)  AP (m) =', ap_m.round());
});
