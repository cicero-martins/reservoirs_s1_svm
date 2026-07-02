// ═══════════════════════════════════════════════════════════════════════════
// RESERVOIR SAR MONITOR — Global GEE App  (v4 — global dataset support)
//
// Catalogue: set CFG.dataset = 'sicily' (personal validated asset) or 'global'
//   (Global Dam Watch v1.0, GDW_RESERVOIRS_V1_0, 35,295 reservoir polygons,
//   fields: RES_NAME / DAM_NAME / ALT_NAME / GDW_ID / CAP_MCM / AREA_SKM).
//   Global mode searches all three name fields; no pre-filter needed.
//
// Method mirrors trainSVMfromJRC / executeMainPipeline in reservoirs_s1_svm.js,
// adapted for interactive single-reservoir use:
//   • AOI → JRC max_extent polygon (NOT HydroLAKES polygon — see note below)
//   • Training composite → fixed 2023 annual MOSAIC (not user-period median)
//   • Training samples → JRC occurrence ≥ 95% water + occurrence = 0 land in a
//                        500 m near-shore ring (auto, pure JRC)
//   • Orbit → highest-coverage 3° incidence-angle bin (PrioritizeDescending…)
//   • Classifier → SVM RBF (cost=1, gamma=0.01)
//   • Post-proc → gap-fill + largest connected region; then 4-pass outlier
//                 removal + LOWESS smoothing of the area series
//   • A/P → computed from JRC polygon geometry (area / perimeter)
//   • Volume → polynomial V=a·A²+b·A+c on the SMOOTHED area (Sicilian reservoirs)
//
// NOTE on AOI source: GDW/HydroLAKES polygons are used as the spatial anchor
//   to locate the JRC max_extent water body (buffered 2 km). The JRC polygon
//   (not the GDW polygon) is the definitive AOI for training and A/P — it
//   captures the full high-water extent rather than the GDW snapshot geometry.
//
// PASTE INTO GEE CODE EDITOR AND CLICK "RUN"
// ═══════════════════════════════════════════════════════════════════════════

// ─── 1. CONFIGURATION ────────────────────────────────────────────────────
var CFG = {
  // ── DATASET SELECTION ──────────────────────────────────────────────────
  // Switch the reservoir catalogue here:
  //   'sicily' = personal validated asset (4+ Sicilian reservoirs, field res_name)
  //   'global' = Awesome GEE Community Catalog HydroLAKES v1.0 (1.43M water bodies)
  dataset: 'sicily',

  datasets: {
    sicily: {
      path:        'projects/ee-ciceromartinsjr/assets/HydroLAKES_sicily4',
      name_fields: ['res_name', 'Lake_name'],
      id_field:    'Hylak_id',
      type_filter: null,
      browsable:   true,
      poly_path:   null,             // search layer IS the polygon layer
    },
    global: {
      // Global Dam Watch v1.0 (Lehner et al. 2024) — 35,295 reservoir polygons.
      // Using the RESERVOIRS layer directly for both name search and map display:
      // it has polygon geometry (needed for AOI zoom + click), all name fields
      // (RES_NAME, DAM_NAME, ALT_NAME — all uppercase), and the GDW_ID join key. The BARRIERS
      // point layer (41k features) is skipped — it added complexity without benefit
      // since reservoirs already carries the same metadata.
      path:        'projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0',
      poly_path:   null,             // path IS the polygon layer — no separate poly needed
      name_fields: ['RES_NAME', 'DAM_NAME', 'ALT_NAME'],
      id_field:    'GDW_ID',
      type_filter: null,
      browsable:   false,
    },
  },

  // JRC max_extent search: margin around the reservoir polygon boundary.
  // A small buffer (500–2000 m) captures JRC water that slightly exceeds the
  // GDW/HydroLAKES boundary. Larger values risk merging adjacent water bodies.
  jrc_search_buffer_m: 2000,

  // Auto-training thresholds  (aligned with trainSVMfromJRC in original app)
  jrc_water_thresh:    95,   // JRC occurrence >= 95% → water sample (original: 95%)
  jrc_water_fallback:  80,   // fallback only if <10 strict points (raised from 50)
  // Land samples are taken in an ANNULUS around the reservoir: the inner radius
  // skips the seasonally-wet fluctuation zone; the outer radius reaches stable
  // land. The training composite MUST be clipped to cover this whole annulus,
  // otherwise land sample points fall on masked pixels and are dropped.
  land_ring_inner_m:   500,
  land_ring_outer_m:   2000,

  // Training composite reference year (original trainSVMfromJRC uses a fixed
  // 2023 annual MOSAIC, decoupled from the user's analysis period).
  train_ref_year: 2023,

  // SVM parameters (RBF kernel — same as original app)
  svm_cost:   1,
  svm_gamma:  0.01,

  // Orbit angle tolerance (degrees)
  orbit_angle_tol: 3,

  // SAR filter
  sar_scale_m:   10,   // native S1 resolution
  // Scale for the per-image area reduceRegion. The reduceRegion scale drives how
  // many pixels flow through the SVM classify + cleaning per image, so coarsening
  // it from 10 m → 20 m cuts the dominant cost ~4× (pixels scale with 1/scale²).
  // 20 m is ample for reservoir area (the SAR is focal_mean-smoothed at 30 m anyway).
  area_scale_m:  20,
  clean_scale_m: 30,   // resolution for connected-component vectorization (keep_largest_only:true path)
  max_pixels:   1e9,

  // Connected-component cleaning:
  //   true  = single largest polygon only (original app; safe for compact reservoirs,
  //           but drops valid dendritic arms in complex / irregular shapes).
  //   false = keep all polygons whose centroid falls INSIDE the JRC max_extent
  //           footprint. Excludes external spurious bodies (irrigated fields, ponds
  //           whose centroids are outside lakePoly) while preserving reservoir arms.
  keep_largest_only: false,

  // FAST area mode — skip per-scene reduceToVectors (the dominant per-image cost,
  // which scales with the number of S1 scenes and triggers "capacity exceeded" on
  // long date ranges). Instead derive WaterCleaned by clipping WaterFilled to the
  // pool polygon (same masking the map display already trusts), with a cheap raster
  // connected-component despeckle. Lets the app run multi-year ranges interactively.
  // Trade-off: no centroid-inside vectorisation — but with keep_largest_only:false
  // the difference is small (spurious blobs inside lakePoly were already counted).
  fast_area:     false,   // set true to enable the cheap area path
  min_blob_px:   5,       // FAST despeckle: drop connected water blobs < this many px

  // Minimum S1 AOI coverage (%) required per image in orbit fallback.
  // The primary selection requires ≥90%; if no orbit achieves that, images with
  // coverage ≥ min_coverage_pct are used. Lowering helps global reservoirs at
  // swath edges (e.g., partial IW sub-swath coverage).
  min_coverage_pct: 50,

  // Temporal window (days) for compositing partial-coverage images.
  // Images within ±N days of each acquisition are mosaicked to fill gaps.
  // Set to 0 to disable. Useful for global reservoirs at swath edges.
  // For Sicily (full coverage): 0. For global: 6–12.
  composite_window_days: 6,

  // Area series is computed in chunks of this many months, evaluated concurrently
  // and rendered progressively. Smaller = finer progress + more parallelism (more
  // concurrent requests, faster first result). 3 = quarterly, 12 = yearly.
  chunk_months: 3,

  // A/P thresholds (this study): drive both classifier selection and the
  // reliability display. The dual-pol (VV+VH) SVM helps only at low A/P; above
  // it the single-pol VV Otsu is equal or better on accuracy, and cheaper.
  ap_low:  120,   // < 120 m  → select dual-pol SVM (complex / dendritic shoreline)
  ap_high: 250,   // >= 250 m → high reliability; 120–250 m → medium

  // Optional: GRDL FeatureCollection asset for global volume curves
  grdl_path: null,
};

// ─── 2. DATA SOURCES ─────────────────────────────────────────────────────
var DS         = CFG.datasets[CFG.dataset];   // active dataset config
var HYDROLAKES = ee.FeatureCollection(DS.path);   // active dataset (polygons for GDW, sicily)
var HYDROLAKES_CACHE = null;   // pre-loaded client-side for browsable (sicily) datasets
var JRC_GSW    = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var JRC_MON    = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory');
var WORLDCOVER = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var S1_RAW     = ee.ImageCollection('COPERNICUS/S1_GRD');

var BANDS = ['VV', 'VH'];   // classification bands (consistent with original app)

// ─── 3. S1 PREPROCESSING (speckle filter + clip) ─────────────────────────
function preprocessS1(img, aoi) {
  return img.select(BANDS)
    .focal_mean(30, 'circle', 'meters')   // speckle reduction (same as original)
    .clip(aoi)
    .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                          'relativeOrbitNumber_start', 'angle'])
    .set('date', img.date().format('YYYY-MM-dd'));
}

// ─── 4. JRC MAX_EXTENT → AOI POLYGON ─────────────────────────────────────
// Returns the largest contiguous JRC water body within searchBuffer of the
// reservoir polygon. Uses the actual polygon geometry (not its centroid) so
// large/elongated reservoirs are not clipped to a fixed-radius circle.
function jrcMaxExtentPoly(hydroGeom, searchBuffer) {
  // Buffer the polygon boundary (not its centroid) to get the search area.
  // Old: centroid.buffer(5000) → clips large reservoirs to a ~10 km circle.
  var searchArea = hydroGeom.buffer(searchBuffer);
  var jrcMax     = JRC_GSW.select('max_extent').eq(1).selfMask();
  var waterVecs  = jrcMax.reduceToVectors({
    geometry:       searchArea,
    scale:          30,
    maxPixels:      1e9,
    bestEffort:     true,
    geometryType:   'polygon',
    eightConnected: true,
    tileScale:      4,   // splits AOI into smaller tiles → lower peak memory per tile
  });
  // Largest polygon by pixel count = main reservoir body
  return waterVecs.sort('count', false).first().geometry();
}

// ─── 5. AUTO-TRAINING SAMPLES ────────────────────────────────────────────
// lakePoly: JRC max_extent polygon (water samples + A/P source)
// aoi     : lakePoly.buffer(100) — classification zone
function autoTrainingSamples(lakePoly, aoi) {
  var occ = JRC_GSW.select('occurrence');

  // Water: high-confidence permanent water (≥95%); fallback (≥80%) only if
  // the strict threshold yields too few points (small/shrinking reservoirs).
  var waterStrict = occ.gte(CFG.jrc_water_thresh).selfMask().sample({
    region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true,
  }).map(function(f) { return f.set('landcover', 1); });

  var waterFallback = occ.gte(CFG.jrc_water_fallback).selfMask().sample({
    region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true,
  }).map(function(f) { return f.set('landcover', 1); });

  var waterSamples = ee.FeatureCollection(
    ee.Algorithms.If(waterStrict.size().gt(10), waterStrict, waterFallback)
  );

  // Land: persistent non-water in an ANNULUS (inner..outer) around the lake.
  // The inner radius skips the seasonally-wet shoreline (mixed pixels); the
  // outer radius reaches stable land. Land = WorldCover non-water classes AND
  // JRC occurrence == 0 (robust: avoids any small ponds/wetland in the ring).
  var landRing = lakePoly.buffer(CFG.land_ring_outer_m)
                   .difference(lakePoly.buffer(CFG.land_ring_inner_m));
  var landMask = WORLDCOVER.neq(80).and(WORLDCOVER.neq(90)).and(WORLDCOVER.neq(95))
    .and(occ.unmask(0).eq(0)).selfMask();
  var landSamples = landMask.sample({
    region: landRing, scale: 30, numPixels: 500, seed: 42, geometries: true,
  }).map(function(f) { return f.set('landcover', 2); });

  return waterSamples.merge(landSamples);
}

// ─── 6. SVM CLASSIFIER ───────────────────────────────────────────────────
function trainSVM(trainingFC, s1Composite) {
  var samples = s1Composite.select(BANDS).sampleRegions({
    collection: trainingFC,
    properties: ['landcover'],
    scale: 30,
  }).filter(ee.Filter.inList('landcover', ee.List([1, 2])))
    .filter(ee.Filter.notNull(BANDS));

  return ee.Classifier.libsvm({
    kernelType: 'RBF',
    cost:       CFG.svm_cost,
    gamma:      CFG.svm_gamma,
  }).train({
    features:        samples,
    classProperty:   'landcover',
    inputProperties: BANDS,
  });
}

// ─── 7. ORBIT AUTO-SELECTION ─────────────────────────────────────────────
// Exact port of PrioritizeDescendingAngleBins() from reservoirs_s1_svm.js.
// Iterates 3-degree angle bins from highest to lowest; picks the first bin
// where all images cover ≥90% of the AOI. Falls back to full collection.
function selectBestOrbit(s1Raw, aoi, callback) {
  var aoiArea = aoi.area();

  var withAngle = s1Raw.map(function(img) {
    var stats     = img.select('angle').reduceRegion({
      reducer: ee.Reducer.mean(), geometry: aoi, scale: 10, maxPixels: 1e9,
    });
    var meanAngle = ee.Number(stats.get('angle'));
    return ee.Algorithms.If(
      meanAngle,
      img.set('angle_mean', meanAngle),
      img
    );
  }).filter(ee.Filter.notNull(['angle_mean']));

  // Unique angles sorted descending (high → low)
  var angleList = withAngle.aggregate_array('angle_mean').distinct().sort().reverse();

  function tryNextBin(angleList, i, nBins) {
    var current     = ee.Number(angleList.get(i));
    var next        = current.subtract(3);
    var binFiltered = withAngle.filter(ee.Filter.and(
      ee.Filter.gte('angle_mean', next),
      ee.Filter.lt('angle_mean', current)
    ));

    var withCoverage = binFiltered.map(function(img) {
      var mask        = img.select('VV').mask();
      // 100 m scale is sufficient for orbit coverage % — avoids pixel overflow on
      // large reservoirs (Itaipu: 13.5 B px at 10 m → 1.35 M px at 100 m).
      var coveredArea = mask.multiply(ee.Image.pixelArea())
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                       scale: 100, maxPixels: 1e7, bestEffort: true})
        .get('VV');
      var pct = ee.Number(coveredArea).divide(aoiArea).multiply(100);
      return img.set('percentCovered', pct);
    });

    var valid = withCoverage
      .filter(ee.Filter.gte('percentCovered', 90))
      .sort('system:time_start');

    return valid.size().evaluate(function(size) {
      if (size > 0) {
        callback(valid);
      } else if (i < nBins - 1) {
        // nBins was pre-evaluated client-side — avoids a blocking .getInfo() per iteration
        tryNextBin(angleList, i + 1, nBins);
      } else {
        // Final fallback: accept images with ≥ min_coverage_pct from any orbit.
        // This handles global reservoirs at Sentinel-1 swath edges where no
        // single orbit achieves 90% coverage of the AOI.
        var withCovFb = withAngle.map(function(img) {
          var pct = img.select('VV').mask().multiply(ee.Image.pixelArea())
            .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                           scale: 100, maxPixels: 1e7, bestEffort: true})
            .getNumber('VV').divide(aoiArea).multiply(100);
          return img.set('percentCovered', pct);
        });
        var validFb = withCovFb
          .filter(ee.Filter.gte('percentCovered', CFG.min_coverage_pct))
          .sort('system:time_start');
        validFb.size().evaluate(function(nFb) {
          callback(nFb > 0 ? validFb : withAngle.sort('system:time_start'));
        });
      }
    });
  }

  angleList.evaluate(function(list) {
    if (list && list.length > 1) {
      tryNextBin(ee.List(list), 0, list.length);   // pass JS length — no .getInfo() needed
    } else {
      callback(withAngle.sort('system:time_start'));
    }
  });
}

// ─── 7b. TEMPORAL COMPOSITING FOR PARTIAL COVERAGE ───────────────────────
// For images with partial AOI coverage (e.g. swath-edge reservoirs), create
// a mosaic from images within ±windowDays of each acquisition date.
// Preserves one data point per original date; nearby images fill gaps.
// Only useful when windowDays > 0 AND images from nearby dates exist.
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

// ─── 7c. PER-SCENE VV OTSU DETECTOR ──────────────────────────────────────
// Single-polarisation water detector: a per-scene Otsu threshold on VV taken
// over a land-ringed buffer (bimodal histogram); water = VV < T. Selected in
// place of the dual-pol SVM when A/P is not low (see A/P classifier selector).
var OTSU = {band: 'VV', hist_buffer_m: 500, hist_buckets: 256};

function otsuThreshold(histogram) {
  histogram   = ee.Dictionary(histogram);
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

function computeOtsuWater(img, lakePoly, aoi) {
  var histRegion = lakePoly.buffer(OTSU.hist_buffer_m);
  var hist = img.select(OTSU.band).reduceRegion({
    reducer:  ee.Reducer.histogram(OTSU.hist_buckets),
    geometry: histRegion, scale: 30, maxPixels: 1e8, bestEffort: true,
  }).get(OTSU.band);
  var threshold = ee.Number(otsuThreshold(hist));
  return img.select(OTSU.band).lt(threshold).rename('Water').clip(aoi);
}

// ─── 8. CLASSIFY + CLEAN WATER MASK ─────────────────────────────────────
// Returns collection with 'Water', 'WaterFilled', 'WaterCleaned' bands added.
// Gap-fill (fastDistanceTransform) + connected-component cleaning.
//
// Cleaning strategy (CFG.keep_largest_only):
//   false (default) — centroid-inside-lakePoly filter: keeps every water polygon
//     whose centroid falls inside the JRC max_extent footprint. Captures dendritic
//     reservoir arms without admitting external spurious bodies (irrigated fields,
//     flooded roads) whose centroids lie outside the reservoir boundary.
//   true — single largest polygon only (original app behaviour); safe for compact
//     reservoirs but physically wrong for irregular / branching shapes.
function classifyCollection(s1Proc, classifier, aoi, lakePoly, useDual) {
  // Step 1 — water detection: dual-pol SVM at low A/P, else single-pol VV Otsu.
  var withWater = s1Proc.map(function(img) {
    var water = useDual
      ? img.select(BANDS).classify(classifier).eq(1).clip(aoi).rename('Water')
      : computeOtsuWater(img, lakePoly, aoi);
    return img.addBands(water);
  });

  // Step 2 — morphological gap-fill (close small holes)
  var withFilled = withWater.map(function(img) {
    var mask   = img.select('Water').unmask(0).clip(aoi);
    var dist   = mask.fastDistanceTransform(30).clip(aoi);
    var filled = dist.lte(0.5).updateMask(dist.lte(0.5));
    return img.addBands(filled.where(mask, 1).rename('WaterFilled'));
  });

  // Step 3-FAST — raster-only cleaning (no reduceToVectors). Despeckle with a cheap
  // connected-component count, then clip to the pool polygon. Produces a 'WaterCleaned'
  // band so computeAreaSeries works unchanged. This is the high-leverage fix for the
  // "capacity exceeded" limit on long date ranges (vectorisation removed entirely).
  if (CFG.fast_area) {
    return withFilled.map(function(img) {
      var filled     = img.select('WaterFilled');
      var nConn      = filled.connectedPixelCount(CFG.min_blob_px, true);
      var despeckled = filled.updateMask(nConn.gte(CFG.min_blob_px));
      return img.addBands(despeckled.clip(lakePoly).rename('WaterCleaned'));
    });
  }

  // Step 3 — connected-component cleaning → WaterCleaned (used for area series).
  //
  // reduceToVectors at clean_scale_m (30 m) produces vector polygons; those
  // whose centroid falls inside lakePoly are kept (centroid-inside filter).
  // tileScale: 4 reduces peak memory per GEE computation tile.
  //
  // IMPORTANT — display vs. computation split:
  //   WaterCleaned is correct for the area chart because GEE runs one batch
  //   computation per image. It must NOT be used as a map display layer
  //   (addLayer) because GEE re-evaluates reduceToVectors independently for
  //   each 256×256 map tile → memory overflow on large reservoirs (Itaipu etc.).
  //   Map display uses WaterFilled.clip(lakePoly) instead (see §18), which is
  //   functionally equivalent but avoids any neighbourhood vectorization.
  return withFilled.map(function(img) {
    var mask  = img.select('WaterFilled');
    var polys = mask.reduceToVectors({
      geometryType: 'polygon', reducer: ee.Reducer.countEvery(),
      scale: CFG.clean_scale_m, maxPixels: CFG.max_pixels,
      bestEffort: true, tileScale: 4,
    });

    var keptPolys;
    if (CFG.keep_largest_only) {
      keptPolys = ee.FeatureCollection([
        polys.map(function(f) {
          return f.set('_area', f.geometry().area({maxError: 1}));
        }).sort('_area', false).first()
      ]);
    } else {
      var withFlag = polys.map(function(f) {
        var inside = lakePoly.contains(f.geometry().centroid({maxError: 1}),
                                       ee.ErrorMargin(1));
        return f.set('_inside', inside);
      });
      keptPolys = withFlag.filter(ee.Filter.eq('_inside', 1));
      keptPolys = ee.FeatureCollection(ee.Algorithms.If(
        keptPolys.size().gt(0),
        keptPolys,
        ee.FeatureCollection([polys.map(function(f) {
          return f.set('_area', f.geometry().area({maxError: 1}));
        }).sort('_area', false).first()])
      ));
    }

    var keptMask = ee.Image().paint({featureCollection: keptPolys, color: 1})
                             .rename('KeptRegionMask');
    return img.addBands(mask.updateMask(keptMask).rename('WaterCleaned'));
  });
}

// ─── 9. AREA TIME SERIES ─────────────────────────────────────────────────
// Takes the classified + cleaned collection; extracts area from WaterCleaned.
function computeAreaSeries(waterMaskCleaned, aoi) {
  return ee.FeatureCollection(waterMaskCleaned.map(function(img) {
    var area_m2 = img.select('WaterCleaned').multiply(ee.Image.pixelArea())
      .reduceRegion({
        reducer: ee.Reducer.sum(), geometry: aoi,
        scale: CFG.area_scale_m, maxPixels: CFG.max_pixels, bestEffort: true,
        tileScale: 4,   // same memory guard as JRC — keeps long ranges within limits
      }).get('WaterCleaned');
    return ee.Feature(null, {
      'system:time_start': img.date().millis(),
      'date': img.date().format('YYYY-MM-dd'),
      'area_ha': ee.Number(area_m2).divide(1e4),
    });
  })).filter(ee.Filter.gt('area_ha', 0));
}

// ─── 9b. OUTLIER REMOVAL + LOWESS SMOOTHING ──────────────────────────────
// Faithful port of removeOutliers / detectAndRemoveLocalOutliers /
// lowessSmoothing from reservoirs_s1_svm.js (operates on 'area_ha').

// Global outlier removal: drop points beyond `threshold` σ from the mean.
function removeOutliers(fc, threshold) {
  var areas  = fc.aggregate_array('area_ha');
  var mean   = areas.reduce(ee.Reducer.mean());
  var stdDev = areas.reduce(ee.Reducer.stdDev());
  return fc.map(function(f) {
    var a   = ee.Number(f.get('area_ha'));
    var dev = a.subtract(mean).abs().divide(stdDev);
    return f.set('isOutlier', dev.gt(threshold));
  }).filter(ee.Filter.eq('isOutlier', 0));
}

// Local outlier removal over a sliding window of `windowSize` samples.
function detectAndRemoveLocalOutliers(fc, windowSize, stdDevThreshold) {
  var features = fc.sort('system:time_start').toList(fc.size());
  var flagged  = ee.List.sequence(0, features.size().subtract(1)).map(function(i) {
    var index   = ee.Number(i);
    var current = ee.Feature(features.get(index));
    var val     = ee.Number(current.get('area_ha'));
    var half    = ee.Number(windowSize).divide(2).int();
    var start   = index.subtract(half).max(0);
    var end     = index.add(half).min(features.size());
    var window  = ee.FeatureCollection(features.slice(start, end));
    var mean    = window.aggregate_mean('area_ha');
    var sd      = window.aggregate_total_sd('area_ha');
    var dev     = val.subtract(mean).abs().divide(sd);
    return current.set('isOutlierLocal', dev.gt(stdDevThreshold));
  });
  return ee.FeatureCollection(flagged).filter(ee.Filter.eq('isOutlierLocal', 0));
}

// LOWESS smoothing via Gaussian-weighted temporal window (±windowDays).
function lowessSmoothing(fc, windowDays, bandwidth) {
  var features = fc.sort('system:time_start').toList(fc.size());
  return ee.FeatureCollection(ee.List.sequence(0, features.size().subtract(1)).map(function(i) {
    var current     = ee.Feature(features.get(i));
    var currentDate = ee.Date(current.get('system:time_start'));
    var wStart      = currentDate.advance(-windowDays, 'day');
    var wEnd        = currentDate.advance(windowDays,  'day');
    var neighbors   = ee.FeatureCollection(features).filter(ee.Filter.date(wStart, wEnd));
    var weighted    = neighbors.map(function(f) {
      var fDate    = ee.Date(f.get('system:time_start'));
      var diffDays = currentDate.difference(fDate, 'day').abs();
      var weight   = diffDays.divide(bandwidth).pow(2).multiply(-1).exp();
      var value    = ee.Number(f.get('area_ha'));
      return f.set({'weight': weight, 'weightedValue': value.multiply(weight)});
    });
    var wSum  = weighted.aggregate_sum('weight');
    var wvSum = weighted.aggregate_sum('weightedValue');
    return current.set('area_ha_smoothed', ee.Number(wvSum).divide(wSum));
  }));
}

// Full original chain: 1 global pass + 3 local passes + LOWESS.
// NOTE: the server-side functions above are kept as a faithful reference (and for
// any batch use), but the INTERACTIVE app uses the client-side port below
// (cleanAndSmoothJS). The server version is O(N²) via toList + nested per-point
// FeatureCollection filters, which (a) hit "user memory limit exceeded" on long
// ranges and (b) crash with "toList: count must be positive. Got: 0" on an empty
// series. The JS port runs on the already-fetched raw points — instant for a few
// hundred points, zero server memory.
function cleanAndSmooth(areaFC) {
  var ts1 = removeOutliers(areaFC, 2);
  var ts2 = detectAndRemoveLocalOutliers(ts1, 5, 1.5);
  var ts3 = detectAndRemoveLocalOutliers(ts2, 5, 1.5);
  var ts4 = detectAndRemoveLocalOutliers(ts3, 10, 1.5);
  return lowessSmoothing(ts4, 20, 7);
}

// ── Client-side O(N) port (instant for ≤ a few hundred points) ───────────────
// Operates on plain rows: [{date:'YYYY-MM-DD', t:<ms>, area_ha:<num>}] sorted by t.
function _meanSd(vals) {
  var n = vals.length;
  if (!n) return {mean: 0, sd: 0};
  var mean = 0, i;
  for (i = 0; i < n; i++) mean += vals[i];
  mean /= n;
  var v = 0;
  for (i = 0; i < n; i++) v += (vals[i] - mean) * (vals[i] - mean);
  return {mean: mean, sd: Math.sqrt(v / n)};   // population SD (matches GEE stdDev)
}

function removeOutliersJS(rows, threshold) {
  var ms = _meanSd(rows.map(function(r) { return r.area_ha; }));
  if (ms.sd === 0) return rows.slice();
  return rows.filter(function(r) {
    return Math.abs(r.area_ha - ms.mean) / ms.sd <= threshold;
  });
}

function detectLocalOutliersJS(rows, windowSize, threshold) {
  if (rows.length < 2) return rows.slice();
  var half = Math.floor(windowSize / 2), keep = [];
  for (var i = 0; i < rows.length; i++) {
    var lo = Math.max(0, i - half), hi = Math.min(rows.length, i + half);  // [lo, hi)
    var ms  = _meanSd(rows.slice(lo, hi).map(function(r) { return r.area_ha; }));
    var dev = ms.sd > 0 ? Math.abs(rows[i].area_ha - ms.mean) / ms.sd : 0;
    if (dev <= threshold) keep.push(rows[i]);
  }
  return keep;
}

function lowessJS(rows, windowDays, bandwidth) {
  var dayMs = 86400000;
  return rows.map(function(r) {
    var wSum = 0, wvSum = 0;
    for (var j = 0; j < rows.length; j++) {
      var dd = Math.abs(r.t - rows[j].t) / dayMs;
      if (dd <= windowDays) {
        var w = Math.exp(-Math.pow(dd / bandwidth, 2));
        wSum += w; wvSum += rows[j].area_ha * w;
      }
    }
    return {date: r.date, t: r.t, area_ha: r.area_ha,
            area_ha_smoothed: wSum > 0 ? wvSum / wSum : r.area_ha};
  });
}

// Same chain as cleanAndSmooth (server): removeOutliers(2) + 3× local + LOWESS(20d,7).
function cleanAndSmoothJS(rows) {
  if (!rows || !rows.length) return [];
  var s = removeOutliersJS(rows, 2);
  s = detectLocalOutliersJS(s, 5,  1.5);
  s = detectLocalOutliersJS(s, 5,  1.5);
  s = detectLocalOutliersJS(s, 10, 1.5);
  return lowessJS(s, 20, 7);
}

// ─── 10. JRC REFERENCE AREA ──────────────────────────────────────────────
function computeJRCSeries(aoi, start, end) {
  return ee.FeatureCollection(
    JRC_MON.filterBounds(aoi).filterDate(start, end).map(function(img) {
      var water   = img.eq(2).clip(aoi);
      var area_m2 = water.multiply(ee.Image.pixelArea()).reduceRegion({
        reducer: ee.Reducer.sum(), geometry: aoi,
        scale: 30, maxPixels: 1e8, bestEffort: true,
        tileScale: 4,   // split region into smaller tiles → avoids "user memory limit exceeded"
      }).get('water');
      return ee.Feature(null, {
        'system:time_start': img.date().millis(),
        'date': img.date().format('YYYY-MM-dd'),
        'jrc_area_ha': ee.Number(area_m2).divide(1e4),
      });
    })
  ).filter(ee.Filter.gt('jrc_area_ha', 0));
}

// ─── 11. VOLUME (SICILIAN POLYNOMIAL + GLOBAL PLACEHOLDER) ──────────────
// V (Mm³) = a·A² + b·A + c  where A = water area in hectares.
// Coefficients from original app / Sicilian Water Authority AEV tables.
var VOLUME_POLY = {
  'Ancipa':       {a:  0.0009, b:  0.0938, c: -2.0159},
  'Rosamarina':   {a:  0.0001, b:  0.1124, c: -2.9694},
  'Poma':         {a:  0.0001, b:  0.0963, c: -7.4893},
  'Pozzillo':     {a:  0.0002, b:  0.0743, c: -3.1034},
  'Arancio':      {a:  0.0002, b:  0.0376, c: -0.5576},
  'Garcia':       {a:  0.0003, b:  0.0256, c: -0.1278},
  'Fanaco':       {a:  0.0005, b:  0.0912, c: -0.4439},
  'Lentini':      {a:  0.0014, b: -2.0234, c:  732.36},
};

function volumeFromArea(area_ha, lakeName) {
  for (var key in VOLUME_POLY) {
    if (lakeName.indexOf(key) !== -1) {
      var c = VOLUME_POLY[key];
      return c.a * area_ha * area_ha + c.b * area_ha + c.c;
    }
  }
  return null;
}

// ─── 12. A/P FROM JRC POLYGON ────────────────────────────────────────────
function computeAP_fromGeom(geom) {
  // GEE geodesic computation (m²  /  m) — no external CRS needed
  var area  = geom.area(1);        // maxError = 1 m
  var perim = geom.perimeter(1);   // maxError = 1 m
  return area.divide(perim);       // metres
}

// Client-side centroid from GeoJSON geometry (for hover/click distance checks)
function geomCentroid(geom) {
  if (!geom) return null;
  var coords = geom.type === 'Point'        ? [geom.coordinates]
             : geom.type === 'Polygon'      ? geom.coordinates[0]
             : geom.type === 'MultiPolygon' ? geom.coordinates[0][0]
             : null;
  if (!coords || !coords.length) return null;
  var lon = 0, lat = 0;
  for (var i = 0; i < coords.length; i++) { lon += coords[i][0]; lat += coords[i][1]; }
  return {lon: lon / coords.length, lat: lat / coords.length};
}

// ─── 12. INITIALISE MAP ──────────────────────────────────────────────────
var mapObj = ui.Map();
mapObj.setOptions('HYBRID');
mapObj.setControlVisibility({all: true});

// ─── 13. LEFT PANEL ──────────────────────────────────────────────────────
var panel = ui.Panel({style: {width: '370px', padding: '10px'}});

// Title
panel.add(ui.Label('Reservoir SAR Monitor', {
  fontSize: '17px', fontWeight: 'bold', margin: '0 0 2px 0',
}));
panel.add(ui.Label('Global · JRC auto-training · A/P-selected classifier', {
  fontSize: '11px', color: '#555', margin: '0 0 8px 0',
}));

// ── Search ─────────────────────────────────────────────────────
panel.add(divider());
panel.add(sectionLabel('1 · Find reservoir'));

var searchBox = ui.Textbox({
  placeholder: DS.browsable
    ? 'Name or Hylak_id (e.g. Pozzillo, 12345)…'
    : 'Name or GDW_ID (e.g. Chichester, Mrica, 12345)…',
  style: {width: '245px'},
});
var searchBtn = ui.Button({label: 'Search', style: {margin: '0 0 0 6px'}});
panel.add(ui.Panel([searchBox, searchBtn], ui.Panel.Layout.Flow('horizontal')));

// GEE ui.Select does not support setItems() after creation —
// use a container Panel and swap the widget on each search.
var selectContainer = ui.Panel({style: {margin: '6px 0 2px 0'}});
panel.add(selectContainer);

// Prominent label showing the reservoir currently loaded for analysis
var selectedLabel = ui.Label('No reservoir selected', {
  fontSize: '12px', color: '#888', fontStyle: 'italic', margin: '0 0 2px 2px',
});
panel.add(selectedLabel);

// Active select widget reference (replaced on each search)
var resultSelect = null;

function showSelectItems(items) {
  selectContainer.clear();
  if (!items || !items.length) {
    selectContainer.add(ui.Label(
      'No results.', {color: '#888', fontSize: '12px'}
    ));
    resultSelect = null;
    return;
  }
  resultSelect = ui.Select({
    items:       items,
    placeholder: '— select reservoir —',
    onChange:    onReservoirSelected,
    style:       {width: '350px'},
  });
  selectContainer.add(resultSelect);
}

// ── Period ─────────────────────────────────────────────────────
panel.add(divider());
panel.add(sectionLabel('2 · Analysis period'));

var startBox = ui.Textbox({value: '2019-01-01', style: {width: '115px'}});
var endBox   = ui.Textbox({value: '2023-12-31', style: {width: '115px'}});
panel.add(ui.Panel(
  [ui.Label('From:', {margin: '5px 4px 0 0'}), startBox,
   ui.Label('To:',   {margin: '5px 6px 0 6px'}), endBox],
  ui.Panel.Layout.Flow('horizontal')
));

// ── Run ────────────────────────────────────────────────────────
panel.add(divider());
var runBtn = ui.Button({
  label: '▶  Run Analysis',
  style: {width: '213px', margin: '8px 4px 8px 0', backgroundColor: '#1565c0'},
  disabled: true,
});
var resetBtn = ui.Button({
  label: '⟳  New reservoir',
  style: {width: '129px', margin: '8px 0'},
});
var statusLabel = ui.Label('Select a reservoir above to begin.', {
  color: '#444', fontSize: '12px', margin: '2px 0 6px 0',
});
panel.add(ui.Panel([runBtn, resetBtn], ui.Panel.Layout.Flow('horizontal')));
panel.add(statusLabel);

// ── A/P indicator ──────────────────────────────────────────────
panel.add(divider());
panel.add(sectionLabel('A/P classifier selector'));

var apValueLabel = ui.Label('— m', {
  fontSize: '26px', fontWeight: 'bold', margin: '2px 0',
});
var apBadgeLabel = ui.Label('Select a reservoir', {
  fontSize: '12px', padding: '3px 10px', margin: '4px 0',
  backgroundColor: '#e0e0e0', color: '#555',
  border: '1px solid #bdbdbd', borderRadius: '4px',
});
var apDescLabel = ui.Label('', {
  fontSize: '11px', color: '#555', margin: '4px 0 0 0',
});
panel.add(apValueLabel);
panel.add(apBadgeLabel);
panel.add(apDescLabel);

// A/P colour legend (matches the reservoir polygon drawn on the map)
var apLegend = ui.Panel({
  layout: ui.Panel.Layout.Flow('horizontal'),
  style:  {margin: '6px 0 0 0'},
});
[['f88f4d', 'Low <120'], ['d64a02', 'Med 120–250'], ['8a2d04', 'High ≥250']]
  .forEach(function(e) {
    apLegend.add(ui.Label(' ', {
      backgroundColor: '#' + e[0], padding: '0 6px', margin: '0 3px 0 0',
      border: '1px solid #999',
    }));
    apLegend.add(ui.Label(e[1], {fontSize: '10px', color: '#555', margin: '0 8px 0 0'}));
  });
panel.add(apLegend);

// ── Charts ─────────────────────────────────────────────────────
panel.add(divider());
panel.add(sectionLabel('Results'));
var chartPanel = ui.Panel();
panel.add(chartPanel);

// ─── 14. LAYOUT ──────────────────────────────────────────────────────────
ui.root.clear();
ui.root.add(ui.SplitPanel({
  firstPanel:  panel,
  secondPanel: mapObj,
  orientation: 'horizontal',
}));

// ─── 15. STATE ───────────────────────────────────────────────────────────
var S = {
  hylak_id:  null,
  lake_name: null,
  hydroGeom: null,   // HydroLAKES geometry (for map zoom only)
  lakePoly:  null,   // JRC max_extent polygon (main AOI source)
  aoi:       null,   // lakePoly.buffer(100)
  ap_m:      null,
};

// ─── 16. SEARCH ──────────────────────────────────────────────────────────
// Helper: build a label string from GDW or HydroLAKES feature properties
function lakeLabel(p) {
  var nm   = p.RES_NAME  || p.DAM_NAME  || p.ALT_NAME  ||
             p.Res_name  || p.Dam_name  || p.res_name  ||
             p.Lake_name || p.lake_name || p.name || p.Name || '';
  var area = p.AREA_SKM  || p.Area_skm  || p.Lake_area || p.lake_area || p.LAKE_AREA || 0;
  var ctry = p.COUNTRY   || p.Country   || p.country   || '';
  var id   = p.GDW_ID    || p.Hylak_id  || p.Hylak_ID || p.hylak_id || p.Grand_id || p.FID || '?';
  var cap  = p.CAP_MCM   || p.Cap_mcm;
  var extra = cap ? ' ~' + (+cap).toFixed(0) + ' mcm'
            : (area ? ' (' + (+area).toFixed(1) + ' km²)' : '');
  return {
    label: (nm || 'Unnamed reservoir') + (ctry ? ' — ' + ctry : '') + extra +
           (nm   ? '' : '  [#' + id + ']'),
    value: String(id),
    name:  nm,
  };
}

// Build a case-insensitive name filter over the active dataset's name fields.
function buildNameFilter(name) {
  var variants = [
    name,
    name.charAt(0).toUpperCase() + name.slice(1).toLowerCase(),
    name.toUpperCase(),
    name.toLowerCase(),
  ];
  var ors = [];
  DS.name_fields.forEach(function(field) {
    variants.forEach(function(v) { ors.push(ee.Filter.stringContains(field, v)); });
  });
  return ee.Filter.or.apply(null, ors);
}

// Pre-load all features for browsable datasets (instant client-side filter + hover)
function loadCacheIfBrowsable(callback) {
  if (!DS.browsable) { if (callback) callback(null); return; }
  if (HYDROLAKES_CACHE) { if (callback) callback(HYDROLAKES_CACHE); return; }
  HYDROLAKES.limit(200).evaluate(function(fc, err) {
    if (err || !fc || !fc.features || !fc.features.length) {
      if (callback) callback(null);
      return;
    }
    HYDROLAKES_CACHE = {
      features:  fc.features,
      items:     fc.features.map(function(f) { return lakeLabel(f.properties); }),
      centroids: fc.features.map(function(f) { return geomCentroid(f.geometry); }),
    };
    if (callback) callback(HYDROLAKES_CACHE);
  });
}

// Unified search: called by button click AND textbox onChange (Enter/blur)
function doSearch(name) {
  name = (name || '').trim();
  selectContainer.clear();

  // Empty query
  if (!name) {
    if (DS.browsable) {
      if (HYDROLAKES_CACHE) {
        showSelectItems(HYDROLAKES_CACHE.items);
        statusLabel.setValue(HYDROLAKES_CACHE.items.length + ' reservoirs — select one or type to filter.');
      } else {
        statusLabel.setValue('Loading reservoirs…');
        loadCacheIfBrowsable(function(cache) {
          if (!cache) { statusLabel.setValue('⚠ Asset empty or inaccessible.'); return; }
          showSelectItems(cache.items);
          statusLabel.setValue(cache.items.length + ' reservoirs — select one or type to filter.');
        });
      }
    } else {
      statusLabel.setValue('Type a reservoir name — the global dataset (1.4M) is too large to list.');
    }
    return;
  }

  // Client-side instant filter for browsable datasets (Sicily)
  if (DS.browsable && HYDROLAKES_CACHE) {
    var q = name.toLowerCase();
    var matches = HYDROLAKES_CACHE.items.filter(function(item) {
      return item.label.toLowerCase().indexOf(q) !== -1;
    });
    showSelectItems(matches);
    statusLabel.setValue(matches.length
      ? matches.length + ' result(s) — select one.'
      : '"' + name + '" not found — try a different spelling.');
    return;
  }

  // Cache not ready yet for browsable: load first, then re-run search
  if (DS.browsable && !HYDROLAKES_CACHE) {
    statusLabel.setValue('Loading…');
    loadCacheIfBrowsable(function(cache) {
      if (cache) { doSearch(name); }
      else { statusLabel.setValue('⚠ Could not load asset.'); }
    });
    return;
  }

  // Server-side search. Supports name substring OR numeric ID (GDW_ID or Hylak_id).
  statusLabel.setValue('Searching for "' + name + '"…');
  var id_field_s  = DS.id_field || 'GDW_ID';
  var nameFilter  = buildNameFilter(name);
  var numId       = parseInt(name, 10);
  var isNumeric   = !isNaN(numId) && String(numId) === name.trim();
  var searchFilter = isNumeric
    ? ee.Filter.or(nameFilter, ee.Filter.eq(id_field_s, numId))
    : nameFilter;
  var col = HYDROLAKES.filter(searchFilter);
  col.limit(50).evaluate(function(fc, err) {
    if (err) { statusLabel.setValue('⚠ Error: ' + err); return; }
    if (fc && fc.features && fc.features.length) {
      showSelectItems(fc.features.map(function(f) { return lakeLabel(f.properties); }));
      statusLabel.setValue(fc.features.length + ' result(s) — select one.');
      return;
    }
    statusLabel.setValue('"' + name + '" not found. Try a different spelling, ' +
      'click the reservoir on the map, or enter its numeric ' + id_field_s + '.');
  });
}

searchBtn.onClick(function() { doSearch(searchBox.getValue()); });
searchBox.onChange(function(v)  { doSearch(v); });   // fires on Enter or blur

// ─── 17. RESERVOIR SELECTED ──────────────────────────────────────────────
// Called by onChange of the dynamically-created ui.Select inside selectContainer
function onReservoirSelected(id_str) {
  if (!id_str) return;

  statusLabel.setValue('Loading lake…');
  runBtn.setDisabled(true);
  chartPanel.clear();
  mapObj.layers().reset();
  resetBadge();

  var numeric_id = parseInt(id_str, 10);
  var id_field_r = DS.id_field || 'GDW_ID';
  var lake;
  if (!isNaN(numeric_id) && id_str !== '?') {
    lake = HYDROLAKES.filter(ee.Filter.eq(id_field_r, numeric_id)).first();
  } else {
    lake = HYDROLAKES.first();
  }

  lake.evaluate(function(f, err) {
    if (err) { statusLabel.setValue('⚠ Lake load error: ' + err); return; }
    if (!f)  { statusLabel.setValue('Could not load reservoir — check dataset path.'); return; }
    var props     = f.properties;
    var hydroGeom = ee.Feature(f).geometry();
    var name      = props.RES_NAME  || props.DAM_NAME  || props.ALT_NAME  ||
                    props.Res_name  || props.Dam_name  || props.res_name  ||
                    props.Lake_name || props.lake_name ||
                    props.name || props.Name || ('Reservoir #' + id_str);

    selectedLabel.setValue('▸ ' + name);
    selectedLabel.style().set({color: '#1565c0', fontStyle: 'normal'});

    mapNameLabel.setValue(name);
    mapNameLabel.style().set({shown: true});
    mapDateLabel.setValue('');
    mapDateLabel.style().set({shown: false});

    // Derive JRC max_extent polygon (server-side, then evaluate for A/P)
    var lakePoly = jrcMaxExtentPoly(hydroGeom, CFG.jrc_search_buffer_m);
    var aoi      = lakePoly.buffer(100);

    // Compute A/P from JRC polygon geometry
    var ap_ee = computeAP_fromGeom(lakePoly);

    ap_ee.evaluate(function(ap_m, err) {
      if (err || ap_m == null || isNaN(+ap_m) || ap_m <= 0) {
        statusLabel.setValue('⚠ JRC water not found near this lake. ' +
          'Try increasing jrc_search_buffer_m or check reservoir location.' +
          (err ? ' Error: ' + err : ''));
        runBtn.setDisabled(true);
        return;
      }
      S.hylak_id  = numeric_id;
      S.lake_name = name;
      S.hydroGeom = hydroGeom;
      S.lakePoly  = lakePoly;
      S.aoi       = aoi;
      S.ap_m      = ap_m;

      updateBadge(ap_m);

      // Map
      mapObj.centerObject(hydroGeom, 13);
      mapObj.addLayer(
        ee.FeatureCollection([ee.Feature(lakePoly)]).style({
          color: apColorHex(ap_m), fillColor: apColorHex(ap_m) + '80', width: 2,
        }), {}, name + ' · A/P ' + ap_m.toFixed(0) + ' m'
      );
      mapObj.addLayer(
        JRC_GSW.select('occurrence').clip(aoi.buffer(500)),
        {min: 0, max: 100, palette: ['white', '006699'], opacity: 0.5},
        'JRC occurrence'
      );

      runBtn.setDisabled(false);
      statusLabel.setValue(
        name + '  •  A/P = ' + ap_m.toFixed(0) + ' m  •  Press Run ▶'
      );
    });
  });
}

// ─── 18. RUN ANALYSIS ────────────────────────────────────────────────────
// Holds the classified+cleaned collection between run and chart-click events
var G_waterMaskCleaned = null;
var G_aoi              = null;
var G_name             = null;

// Helper: load a specific date's classified image on the map
function loadDateOnMap(dateStr) {
  if (!G_waterMaskCleaned || !G_aoi) return;
  var img = G_waterMaskCleaned
    .filter(ee.Filter.date(ee.Date(dateStr), ee.Date(dateStr).advance(1, 'day')))
    .first();
  img.evaluate(function(info) {
    if (!info) { statusLabel.setValue('No image for ' + dateStr); return; }
    mapObj.layers().reset();
    mapObj.addLayer(
      ee.FeatureCollection([ee.Feature(S.lakePoly)]).style({
        color: apColorHex(S.ap_m), fillColor: '00000000', width: 2,
      }), {}, G_name + ' boundary (A/P)'
    );
    mapObj.addLayer(
      ee.Image(img).select('VV').clip(G_aoi),
      {min: -20, max: 0, palette: ['000000','ffffff'], opacity: 0.7},
      'SAR VV  ' + dateStr
    );
    mapObj.addLayer(
      ee.Image(img).select('WaterFilled').updateMask(ee.Image.constant(1).clip(S.lakePoly)),
      {palette: ['1565c0'], opacity: 0.85},
      'Water  ' + dateStr
    );
    mapDateLabel.setValue(dateStr);
    mapDateLabel.style().set({shown: true});
    statusLabel.setValue('Showing: ' + dateStr + '  •  Click another point to update map.');
  });
}

runBtn.onClick(function() {
  if (!S.lakePoly || !S.aoi) return;

  var start    = startBox.getValue();
  var end      = endBox.getValue();
  var name     = S.lake_name;
  var lakePoly = S.lakePoly;
  var aoi      = S.aoi;

  runBtn.setDisabled(true);
  chartPanel.clear();
  showLoader('Preparing…  (orbit selection + training)');
  statusLabel.setValue('Getting S1 images for ' + name + '…');

  // ── Raw S1 collection ────────────────────────────────────────
  var s1Raw = S1_RAW
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  // ── Orbit auto-selection ─────────────────────────────────────
  selectBestOrbit(s1Raw, aoi, function(s1Filtered) {

    statusLabel.setValue('Running JRC auto-training…');

    // ── Training composite ───────────────────────────────────
    // Fixed reference-year annual MOSAIC (not a median of the user period).
    // Exactly mirrors trainSVMfromJRC: mosaic → focal_mean(30) → clip.
    // mosaic() preserves instantaneous-image backscatter statistics, so the
    // SVM boundary matches the per-image data it is later applied to.
    // CRITICAL: clip to aoi + outer land ring, so that land sample points
    // (in the 500–2000 m annulus, outside aoi) sample VALID pixels. If clipped
    // to aoi only, those points hit masked pixels → null → dropped → the SVM
    // sees water-only training → classifies the whole AOI as water.
    var trainClip   = aoi.buffer(CFG.land_ring_outer_m);
    var refStart    = CFG.train_ref_year + '-01-01';
    var refEnd      = (CFG.train_ref_year + 1) + '-01-01';
    var s1Composite = S1_RAW
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.eq('resolution_meters', 10))
      .filterBounds(trainClip)
      .filterDate(refStart, refEnd)
      .select(BANDS)
      .mosaic()
      .focal_mean(30, 'circle', 'meters')
      .clip(trainClip);

    // ── A/P-based classifier selection ───────────────────────
    // Dual-pol SVM only where the shoreline is complex (low A/P); the cheaper
    // single-pol VV Otsu is used elsewhere (equal accuracy above A/P ~120 m).
    var useDual    = (S.ap_m != null && S.ap_m < CFG.ap_low);
    var methodName = useDual ? 'dual-pol SVM (VV+VH)' : 'single-pol VV Otsu';

    // ── Training samples + SVM (only needed for the dual-pol branch) ──
    var trainingFC = autoTrainingSamples(lakePoly, aoi);
    var classifier = useDual ? trainSVM(trainingFC, s1Composite) : null;

    // ── Preprocess + classify + clean ────────────────────────
    statusLabel.setValue('A/P = ' + (S.ap_m != null ? S.ap_m.toFixed(0) : '?') +
                         ' m → ' + methodName + '; classifying ' + name + '…');
    var s1Proc = s1Filtered.map(function(img) { return preprocessS1(img, aoi); });
    // For partial-coverage images (global reservoirs at S1 swath edges), mosaic
    // with images from within ±composite_window_days to fill coverage gaps.
    s1Proc = fillCoverageGaps(s1Proc, CFG.composite_window_days);
    var waterMaskCleaned = classifyCollection(s1Proc, classifier, aoi, lakePoly, useDual);

    // Store globally for chart-click callbacks
    G_waterMaskCleaned = waterMaskCleaned;
    G_aoi              = aoi;
    G_name             = name;

    // ── JRC reference series (SAR area is computed in yearly chunks below) ──
    var jrcFC = computeJRCSeries(aoi, start, end);

    // Fixed chart order via placeholder panels (async callbacks fill them):
    //   SAR area (top) · JRC reference (middle) · Volume (bottom).
    var sarPanel = ui.Panel();
    var jrcPanel = ui.Panel();
    var volPanel = ui.Panel();
    chartPanel.add(sarPanel);
    chartPanel.add(jrcPanel);
    chartPanel.add(volPanel);

    // ── Map: show most recent image ──────────────────────────
    var lastImg = waterMaskCleaned.sort('system:time_start', false).first();
    ee.Image(lastImg).date().format('YYYY-MM-dd').evaluate(function(lastDate) {
      if (lastDate) {
        mapDateLabel.setValue(lastDate);
        mapDateLabel.style().set({shown: true});
      }
    });
    mapObj.layers().reset();
    mapObj.addLayer(
      ee.FeatureCollection([ee.Feature(lakePoly)]).style({
        color: apColorHex(S.ap_m), fillColor: '00000000', width: 2,
      }), {}, name + ' boundary (A/P)'
    );
    mapObj.addLayer(
      lastImg.select('VV').clip(aoi),
      {min: -20, max: 0, palette: ['000000','ffffff'], opacity: 0.7},
      'SAR VV (latest)'
    );
    mapObj.addLayer(
      lastImg.select('WaterFilled').updateMask(ee.Image.constant(1).clip(lakePoly)),
      {palette: ['1565c0'], opacity: 0.85},
      'Water (latest)'
    );

    // ── JRC reference chart (guard: GSW Monthly ends ~2021) ──
    jrcFC.size().evaluate(function(nJrc) {
      if (!nJrc) {
        jrcPanel.add(ui.Label(
          'JRC optical reference unavailable for this period ' +
          '(GSW Monthly History ends ~2021).',
          {color: '#888', fontSize: '11px', margin: '4px 0'}
        ));
        return;
      }
      var jrcChart = ui.Chart.feature.byFeature(jrcFC.sort('date'), 'date', ['jrc_area_ha'])
        .setChartType('LineChart').setOptions({
          title: name + '  —  JRC optical reference (monthly, 30 m)',
          hAxis: {title: 'Date', format: 'YYYY-MM-dd', gridlines: {count: -1}},
          vAxis: {title: 'Area (ha)', minValue: 0},
          series: {0: {color: 'e65100', lineWidth: 1.5,
                       pointSize: 3, lineDashStyle: [4, 2]}},
          legend: {position: 'none'},
          height: 210,
          chartArea: {left: 60, right: 10, top: 40, bottom: 60},
        });
      jrcPanel.add(jrcChart);
    });

    // ── SAR area + Volume — chunked (CFG.chunk_months), concurrent, progressive ──
    // The area series is the dominant cost (one SVM classify + reduceRegion per
    // image). Chunking the date range lets GEE parallelise the work AND lets the
    // chart fill in as each chunk returns. The chart opens immediately spanning
    // the FULL period (grey while loading) via a fixed date-axis viewWindow, and
    // turns white when complete. Cleaning + LOWESS run client-side (cleanAndSmoothJS).
    function _isoDate(d) { return d.toISOString().slice(0, 10); }
    function _addMonths(d, m) {
      return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + m, d.getUTCDate()));
    }
    var startD   = new Date(start + 'T00:00:00Z');
    var endView  = new Date(end + 'T00:00:00Z');
    var endExcl  = new Date(endView.getTime() + 86400000);   // end day inclusive
    var chunks   = [];
    var cur      = startD;
    while (cur < endExcl) {
      var nxt = _addMonths(cur, CFG.chunk_months || 3);
      if (nxt > endExcl) nxt = endExcl;
      chunks.push([_isoDate(cur), _isoDate(nxt)]);   // half-open [cs, ce)
      cur = nxt;
    }

    var allRaw = [];
    var nDone  = 0;
    var nTotal = chunks.length;

    // Numeric (ms) x-axis: continuous (so viewWindow can fix the full period) and
    // it survives GEE's chart serialisation, unlike a Date-typed column (which gets
    // stringified → "axis #0 cannot be of type string"). Date labels come from
    // custom ticks instead of an axis format.
    var startMs = startD.getTime();
    var endMs   = endView.getTime();

    function fmtYM(ms) {
      var d = new Date(ms);
      return d.getUTCFullYear() + '-' + ('0' + (d.getUTCMonth() + 1)).slice(-2);
    }
    function fmtDate(ms) {
      var d = new Date(ms);
      return d.getUTCFullYear() + '-' + ('0' + (d.getUTCMonth() + 1)).slice(-2) +
             '-' + ('0' + d.getUTCDate()).slice(-2);
    }
    function makeTicks(minMs, maxMs, n) {
      var ticks = [];
      for (var k = 0; k <= n; k++) {
        var v = Math.round(minMs + (maxMs - minMs) * k / n);
        ticks.push({v: v, f: fmtYM(v)});
      }
      return ticks;
    }
    var X_TICKS = makeTicks(startMs, endMs, Math.min(8, Math.max(3, nTotal)));

    function clickToDate(x) {
      if (x == null) return;
      var ms = (x instanceof Date) ? x.getTime() : x;
      loadDateOnMap(new Date(ms).toISOString().slice(0, 10));
    }

    function renderArea(isFinal) {
      var rows = allRaw.filter(function(r) { return r.area_ha != null && r.t != null; })
                       .sort(function(a, b) { return a.t - b.t; });
      var cleaned = rows.length ? cleanAndSmoothJS(rows) : [];

      // No points yet → show a label, not an empty chart (an all-null DataTable
      // makes gviz throw transiently before data arrives).
      if (!cleaned.length) {
        sarPanel.clear();
        sarPanel.add(ui.Label(
          isFinal ? 'No SAR water area for this period.'
                  : 'Computing area series…  [0/' + nTotal + ' periods]',
          {color: '#888', fontSize: '12px', margin: '8px 4px'}));
        if (isFinal) volPanel.clear();
        return;
      }

      // x cells use {v: ms, f: 'YYYY-MM-DD'}: v keeps the axis numeric/continuous
      // (so viewWindow works and it survives serialisation); f is what the tooltip
      // shows (otherwise the raw ms timestamp appears). Axis labels come from X_TICKS.
      // Boundary nulls (start/end ms) fix the axis span even with few points.
      var dt = [['Time', 'Raw area', 'LOWESS smoothed'],
                [{v: startMs, f: fmtDate(startMs)}, null, null]];
      cleaned.forEach(function(r) {
        dt.push([{v: r.t, f: fmtDate(r.t)}, r.area_ha, r.area_ha_smoothed]);
      });
      dt.push([{v: endMs, f: fmtDate(endMs)}, null, null]);

      var sarChart = ui.Chart(dt, 'LineChart', {
        title: name + '  —  SAR water area (SVM auto-training)' +
               (isFinal ? '\n(click a point to load that date on map)'
                        : '   [' + nDone + '/' + nTotal + ' periods…]'),
        hAxis: {title: 'Date', viewWindow: {min: startMs, max: endMs}, ticks: X_TICKS},
        vAxis: {title: 'Area (ha)', minValue: 0},
        series: {0: {color: '90caf9', lineWidth: 1, pointSize: 3},
                 1: {color: 'd32f2f', lineWidth: 2.5, pointSize: 0}},
        legend: {position: 'top', maxLines: 2},
        height: 280,
        chartArea: {left: 60, right: 10, top: 60, bottom: 60},
        backgroundColor: {fill: isFinal ? '#ffffff' : '#f0f0f0'},   // grey while loading
        interpolateNulls: false,
      });
      sarChart.onClick(clickToDate);
      sarPanel.clear();
      sarPanel.add(sarChart);

      if (!isFinal) return;

      // Volume only on the final (complete) smoothed series.
      var hasVolCoeff = false;
      for (var key in VOLUME_POLY) {
        if (name.indexOf(key) !== -1) { hasVolCoeff = true; break; }
      }
      volPanel.clear();
      if (hasVolCoeff) {
        var volDT = [['Time', 'Volume (Mm³)'], [{v: startMs, f: fmtDate(startMs)}, null]];
        cleaned.forEach(function(r) {
          var a   = (r.area_ha_smoothed != null) ? r.area_ha_smoothed : r.area_ha;
          var vol = volumeFromArea(a, name);
          if (vol !== null) volDT.push([{v: r.t, f: fmtDate(r.t)}, Math.max(vol, 0)]);
        });
        volDT.push([{v: endMs, f: fmtDate(endMs)}, null]);
        if (volDT.length > 3) {
          var volChart = ui.Chart(volDT, 'LineChart', {
            title: name + '  —  Volume (Mm³) from AEV polynomial' +
                   '\n(click a point to load that date on map)',
            hAxis: {title: 'Date', viewWindow: {min: startMs, max: endMs}, ticks: X_TICKS},
            vAxis: {title: 'Volume (Mm³)', minValue: 0},
            series: {0: {color: '2e7d32', lineWidth: 2, pointSize: 3}},
            legend: {position: 'none'},
            height: 210,
            chartArea: {left: 60, right: 10, top: 50, bottom: 60},
            interpolateNulls: false,
          });
          volChart.onClick(clickToDate);
          volPanel.add(volChart);
        }
      } else if (CFG.grdl_path) {
        volPanel.add(ui.Label('Volume: looking up GRDL asset…',
          {color: '#888', fontSize: '11px', margin: '4px 0'}));
      } else {
        volPanel.add(ui.Label(
          'Volume: set CFG.grdl_path to a GRDL GEE asset to enable global volume.',
          {color: '#888', fontSize: '11px', margin: '4px 0'}));
      }
    }

    // Seed the empty, full-period (grey) chart immediately; advance the loader.
    renderArea(false);
    showLoader('Computing area series…  0/' + nTotal);
    statusLabel.setValue('Computing area series… 0/' + nTotal + ' periods');

    chunks.forEach(function(rng) {
      var chunkFC = computeAreaSeries(waterMaskCleaned.filterDate(rng[0], rng[1]), aoi);
      chunkFC.evaluate(function(fcEval, err) {
        nDone++;
        if (!err && fcEval && fcEval.features) {
          fcEval.features.forEach(function(f) {
            allRaw.push({date:    f.properties.date,
                         t:       f.properties['system:time_start'],
                         area_ha: f.properties.area_ha});
          });
        }
        var isFinal = (nDone === nTotal);
        renderArea(isFinal);
        updateLoader(nDone / nTotal, 'Computing area series…  ' + nDone + '/' + nTotal);
        statusLabel.setValue(isFinal
          ? '✓ Done — ' + name + '  •  ' + allRaw.length + ' SAR points  •  ' +
            start.slice(0, 4) + '–' + end.slice(0, 4) +
            '  •  A/P = ' + (S.ap_m != null ? S.ap_m.toFixed(0) : '?') + ' m'
          : 'Computing area series… ' + nDone + '/' + nTotal +
            ' periods (' + allRaw.length + ' pts so far)');
        if (isFinal) { hideLoader(); runBtn.setDisabled(false); }
      });
    });
  });
});

resetBtn.onClick(function() {
  S.hylak_id  = null; S.lake_name = null;
  S.hydroGeom = null; S.lakePoly  = null;
  S.aoi       = null; S.ap_m      = null;
  G_waterMaskCleaned = null; G_aoi = null; G_name = null;

  selectContainer.clear();
  chartPanel.clear();
  resetBadge();
  selectedLabel.setValue('No reservoir selected');
  selectedLabel.style().set({color: '#888', fontStyle: 'italic'});
  mapNameLabel.setValue(''); mapNameLabel.style().set({shown: false});
  mapDateLabel.setValue(''); mapDateLabel.style().set({shown: false});
  runBtn.setDisabled(true);
  statusLabel.setValue('Select a reservoir above to begin.');

  mapObj.layers().reset();
  showInitialPoints();
  if (CFG.dataset === 'sicily') { mapObj.setCenter(14.0, 37.5, 8); }
  else { mapObj.setCenter(0, 20, 3); }
});

// ─── 19. A/P BADGE ───────────────────────────────────────────────────────
// Sequential A/P colour (matches the study-area figure): light -> dark = low -> high A/P.
function apColorHex(ap) {
  if (ap == null || isNaN(+ap)) return '9e9e9e';
  return (ap < CFG.ap_low) ? 'f88f4d' : (ap < CFG.ap_high) ? 'd64a02' : '8a2d04';
}

function updateBadge(ap_m) {
  if (ap_m == null || isNaN(+ap_m)) return;
  apValueLabel.setValue((+ap_m).toFixed(0) + ' m');
  if (ap_m >= CFG.ap_high) {
    apBadgeLabel.setValue('● High reliability  (A/P ≥ ' + CFG.ap_high + ' m)  ·  VV Otsu');
    apBadgeLabel.style().set({
      backgroundColor: '#c8e6c9', color: '#1b5e20', border: '1px solid #a5d6a7',
    });
    apDescLabel.setValue(
      'Compact shoreline: few mixed pixels at SAR scale. The single-pol VV Otsu ' +
      'is selected (equal accuracy to dual-pol here, at lower cost).'
    );
  } else if (ap_m >= CFG.ap_low) {
    apBadgeLabel.setValue('◐ Medium reliability  (' + CFG.ap_low + '–' + CFG.ap_high + ' m)  ·  VV Otsu');
    apBadgeLabel.style().set({
      backgroundColor: '#fff9c4', color: '#e65100', border: '1px solid #ffe082',
    });
    apDescLabel.setValue(
      'Moderate shoreline complexity. The single-pol VV Otsu is selected; dual-pol ' +
      'gives no consistent gain above A/P ~120 m. Cross-check against JRC if in doubt.'
    );
  } else {
    apBadgeLabel.setValue('○ Low A/P  (< ' + CFG.ap_low + ' m)  ·  dual-pol SVM');
    apBadgeLabel.style().set({
      backgroundColor: '#ffcdd2', color: '#b71c1c', border: '1px solid #ef9a9a',
    });
    apDescLabel.setValue(
      'Narrow / dendritic shoreline with many mixed pixels. The dual-pol (VV+VH) ' +
      'SVM is selected to recover water that a single-band threshold misses.'
    );
  }
}

function resetBadge() {
  apValueLabel.setValue('— m');
  apBadgeLabel.setValue('Computing…');
  apBadgeLabel.style().set({
    backgroundColor: '#e0e0e0', color: '#555', border: '1px solid #bdbdbd',
  });
  apDescLabel.setValue('');
}

// ─── 20. UI HELPERS ──────────────────────────────────────────────────────
function divider() {
  return ui.Panel([], null, {
    backgroundColor: '#e0e0e0', height: '1px', margin: '7px 0',
  });
}

function sectionLabel(text) {
  return ui.Label(text, {fontWeight: 'bold', margin: '4px 0 4px 0'});
}

// ─── 21. INITIAL MAP LAYER ───────────────────────────────────────────────
// Render reservoir locations on startup so users can click.
// HYDROLAKES holds the active dataset's polygon/feature collection for both modes.
function showInitialPoints() {
  mapObj.addLayer(
    HYDROLAKES.style({color: '1E88E5', fillColor: '1E88E544', pointSize: 7, width: 2}),
    {}, 'Reservoirs — click to select'
  );
}

// ─── 22. MAP CLICK → SELECT RESERVOIR ───────────────────────────────────
mapObj.onClick(function(coords) {
  // Sicily: client-side nearest-centroid from pre-loaded cache (instant)
  if (DS.browsable && HYDROLAKES_CACHE && HYDROLAKES_CACHE.centroids.length) {
    var minDist = Infinity, nearestItem = null;
    HYDROLAKES_CACHE.centroids.forEach(function(c, idx) {
      if (!c) return;
      var dx = c.lon - coords.lon, dy = c.lat - coords.lat;
      var d  = Math.sqrt(dx * dx + dy * dy);
      if (d < minDist) { minDist = d; nearestItem = HYDROLAKES_CACHE.items[idx]; }
    });
    if (nearestItem && minDist < 0.5) {
      showSelectItems([nearestItem]);
      statusLabel.setValue('Clicked: ' + nearestItem.label + '  •  Press Run ▶');
      onReservoirSelected(nearestItem.value);
    } else {
      statusLabel.setValue('No reservoir near click. Zoom in and try again.');
    }
    return;
  }

  // Global (or Sicily before cache loads): server-side spatial query.
  // Buffer: 5 km for polygons (click inside reservoir body) catches edge clicks.
  statusLabel.setValue('Finding reservoir at click…');
  var clickPt = ee.Geometry.Point([coords.lon, coords.lat]);
  var nearby  = HYDROLAKES.filterBounds(clickPt.buffer(5000));
  if (DS.type_filter) {
    nearby = nearby.filter(ee.Filter.eq(DS.type_filter[0], DS.type_filter[1]));
  }
  nearby.limit(5).evaluate(function(fc, err) {
    if (err || !fc || !fc.features || !fc.features.length) {
      statusLabel.setValue('No reservoir near click. Zoom in and try again.');
      return;
    }
    var items = fc.features.map(function(f) { return lakeLabel(f.properties); });
    showSelectItems(items);
    if (items.length === 1) {
      statusLabel.setValue('Clicked: ' + items[0].label + '  •  Press Run ▶');
      onReservoirSelected(items[0].value);
    } else {
      statusLabel.setValue(items.length + ' reservoirs near click — select one above.');
    }
  });
});

// ─── 23. STARTUP ─────────────────────────────────────────────────────────
if (CFG.dataset === 'sicily') {
  mapObj.setCenter(14.0, 37.5, 8);
} else {
  mapObj.setCenter(0, 20, 2);
}
showInitialPoints();
loadCacheIfBrowsable(null);   // background pre-load for instant search + hover

// Map legend overlay
(function() {
  function colorBox(hex) {
    return ui.Panel([], null, {
      backgroundColor: hex, width: '14px', height: '14px',
      margin: '3px 6px 3px 0', border: '1px solid rgba(0,0,0,0.25)',
    });
  }
  function legendRow(hex, text) {
    return ui.Panel([colorBox(hex), ui.Label(text, {fontSize: '11px', margin: '2px 0'})],
                    ui.Panel.Layout.Flow('horizontal'));
  }
  var lg = ui.Panel({
    style: {position: 'top-right', padding: '8px 10px',
            backgroundColor: 'rgba(255,255,255,0.92)'},
  });
  lg.add(ui.Label('Legend', {fontWeight: 'bold', fontSize: '12px', margin: '0 0 4px 0'}));
  lg.add(legendRow('#1E88E5', 'Reservoir locations (click to select)'));
  lg.add(legendRow('#00FFFF', 'Reservoir boundary (JRC max extent)'));
  lg.add(legendRow('#006699', 'JRC occurrence (background)'));
  lg.add(legendRow('#1565c0', 'Detected water (SAR-SVM)'));
  mapObj.add(lg);
})();

// Persistent map overlay: reservoir name (bottom-left) and current image date (bottom-right)
var mapNameLabel = ui.Label('', {
  fontSize: '30px', fontWeight: 'bold', color: 'white',
  backgroundColor: 'rgba(0,0,0,0.55)', padding: '5px 14px',
  position: 'bottom-left', margin: '0', shown: false,
});
var mapDateLabel = ui.Label('', {
  fontSize: '22px', fontWeight: 'bold', color: 'white',
  backgroundColor: 'rgba(0,0,0,0.55)', padding: '5px 14px',
  position: 'bottom-right', margin: '0', shown: false,
});
mapObj.add(mapNameLabel);
mapObj.add(mapDateLabel);

// ─── 24. MAP LOADING BAR (determinate progress) ──────────────────────────
// GEE ui has no timer/spinner, so this is a DETERMINATE bar driven by chunk
// completion (nDone/nTotal) — it advances each time a chunk returns. Shown over
// the map while the area series computes; hidden when complete.
var LOADER_W = 320;
var loaderFill = ui.Panel([], null,
  {backgroundColor: '#1565c0', height: '12px', width: '0px', margin: '0'});
var loaderTrack = ui.Panel([loaderFill], null,
  {backgroundColor: '#cfd8dc', width: LOADER_W + 'px', height: '12px',
   margin: '6px 0 0 0', border: '1px solid #90a4ae'});
var loaderText = ui.Label('', {fontSize: '12px', fontWeight: 'bold',
                               color: '#1565c0', margin: '0', textAlign: 'center'});
var loaderBox = ui.Panel([loaderText, loaderTrack], ui.Panel.Layout.Flow('vertical'),
  {position: 'top-center', padding: '10px 16px', shown: false,
   backgroundColor: 'rgba(255,255,255,0.95)', border: '1px solid #b0bec5'});
mapObj.add(loaderBox);

function showLoader(text) {
  loaderText.setValue(text || 'Loading…');
  loaderFill.style().set('width', '8px');   // small sliver = "started"
  loaderBox.style().set('shown', true);
}
function updateLoader(frac, text) {
  var w = Math.round(LOADER_W * Math.max(0, Math.min(1, frac)));
  loaderFill.style().set('width', w + 'px');
  if (text) loaderText.setValue(text);
}
function hideLoader() {
  loaderBox.style().set('shown', false);
}
