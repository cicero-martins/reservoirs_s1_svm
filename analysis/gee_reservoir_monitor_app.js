// ═══════════════════════════════════════════════════════════════════════════
// RESERVOIR SAR MONITOR — Global GEE App  (v3 — aligned with original)
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
// NOTE on AOI source: HydroLAKES polygons represent average fill; they
//   underestimate A/P by 33–53% vs. JRC max_extent for Sicilian reservoirs.
//   The app uses HydroLAKES ONLY for name search + initial map zoom.
//
// PASTE INTO GEE CODE EDITOR AND CLICK "RUN"
// ═══════════════════════════════════════════════════════════════════════════

// ─── 1. CONFIGURATION ────────────────────────────────────────────────────
var CFG = {
  // HydroLAKES — choose ONE path (comment out the other)
  //
  // Option A: personal Sicily asset (known to work for 4 Sicilian reservoirs):
  hydrolakes_path: 'projects/ee-ciceromartinsjr/assets/HydroLAKES_sicily4',
  //
  // Option B: Awesome GEE Community Catalog — global coverage, but requires
  // that your account can access sat-io datasets (check in Assets panel):
  // hydrolakes_path: 'projects/sat-io/open-datasets/HydroLakes/HydroLAKES_polys_v10',

  // JRC max_extent search: radius around HydroLAKES centroid
  jrc_search_buffer_m: 5000,

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
  sar_scale_m:  10,
  max_pixels:   1e9,

  // Connected-component cleaning: false = keep all polygons intersecting the
  // JRC footprint (recovers disconnected reservoir arms); true = original
  // behaviour (single largest polygon only).
  keep_largest_only: false,

  // A/P reliability thresholds (from ROC pilot study, AUC = 0.71, N = 20)
  ap_high: 333,   // ≥ 333 m → 88% precision at KGE ≥ 0.5
  ap_med:  200,   // 200–333 m → moderate

  // Optional: GRDL FeatureCollection asset for global volume curves
  grdl_path: null,
};

// ─── 2. DATA SOURCES ─────────────────────────────────────────────────────
var HYDROLAKES = ee.FeatureCollection(CFG.hydrolakes_path);
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
                          'relativeOrbitNumber_start', 'angle']);
}

// ─── 4. JRC MAX_EXTENT → AOI POLYGON ─────────────────────────────────────
// Returns the largest contiguous JRC water body within searchBuffer of the
// HydroLAKES centroid. This is the definitive AOI for training and classification.
function jrcMaxExtentPoly(hydroGeom, searchBuffer) {
  var searchArea = hydroGeom.centroid().buffer(searchBuffer);
  var jrcMax     = JRC_GSW.select('max_extent').eq(1).selfMask();
  var waterVecs  = jrcMax.reduceToVectors({
    geometry:       searchArea,
    scale:          30,
    maxPixels:      1e9,
    bestEffort:     true,
    geometryType:   'polygon',
    eightConnected: true,
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

  function tryNextBin(angleList, i) {
    var current     = ee.Number(angleList.get(i));
    var next        = current.subtract(3);
    var binFiltered = withAngle.filter(ee.Filter.and(
      ee.Filter.gte('angle_mean', next),
      ee.Filter.lt('angle_mean', current)
    ));

    var withCoverage = binFiltered.map(function(img) {
      var mask        = img.select('VV').mask();
      var coveredArea = mask.multiply(ee.Image.pixelArea())
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi, scale: 10, maxPixels: 1e9})
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
      } else if (i < angleList.length().getInfo() - 1) {
        tryNextBin(angleList, i + 1);
      } else {
        callback(withAngle.sort('system:time_start'));
      }
    });
  }

  angleList.evaluate(function(list) {
    if (list && list.length > 1) {
      tryNextBin(ee.List(list), 0);
    } else {
      callback(withAngle.sort('system:time_start'));
    }
  });
}

// ─── 8. CLASSIFY + CLEAN WATER MASK ─────────────────────────────────────
// Returns collection with 'Water', 'WaterFilled', 'WaterCleaned' bands added.
// Gap-fill (fastDistanceTransform) + connected-component cleaning.
//
// CLEANING CHANGE vs original: the original kept ONLY the single largest
// connected polygon, which dropped genuine reservoir arms/parcels whenever
// classification noise disconnected them (e.g. the dam pool of dendritic
// Ancipa) → systematic low bias. Here we instead keep EVERY connected water
// polygon that intersects the JRC max-extent polygon (lakePoly), i.e. the
// historical reservoir footprint. Spurious patches outside that footprint are
// still discarded. Set CFG keep_largest_only = true to restore old behaviour.
function classifyCollection(s1Proc, classifier, aoi, lakePoly) {
  // Step 1 — SVM classification
  var withWater = s1Proc.map(function(img) {
    var water = img.select(BANDS).classify(classifier).eq(1).clip(aoi);
    return img.addBands(water.rename('Water'));
  });

  // Step 2 — morphological gap-fill (close small holes)
  var withFilled = withWater.map(function(img) {
    var mask   = img.select('Water').unmask(0).clip(aoi);
    var dist   = mask.fastDistanceTransform(30).clip(aoi);
    var filled = dist.lte(0.5).updateMask(dist.lte(0.5));
    return img.addBands(filled.where(mask, 1).rename('WaterFilled'));
  });

  // Step 3 — connected-component cleaning
  return withFilled.map(function(img) {
    var mask   = img.select('WaterFilled');
    var polys  = mask.reduceToVectors({
      geometryType: 'polygon', reducer: ee.Reducer.countEvery(),
      scale: 10, maxPixels: CFG.max_pixels, bestEffort: true,
    });

    var keptPolys;
    if (CFG.keep_largest_only) {
      keptPolys = ee.FeatureCollection([
        polys.map(function(f) {
          return f.set('area', f.geometry().area({maxError: 1}));
        }).sort('area', false).first()
      ]);
    } else {
      // Keep all polygons that intersect the reservoir's historical footprint.
      keptPolys = polys.filterBounds(lakePoly);
    }

    var keptMask = ee.Image().paint({
      featureCollection: keptPolys, color: 1,
    }).rename('KeptRegionMask');
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
        scale: CFG.sar_scale_m, maxPixels: CFG.max_pixels,
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
function cleanAndSmooth(areaFC) {
  var ts1 = removeOutliers(areaFC, 2);
  var ts2 = detectAndRemoveLocalOutliers(ts1, 5, 1.5);
  var ts3 = detectAndRemoveLocalOutliers(ts2, 5, 1.5);
  var ts4 = detectAndRemoveLocalOutliers(ts3, 10, 1.5);
  return lowessSmoothing(ts4, 20, 7);
}

// ─── 10. JRC REFERENCE AREA ──────────────────────────────────────────────
function computeJRCSeries(aoi, start, end) {
  return ee.FeatureCollection(
    JRC_MON.filterBounds(aoi).filterDate(start, end).map(function(img) {
      var water   = img.eq(2).clip(aoi);
      var area_m2 = water.multiply(ee.Image.pixelArea()).reduceRegion({
        reducer: ee.Reducer.sum(), geometry: aoi,
        scale: 30, maxPixels: 1e8,
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
panel.add(ui.Label('Global · JRC auto-training · SVM RBF · A/P reliability', {
  fontSize: '11px', color: '#555', margin: '0 0 8px 0',
}));

// ── Search ─────────────────────────────────────────────────────
panel.add(divider());
panel.add(sectionLabel('1 · Find reservoir'));

var searchBox = ui.Textbox({
  placeholder: 'Type name (e.g. Pozzillo, Manikdoh, Chichester)…',
  style: {width: '245px'},
});
var searchBtn = ui.Button({label: 'Search', style: {margin: '0 0 0 6px'}});
panel.add(ui.Panel([searchBox, searchBtn], ui.Panel.Layout.Flow('horizontal')));

// GEE ui.Select does not support setItems() after creation —
// use a container Panel and swap the widget on each search.
var selectContainer = ui.Panel({style: {margin: '6px 0 2px 0'}});
panel.add(selectContainer);

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
  style: {width: '350px', margin: '8px 0', backgroundColor: '#1565c0'},
  disabled: true,
});
var statusLabel = ui.Label('Select a reservoir above to begin.', {
  color: '#444', fontSize: '12px', margin: '2px 0 6px 0',
});
panel.add(runBtn);
panel.add(statusLabel);

// ── A/P indicator ──────────────────────────────────────────────
panel.add(divider());
panel.add(sectionLabel('A/P reliability indicator'));

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
// Helper: build a label string from whatever properties a HydroLAKES feature has
function lakeLabel(p) {
  // res_name is the confirmed field name in HydroLAKES_sicily4
  var nm   = p.res_name  || p.Lake_name || p.lake_name || p.name || p.Name || '';
  var area = p.Lake_area || p.lake_area  || p.LAKE_AREA || 0;
  var ctry = p.Country   || p.country    || p.COUNTRY   || '';
  var id   = p.Hylak_id  || p.Hylak_ID  || p.hylak_id  || p.Grand_id || p.FID || '?';
  return {
    label: (nm || 'ID=' + id) + (ctry ? ' — ' + ctry : '') +
           (area ? ' (' + (+area).toFixed(1) + ' km²)' : ''),
    value: String(id),
    name:  nm,
  };
}

searchBtn.onClick(function() {
  var name = (searchBox.getValue() || '').trim();
  statusLabel.setValue(name ? 'Searching for "' + name + '"…' : 'Loading all reservoirs…');
  selectContainer.clear();

  var col;
  if (name) {
    var u = name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
    col = HYDROLAKES.filter(ee.Filter.or(
      ee.Filter.stringContains('res_name',  name),
      ee.Filter.stringContains('res_name',  u),
      ee.Filter.stringContains('res_name',  name.toUpperCase()),
      ee.Filter.stringContains('res_name',  name.toLowerCase()),
      ee.Filter.stringContains('Lake_name', name),
      ee.Filter.stringContains('Lake_name', u)
    ));
  } else {
    col = HYDROLAKES;
  }

  col.limit(50).evaluate(function(fc, err) {
    if (err) {
      statusLabel.setValue('⚠ Error: ' + err);
      return;
    }
    if (!fc || !fc.features || !fc.features.length) {
      statusLabel.setValue('"' + name + '" not found — loading full asset…');
      HYDROLAKES.limit(100).evaluate(function(fc2, err2) {
        if (err2 || !fc2 || !fc2.features || !fc2.features.length) {
          statusLabel.setValue('⚠ Asset empty or inaccessible: ' + (err2 || 'no features'));
          return;
        }
        showSelectItems(fc2.features.map(function(f) { return lakeLabel(f.properties); }));
        statusLabel.setValue(
          '"' + name + '" not found. Showing all ' + fc2.features.length + ' features.'
        );
      });
      return;
    }
    showSelectItems(fc.features.map(function(f) { return lakeLabel(f.properties); }));
    statusLabel.setValue(fc.features.length + ' result(s). Select one.');
  });
});

// ─── 17. RESERVOIR SELECTED ──────────────────────────────────────────────
// Called by onChange of the dynamically-created ui.Select inside selectContainer
function onReservoirSelected(id_str) {
  if (!id_str) return;

  statusLabel.setValue('Loading lake…');
  runBtn.setDisabled(true);
  chartPanel.clear();
  mapObj.layers().reset();
  resetBadge();

  var hylak_id = parseInt(id_str, 10);
  var lake;
  if (!isNaN(hylak_id) && id_str !== '?') {
    lake = HYDROLAKES.filter(
      ee.Filter.or(
        ee.Filter.eq('Hylak_id', hylak_id),   // confirmed field name
        ee.Filter.eq('Hylak_ID', hylak_id),   // fallback for global HydroLAKES
        ee.Filter.eq('Grand_id', hylak_id)    // GRanD ID fallback
      )
    ).first();
  } else {
    lake = HYDROLAKES.first();
  }

  lake.evaluate(function(f, err) {
    if (err) { statusLabel.setValue('⚠ Lake load error: ' + err); return; }
    if (!f)  { statusLabel.setValue('Could not load lake — check HydroLAKES path.'); return; }
    var props     = f.properties;
    var hydroGeom = ee.Feature(f).geometry();
    var name      = props.res_name  || props.Lake_name || props.lake_name ||
                    props.name || props.Name || ('Lake #' + id_str);

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
      S.hylak_id  = hylak_id;
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
          color: '00FFFF', fillColor: '006699AA', width: 2,
        }), {}, name + ' (JRC max extent)'
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
        color: '00FFFF', fillColor: '00000000', width: 2,
      }), {}, G_name + ' boundary'
    );
    mapObj.addLayer(
      ee.Image(img).select('VV').clip(G_aoi),
      {min: -20, max: 0, palette: ['000000','ffffff'], opacity: 0.7},
      'SAR VV  ' + dateStr
    );
    mapObj.addLayer(
      ee.Image(img).select('WaterCleaned').selfMask().clip(G_aoi),
      {palette: ['1565c0'], opacity: 0.85},
      'Water  ' + dateStr
    );
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

    // ── Training samples ─────────────────────────────────────
    var trainingFC = autoTrainingSamples(lakePoly, aoi);

    // ── Train SVM ────────────────────────────────────────────
    var classifier = trainSVM(trainingFC, s1Composite);

    // ── Preprocess + classify + clean ────────────────────────
    statusLabel.setValue('Classifying ' + name + '…');
    var s1Proc           = s1Filtered.map(function(img) { return preprocessS1(img, aoi); });
    var waterMaskCleaned = classifyCollection(s1Proc, classifier, aoi, lakePoly);

    // Store globally for chart-click callbacks
    G_waterMaskCleaned = waterMaskCleaned;
    G_aoi              = aoi;
    G_name             = name;

    // ── Area + JRC series ────────────────────────────────────
    var areaFC    = computeAreaSeries(waterMaskCleaned, aoi);
    var areaClean = cleanAndSmooth(areaFC);  // outlier removal (4 passes) + LOWESS
    var jrcFC     = computeJRCSeries(aoi, start, end);

    // ── Map: show most recent image ──────────────────────────
    var lastImg = waterMaskCleaned.sort('system:time_start', false).first();
    mapObj.layers().reset();
    mapObj.addLayer(
      ee.FeatureCollection([ee.Feature(lakePoly)]).style({
        color: '00FFFF', fillColor: '00000000', width: 2,
      }), {}, name + ' boundary'
    );
    mapObj.addLayer(
      lastImg.select('VV').clip(aoi),
      {min: -20, max: 0, palette: ['000000','ffffff'], opacity: 0.7},
      'SAR VV (latest)'
    );
    mapObj.addLayer(
      lastImg.select('WaterCleaned').selfMask().clip(aoi),
      {palette: ['1565c0'], opacity: 0.85},
      'Water (latest)'
    );

    // ── SAR area chart — raw + LOWESS, click updates map ─────
    var sarChart = ui.Chart.feature.byFeature(
        areaClean.sort('date'), 'date', ['area_ha', 'area_ha_smoothed'])
      .setChartType('LineChart').setOptions({
        title: name + '  —  SAR water area (SVM auto-training)' +
               '\n(click a point to load that date on map)',
        hAxis: {title: 'Date', format: 'YYYY-MM-dd', gridlines: {count: -1}},
        vAxis: {title: 'Area (ha)', minValue: 0},
        series: {
          0: {color: '90caf9', lineWidth: 1,   pointSize: 3, labelInLegend: 'Raw area'},
          1: {color: 'd32f2f', lineWidth: 2.5, pointSize: 0, labelInLegend: 'LOWESS smoothed'},
        },
        legend: {position: 'top', maxLines: 2},
        height: 280,
        chartArea: {left: 60, right: 10, top: 60, bottom: 60},
      });
    sarChart.onClick(function(xValue) {
      if (!xValue) return;
      loadDateOnMap(xValue);
    });
    chartPanel.add(sarChart);

    // ── JRC reference chart (guard: GSW Monthly ends ~2021) ──
    jrcFC.size().evaluate(function(nJrc) {
      if (!nJrc) {
        chartPanel.add(ui.Label(
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
      chartPanel.add(jrcChart);
    });

    // ── Volume chart (polynomial, Sicilian reservoirs) ───────
    var hasVolCoeff = false;
    for (var key in VOLUME_POLY) {
      if (name.indexOf(key) !== -1) { hasVolCoeff = true; break; }
    }
    if (hasVolCoeff) {
      // Volume is derived from the LOWESS-smoothed area (original uses
      // areaLago_smoothed), so the volume curve is not driven by noise spikes.
      areaClean.sort('date').evaluate(function(fc) {
        if (!fc || !fc.features) return;
        var volFeats = fc.features.map(function(f) {
          var a   = (f.properties.area_ha_smoothed != null)
                      ? f.properties.area_ha_smoothed
                      : (f.properties.area_ha || 0);
          var vol = volumeFromArea(a, name);
          return ee.Feature(null, {
            'date': f.properties.date,
            'volume_Mm3': vol !== null ? Math.max(vol, 0) : null,
          });
        });
        var volFC    = ee.FeatureCollection(volFeats).filter(ee.Filter.notNull(['volume_Mm3']));
        var volChart = ui.Chart.feature.byFeature(volFC, 'date', ['volume_Mm3'])
          .setChartType('LineChart').setOptions({
            title: name + '  —  Volume (Mm³) from AEV polynomial' +
                   '\n(click a point to load that date on map)',
            hAxis: {title: 'Date', format: 'YYYY-MM-dd', gridlines: {count: -1}},
            vAxis: {title: 'Volume (Mm³)', minValue: 0},
            series: {0: {color: '2e7d32', lineWidth: 2, pointSize: 3}},
            legend: {position: 'none'},
            height: 210,
            chartArea: {left: 60, right: 10, top: 50, bottom: 60},
          });
        volChart.onClick(function(xValue) {
          if (!xValue) return;
          loadDateOnMap(xValue);
        });
        chartPanel.add(volChart);
      });
    } else if (CFG.grdl_path) {
      chartPanel.add(ui.Label(
        'Volume: looking up GRDL asset…', {color: '#888', fontSize: '11px', margin: '4px 0'}
      ));
    } else {
      chartPanel.add(ui.Label(
        'Volume: set CFG.grdl_path to a GRDL GEE asset to enable global volume.',
        {color: '#888', fontSize: '11px', margin: '4px 0'}
      ));
    }

    // ── Status ───────────────────────────────────────────────
    s1Proc.size().evaluate(function(n) {
      statusLabel.setValue(
        '✓ Done  —  ' + name + '  •  ' + (n || '?') + ' S1 images  •  ' +
        start.slice(0, 4) + '–' + end.slice(0, 4) +
        '  •  A/P = ' + (S.ap_m != null ? S.ap_m.toFixed(0) : '?') + ' m'
      );
      runBtn.setDisabled(false);
    });
  });
});

// ─── 19. A/P BADGE ───────────────────────────────────────────────────────
function updateBadge(ap_m) {
  if (ap_m == null || isNaN(+ap_m)) return;
  apValueLabel.setValue((+ap_m).toFixed(0) + ' m');
  if (ap_m >= CFG.ap_high) {
    apBadgeLabel.setValue('● High reliability  (A/P ≥ ' + CFG.ap_high + ' m)');
    apBadgeLabel.style().set({
      backgroundColor: '#c8e6c9', color: '#1b5e20', border: '1px solid #a5d6a7',
    });
    apDescLabel.setValue(
      '88% of reservoirs at this A/P achieved KGE ≥ 0.5 (global pilot, N=20, AUC=0.71). ' +
      'Compact / simple shoreline → low mixed-pixel contamination at SAR scale.'
    );
  } else if (ap_m >= CFG.ap_med) {
    apBadgeLabel.setValue('◐ Moderate reliability  (' + CFG.ap_med + '–' + CFG.ap_high + ' m)');
    apBadgeLabel.style().set({
      backgroundColor: '#fff9c4', color: '#e65100', border: '1px solid #ffe082',
    });
    apDescLabel.setValue(
      'Mixed classification results expected at this shoreline complexity. ' +
      'Validate results against JRC / optical imagery before use.'
    );
  } else {
    apBadgeLabel.setValue('○ Low reliability  (A/P < ' + CFG.ap_med + ' m)');
    apBadgeLabel.style().set({
      backgroundColor: '#ffcdd2', color: '#b71c1c', border: '1px solid #ef9a9a',
    });
    apDescLabel.setValue(
      'Highly irregular or dendritic shoreline. SAR classification likely unreliable ' +
      '(many mixed shore pixels). Cross-validate carefully with optical data.'
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
