/**
 * exportGlobalPilotV4.js
 *
 * Batch GEE export of SAR water-area time series for the global pilot v4 reservoir set.
 * 47 reservoirs (≤1000 ha, man-made only) — trimmed from 60 after GDW lookup scan.
 * Pipeline is identical to exportGlobalPilotV2.js (v226 alignment).
 *
 * Output: CSV in Google Drive folder GEE_GlobalPilotV4 (SAR) / GEE_GlobalPilotV4_JRC (JRC).
 *
 * Batching — run one batch per Code Editor session (GEE memory limit ~6 reservoirs/run):
 *   B01: [0,  6]  — Sau, Susqueda, El_Atazar, Siurana, Bleiloch, Rappbode
 *   B02: [6,  12] — Castillon, Saint_Cassien, Salto, Turano, Katse, Mohale
 *   B03: [12, 18] — Blyde, Cachi, Miyagase, Yamba, El_Burguillo, Boadella
 *   B04: [18, 24] — Puentes_Viejas, Guajaraz, Panneciere, Sarrans, Bilancino, Ampollino
 *   B05: [24, 30] — Arvo, Cecita, Oued_Makhazine, Karapuzha, Saguaro, Canyon_Lake
 *   B06: [30, 35] — Boegoeberg, Woodstock, Tzaneen, Googong, Cardinia
 *   B07: [35, 41] — Triouzoune, Grandval, Deer_Creek, East_Canyon, Pineview, Rockport
 *   B08: [41, 45] — Antero, Shaharchay, Welbedacht, Occhito
 *
 * Dropped 15 total (13 from GDW lookup + 2 from JRC area scan):
 *   Area > 1000 ha (GDW confirmed): Plastiras 1973, Almus 2016, Chelmsford 2970,
 *     Riano 1557, Ebro_Embalse 5438, Eleven_Mile 1237, Demirkopru 3587
 *   Area > 1000 ha (JRC scan confirmed): Aguilar 1410, Cruz_del_Eje 1167
 *   Wrong GDW match / bad polygon: Blue_Rock, La_Vina, Nagle
 *   Not in GDW within 25 km: Abdelmoumen, Suat_Ugurlu, Wadi_Dayqah
 *
 * NOTES:
 *   - 9 reservoirs have gdwId filled from lookup (inline comments); 6 reverted to null
 *     after confirming GDW polygon was too small (Siurana, Bleiloch, Katse, Mohale,
 *     Cecita, Triouzoune) — coordinates adjusted to reservoir body center.
 *   - Three Sila plateau lakes (Ampollino, Arvo, Cecita) are 7–15 km apart — verify
 *     the correct polygon was matched after export (check print() output for ap_m).
 *   - Yamba Dam completed 2015: S1 coverage starts 2014, reservoir fills 2015–2020.
 *   - Oued_Makhazine/Shaharchay: GDW centroids are >19 km away — verify polygon match.
 */

// ── Configuration ─────────────────────────────────────────────────────────────
var CFG = {
  s1_start:               '2014-10-01',
  s1_end:                 '2021-12-31',
  jrc_occ_thresh:         95,
  jrc_occ_fallback:       80,
  train_year:             2023,
  clean_scale_m:          30,
  max_pixels:             1e9,
  keep_largest_only:      false,
  land_ring_inner_m:      500,
  land_ring_outer_m:      2000,
  drive_folder:           'GEE_GlobalPilotV4',
  drive_folder_jrc:       'GEE_GlobalPilotV4_JRC',
  composite_window_days:  6,
  coverage_strict_pct:    90,
  min_coverage_pct:       50,
};

var JRC_ONLY = false;

// ── Classifier selection (method-comparison experiment) ───────────────────────
// 'SVM'          : dual-pol (VV+VH) RBF SVM, JRC auto-trained — original method (Tier 3).
// 'VV_OTSU'      : single-pol VV per-scene Otsu, SAME full post-processing as SVM (Tier 1).
//                  KGE/EECU difference vs SVM isolates the *classifier* effect.
// 'VV_OTSU_FAST' : VV Otsu + pixel-count area inside the pool polygon — NO fill, NO
//                  vectorisation, NO keep-polygon, NO dynamic A/P (Tier 1-fast). Cuts the
//                  dominant (vectorisation) cost to isolate the *pipeline architecture* cost.
var CLASSIFIER = 'VV_OTSU';  // 'SVM' | 'VV_OTSU' | 'VV_OTSU_FAST'

// Convenience flags derived from CLASSIFIER.
var USE_OTSU = (CLASSIFIER === 'VV_OTSU' || CLASSIFIER === 'VV_OTSU_FAST');
var FAST     = (CLASSIFIER === 'VV_OTSU_FAST');

var OTSU = {
  band:          'VV',   // single polarisation thresholded (water = low backscatter)
  hist_buffer_m: 500,    // land ring around the pool for a bimodal histogram
  hist_buckets:  256,    // histogram resolution
};

// JRC reference is classifier-independent — only re-export it on the SVM run.
// For the VV runs the JRC CSVs already exist; leave EXPORT_JRC false to save compute.
var EXPORT_JRC = (CLASSIFIER === 'SVM');

// Output folder is suffixed per mode so the runs never overwrite each other.
var MODE_SUFFIX = (CLASSIFIER === 'VV_OTSU')      ? '_VVotsu'
                : (CLASSIFIER === 'VV_OTSU_FAST') ? '_VVfast'
                : '';
var SAR_FOLDER  = CFG.drive_folder + MODE_SUFFIX;

// ── Pilot v4 reservoir list ───────────────────────────────────────────────────
// Format: [name, lat, lon, gdw_id, dahiti_id, area_ha_approx, hylak_id]
// gdw_id = null → coordinate fallback (largest JRC polygon within 10 km)
// hylak_id = null for all (no Sicily-specific asset needed)
var PILOT_RESERVOIRS = [

  // ── LOW A/P — Dendritic / narrow valley ──────────────────────────────────
  // Spain
  ['Sau',           41.970,   2.385,  null, null,  580, null],  // Ter gorge, Catalonia
  ['Susqueda',      41.933,   2.518,  null, null,  390, null],  // Ter canyon (adj. to Sau — verify polygon)
  ['El_Atazar',     40.892,  -3.637,  null, null,  640, null],  // Lozoya gorge, Madrid supply
  ['Siurana',       41.255,   0.880,  null, null,  260, null],  // coord → reservoir body (was 41.197,0.914 = dam toe)
  // Germany
  ['Bleiloch',      50.695,  11.720,  null, null,  920, null],  // coord → lake center (was 50.637,11.697 = dam wall)
  ['Rappbode',      51.738,  10.913,  null, null,  390, null],  // Harz narrow valley
  // France
  ['Castillon',     43.893,   6.534,  null, null,  640, null],  // Verdon canyon
  ['Saint_Cassien', 43.596,   6.724,  null, null,  430, null],  // Provence; branching
  // Italy
  ['Salto',         42.206,  13.055,  null, null,  475, null],  // Salto gorge, Lazio
  ['Turano',        42.215,  12.966,  null, null,  290, null],  // very narrow Turano gorge
  // DROPPED: Plastiras (GDW 1973 ha > 1000 ha), Almus (GDW 2016 ha), Suat_Ugurlu (not in GDW), Abdelmoumen (not in GDW)
  // Lesotho
  ['Katse',        -29.355,  28.570,  null, null,  325, null],  // coord → lake body E of dam (GDW 2069 area=3330ha → wrong polygon)
  ['Mohale',       -29.505,  28.155,  null, null,  420, null],  // coord → lake center N of dam (GDW 2071 area=2348ha → wrong polygon)
  // South Africa
  ['Blyde',        -24.535,  30.800,  null, null,  240, null],  // Blyde River Canyon gorge
  // Central America
  ['Cachi',          9.810, -83.769,  null, null,  340, null],  // narrow Reventazón valley, CR
  // Japan
  ['Miyagase',      35.557, 139.185,  null, null,  325, null],  // Nakatsu gorge, Kanagawa
  ['Yamba',         36.692, 138.828,  null, null,  340, null],  // Agatsuma gorge, Gunma (dam 2015)

  // ── MED A/P — Moderately irregular shoreline ──────────────────────────────
  // Spain
  ['El_Burguillo',  40.367,  -4.500,  null, null,  700, null],  // Alberche valley
  ['Boadella',      42.329,   2.831,  null, null,  600, null],  // Muga R.
  ['Puentes_Viejas',40.983,  -3.576,  null, null,  400, null],  // Lozoya chain
  ['Guajaraz',      39.675,  -4.107,  6046, null,  350, null],  // GDW 6046 (94 ha — try)
  // France
  ['Panneciere',    47.200,   3.883,  null, null,  480, null],  // Cure R., Morvan
  ['Sarrans',       44.818,   2.763,  null, null,  370, null],  // Truyère tributary
  // Italy
  ['Bilancino',     43.978,  11.202,  null, null,  500, null],  // Sieve R., Mugello
  ['Ampollino',     39.235,  16.553,  null, null,  360, null],  // Sila plateau
  ['Arvo',          39.300,  16.528,  null, null,  280, null],  // Sila
  ['Cecita',        39.350,  16.565,  null, null,  420, null],  // coord → lake center (GDW 16286 area=41ha → wrong polygon)
  // Morocco
  ['Oued_Makhazine',35.167,  -5.533, 37324, null,  660, null],  // GDW 37324 (62 ha, 19.7 km — uncertain; verify polygon)
  // India
  ['Karapuzha',     11.617,  76.033,  null, null,  260, null],  // Kerala Western Ghats
  // USA
  ['Saguaro',       33.655,-111.531,  null, null,  490, null],  // Salt R. canyon, AZ
  ['Canyon_Lake',   33.528,-111.427,  null, null,  240, null],  // Salt R. chain, AZ
  // South Africa
  ['Boegoeberg',   -29.026,  22.155, 25023, null,  297, null],  // GDW 25023 ✓ (2.97 km²)
  ['Woodstock',    -28.920,  29.203,  null, null,  590, null],  // Tugela headwaters
  ['Tzaneen',      -23.813,  30.145,  null, null,  400, null],  // Letaba R.
  // DROPPED: Chelmsford (GDW 2970 ha > 1000 ha), Nagle (wrong match 11 km, 1421 ha)
  // Australia
  ['Googong',      -35.440, 149.225,  null, null,  480, null],  // ACT, Queanbeyan R.
  ['Cardinia',     -37.935, 145.510,  5517, null,  944, null],  // GDW 5517 ✓ (9.44 km²)

  // ── HIGH A/P — Compact bowl / plain ───────────────────────────────────────
  // Spain — DROPPED: Riano (GDW 1557 ha), Ebro_Embalse (GDW 5438 ha), Aguilar (JRC max 1410 ha)
  // France
  ['Triouzoune',    45.516,   2.213,  null, null,  320, null],  // coord → lake center (GDW 15793 area=36ha → wrong polygon; was 45.520,2.265)
  ['Grandval',      45.018,   3.093,  null, null,  750, null],  // Truyère; compact Massif Central
  // USA — DROPPED: Eleven_Mile (GDW 1237 ha > 1000 ha)
  ['Deer_Creek',    40.406,-111.529,  null, null,  890, null],  // UT; compact Wasatch foothills
  ['East_Canyon',   40.930,-111.582,  null, null,  290, null],  // UT; compact bowl
  ['Pineview',      41.273,-111.839,  2611, null,  969, null],  // GDW 2611 ✓ (9.69 km²)
  ['Rockport',      40.766,-111.296,  null, null,  480, null],  // UT; compact Weber R.
  ['Antero',        38.980,-105.860,  null, null,  880, null],  // CO; flat South Park
  // DROPPED: Wadi_Dayqah (not in GDW), Demirkopru (GDW 3587 ha + 24.6 km wrong), Blue_Rock (wrong dam)
  // Iran
  ['Shaharchay',    37.640,  45.009,  6939, null,  350, null],  // GDW 6939 "Shahrchay" (7.43 km², 24.9 km — try)
  // South Africa — DROPPED: Nagle (wrong match)
  ['Welbedacht',   -29.870,  26.820,  null, null,  200, null],  // Caledon R.; small flat bowl
  // Argentina — DROPPED: La_Vina (wrong GDW match), Cruz_del_Eje (JRC max 1167 ha)
  // Italy
  ['Occhito',       41.534,  14.913,  4076, null,  746, null],  // GDW 4076 ✓ (7.46 km²)
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

  // Coordinate fallback: largest JRC polygon with centroid ≤10 km from point.
  var pt      = ee.Geometry.Point([lon, lat]);
  var vecs    = jrcMax.reduceToVectors({
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

// ── Otsu threshold (Donchyts/Markert between-class-variance formulation) ──────
// Returns the bucketMean that maximises between-class variance of a 1-band histogram.
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

// Per-scene VV water mask via Otsu. Histogram taken over a land-ringed buffer (bimodal),
// threshold applied as water = VV < T (smooth water has low backscatter).
// NOTE: on non-bimodal scenes (pool fills the buffer, or wind raises water backscatter
// toward land) Otsu returns a degenerate split — that failure is the signal we want to
// measure, so no clamp is applied. To guard, wrap T with .max(-25).min(-8) if needed.
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
function classifyImage(img, svm, lakePoly) {
  var aoi    = lakePoly.buffer(100);
  var water  = USE_OTSU
    ? computeOtsuWater(img, lakePoly).clip(aoi)
    : img.select(BANDS).classify(svm).eq(1).clip(aoi).rename('Water');

  // ── Tier 1-fast: cheap path — pixel-count area inside the pool polygon, skipping
  // fill / vectorisation / keep-polygon / dynamic perimeter (the dominant cost). ──
  if (FAST) {
    var areaFast_m2 = water.rename('Water').multiply(ee.Image.pixelArea())
      .reduceRegion({reducer: ee.Reducer.sum(), geometry: lakePoly,
                     scale: 10, maxPixels: 1e8, bestEffort: true}).get('Water');
    return img.addBands(water.rename('Water'))
      .set('_area_m2',      areaFast_m2)
      .set('_area_ha',      ee.Number(areaFast_m2).divide(1e4))
      .set('_ap_m_dynamic', -1);   // sentinel: not computed in fast mode
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

  // Dynamic A/P: perimeter of actual water extent at this acquisition date.
  // At high water, dendritic arms are flooded → complex perimeter → low A/P.
  // At low water, only trunk remains → simpler shape → higher A/P.
  var mergedGeom = keptPolys.union(1).geometry();
  var dynPerim_m = mergedGeom.perimeter(1);
  var dynAP_m    = ee.Number(area_m2).divide(ee.Number(dynPerim_m).max(1));

  return img.addBands([water, filled, cleaned])
    .set('_area_m2', area_m2)
    .set('_area_ha', ee.Number(area_m2).divide(1e4))
    .set('_ap_m_dynamic', dynAP_m);
}

// ── Main export loop ───────────────────────────────────────────────────────────
// Change BATCH_SLICE to the desired batch before running (see header comment).
var BATCH_SLICE = JRC_ONLY ? [0, 45] : [41, 45];

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

  // ── JRC monthly reference ────────────────────────────────────────────────
  var JRC_MONTHLY = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory');
  var aoiTotalArea_m2 = ee.Image(1).rename('total').clip(aoi)
    .multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 30, maxPixels: 1e8, bestEffort: true})
    .getNumber('total');

  var jrcWater = JRC_MONTHLY
    .filterDate('2015-01-01', '2021-12-31')
    .map(function(img) {
      var wc       = img.select('water');
      var observed = wc.gte(1).unmask(0).rename('obs');
      var water    = wc.eq(2).unmask(0).rename('wat');
      var stats    = observed.addBands(water).multiply(ee.Image.pixelArea())
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                       scale: 30, maxPixels: 1e8, bestEffort: true});
      var validFrac = ee.Number(stats.getNumber('obs')).divide(aoiTotalArea_m2);
      var jrcAreaHa = ee.Number(stats.getNumber('wat')).divide(1e4);
      return ee.Feature(null, {
        'date':        ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
        'jrc_area_ha': jrcAreaHa,
        'valid_frac':  validFrac,
      });
    })
    .filter(ee.Filter.gt('valid_frac', 0));

  if (EXPORT_JRC) {
    Export.table.toDrive({
      collection:     jrcWater,
      description:    'JRC_area_' + rName,
      folder:         CFG.drive_folder_jrc,
      fileNamePrefix: 'JRC_area_' + rName,
      fileFormat:     'CSV',
    });
  }

  if (!JRC_ONLY) {
    var svm = null;
    if (CLASSIFIER === 'SVM') {
    // ── Training composite (Tier 3 / SVM only) ─────────────────────────────
    var s1Composite = S1_GRD
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.eq('resolution_meters', 10))
      .filterBounds(trainClip)
      .filter(ee.Filter.calendarRange(CFG.train_year, CFG.train_year, 'year'))
      .select(BANDS).mosaic()
      .focal_mean(30, 'circle', 'meters').clip(trainClip);

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

    var trainedSamples = s1Composite.select(BANDS).sampleRegions({
      collection: waterSamples.merge(landSamples),
      properties: ['landcover'], scale: 30,
    }).filter(ee.Filter.inList('landcover', ee.List([1, 2])))
      .filter(ee.Filter.notNull(BANDS));

    svm = ee.Classifier.libsvm({kernelType: 'RBF', cost: 1, gamma: 0.01})
      .train({features: trainedSamples, classProperty: 'landcover', inputProperties: BANDS});
    }  // end if (CLASSIFIER === 'SVM') — VV_OTSU needs no training

    // ── S1 time series ─────────────────────────────────────────────────────
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
      return classifyImage(img, svm, lakePoly);
    }).map(function(img) {
      return ee.Feature(null, {
        'date':          ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
        'area_m2':       img.get('_area_m2'),
        'area_ha':       img.get('_area_ha'),
        'relOrbit':      img.get('relativeOrbitNumber_start'),
        'passDirection': img.get('orbitProperties_pass'),
        'ap_m':          ap_m,           // static: JRC max_extent polygon
        'ap_m_dynamic':  img.get('_ap_m_dynamic'),  // dynamic: water polygon at this date
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
