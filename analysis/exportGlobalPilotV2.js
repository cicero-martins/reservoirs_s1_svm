/**
 * exportGlobalPilotV2.js
 *
 * Batch GEE export of SAR water-area time series for the global pilot reservoir set.
 * Pipeline mirrors gee_reservoir_monitor_app.js exactly:
 *   - BANDS = ['VV','VH']
 *   - Training composite: S1_GRD (dB) + resolution_meters==10 + per-AOI mosaic()
 *     + focal_mean(30) — NOT S1_GRD_FLOAT, NOT median()
 *   - Water samples: JRC occ ≥95% (→ ≥80% fallback if <10 pts), landcover=1, 500 pts
 *   - Land samples: WorldCover neq(80,90,95) + occ==0 in annulus, landcover=2, 500 pts
 *   - SVM: libsvm(RBF, cost=1, gamma=0.01), classProperty='landcover'
 *   - Filter: notNull(BANDS) + inList('landcover',[1,2]) before training
 *   - Classification: .classify(svm).eq(1) → binary mask (1=water, 0=land)
 *   - Orbit: highest mean incidence-angle orbit achieving >=90% AOI coverage
 *     (app v226 PrioritizeDescendingAngleBins); per-scene >=90% gate (fallback >=50%).
 *     Runs on RAW imagery (keeps 'angle' band); preprocess applied afterwards.
 *   - Gap-fill: v226 fastDistanceTransform(30).lte(0.5)
 *   - Cleaning: keep ALL polygons with centroid inside lakePoly (app v226
 *     keep_largest_only=false); fallback to largest polygon if none inside
 *   - Area: WaterCleaned * pixelArea → sum
 *
 * Also exports per reservoir:
 *   - ap_m: area-to-perimeter ratio of JRC max_extent polygon (m)
 *   - jrc_area_ha: mean annual JRC water area 2015-2024 (optical reference for KGE)
 *
 * Output per reservoir: CSV in Google Drive (folder GEE_GlobalPilotV2)
 *   Columns: date, area_m2, area_ha, relOrbit, passDirection
 *
 * Paste into GEE Code Editor and click Run.
 */

// ── Configuration ────────────────────────────────────────────────────────────
var CFG = {
  s1_start:          '2014-10-01',
  s1_end:            '2021-12-31',
  jrc_occ_thresh:    95,
  jrc_occ_fallback:  80,
  train_year:        2023,
  clean_scale_m:     30,
  max_pixels:        1e9,
  keep_largest_only: false,
  land_ring_inner_m: 500,
  land_ring_outer_m: 2000,
  drive_folder:      'GEE_GlobalPilotV2',
  drive_folder_jrc:  'GEE_GlobalPilotV2_JRC',  // corrected JRC exports with valid_frac
  composite_window_days: 6,  // ±6 days mosaic to fill partial swath-edge coverage gaps
  // Per-image AOI coverage gate — mirrors app v226 selectBestOrbit (§7).
  // Images covering < coverage_strict_pct of the AOI classify only the imaged
  // sliver of the reservoir → spurious low-area dips (e.g. Harlan County orbit 34,
  // 2020). The app excludes these entirely. Strict ≥90%; fallback ≥min_coverage_pct.
  coverage_strict_pct: 90,
  min_coverage_pct:    50,
};

// Set true to export only JRC reference series (skips SVM training + S1 classification).
// Use this to re-export corrected JRC with valid_frac without re-running SAR pipeline.
var JRC_ONLY = false;

// ── Pilot reservoir list (pilot_v2.csv, 2026-06-22) ─────────────────────────
// Each entry: [name, lat, lon, gdw_id, dahiti_id, area_ha, hylak_id]
//   gdw_id   — GDW_ID in GDW_RESERVOIRS_V1_0; null → use coordinate fallback
//   hylak_id — Hylak_id in HydroLAKES_sicily4 (Sicily only); null for non-Sicily
//              App uses ee.Filter.eq('Hylak_id', numeric_id) — mirrored exactly here.
//   Note: Forggen gdw_id set to null (GDW match 15667 was wrong polygon, ~1 km²)
var PILOT_RESERVOIRS = [
  // ── Italy / Sicily — Hylak_id from HydroLAKES_sicily4 asset ─────────────
  ['Ancipa',         37.887,  14.565, null,  null,    88, 1369046],  // Hylak_id confirmed; max SAR area ~88 ha (Schwatke 2022)
  ['Poma',           37.994,  13.090, null, 42134,   620,  173610],
  ['Pozzillo',       37.783,  14.635, null,  null,   930,  173729],
  ['Rosamarina',     37.944,  13.640, null, 42122,   440,  173633],
  // Garcia: not in sicily4 asset (Hylak_id=null) → coord fallback; DAHITI 42123
  ['Garcia',         37.799,  13.119, null, 42123,   400, null],

  // ── Europe — Mediterranean ────────────────────────────────────────────────
  ['Yesa',           42.606,  -1.115, 1423, 10297,  1554, null],  // Csa, Aragon ES
  ['Puente_Nuevo',   38.127,  -4.977, 1550, 10304,  1209, null],  // Csa, Andalusia ES
  ['Alcantara',      39.764,  -6.688, 1505, 10310,  4532, null],  // Csa, Extremadura ES
  ['Caia',           39.041,  -7.202, 1523, 10302,  1005, null],  // Csa, Alentejo PT
  ['Kerkini',        41.198,  23.158, null,   null,   800, null],  // Csa, N Greece — managed floodplain reservoir; no DAHITI

  // ── Europe — Temperate oceanic / alpine ───────────────────────────────────
  ['Eder',           51.195,   9.044, 8683, 11148,  1160, null],  // Cfb, Germany
  ['Forggen',        47.632,  10.743, null, 10341,  1460, null],  // Dfb, Bavaria DE (GDW wrong)

  // ── North America — US ────────────────────────────────────────────────────
  ['Caballo',        32.930,-107.295, null,   null,   800, null],  // BSk, New Mexico — Rio Grande; no DAHITI
  ['Curwensville',   40.968, -78.519, null,   null,   520, null],  // Dfb, Pennsylvania — Army Corps; no DAHITI
  ['Hugo_Lake',      34.059, -95.414,  948, 10276,  4787, null],  // Cfa, Oklahoma
  ['Hubbard_Creek',  32.791, -98.999,  981, 10272,  4315, null],  // BSk, Texas
  ['Harlan_County',  40.057, -99.265,  775, 11108,  5001, null],  // Dwa, Nebraska

  // ── Africa ────────────────────────────────────────────────────────────────
  ['Umbuluzi',      -26.110,  32.222, 2050,  1007,  3603, null],  // Cwa, Mozambique
  ['Erfenis',       -28.497,  26.820, null,   null,   700, null],  // BSh, Free State ZA — no DAHITI

  // ── South Asia ────────────────────────────────────────────────────────────
  ['Vani_Vilasa',    13.837,  76.437, 1931, 10479,  3930, null],  // BSh, Karnataka IN

  // ── South America ─────────────────────────────────────────────────────────
  ['Paraibuna',     -23.370, -45.654, 1187, 11410,  1057, null],  // Cfa, SE Brazil
  // Acude_Oros (-6.244, -39.018, BSh NE Brazil, 6125 ha) removed — too large, AOI misses parts
  ['Contas',        -13.845, -40.329, 1152, 10592,  7485, null],  // Aw,  Bahia Brazil
];  // 7th column: hylak_id for Sicily, null for all others

// ── Datasets ─────────────────────────────────────────────────────────────────
var BANDS   = ['VV', 'VH'];  // both bands required — matches app pipeline exactly
var S1_GRD  = ee.ImageCollection('COPERNICUS/S1_GRD');  // dB scale for both training AND classification
var JRC_GSW = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var JRC_OCC = JRC_GSW.select('occurrence');
var WC      = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var GDW            = ee.FeatureCollection('projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0');
var HYDROLAKES_SIC = ee.FeatureCollection('projects/ee-ciceromartinsjr/assets/HydroLAKES_sicily4');

// ── Helper: get JRC max_extent polygon ───────────────────────────────────────
// Mirrors the AOI derivation in gee_reservoir_monitor_app.js (§4 + §17).
// Priority:
//   1. hylakId != null → HydroLAKES_sicily4, ee.Filter.eq('Hylak_id', hylakId)
//      Exact mirror of app: HYDROLAKES.filter(ee.Filter.eq(DS.id_field, numeric_id))
//   2. gdwId != null   → GDW_RESERVOIRS_V1_0, ee.Filter.eq('GDW_ID', gdwId)
//      Exact mirror of app CFG.dataset='global' mode
//   3. Both null       → coordinate fallback (Eder, Forggen — no reliable polygon)
function getLakePoly(lat, lon, gdwId, hylakId) {
  var jrcMax = JRC_GSW.select('max_extent').eq(1).selfMask();

  // Exact copy of jrcMaxExtentPoly() from app (§4, line 147–163)
  function jrcMaxExtentPoly(hydroGeom) {
    var searchArea = hydroGeom.buffer(2000);   // CFG.jrc_search_buffer_m = 2000
    var waterVecs  = jrcMax.reduceToVectors({
      geometry: searchArea, scale: 30, maxPixels: 1e9,
      bestEffort: true, geometryType: 'polygon',
      eightConnected: true, tileScale: 4,
    });
    return waterVecs.sort('count', false).first().geometry();
  }

  if (hylakId !== null) {
    // Sicily: mirror of app onReservoirSelected → HYDROLAKES.filter(eq('Hylak_id', id))
    var hydroGeom = HYDROLAKES_SIC
      .filter(ee.Filter.eq('Hylak_id', hylakId)).first().geometry();
    return jrcMaxExtentPoly(hydroGeom);
  }

  if (gdwId !== null) {
    // Global: mirror of app CFG.dataset='global' → HYDROLAKES.filter(eq('GDW_ID', id))
    var hydroGeom = GDW
      .filter(ee.Filter.eq('GDW_ID', gdwId)).first().geometry();
    return jrcMaxExtentPoly(hydroGeom);
  }

  // Coordinate fallback for Eder / Forggen (no reliable polygon in GDW or Sicily asset).
  // Finds the largest JRC max_extent polygon with centroid ≤10 km from the point.
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

// ── Helper: select best orbit (highest incidence angle + AOI coverage gate) ───
// Faithful port of app v226 selectBestOrbit / PrioritizeDescendingAngleBins (§7).
// The app does NOT pick the orbit with the most pixels — it prioritises the
// HIGHEST mean incidence angle that achieves ≥90% AOI coverage. Higher (far-range)
// incidence angles give a more specular (darker) water return → larger water/land
// backscatter contrast → classification that is robust to wind roughening of the
// water surface. The old pixel-count criterion picked a different orbit than the
// app for swath-edge reservoirs (e.g. Harlan County: export chose orbit 34 with
// episodic wind-driven misclassification dips; the app chose a higher-angle orbit
// giving a stable ~5000 ha series). Confirmed via acquisition-date offset analysis.
//
// IMPORTANT: must run on RAW imagery that still carries the 'angle' band (before
// .select(BANDS) drops it).
function selectBestOrbit(s1Raw, aoi) {
  var aoiArea = aoi.area(1);

  // Per-image mean incidence angle + AOI coverage %
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

  // Score each relative orbit: mean incidence angle, image count, #scenes ≥90%.
  // (Within one relative orbit the incidence angle over the AOI is ~constant, so
  //  ranking orbits by mean angle reproduces the app's angle-bin priority.)
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

  // Prefer orbits with ≥20 images AND ≥1 scene at ≥90% coverage; among those,
  // pick the HIGHEST mean incidence angle. Progressive fallbacks for sparse cases.
  var qualified = scored.filter(ee.Filter.gte('n', 20))
                        .filter(ee.Filter.gt('n_covered', 0));
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

  // Within the chosen orbit, keep ≥90%-coverage scenes (fallback ≥50%).
  var strict = bestCol.filter(ee.Filter.gte('percentCovered', CFG.coverage_strict_pct));
  return ee.ImageCollection(ee.Algorithms.If(
    strict.size().gt(0), strict,
    bestCol.filter(ee.Filter.gte('percentCovered', CFG.min_coverage_pct))
  ));
}

// ── Helper: fill partial swath-edge coverage gaps ────────────────────────────
// For reservoirs at Sentinel-1 swath edges, individual acquisitions may cover
// only part of the AOI. This function mosaics images within ±windowDays of each
// acquisition date so that nearby images fill pixels missing from the primary pass.
// Mirrors composite_window_days logic in gee_reservoir_monitor_app.js §7b.
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

// ── Helper: classify single image and compute water area ──────────────────────
function classifyImage(img, svm, lakePoly) {
  var aoi = lakePoly.buffer(100);

  // SVM → binary water mask: .eq(1) converts class labels (1=water, 2=land) to 0/1
  var water  = img.select(BANDS).classify(svm).eq(1).clip(aoi).rename('Water');

  // Gap-fill: pixels within 0.5 px of a water pixel are filled (closes small holes)
  var mask   = water.unmask(0).clip(aoi);
  var dist   = mask.fastDistanceTransform(30).clip(aoi);
  var filled = dist.lte(0.5).updateMask(dist.lte(0.5))
                   .where(mask, 1).rename('WaterFilled');

  // Centroid-inside cleaning: keep only water polygons whose centroid falls inside
  // the JRC max_extent polygon (excludes spurious bodies outside the reservoir)
  var polys = filled.reduceToVectors({
    geometryType: 'polygon',
    reducer:      ee.Reducer.countEvery(),
    scale:        CFG.clean_scale_m,
    maxPixels:    CFG.max_pixels,
    bestEffort:   true,
    tileScale:    4,
  });

  // Keep ALL polygons whose centroid falls inside the JRC max_extent footprint
  // (lakePoly). Faithful port of app v226 classifyCollection (keep_largest_only =
  // false): captures dendritic reservoir arms while excluding external spurious
  // bodies (irrigated fields, ponds) whose centroids lie outside the reservoir.
  // Safety fallback: if no centroid lands inside (extreme low-water state), use
  // the single largest polygon to avoid an empty mask.
  var withFlag = polys.map(function(f) {
    var inside = lakePoly.contains(f.geometry().centroid({maxError: 1}),
                                   ee.ErrorMargin(1));
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

  var keptMask = ee.Image().paint({featureCollection: keptPolys, color: 1})
                   .rename('KeptRegionMask');
  var cleaned  = filled.updateMask(keptMask).rename('WaterCleaned');

  // Area: multiply by actual pixel area (m²/px at the native 10 m resolution)
  var area_m2 = cleaned.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer:   ee.Reducer.sum(),
    geometry:  aoi,
    scale:     10, maxPixels: 1e8, bestEffort: true,
  }).get('WaterCleaned');

  return img.addBands([water, filled, cleaned])
    .set('_area_m2', area_m2)
    .set('_area_ha', ee.Number(area_m2).divide(1e4));
}

// ── Main export loop ──────────────────────────────────────────────────────────
// GEE Code Editor memory limit: run at most 6–7 reservoirs per script execution.
// Change BATCH_SLICE each time you run:
//   Batch 1: [0, 7]   → Sicily (4) + Iberia/PT (3)
//   Batch 2: [7, 14]  → Iberia/PT (2) + Eder + Forggen + US West+East (2)
//   Batch 3: [14, 22] → US Central (3) + Africa (2) + India (1) + Brazil (3)
// JRC_ONLY = true  → all 22 reservoirs in one run (no SVM, no S1 collection)
// JRC_ONLY = false → run SAR batches of ~5-7 reservoirs (GEE memory limit):
//   Batch 1 (Sicily + Iberia W):  [0,  7]  Ancipa…Puente_Nuevo
//   Batch 2 (Iberia E + Europe):  [7, 12]  Alcantara…Forggen
//   Batch 3 (US):                 [12, 17]  Caballo…Harlan_County
//   Batch 4 (Africa + Asia + SA): [17, 22]  Umbuluzi…Contas
var BATCH_SLICE = JRC_ONLY ? [0, 22] : [0, 7];

PILOT_RESERVOIRS.slice(BATCH_SLICE[0], BATCH_SLICE[1]).forEach(function(res) {
  var rName     = res[0];
  var lat       = res[1];
  var lon       = res[2];
  var gdwId     = res[3];   // GDW_ID or null
  var areaHaEst = res[5];
  var hylakId   = res[6];   // Hylak_id (Sicily only) or null

  var lakePoly = getLakePoly(lat, lon, gdwId, hylakId);
  var aoi      = lakePoly.buffer(100);
  var trainClip = aoi.buffer(CFG.land_ring_outer_m);  // covers lake + land annulus

  // ── A/P ratio of JRC max_extent polygon ──────────────────────────────────
  // area (m²) / perimeter (m) = length-scale index of shoreline compactness
  var lakeArea_m2  = lakePoly.area(1);
  var lakePerim_m  = lakePoly.perimeter(1);
  var ap_m         = ee.Number(lakeArea_m2).divide(lakePerim_m);

  // ── JRC optical reference area (monthly, 2015-2024) ──────────────────────
  // JRC/GSW1_4/MonthlyHistory band 'water': 0=no observation, 1=land, 2=water.
  // valid_frac = fraction of AOI pixels that were actually observed that month
  //   (value 1 or 2). Low valid_frac → cloud contamination → area underestimate.
  // Filter in Python: use only months where valid_frac >= 0.8 (or desired threshold).
  var JRC_MONTHLY = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory');

  // Total AOI pixel area (constant for all months — compute once per reservoir)
  var aoiTotalArea_m2 = ee.Image(1).rename('total').clip(aoi)
    .multiply(ee.Image.pixelArea())
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                   scale: 30, maxPixels: 1e8, bestEffort: true})
    .getNumber('total');

  var jrcWater = JRC_MONTHLY
    .filterDate('2015-01-01', '2021-12-31')
    .map(function(img) {
      var wc = img.select('water');
      // Observed pixels: value 1 (land) or 2 (water) — not 0 (no data)
      var observed = wc.gte(1).unmask(0).rename('obs');
      var water    = wc.eq(2).unmask(0).rename('wat');

      var stats = observed.addBands(water)
        .multiply(ee.Image.pixelArea())
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                       scale: 30, maxPixels: 1e8, bestEffort: true});

      var obsArea_m2 = stats.getNumber('obs');
      var watArea_m2 = stats.getNumber('wat');
      var validFrac  = ee.Number(obsArea_m2).divide(aoiTotalArea_m2);
      var jrcAreaHa  = ee.Number(watArea_m2).divide(1e4);
      var dt = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');

      return ee.Feature(null, {
        'date':         dt,
        'jrc_area_ha':  jrcAreaHa,
        'valid_frac':   validFrac,
      });
    })
    // Keep only months with any observation (valid_frac > 0); area may be 0 if reservoir empty
    .filter(ee.Filter.gt('valid_frac', 0));

  Export.table.toDrive({
    collection:     jrcWater,
    description:    'JRC_area_' + rName,
    folder:         CFG.drive_folder_jrc,
    fileNamePrefix: 'JRC_area_' + rName,
    fileFormat:     'CSV',
  });

  if (!JRC_ONLY) {
  // ── Training composite (per-AOI) ──────────────────────────────────────────
  // S1_GRD (dB), IW, 10 m resolution, 2023 annual mosaic → focal_mean → clip
  // mosaic() must be called after filterBounds() for a per-AOI composite
  var s1Composite = S1_GRD
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filterBounds(trainClip)
    .filter(ee.Filter.calendarRange(CFG.train_year, CFG.train_year, 'year'))
    .select(BANDS)
    .mosaic()
    .focal_mean(30, 'circle', 'meters')
    .clip(trainClip);

  // ── Water training samples ─────────────────────────────────────────────────
  // Strict: JRC occ ≥95%; fallback ≥80% if <10 strict points
  var waterStrict = JRC_OCC.gte(CFG.jrc_occ_thresh).selfMask().sample({
    region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true,
  }).map(function(f) { return f.set('landcover', 1); });

  var waterFallback = JRC_OCC.gte(CFG.jrc_occ_fallback).selfMask().sample({
    region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true,
  }).map(function(f) { return f.set('landcover', 1); });

  var waterSamples = ee.FeatureCollection(
    ee.Algorithms.If(waterStrict.size().gt(10), waterStrict, waterFallback)
  );

  // ── Land training samples ──────────────────────────────────────────────────
  // Annulus 500–2000 m around the lake; exclude WorldCover water (80) + wetland
  // (90) + mangrove (95) AND any pixel with JRC occ > 0
  var landRing = lakePoly.buffer(CFG.land_ring_outer_m)
                   .difference(lakePoly.buffer(CFG.land_ring_inner_m));
  var landMask = WC.neq(80).and(WC.neq(90)).and(WC.neq(95))
    .and(JRC_OCC.unmask(0).eq(0)).selfMask();
  var landSamples = landMask.sample({
    region: landRing, scale: 30, numPixels: 500, seed: 42, geometries: true,
  }).map(function(f) { return f.set('landcover', 2); });

  var trainingFC = waterSamples.merge(landSamples);

  // ── Train SVM ─────────────────────────────────────────────────────────────
  var trainedSamples = s1Composite.select(BANDS).sampleRegions({
    collection: trainingFC,
    properties: ['landcover'],
    scale:      30,
  }).filter(ee.Filter.inList('landcover', ee.List([1, 2])))
    .filter(ee.Filter.notNull(BANDS));

  var svm = ee.Classifier.libsvm({
    kernelType: 'RBF',
    cost:        1,
    gamma:       0.01,
  }).train({
    features:        trainedSamples,
    classProperty:   'landcover',
    inputProperties: BANDS,
  });

  // ── S1 time series — orbit selection on RAW imagery ────────────────────────
  // Keep ALL bands (esp. 'angle') so selectBestOrbit can rank by incidence angle.
  // Do NOT select(BANDS) or focal_mean yet — that happens after orbit selection,
  // exactly like the app (s1Raw → selectBestOrbit → preprocessS1).
  var s1Raw = S1_GRD
    .filterBounds(aoi)
    .filterDate(CFG.s1_start, CFG.s1_end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  // Orbit auto-selection: highest incidence angle achieving ≥90% AOI coverage.
  var bestCol = selectBestOrbit(s1Raw, aoi);

  // Preprocess AFTER orbit selection (mirrors app preprocessS1): speckle filter,
  // select VV+VH, tag date. copyProperties without a list skips system: props, so
  // list system:time_start + orbital props explicitly.
  var s1Proc = bestCol.map(function(img) {
    return img.select(BANDS).focal_mean(30, 'circle', 'meters')
      .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                            'relativeOrbitNumber_start'])
      .set('date', img.date().format('YYYY-MM-dd'));  // required by fillCoverageGaps
  });

  // ── Fill partial swath-edge coverage gaps ─────────────────────────────────
  // After orbit selection, all images share the same orbit/pass. Store those
  // values before mosaic compositing (mosaic() drops non-system properties).
  var bestOrbitNum  = ee.Number(bestCol.first().get('relativeOrbitNumber_start'));
  var bestOrbitPass = ee.String(bestCol.first().get('orbitProperties_pass'));
  var filledCol = fillCoverageGaps(s1Proc, CFG.composite_window_days)
    .map(function(img) {
      return img.set('relativeOrbitNumber_start', bestOrbitNum)
                .set('orbitProperties_pass',      bestOrbitPass);
    });

  // ── Classify and compute area per image ───────────────────────────────────
  var classified = filledCol.map(function(img) {
    return classifyImage(img, svm, lakePoly);
  });

  // ── Reduce to feature collection for export ────────────────────────────────
  var areaSeries = classified.map(function(img) {
    var dt = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    return ee.Feature(null, {
      'date':          dt,
      'area_m2':       img.get('_area_m2'),
      'area_ha':       img.get('_area_ha'),
      'relOrbit':      img.get('relativeOrbitNumber_start'),
      'passDirection': img.get('orbitProperties_pass'),
      'ap_m':          ap_m,   // A/P ratio — same value repeated per row for easy join
    });
  });

  areaSeries = areaSeries.filter(ee.Filter.gt('area_ha', 0));

  Export.table.toDrive({
    collection:     areaSeries,
    description:    'SAR_area_' + rName,
    folder:         CFG.drive_folder,
    fileNamePrefix: 'SAR_area_' + rName,
    fileFormat:     'CSV',
  });
  } // end if (!JRC_ONLY)

  print('Export queued: ' + rName + '  (~' + areaHaEst + ' ha)  AP (m) =', ap_m.round());
});
