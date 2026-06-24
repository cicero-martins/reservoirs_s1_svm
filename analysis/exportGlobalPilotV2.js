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
 *   - Gap-fill: v226 fastDistanceTransform(30).lte(0.5)
 *   - Cleaning: filterBounds(lakePoly) → keep largest intersecting polygon
 *     (replaces centroid-inside which failed for elongated/valley reservoirs)
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
  s1_end:            '2026-06-30',
  jrc_occ_thresh:    95,
  jrc_occ_fallback:  80,
  train_year:        2023,
  clean_scale_m:     30,
  max_pixels:        1e9,
  keep_largest_only: false,
  land_ring_inner_m: 500,
  land_ring_outer_m: 2000,
  drive_folder:      'GEE_GlobalPilotV2',
};

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
  ['Zujar',          38.930,  -5.232, 1527, 10301,  9314, null],  // Csa, Extremadura ES
  ['Caia',           39.041,  -7.202, 1523, 10302,  1005, null],  // Csa, Alentejo PT

  // ── Europe — Temperate oceanic / alpine ───────────────────────────────────
  ['Eder',           51.195,   9.044, null, 11148,  1160, null],  // Cfb, Germany
  ['Forggen',        47.632,  10.743, null, 10341,  1460, null],  // Dfb, Bavaria DE (GDW wrong)

  // ── North America — US (5 biome-diverse picks) ───────────────────────────
  ['Elwell',         48.349,-111.328,  493, 12974,  6514, null],  // BSk, Montana
  ['Allegheny',      41.911, -78.939,  737, 12971,  4140, null],  // Dfb, Pennsylvania
  ['Hugo_Lake',      34.059, -95.414,  948, 10276,  4787, null],  // Cfa, Oklahoma
  ['Hubbard_Creek',  32.791, -98.999,  981, 10272,  4315, null],  // BSk, Texas
  ['Harlan_County',  40.057, -99.265,  775, 11108,  5001, null],  // Dwa, Nebraska

  // ── Africa ────────────────────────────────────────────────────────────────
  ['Umbuluzi',      -26.110,  32.222, 2050,  1007,  3603, null],  // Cwa, Mozambique
  ['Sterkfontein',  -28.411,  29.008, 2062, 11393,  6381, null],  // Cwb, South Africa

  // ── South Asia ────────────────────────────────────────────────────────────
  ['Vani_Vilasa',    13.837,  76.437, 1931, 10479,  3930, null],  // BSh, Karnataka IN

  // ── South America ─────────────────────────────────────────────────────────
  ['Paraibuna',     -23.370, -45.654, 1187, 11410,  1057, null],  // Cfa, SE Brazil
  ['Acude_Oros',     -6.244, -39.018, 1138, 10594,  6125, null],  // BSh, NE Brazil
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

// ── Helper: select best orbit (highest average pixel coverage of AOI) ─────────
// Scores every relative-orbit number by mean VV pixel count within aoi.
// Commissioning-era orbits (N < 20 images) are excluded before scoring:
// a single acquisition during S1A commissioning (Oct 2014) can have a marginally
// higher pixel count than operational orbits with 400+ images, causing it to win
// by <1 pixel — confirmed by diagnostic on Rosamarina (orbit 24, N=1, 574 px vs
// orbit 124, N=478, 573.87 px). Falls back to all orbits if none qualifies.
function selectBestOrbit(col, aoi) {
  var withOrbit = col.map(function(img) {
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
    var cov  = sub.map(function(img) {
      var count = img.reduceRegion({
        reducer: ee.Reducer.count(), geometry: aoi,
        scale: 100, maxPixels: 1e7, bestEffort: true,
      }).getNumber('VV');
      return img.set('_cov', count);
    }).aggregate_mean('_cov');
    return ee.Feature(null, {'orbit': o, 'pass': pass, 'n': n, 'cov': cov});
  }));
  // Require ≥20 images per orbit to exclude commissioning-phase one-offs
  var qualified  = scored.filter(ee.Filter.gte('n', 20));
  var candidates = ee.Algorithms.If(qualified.size().gt(0), qualified, scored);
  var best       = ee.FeatureCollection(candidates).sort('cov', false).first();
  return col.filter(ee.Filter.and(
    ee.Filter.eq('relativeOrbitNumber_start', best.getNumber('orbit')),
    ee.Filter.eq('orbitProperties_pass',      best.getString('pass'))
  ));
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

  // Keep the largest polygon that physically intersects lakePoly (filterBounds).
  // Falls back to the globally largest polygon if none intersects (shouldn't happen
  // for well-formed lake polygons, but avoids crashes on edge cases).
  // filterBounds replaces the old centroid-inside test, which failed for elongated
  // valley reservoirs where the main water body's centroid falls outside the JRC
  // max_extent polygon used as lakePoly.
  var intersecting = polys.filterBounds(lakePoly);
  var keptPolys    = ee.FeatureCollection(ee.Algorithms.If(
    intersecting.size().gt(0),
    ee.FeatureCollection([intersecting.map(function(f) {
      return f.set('_area', f.geometry().area({maxError: 1}));
    }).sort('_area', false).first()]),
    ee.FeatureCollection([polys.map(function(f) {
      return f.set('_area', f.geometry().area({maxError: 1}));
    }).sort('_area', false).first()])
  ));

  var keptMask = ee.Image(0).paint({featureCollection: keptPolys, color: 1});
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
// Run one batch at a time (GEE memory limit ~6-7 reservoirs per execution):
//   Batch 1 (Sicily+Iberia):  [0,  7]
//   Batch 2 (Iberia+Europe):  [7, 13]
//   Batch 3 (US West+East):   [13, 17]
//   Batch 4 (Africa+Asia+SA): [17, 23]
var BATCH_SLICE = [0, 7];

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

  // ── JRC optical reference area (mean annual water area 2015-2024) ─────────
  // Used as the observation series for KGE = f(SAR area, JRC area).
  // JRC monthly water v1.4: band 'water' (0=no obs, 1=no water, 2=water).
  var JRC_MONTHLY = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory');
  var jrcWater = JRC_MONTHLY
    .filterDate('2015-01-01', '2024-12-31')
    .filter(ee.Filter.calendarRange(1, 12, 'month'))
    .map(function(img) {
      var water = img.eq(2).rename('water');
      var area_ha = water.multiply(ee.Image.pixelArea()).reduceRegion({
        reducer: ee.Reducer.sum(), geometry: aoi,
        scale: 30, maxPixels: 1e8, bestEffort: true,
      }).getNumber('water');
      var dt = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
      return ee.Feature(null, {'date': dt, 'jrc_area_ha': ee.Number(area_ha).divide(1e4)});
    })
    .filter(ee.Filter.gt('jrc_area_ha', 0));

  Export.table.toDrive({
    collection:     jrcWater,
    description:    'JRC_area_' + rName,
    folder:         CFG.drive_folder,
    fileNamePrefix: 'JRC_area_' + rName,
    fileFormat:     'CSV',
  });

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

  // ── S1 time series — best orbit ────────────────────────────────────────────
  // Require both VV and VH; copy ALL properties so selectBestOrbit can read
  // relativeOrbitNumber_start and orbitProperties_pass
  var s1col = S1_GRD
    .filterBounds(aoi)
    .filterDate(CFG.s1_start, CFG.s1_end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(BANDS)
    .map(function(img) {
      // copyProperties(img) without a list skips system: properties in GEE.
      // Explicitly list system:time_start and the orbital properties needed by
      // selectBestOrbit and the areaSeries export.
      return img.focal_mean(30, 'circle', 'meters')
        .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                              'relativeOrbitNumber_start', 'angle']);
    });

  var bestCol = selectBestOrbit(s1col, aoi);

  // ── Classify and compute area per image ───────────────────────────────────
  var classified = bestCol.map(function(img) {
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

  print('Export queued: ' + rName + '  (~' + areaHaEst + ' ha, AP=' + ap_m.round() + ' m)');
});
