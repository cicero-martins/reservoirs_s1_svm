// ============================================================
// VISUALIZE TRAINING SAMPLES — JRC max_extent AOI pipeline
//
// Layers shown:
//   1. HydroLAKES polygon (search anchor, NOT the AOI)
//   2. JRC max_extent polygon (lakePoly → AOI source)
//   3. aoi = lakePoly.buffer(100) (SAR/JRC detection zone)
//   4. Land ring: aoi → 2 km (land sample zone)
//   5. Water samples: JRC occurrence ≥90% inside lakePoly (blue)
//   6. Land samples: WorldCover non-water in land ring (red)
//   7. JRC occurrence background
//   8. S1 VV median 2023 (for context)
// ============================================================

// ── Sicily validation reservoirs (HydroLAKES_sicily4) ────────
var ASSET = 'projects/ee-ciceromartinsjr/assets/HydroLAKES_sicily4';
var HYLAK_ID = 173633;   // Rosamarina  ← most interesting (polygon mismatch case)
// var HYLAK_ID = 1369046;  // Ancipa
// var HYLAK_ID = 173610;   // Poma
// var HYLAK_ID = 173729;   // Pozzillo

// ── Global pilot reservoirs (HydroLAKES_pilot24) ─────────────
// var ASSET    = 'projects/ee-ciceromartinsjr/assets/HydroLAKES_pilot24';
// var HYLAK_ID = 172818;   // Borbollon_ES
// var HYLAK_ID = 15647;    // Manikdoh
// var HYLAK_ID = 172148;   // San_Esteban_ES
// var HYLAK_ID = 14552;    // Gabriel_y_Galan_ES

var SEARCH_BUFFER = 2000;   // metres — search radius around HydroLAKES centroid
var AOI_BUFFER    = 100;    // metres — margin over JRC max_extent polygon

// ── Load HydroLAKES anchor ────────────────────────────────────
var hydroLakes = ee.FeatureCollection(ASSET);
var hydroPoly  = hydroLakes
  .filter(ee.Filter.eq('Hylak_id', HYLAK_ID)).first().geometry();
var searchArea = hydroPoly.buffer(SEARCH_BUFFER);

// ── Derive JRC max_extent polygon ────────────────────────────
var occ    = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
var jrcMax = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('max_extent').eq(1).selfMask();

var waterVecs = jrcMax.reduceToVectors({
  geometry:       searchArea,
  scale:          30,
  maxPixels:      1e9,
  bestEffort:     true,
  geometryType:   'polygon',
  eightConnected: true,
});

// Largest contiguous water body = main reservoir
var lakePoly = waterVecs.sort('count', false).first().geometry();
var aoi      = lakePoly.buffer(AOI_BUFFER);
var landRing = lakePoly.buffer(2000).difference(aoi);

// ── A/P diagnostics ───────────────────────────────────────────
var lake_area_m2 = lakePoly.area(1);
var lake_perim_m = lakePoly.perimeter(1);
var ap_jrc       = lake_area_m2.divide(lake_perim_m);

var hydro_area_m2 = hydroPoly.area(1);
var hydro_perim_m = hydroPoly.perimeter(1);
var ap_hydro      = hydro_area_m2.divide(hydro_perim_m);

print('=== AOI comparison ===');
print('HydroLAKES  area (ha):', hydro_area_m2.divide(10000).round());
print('HydroLAKES  perim (m):', hydro_perim_m.round());
print('HydroLAKES  A/P  (m):', ap_hydro.round());
print('JRC max_ext area (ha):', lake_area_m2.divide(10000).round());
print('JRC max_ext perim (m):', lake_perim_m.round());
print('JRC max_ext A/P  (m):', ap_jrc.round());
print('Detection aoi area (ha):', aoi.area(1).divide(10000).round());

// ── Training samples ──────────────────────────────────────────
// Water: JRC ≥90% inside lakePoly core
var water90 = occ.gte(90).selfMask().sample({
  region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true
}).map(function(f) { return f.set('landcover', 1, 'label', 'water≥90%'); });

var water50 = occ.gte(50).selfMask().sample({
  region: lakePoly, scale: 30, numPixels: 500, seed: 42, geometries: true
}).map(function(f) { return f.set('landcover', 1, 'label', 'water≥50%'); });

var waterSamples = ee.FeatureCollection(
  ee.Algorithms.If(water90.size().gt(10), water90, water50)
);

// Land: WorldCover non-water in land ring (aoi → 2 km)
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var landMask = wc.neq(80).and(wc.neq(90)).and(wc.neq(95))
  .and(occ.unmask(0).eq(0)).selfMask();
var landSamples = landMask.sample({
  region: landRing, scale: 10, numPixels: 500, seed: 42, geometries: true
}).map(function(f) { return f.set('landcover', 2, 'label', 'land'); });

print('=== Training samples ===');
print('water90 size:', water90.size());
print('water50 size:', water50.size());
print('waterSamples used:', waterSamples.size());
print('landSamples:', landSamples.size());

// ── Map setup ─────────────────────────────────────────────────
Map.centerObject(aoi, 13);
Map.setOptions('SATELLITE');

// JRC occurrence (background)
Map.addLayer(
  occ.updateMask(occ.gt(0)),
  {min: 0, max: 100, palette: ['white', 'lightblue', '0044ff']},
  'JRC Occurrence (%)', true
);

// JRC max_extent raster (the source used for lakePoly)
Map.addLayer(
  jrcMax.clip(searchArea),
  {palette: ['00ccff']}, 'JRC max_extent pixels (cyan)', false
);

// S1 VV median 2023 (context for SAR appearance)
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate('2023-01-01', '2024-01-01')
  .filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV').median();
Map.addLayer(s1, {min: -20, max: 0}, 'S1 VV median 2023', false);

// ESA WorldCover
Map.addLayer(
  wc,
  {min: 10, max: 100,
   palette: ['006400','ffbb22','ffff4c','f096ff','fa0000',
             'b4b4b4','f0f0f0','0064c8','0096a0','00cf75','fae6a0']},
  'ESA WorldCover', false
);

// ── Polygon boundaries ────────────────────────────────────────
// Helper: paint outline only
function outline(geom, color, width) {
  return ee.Image().byte()
    .paint(ee.FeatureCollection([ee.Feature(geom)]), 1, width)
    .visualize({palette: [color]});
}

// Land ring (sample zone) — orange fill at low opacity
Map.addLayer(
  ee.Image(1).clip(landRing).visualize({palette: ['ff8800'], opacity: 0.15}),
  {}, 'Land sample ring (fill)', true
);
Map.addLayer(outline(landRing.bounds(), 'ff8800', 1), {}, 'Land ring outer bound', false);

// Search area (2 km around HydroLAKES)
Map.addLayer(outline(searchArea, 'ffaa00', 1), {}, 'Search area (HydroLAKES +2km)', false);

// HydroLAKES polygon — dashed yellow
Map.addLayer(outline(hydroPoly, 'ffff00', 2), {}, 'HydroLAKES polygon (yellow)');

// JRC max_extent polygon (lakePoly) — solid white
Map.addLayer(outline(lakePoly, 'ffffff', 2), {}, 'JRC max_extent polygon / lakePoly (white)');

// AOI = lakePoly + 100m buffer — dashed green
Map.addLayer(outline(aoi, '00ff44', 1), {}, 'AOI = lakePoly.buffer(100) (green)');

// ── Training samples ──────────────────────────────────────────
Map.addLayer(landSamples,  {color: 'ff2222'}, 'Land samples (red)');
Map.addLayer(waterSamples, {color: '2222ff'}, 'Water samples (blue)');

// ── Legend ────────────────────────────────────────────────────
var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px 12px', width: '230px'}});
legend.add(ui.Label('Polygons & samples', {fontWeight: 'bold', fontSize: '13px'}));

[
  ['HydroLAKES polygon',         '#ffff00'],
  ['JRC max_extent (lakePoly)',   '#ffffff'],
  ['AOI = lakePoly+100m',        '#00ff44'],
  ['Land sample ring',           '#ff8800'],
  ['Water samples (JRC ≥90%)',   '#2222ff'],
  ['Land samples (WorldCover)',  '#ff2222'],
].forEach(function(item) {
  legend.add(ui.Panel([
    ui.Label('■', {color: item[1], fontSize: '18px', margin: '0 6px 0 0'}),
    ui.Label(item[0], {fontSize: '11px', margin: '4px 0'})
  ], ui.Panel.Layout.flow('horizontal')));
});
Map.add(legend);
