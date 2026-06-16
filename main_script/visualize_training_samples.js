// ============================================================
// VISUALIZE JRC AUTO-TRAINING SAMPLES
// Shows water/land sample locations for any pilot reservoir
// ============================================================

var hydroLakes = ee.FeatureCollection(
  'projects/ee-ciceromartinsjr/assets/HydroLAKES_pilot24'
);

// ── Change this to inspect a different reservoir ──────────────
var HYLAK_ID = 172818;   // Borbollon_ES
// var HYLAK_ID = 15647;    // Manikdoh
// var HYLAK_ID = 9794;     // Benito_Juarez_MX
// var HYLAK_ID = 15553;    // Panam
// var HYLAK_ID = 15840;    // Periyar
// var HYLAK_ID = 172148;   // San_Esteban_ES
// var HYLAK_ID = 14552;    // Gabriel_y_Galan_ES
// var HYLAK_ID = 14457;    // Ricobayo_ES
// var HYLAK_ID = 184942;   // Maroondah_AU
// var HYLAK_ID = 735;      // Winnibigoshish_US
// var HYLAK_ID = 738;      // Leech_US

// ── Load AOI ──────────────────────────────────────────────────
var feat = hydroLakes.filter(ee.Filter.eq('Hylak_id', HYLAK_ID)).first();
var aoi  = feat.geometry();

// ── JRC occurrence layer ──────────────────────────────────────
var occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');

// ── Water samples: occurrence ≥90% inside AOI ────────────────
var water90 = occ.gte(90).selfMask().sample({
  region: aoi, scale: 30, numPixels: 500, seed: 42, geometries: true
}).map(function(f) { return f.set('landcover', 1, 'label', 'water≥90%'); });

var water50 = occ.gte(50).selfMask().sample({
  region: aoi, scale: 30, numPixels: 500, seed: 42, geometries: true
}).map(function(f) { return f.set('landcover', 1, 'label', 'water≥50%'); });

var waterSamples = ee.FeatureCollection(
  ee.Algorithms.If(water90.size().gt(10), water90, water50)
);

// ── Land samples: ESA WorldCover (independent of JRC) ────────
// WorldCover classes: 10=Trees 20=Shrub 30=Grass 40=Cropland
//   50=Built 60=Bare  80=Water 90=Wetland 95=Mangrove
// Land classes = anything that is NOT water (80) or wetland (90/95)
var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var jrcMaxExtent = occ.gt(0);   // all pixels ever observed as water by JRC

// Exclude: permanent water (80), wetland (90), mangrove (95), JRC max extent
var landMask = worldcover.neq(80)
  .and(worldcover.neq(90))
  .and(worldcover.neq(95))
  .and(jrcMaxExtent.unmask(0).eq(0))   // also exclude anything JRC ever saw as water
  .selfMask();

var searchZone = aoi.buffer(2000);

var landSamples = landMask.sample({
  region: searchZone, scale: 10, numPixels: 500, seed: 42, geometries: true
}).map(function(f) { return f.set('landcover', 2, 'label', 'land_worldcover'); });

print('landSamples (WorldCover):', landSamples.size());

// ── Print counts ──────────────────────────────────────────────
print('Hylak_ID:', HYLAK_ID);
print('water90 size:', water90.size());
print('water50 size:', water50.size());
print('waterSamples used:', waterSamples.size());
print('landSamples:', landSamples.size());
print('AOI area (ha):', aoi.area(1).divide(10000));

// ── Map visualization ─────────────────────────────────────────
Map.centerObject(aoi, 12);

// ESA WorldCover (land cover classes)
Map.addLayer(
  worldcover,
  {min: 10, max: 100,
   palette: ['006400','ffbb22','ffff4c','f096ff','fa0000',
             'b4b4b4','f0f0f0','0064c8','0096a0','00cf75','fae6a0']},
  'ESA WorldCover', false
);

// JRC occurrence as background
Map.addLayer(
  occ.updateMask(occ.gt(0)),
  {min: 0, max: 100, palette: ['white','lightblue','blue']},
  'JRC Occurrence (%)', true
);

// S1 composite for context (recent 1 year)
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate('2023-01-01', '2024-01-01')
  .filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV')
  .median();
Map.addLayer(s1, {min: -20, max: 0}, 'S1 VV median (2023)', false);

// AOI boundary
Map.addLayer(
  ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(aoi)]), 1, 2),
  {palette: ['yellow']}, 'AOI boundary'
);

// JRC maximum extent (ever water)
Map.addLayer(
  jrcMaxExtent.selfMask(),
  {palette: ['cyan']}, 'JRC max extent (occ > 0)', false
);

// Search zone boundary (2 km buffer)
Map.addLayer(
  ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(searchZone)]), 1, 1),
  {palette: ['orange']}, 'Search zone (2 km buffer)', false
);

// Water samples
Map.addLayer(
  waterSamples,
  {color: '0000FF'}, 'Water samples (blue)'
);

// Land samples
Map.addLayer(
  landSamples,
  {color: 'FF0000'}, 'Land samples (red)'
);

// ── Legend panel ──────────────────────────────────────────────
var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
legend.add(ui.Label('Training samples', {fontWeight: 'bold'}));
[['Water samples', '#0000FF'], ['Land samples', '#FF0000'],
 ['AOI boundary', '#FFFF00'], ['Land buffer', '#FFA500']
].forEach(function(item) {
  var row = ui.Panel([
    ui.Label('', {backgroundColor: item[1], padding: '8px', margin: '0 6px 0 0'}),
    ui.Label(item[0])
  ], ui.Panel.Layout.flow('horizontal'));
  legend.add(row);
});
Map.add(legend);
