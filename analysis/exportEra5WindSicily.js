/**
 * exportEra5WindSicily.js
 *
 * ERA5 10 m wind at every Sentinel-1 overpass for the 4 PlanetScope-validated
 * Sicilian reservoirs, over the PlanetScope window (2024–2025). Companion to
 * exportEra5Wind.js (the 34 global reservoirs) — same sampler, Sicily list + dates.
 *
 * Feeds the Sicily wind analyses (wind divergence; ΔKGE-vs-wind context — note
 * ΔKGE-vs-wind is only N=4 reservoirs, not a meaningful correlation).
 *
 * Output: CSV per reservoir in Drive folder GEE_SicilyPlanet_Era5Wind, columns:
 *   date, datetime_utc, wind_ms, wind_dir_deg, relOrbit, pass
 * Hourly ERA5 (S1 ~06 h/18 h, diurnal wind). Join to SAR_area_<name>.csv on `date`.
 */

// ── Config ────────────────────────────────────────────────────────────────────
var CFG = {
  s1_start:     '2024-01-01',          // PlanetScope window (2024-05 → 2025-05) + margin
  s1_end:       '2025-12-31',
  drive_folder: 'GEE_SicilyPlanet_Era5Wind',
  era5_scale_m: 27830,                 // ERA5 native ~0.25°
};

var ERA5  = ee.ImageCollection('ECMWF/ERA5/HOURLY');
var U_BND = 'u_component_of_wind_10m';
var V_BND = 'v_component_of_wind_10m';
var S1_GRD = ee.ImageCollection('COPERNICUS/S1_GRD');

// [name, lat, lon] — same coords as exportSicilyPlanet.js (Pozzillo FIXED to 37.700,14.530).
var RESERVOIRS = [
  ['Ancipa',     37.887, 14.565],
  ['Pozzillo',   37.700, 14.530],
  ['Poma',       37.994, 13.090],
  ['Rosamarina', 37.944, 13.640],
];

var BATCH_SLICE = [0, RESERVOIRS.length];

// ── Per-overpass wind sampler (identical to exportEra5Wind.js) ────────────────
function windAtOverpass(s1Img, point) {
  var t   = ee.Date(s1Img.get('system:time_start'));
  var era = ee.Image(ERA5.filterDate(t.advance(-1, 'hour'), t.advance(1, 'hour')).first());
  var uv  = era.select([U_BND, V_BND]).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: point,
    scale: CFG.era5_scale_m, bestEffort: true,
  });
  var u   = ee.Number(uv.get(U_BND));
  var v   = ee.Number(uv.get(V_BND));
  var spd = u.pow(2).add(v.pow(2)).sqrt();
  var dir = u.multiply(-1).atan2(v.multiply(-1)).multiply(180 / Math.PI).mod(360);
  return ee.Feature(null, {
    'date':         t.format('YYYY-MM-dd'),
    'datetime_utc': t.format('YYYY-MM-dd HH:mm'),
    'wind_ms':      spd,
    'wind_dir_deg': dir,
    'relOrbit':     s1Img.get('relativeOrbitNumber_start'),
    'pass':         s1Img.get('orbitProperties_pass'),
  });
}

// ── Main loop ─────────────────────────────────────────────────────────────────
RESERVOIRS.slice(BATCH_SLICE[0], BATCH_SLICE[1]).forEach(function(res) {
  var rName = res[0];
  var point = ee.Geometry.Point([res[2], res[1]]);   // [lon, lat]

  var s1 = S1_GRD
    .filterBounds(point).filterDate(CFG.s1_start, CFG.s1_end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  var windSeries = ee.FeatureCollection(s1.map(function(img) {
    return windAtOverpass(img, point);
  })).filter(ee.Filter.notNull(['wind_ms']));

  Export.table.toDrive({
    collection:     windSeries,
    description:    'Era5Wind_' + rName,
    folder:         CFG.drive_folder,
    fileNamePrefix: 'Era5Wind_' + rName,
    fileFormat:     'CSV',
  });

  print('Wind export queued: ' + rName);
});
