/**
 * exportEra5Wind.js
 *
 * Downloads ERA5 10 m wind at the instant of every Sentinel-1 overpass, per reservoir,
 * for Experiment C (wind mechanism): does the VV↔dual-pol divergence grow with wind?
 *
 * For each reservoir point we take the same S1 acquisitions as the area pipeline
 * (IW, 10 m, VV+VH, full date range), and for each overpass timestamp we sample the
 * nearest ERA5 hourly field. Sentinel-1 is sun-synchronous (~06 h descending /
 * ~18 h ascending local), and wind has a strong diurnal cycle, so HOURLY ERA5 — not
 * daily — is required to capture the wind actually present at imaging time.
 *
 * Output: CSV per reservoir in Drive folder GEE_Era5Wind, columns:
 *   date, datetime_utc, wind_ms, wind_dir_deg, relOrbit, pass
 * Join to SAR_area_<name>.csv on `date` (relOrbit disambiguates duplicate-date overpasses).
 *
 * Reference resolution caveat: ERA5 is ~0.25° (~28 km) — this is regional wind, not the
 * local over-water field. Adequate as a relative wind-exposure proxy; state the limitation.
 *
 * Batching: ERA5 point-sampling is light; all reservoirs usually fit one run. If a run
 * times out, set BATCH_SLICE to a sub-range (e.g. [0, 17]) and run twice.
 */

// ── Config ────────────────────────────────────────────────────────────────────
var CFG = {
  s1_start:     '2014-10-01',
  s1_end:       '2021-12-31',
  drive_folder: 'GEE_Era5Wind',
  era5_scale_m: 27830,           // ERA5 native ~0.25°
};

// ERA5 hourly reanalysis (covers land AND water, unlike ERA5-Land which is land-masked).
// Verify band names in the Code Editor if the dataset is revised.
var ERA5  = ee.ImageCollection('ECMWF/ERA5/HOURLY');
var U_BND = 'u_component_of_wind_10m';
var V_BND = 'v_component_of_wind_10m';

var S1_GRD = ee.ImageCollection('COPERNICUS/S1_GRD');

// ── Reservoir list [name, lat, lon] (from global_pilot_v4_candidates.csv) ─────
var RESERVOIRS = [
  ['Sau', 41.9700, 2.3850],
  ['Susqueda', 41.9330, 2.5180],
  ['El_Atazar', 40.8920, -3.6370],
  ['Rappbode', 51.7380, 10.9130],
  ['Castillon', 43.8930, 6.5340],
  ['Saint_Cassien', 43.5960, 6.7240],
  ['Salto', 42.2060, 13.0550],
  ['Blyde', -24.5350, 30.8000],
  ['Cachi', 9.8100, -83.7690],
  ['Miyagase', 35.5570, 139.1850],
  ['Yamba', 36.6920, 138.8280],
  ['El_Burguillo', 40.3670, -4.5000],
  ['Boadella', 42.3290, 2.8310],
  ['Puentes_Viejas', 40.9830, -3.5760],
  ['Guajaraz', 39.6750, -4.1070],
  ['Panneciere', 47.2000, 3.8830],
  ['Sarrans', 44.8180, 2.7630],
  ['Bilancino', 43.9780, 11.2020],
  ['Cecita', 39.3500, 16.5650],
  ['Oued_Makhazine', 35.1670, -5.5330],
  ['Karapuzha', 11.6170, 76.0330],
  ['Saguaro', 33.6550, -111.5310],
  ['Boegoeberg', -29.0260, 22.1550],
  ['Tzaneen', -23.8130, 30.1450],
  ['Googong', -35.4400, 149.2250],
  ['Cardinia', -37.9350, 145.5100],
  ['Grandval', 45.0180, 3.0930],
  ['Deer_Creek', 40.4060, -111.5290],
  ['East_Canyon', 40.9300, -111.5820],
  ['Pineview', 41.2730, -111.8390],
  ['Rockport', 40.7660, -111.2960],
  ['Antero', 38.9800, -105.8600],
  ['Shaharchay', 37.6400, 45.0090],
  ['Welbedacht', -29.8700, 26.8200],
];

var BATCH_SLICE = [0, RESERVOIRS.length];   // narrow if a run times out

// ── Per-overpass wind sampler ─────────────────────────────────────────────────
function windAtOverpass(s1Img, point) {
  var t   = ee.Date(s1Img.get('system:time_start'));
  // nearest hourly ERA5 field (top-of-hour images; ±1 h window brackets the overpass)
  var era = ee.Image(ERA5.filterDate(t.advance(-1, 'hour'), t.advance(1, 'hour')).first());
  var uv  = era.select([U_BND, V_BND]).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: point,
    scale: CFG.era5_scale_m, bestEffort: true,
  });
  var u   = ee.Number(uv.get(U_BND));
  var v   = ee.Number(uv.get(V_BND));
  var spd = u.pow(2).add(v.pow(2)).sqrt();
  // meteorological direction (degrees FROM which the wind blows)
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
