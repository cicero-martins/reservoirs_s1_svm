/**
 * exportSicilyPlanetMasks.js
 *
 * Exports the per-date PlanetScope NDWI WATER MASKS (the ones tuned for the
 * validation) to Google Drive, so the optical waterlines can feed the Schwatke
 * bathymetry reconstruction (analysis/schwatke_bathymetry_3d.py) exactly like the
 * SAR masks. This lets us reconstruct an OPTICAL (3 m PlanetScope) DEM and
 * cross-validate the SAR-derived bathymetry — crucially giving Ancipa and
 * Pozzillo (which have no modern field survey) an independent near-truth.
 *
 * ── HOW TO USE ──────────────────────────────────────────────────────────────
 * PASTE THIS BLOCK AT THE END of your `SicilianLakesPlanet` /
 * GEEndwiValidation_app.js script, which already defines:
 *   - the site AOIs (aoiPozzillo, aoiAncipa, aoiRosamarina, aoiPoma, ...),
 *   - `assetData`      (per site: { aoi, images:{ "Site - YYYY/MM/DD": assetId } }),
 *   - `paperThresholds`("Site - YYYY/MM/DD" -> chosen NDWI threshold).
 * Then run and start the Export tasks. It exports ONLY the dates that have a
 * chosen threshold (the validated masks); dates without one are skipped.
 *
 * Grid: 10 m, EPSG:32633 — matches raw_data/GEE_SicilyMasks so the PlanetScope
 * DEM is directly comparable to the SAR DEM and reuses the same reconstruction.
 * (Set EXPORT_SCALE = 3 for a native-resolution optical product instead.)
 * NDWI = normalizedDifference(['b2','b4']) = (green - NIR)/(green + NIR); water = NDWI > threshold.
 * ────────────────────────────────────────────────────────────────────────────
 */

// Run a subset per batch to keep the task list manageable; [] = all four sites.
// Ancipa + Pozzillo first: they lack a modern survey, so PlanetScope is their truth.
var EXPORT_ONLY   = ['Ancipa', 'Pozzillo'];
var EXPORT_SCALE  = 10;                       // m (10 = match SAR masks; 3 = native PlanetScope)
var DRIVE_FOLDER  = 'GEE_SicilyPlanetMasks';

// "Ancipa - 2024/04/29" -> "2024-04-29"
function isoDate(dateKey) {
  return dateKey.split(' - ')[1].replace(/\//g, '-');
}

var nQueued = 0;
Object.keys(assetData).forEach(function (site) {
  if (EXPORT_ONLY.length && EXPORT_ONLY.indexOf(site) < 0) return;
  var aoi  = assetData[site].aoi;
  var imgs = assetData[site].images;

  Object.keys(imgs).forEach(function (dateKey) {
    var thr = paperThresholds[dateKey];
    if (thr === undefined) return;            // only the validated (chosen-threshold) dates

    var ndwi  = ee.Image(imgs[dateKey]).normalizedDifference(['b2', 'b4']);
    var water = ndwi.gt(thr).clip(aoi).unmask(0).toByte().rename('water');

    Export.image.toDrive({
      image:          water,
      description:    'maskPlanet_' + site + '_' + isoDate(dateKey),
      folder:         DRIVE_FOLDER,
      fileNamePrefix: 'mask_Planet_' + site + '_' + isoDate(dateKey),
      region:         aoi,
      scale:          EXPORT_SCALE,
      crs:            'EPSG:32633',
      maxPixels:      1e12
    });
    nQueued += 1;
  });
});
print('Queued ' + nQueued + ' PlanetScope mask exports (' +
      (EXPORT_ONLY.length ? EXPORT_ONLY.join(', ') : 'all sites') +
      ') -> Drive/' + DRIVE_FOLDER + ' at ' + EXPORT_SCALE + ' m EPSG:32633.');
