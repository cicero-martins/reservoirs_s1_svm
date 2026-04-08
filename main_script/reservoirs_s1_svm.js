

/*******************************************************************************
 * MODEL
 *
 * Defines data model: reservoir geometries, volume coefficients,
 * and central coordinates for the map initialization.
 ******************************************************************************/

var m = {};

// Reservoir AOIs — sorted alphabetically
m.AOI = {
  'Invaso Ancipa': Ancipa,
  'Invaso Arancio': Arancio,
  'Invaso Castello': Castello,
  'Invaso Cimia': Cimia,
  'Invaso Comunelli': Comunelli,
  'Invaso di Gela': DiGela,
  'Invaso di Gammauta': Gammauta,
  'Invaso di Piana degli Albanesi': Albanesi,
  'Invaso Diddino': Diddino,
  'Invaso Dirillo': Dirillo,
  'Invaso Disueri': Disueri,
  'Invaso Don Sturzo': DonSturzo,
  'Invaso Fanaco': Fanaco,
  'Invaso Fiumara Grande': FiumaraGrande,
  'Invaso Furore': Furore,
  'Invaso Garcia': Garcia,
  'Invaso Gibbesi': Gibbesi,
  'Invaso Gorgo': Gorgo,
  'Invaso Guadalami': Guadalami,
  'Invaso Lentini': Lentini,
  'Invaso Monte Cavallaro': MonteCavallaro,
  'Invaso Nicoletti': Nicoletti,
  'Invaso Olivo': Olivo,
  'Invaso Paceco': Paceco,
  'Invaso Pergusa': Pergusa,
  'Invaso Piano del Leone': Leone,
  'Invaso Pietrarossa': Pietrarossa,
  'Invaso Poma': Poma,
  'Invaso Ponte Barca': PonteBarca,
  'Invaso Pozzillo': Pozzillo,
  'Invaso Prizzi': Prizzi,
  'Invaso Rosamarina': Rosamarina,
  'Invaso Rubino': Rubino,
  'Invaso San Giovanni': SanGiovanni,
  'Invaso Santa Rosalia': SantaRosalia,
  'Invaso Scanzano': Scanzano,
  'Invaso Sciaguana': Sciaguana,
  'Invaso Trinità': Trinita,
  'Invaso Vasca Ogliastro': VascaOgliastro,
  'Invaso Villarosa': Villarosa,
  'Invaso Zaffarana': Zaffarana
};

// Volume-from-area polynomial coefficients: V = a*A² + b*A + c
// where A is water surface area (hectares) and V is volume (10^6 m³)
// Coefficients derived from the Water Agency of Sicily (2006) area-volume curves
var lakeCoefficients = {
  'Invaso Ancipa':                  {a: 0.0009,      b: 0.0938,  c: -2.0159},
  'Invaso Arancio':                 {a: 0.0002,      b: 0.0376,  c: -0.5576},
  'Invaso Castello':                {a: 0.0005,      b: 0.0491,  c: -0.2771},
  'Invaso Cimia':                   {a: 0.0009,      b: 0.0335,  c: -0.0076},
  'Invaso Comunelli':               {a: 0.001,       b: 0.0116,  c:  0.0935},
  'Invaso di Gela':                 {a: 0.000,       b: 0.0,     c:  0},
  'Invaso di Gammauta':             {a: 0.0012,      b: 0.0118,  c:  0.01},
  'Invaso di Piana degli Albanesi': {a: 0.0002,      b: 0.0288,  c: -0.4372},
  'Invaso Diddino':                 {a: 0.000,       b: 0.0,     c:  0},
  'Invaso Dirillo':                 {a: 0.000,       b: 0.0,     c:  0},
  'Invaso Disueri':                 {a: 0.0009,      b: -0.0186, c:  0.2035},
  'Invaso Don Sturzo':              {a: 0.0002,      b: 0.0554,  c: -1.3307},
  'Invaso Fanaco':                  {a: 0.0005,      b: 0.0912,  c: -0.4439},
  'Invaso Fiumara Grande':          {a: 0.0077,      b: -0.0023, c:  0.0048},
  'Invaso Furore':                  {a: 0.0013,      b: 0.0609,  c: -0.1161},
  'Invaso Garcia':                  {a: 0.0003,      b: 0.0256,  c: -0.1278},
  'Invaso Gibbesi':                 {a: 0.0006,      b: 0.0465,  c: -0.2596},
  'Invaso Gorgo':                   {a: 0.0039,      b: -0.1774, c:  2.2172},
  'Invaso Guadalami':               {a: 0.0067,      b: 0.0046,  c: -0.0086},
  'Invaso Lentini':                 {a: 0.0014,      b: -2.0234, c:  732.36},
  'Invaso Monte Cavallaro':         {a: 0.0068,      b: 0.3088,  c: -12.377},
  'Invaso Nicoletti':               {a: 0.0005,      b: 0.0475,  c: -0.1694},
  'Invaso Olivo':                   {a: 0.0008,      b: 0.0618,  c: -0.1773},
  'Invaso Paceco':                  {a: 0.0001,      b: 0.045,   c: -0.0668},
  'Invaso Pergusa':                 {a: 0.000,       b: 0.0,     c:  0},
  'Invaso Piano del Leone':         {a: 0.0009,      b: 0.0277,  c: -0.0024},
  'Invaso Pietrarossa':             {a: 0.000,       b: 0.0,     c:  0},
  'Invaso Poma':                    {a: 0.0001,      b: 0.0817,  c: -2.114},
  'Invaso Ponte Barca':             {a: 0.0000003,   b: 0.0013,  c: -0.0207},
  'Invaso Pozzillo':                {a: 0.0003,      b: -0.061,  c:  2.7448},
  'Invaso Prizzi':                  {a: 0.0008,      b: 0.0275,  c: -0.0968},
  'Invaso Rosamarina':              {a: 0.0003,      b: 0.0817,  c: -0.5108},
  'Invaso Rubino':                  {a: 0.0003,      b: 0.0428,  c: -0.1349},
  'Invaso San Giovanni':            {a: 0.0002,      b: 0.0416,  c: -0.2955},
  'Invaso Santa Rosalia':           {a: 0.0007,      b: 0.0744,  c: -0.358},
  'Invaso Scanzano':                {a: 0.0005,      b: 0.0463,  c: -0.1147},
  'Invaso Sciaguana':               {a: 0.001,       b: 0.0124,  c:  0.0445},
  'Invaso Trinità':                 {a: 0.0003,      b: 0.0147,  c: -0.197},
  'Invaso Vasca Ogliastro':         {a: 0.0024,      b: -0.0243, c:  0.0672},
  'Invaso Villarosa':               {a: 0.0006,      b: 0.0381,  c: -0.2521},
  'Invaso Zaffarana':               {a: 0.0021,      b: 0.0007,  c:  0.0144}
};

// Central coordinates of Sicily used for initial map view
m.siciliaCoordinates = ee.Geometry.Point([13.92973509314157, 37.63464221279454]);


/*******************************************************************************
 * COMPONENTS
 *
 * Defines all UI widgets that compose the application.
 ******************************************************************************/

var c = {};

c.controlPanel = ui.Panel();
c.map = ui.Map();
c.aoiOutlineLayer = null;

// Title and description
c.info = {};
c.info.titleLabel = ui.Label('Water Surface and Volume in Reservoirs Using Sentinel-1', {
  fontSize: '24px',
  fontWeight: 'bold',
  color: '383838'
});
c.info.aboutLabel = ui.Label(
  'This application allows you to calculate the area and volume of water in a ' +
  'user-defined area of interest, using SAR images from the Sentinel-1 mission. ' +
  'Draw the area of interest or select from the list (Sicilian reservoirs), ' +
  'define a period, and click RUN! to view the data. ' +
  'Clicking on a point on the chart loads the map with the image for that date.');
c.info.panel = ui.Panel([c.info.titleLabel, c.info.aboutLabel]);

// Reservoir selector
c.selectAOI = {};
c.selectAOI.label = ui.Label('Click on the map or select from the list');
c.selectAOI.selector = ui.Select(Object.keys(m.AOI), null, 'Invaso Rosamarina');

c.selectAOI.selector.onChange(function(reservoirName) {
  if (!reservoirName) return;

  // Remove timelapse panel if present
  if (c.thumbnailPanel) {
    c.map.remove(c.thumbnailPanel);
    c.thumbnailPanel = null;
  }

  var geometry = m.AOI[reservoirName];
  c.map.centerObject(geometry, 12);

  // Remove previous AOI outline layer
  if (c.aoiOutlineLayer) {
    c.map.layers().remove(c.aoiOutlineLayer);
  }

  // Add new AOI outline layer
  var outlineFeature = ee.Feature(geometry);
  c.aoiOutlineLayer = ui.Map.Layer(outlineFeature, {
    color: 'orange',
    fillColor: '00000000',
    width: 1,
    opacity: 0.6
  }, 'Reservoir');
  c.map.layers().add(c.aoiOutlineLayer);

  c.labelNomeReservatorio.setValue('📍 ' + reservoirName);

  if (c.etiquetaFlutuante) {
    c.etiquetaFlutuante.setValue('');
    c.etiquetaFlutuante.style().set('shown', false);
  }

  geometry.centroid(1).coordinates().getInfo(function(coord) {
    if (coord) {
      c.etiquetaFlutuante.setValue('📍 ' + reservoirName);
      c.etiquetaFlutuante.style().set({shown: true, position: 'top-center'});
    }
  });
});

c.selectAOI.panel = ui.Panel([c.selectAOI.label, c.selectAOI.selector]);

// --- Execution Logic (Controller) ---

c.executeAnalysis = function() {
  var isCustom = c.toggleCustom.getValue();
  var finalGeometry;

  if (isCustom) {
    // 1. Get geometry from drawing tools or uploaded asset
    var layers = c.map.drawingTools().layers();
    if (layers.length() > 0) {
      finalGeometry = layers.get(0).getEeObject();
    } else {
      print('Error: No custom geometry found. Please draw an AOI on the map.');
      return;
    }
  } else {
    // 2. Get geometry from the Predefined Selector
    var selectedName = c.selectAOI.selector.getValue();
    finalGeometry = m.AOI[selectedName];
  }

  // Proceed with the SVM Classification using finalGeometry
  print('Running analysis for:', isCustom ? 'Custom AOI' : selectedName);
  
  // Call your processing function here, e.g.:
  // runSVMClassification(finalGeometry);
};

// Custom AOI drawing controls
var customAOI = null;
var customAOIFeature = null;
var customAOIName = null;

c.customAOI = {};
c.customAOI.checkbox = ui.Checkbox('Use custom AOI (draw on map)', true);

c.customAOI.startButton = ui.Button('Start drawing', function() {
  var drawingTools = c.map.drawingTools();
  drawingTools.setShown(true);
  try { drawingTools.layers().reset(); } catch (e) {}
  drawingTools.setShape('polygon');
  drawingTools.draw();
  toast('Drawing mode activated. Draw a polygon (double-click to finish), then click "Confirm AOI".');
});

c.customAOI.confirmButton = ui.Button('Confirm AOI', function() {
  var drawingTools = c.map.drawingTools();
  var fcClient = drawingTools.toFeatureCollection();

  fcClient.evaluate(function(fcObj) {
    if (!fcObj || !fcObj.features || fcObj.features.length === 0) {
      toast('No geometry found. Draw a polygon and finish with a double-click.');
      return;
    }

    var lastFeat = fcObj.features[fcObj.features.length - 1];
    var geom = ee.Geometry(lastFeat.geometry);

    var nameVal = (c.customAOI.nameBox && c.customAOI.nameBox.getValue &&
                   c.customAOI.nameBox.getValue().trim()) || '';
    if (!nameVal) {
      var now = new Date();
      nameVal = 'AOI_' + now.toISOString().replace(/[:.]/g, '-');
    }
    customAOIName = nameVal;
    c.labelNomeReservatorio.setValue('📍 ' + customAOIName);

    customAOIFeature = ee.Feature(geom, {name: customAOIName});
    customAOI = customAOIFeature.geometry();

    if (c.aoiOutlineLayer) {
      try { c.map.layers().remove(c.aoiOutlineLayer); } catch (e) {}
    }
    c.aoiOutlineLayer = ui.Map.Layer(
      customAOIFeature,
      {color: 'magenta', fillColor: '00000000'},
      'Custom AOI: ' + customAOIName
    );
    c.map.layers().add(c.aoiOutlineLayer);

    toast('AOI confirmed: ' + customAOIName + '. Check "Use custom AOI" to process this area.');
  });
});

c.customAOI.clearButton = ui.Button('Clear drawings', function() {
  var drawingTools = c.map.drawingTools();
  try { drawingTools.layers().reset(); } catch (e) {}
  if (c.aoiOutlineLayer) {
    try { c.map.layers().remove(c.aoiOutlineLayer); } catch (e) {}
    c.aoiOutlineLayer = null;
  }
  customAOI = null;
  toast('Drawings cleared and custom AOI removed.');
});

c.customAOI.nameBox = ui.Textbox({
  placeholder: 'AOI name',
  value: '',
  style: {width: '100%'}
});

c.customAOI.panel = ui.Panel([
  ui.Label('Custom AOI'),
  c.customAOI.startButton,
  ui.Label('AOI name:'),
  c.customAOI.nameBox,
  c.customAOI.confirmButton,
  c.customAOI.clearButton,
  ui.Label({value: 'Workflow: Start drawing → draw on map → Confirm AOI',
            style: {fontSize: '11px', color: '#777'}})
], ui.Panel.Layout.flow('vertical'), {margin: '6px 0'});

// Custom area-to-volume coefficient controls
c.customCoeff = {};
c.customCoeff.checkbox = ui.Checkbox('Use custom area→volume coefficients (a, b, c)', false);
c.customCoeff.a = ui.Textbox({placeholder: 'a (e.g. 0.0003)', value: '', style: {width: '90px'}});
c.customCoeff.b = ui.Textbox({placeholder: 'b (e.g. 0.0817)', value: '', style: {width: '90px'}});
c.customCoeff.c = ui.Textbox({placeholder: 'c (e.g. -0.5108)', value: '', style: {width: '90px'}});

c.customCoeff.panel = ui.Panel([
  ui.Label('Area→volume coefficients:'),
  
  ui.Panel(
    [ui.Label('a'), c.customCoeff.a, ui.Label('b'), c.customCoeff.b, ui.Label('c'), c.customCoeff.c],
    ui.Panel.Layout.flow('horizontal')
  ),
  c.customCoeff.checkbox
], null, {margin: '6px 0'});

// Default date range: last 12 months
var today = new Date();

// Calculate 3 months ago date
var threeMonthsAgo = new Date();
threeMonthsAgo.setMonth(today.getMonth() - 3);

// Format dates as strings like 'YYYY-MM-DD'
var startDateDefault = threeMonthsAgo.toISOString().split('T')[0];
var endDateDefault   = today.toISOString().split('T')[0];

// Start date input
c.selectDateInitial = {};
c.selectDateInitial.label = ui.Label('Start');
c.selectDateInitial.textbox = ui.Textbox({
  placeholder: 'YYYY-MM-DD', value: startDateDefault, style: {width: '100px'}
});
c.selectDateInitial.panel = ui.Panel({
  widgets: [c.selectDateInitial.label, c.selectDateInitial.textbox],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {stretch: 'horizontal', margin: '10px 0'}
});

// End date input
c.selectDateFinal = {};
c.selectDateFinal.label = ui.Label('End');
c.selectDateFinal.textbox = ui.Textbox({
  placeholder: 'YYYY-MM-DD', value: endDateDefault, style: {width: '100px'}
});
c.selectDateFinal.panel = ui.Panel({
  widgets: [c.selectDateFinal.label, c.selectDateFinal.textbox],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {stretch: 'horizontal', margin: '10px 0'}
});

// Combined date panel
c.selectDate = {};
c.selectDate.panel = ui.Panel({
  widgets: [c.selectDateInitial.panel, c.selectDateFinal.panel],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {stretch: 'horizontal', margin: '10px 0'}
});

// RUN button
c.selectedDates = {};
c.selectedDates.button = ui.Button({
  label: '▶️ RUN!',
  onClick: atualizarMapa,
  style: {
    backgroundColor: 'orange',
    color: 'black',
    fontWeight: 'bold',
    fontSize: '20px',
    margin: '4px',
    padding: '8px'
  }
});

// Reset button
c.resetButton = ui.Button({
  label: '🔄 Reset',
  onClick: function() {
    c.map.layers().reset();
    c.map.centerObject(m.siciliaCoordinates, 8.5);
    if (c.thumbnailPanel) {
      c.map.remove(c.thumbnailPanel);
      c.thumbnailPanel = null;
    }
    var freshMarkerLayer = ui.Map.Layer(
      reservoirMarkers.style({color: 'cyan', pointSize: 6, width: 1}),
      {}, 'Reservoirs', true
    );
    c.map.add(freshMarkerLayer);
    c.exportButton.style().set('shown', false);
    c.graphicResults.panel.clear();
    c.labelNomeReservatorio.setValue('');
    c.dataAtual.label.setValue('Date: ---');
  },
  style: {
    backgroundColor: '#ccc',
    color: 'black',
    fontWeight: 'bold',
    fontSize: '14px',
    margin: '4px',
    padding: '8px'
  }
});

// Timelapse button
c.timelapseButton = ui.Button({
  label: '🎞️ Timelapse',
  onClick: gerarTimelapse,
  style: {
    backgroundColor: '#eeeee4',
    color: 'black',
    fontWeight: 'bold',
    fontSize: '12px',
    margin: '4px',
    padding: '8px'
  }
});

// Selected image date label
c.dataAtual = {};
c.dataAtual.label = ui.Label('Date: ---');
c.dataAtual.label.style().set({fontSize: '16px', fontWeight: 'bold', margin: '5px 0'});

// Chart panel
c.graphicResults = {};
c.graphicResults.panel = ui.Panel({style: {height: '500px', margin: '20px 0'}});
c.graphicResults.disclaimer = ui.Label(
  '*Volumes of Sicilian reservoirs computed using smoothed surface area from classified Sentinel-1 images ' +
  'and the area-to-volume relationships from original design documentation, compiled in Le Grandi Dighe in Sicilia (Bruno, Cannarozzo e Ciraolo, 2003).');
c.graphicResults.disclaimer.style().set({fontSize: '10px', margin: '10px 0'});

// Map legends
c.legenda = {};

c.legenda.data = {};
c.legenda.data.panel = ui.Panel({style: {position: 'bottom-right', padding: '8px 15px'}});
c.legenda.data.label = c.dataAtual.label;
c.legenda.data.panel.add(c.legenda.data.label);

c.legenda.sar = {};
c.legenda.sar.panel = ui.Panel({style: {position: 'bottom-right', padding: '8px 15px'}});
c.legenda.sar.label = ui.Label('Sentinel-1 VV', {fontWeight: 'bold'});
c.legenda.sar.gradiente = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: {bbox: [0, 0, 1, 0.1], dimensions: '100x10', palette: ['black', 'white']},
  style: {stretch: 'horizontal', margin: '0 8px'}
});
c.legenda.sar.escala = ui.Panel({
  widgets: [
    ui.Label('-24', {fontSize: '10px', textAlign: 'left'}),
    ui.Panel({style: {stretch: 'horizontal', margin: '0 10px'}}),
    ui.Label('0',   {fontSize: '10px', textAlign: 'right'})
  ],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {margin: '0'}
});
c.legenda.sar.panel.add(c.legenda.sar.label);
c.legenda.sar.panel.add(c.legenda.sar.gradiente);
c.legenda.sar.panel.add(c.legenda.sar.escala);

c.legenda.agua = {};
c.legenda.agua.panel = ui.Panel({style: {position: 'bottom-right', padding: '8px 15px'}});
c.legenda.agua.label = ui.Label('Water', {fontWeight: 'bold'});
c.legenda.agua.gradiente = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: {bbox: [0, 0, 1, 0.1], dimensions: '100x10', palette: ['#68C1FF']},
  style: {stretch: 'horizontal', margin: '0 8px'}
});
c.legenda.agua.panel.add(c.legenda.agua.label);
c.legenda.agua.panel.add(c.legenda.agua.gradiente);

// Export feedback label (initially hidden)
c.exportMessage = ui.Label('', {color: 'green'});

// Export button (initially hidden)
c.exportButton = ui.Button({
  label: 'Export to Google Drive',
  onClick: function() {
    if (!c.selectedImage) {
      ui.alert('No image selected for export.');
      return;
    }

    var aoi = getActiveAOI();
    var waterMask = c.selectedImage.select('WaterCleaned');
    var sarVV = c.selectedImage.select('VV').clip(aoi);

    var sarShifted   = shiftImage(sarVV,     18.393, -7.035);
    var waterShifted = shiftImage(waterMask, 18.393, -7.035);

    var waterBin = waterShifted.eq(1).selfMask();
    var proj_utm33n = ee.Projection('EPSG:32633').atScale(10);
    var waterBin_utm = waterBin.reproject({crs: proj_utm33n});

    var waterPolygons = waterBin_utm.reduceToVectors({
      geometry: aoi,
      geometryType: 'polygon',
      reducer: ee.Reducer.countEvery(),
      scale: 10,
      maxPixels: 1e9,
      bestEffort: true
    });

    Export.table.toDrive({
      collection: waterPolygons,
      description: 'WaterMask_' +
        ee.Date(c.selectedImage.get('system:time_start')).format('YYYY-MM-dd').getInfo(),
      folder: 'EarthEngineExports',
      fileFormat: 'GeoJSON'
    });

    Export.image.toDrive({
      image: sarShifted,
      description: 'SAR_VV_' +
        ee.Date(c.selectedImage.get('system:time_start')).format('YYYY-MM-dd').getInfo(),
      folder: 'EarthEngineExports',
      scale: 10,
      crs: 'EPSG:32633',
      fileFormat: 'GeoTIFF',
      maxPixels: 1e9
    });

    c.exportMessage.setValue('Export started! Check your Google Drive.');
  },
  style: {shown: false}
});

// Reservoir name label displayed on map
c.labelNomeReservatorio = ui.Label({
  value: '',
  style: {
    position: 'bottom-left',
    padding: '8px',
    fontWeight: 'bold',
    fontSize: '20px',
    backgroundColor: 'rgba(255, 255, 255, 0.7)'
  }
});
c.map.add(c.labelNomeReservatorio);

// Floating label (shown near reservoir centroid after selection)
c.etiquetaFlutuante = ui.Label({
  value: '',
  style: {
    position: 'bottom-left',
    padding: '6px',
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    fontWeight: 'bold',
    fontSize: '14px'
  }
});


/*******************************************************************************
 * COMPOSITION
 *
 * Assembles widgets into the app layout.
 ******************************************************************************/

ui.root.clear();
ui.root.add(c.controlPanel);
ui.root.add(c.map);

c.controlPanel.style().set({width: '400px', padding: '10px'});

var bloccoIntro = ui.Panel(
  [c.info.titleLabel, c.info.aboutLabel],
  ui.Panel.Layout.flow('vertical'),
  {margin: '0 0 10px 0'}
);

// --- Mode Switch Toggle ---
// --- Mode Switch Toggle ---
c.toggleCustom = ui.Checkbox({
  label: 'Activate Custom AOI Mode (Draw/Upload)',
  value: false,
  onChange: function(checked) {
    // 1. Refresh Side Panel
    c.updatePanelWidgets();
    
    // 2. Manage Map state
    if (checked) {
      // If entering Custom Mode: remove the predefined AOI outline to clear the view
      if (c.aoiOutlineLayer) {
        c.map.layers().remove(c.aoiOutlineLayer);
        c.aoiOutlineLayer = null; // Clear reference
      }
      c.map.drawingTools().setShown(true);
    } else {
      // If returning to Preset Mode: correctly clear the drawing tools
      var drawingLayers = c.map.drawingTools().layers();
      drawingLayers.forEach(function(layer) {
        // Correct way to clear geometries from a ui.Map.GeometryLayer
        var geometries = layer.geometries();
        while (geometries.length() > 0) {
          geometries.remove(geometries.get(0));
        }
      });
      c.map.drawingTools().setShown(false);
    }
  },
  style: {margin: '10px 0 10px 10px', fontWeight: 'bold'}
});

var bloccoSelezione = ui.Panel([
  ui.Label('🗺️ Select a Reservoir', {fontWeight: 'bold'}),
  c.selectAOI.panel
], ui.Panel.Layout.flow('vertical'), {margin: '5px 0', padding: '4px', border: '1px solid #ccc'});

var bloccoCustomAOI = ui.Panel([
  ui.Label('🗺️ Draw your own AOI', {fontWeight: 'bold'}),
  //c.customAOI.checkbox,
  c.customAOI.panel,
  c.customCoeff.panel
], ui.Panel.Layout.flow('vertical'), {margin: '5px 0', padding: '4px', border: '1px solid #ccc'});

var bloccoData = ui.Panel([
  ui.Label('📅 Analysis Period', {fontWeight: 'bold'}),
  c.selectDate.panel
], ui.Panel.Layout.flow('vertical'), {margin: '5px 0', padding: '4px', border: '1px solid #ccc'});

var botoesPanel = ui.Panel({
  widgets: [c.selectedDates.button, c.resetButton, c.timelapseButton],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {stretch: 'horizontal', margin: '10px 0'}
});

/**
 * Dynamic UI Controller
 * Rebuilds the control panel based on the selection mode (Preset vs. Custom).
 */
c.updatePanelWidgets = function() {
  var isCustom = c.toggleCustom.getValue();

  var widgets = [
    bloccoIntro,
    c.toggleCustom
  ];

  if (isCustom) {
    widgets.push(bloccoCustomAOI);
  } else {
    widgets.push(bloccoSelezione);
  }

  // Bottom common blocks
  widgets.push(bloccoData);
  widgets.push(botoesPanel);
  widgets.push(c.graphicResults.panel);

  c.controlPanel.widgets().reset(widgets);
};

// Initialization: Set up the initial state of the side panel
c.updatePanelWidgets();

c.map.add(c.legenda.data.panel);
c.map.add(c.legenda.sar.panel);
c.map.add(c.legenda.agua.panel);

c.map.setOptions('SATELLITE');
c.map.centerObject(m.siciliaCoordinates, 8.5);


/*******************************************************************************
 * STYLING
 ******************************************************************************/

var visParamWater = {min: 0, max: 1, palette: ['#68C1FF'], opacity: 0.8};
var visParamSAR   = {min: -24, max: 0, palette: ['black', 'white'], opacity: 0.8};


/*******************************************************************************
 * BEHAVIOURS
 *
 * Helper functions and event handlers.
 ******************************************************************************/

// Returns true when the user has enabled the custom AOI option and a geometry exists
function usingCustomAOI() {
  return c.customAOI && c.customAOI.checkbox && c.customAOI.checkbox.getValue &&
         c.customAOI.checkbox.getValue() && customAOI;
}

// Returns the active AOI geometry (custom or selected reservoir)
function getActiveAOI() {
  return usingCustomAOI() ? customAOI : m.AOI[c.selectAOI.selector.getValue()];
}

// Returns the active reservoir name (custom or selected)
function getActiveName() {
  return usingCustomAOI() ? customAOIName : c.selectAOI.selector.getValue();
}

// Displays a temporary toast message on the map
function toast(msg, seconds) {
  seconds = seconds || 3;
  var label = ui.Label(String(msg), {
    fontWeight: 'bold', fontSize: '11px', color: 'white', textAlign: 'center'
  });
  var panel = ui.Panel({
    widgets: [label],
    style: {
      position: 'top-center',
      padding: '6px 10px',
      backgroundColor: 'rgba(0,0,0,0.7)',
      border: '1px solid gray',
      borderRadius: '3px'
    }
  });
  c.map.add(panel);
  ui.util.setTimeout(function() {
    try { c.map.remove(panel); } catch (e) {}
  }, seconds * 1000);
}

// Shows an animated loading indicator on the map
function showMapLoadingMessage(textBase, intervalMs) {
  var label = ui.Label(textBase, {
    fontWeight: 'bold', fontSize: '8px', color: 'white', textAlign: 'center'
  });
  var panel = ui.Panel({
    widgets: [label],
    style: {
      position: 'top-center',
      padding: '3px',
      backgroundColor: 'rgba(0, 0, 0, 0.6)',
      border: '1px solid gray',
      borderRadius: '2px'
    }
  });
  c.map.add(panel);

  var dots = 0;
  var maxDots = 3;
  var running = true;

  function animate() {
    if (!running) return;
    dots = (dots + 1) % (maxDots + 1);
    label.setValue(String(textBase) + Array(dots + 1).join('.'));
    ui.util.setTimeout(animate, intervalMs || 500);
  }
  animate();

  return {
    panel: panel,
    stop: function() {
      running = false;
      c.map.remove(panel);
    }
  };
}

// Shifts an image by offsetX_m meters east and offsetY_m meters north
// (used to correct minor geometric offset between SAR and optical basemap)
function shiftImage(image, offsetX_m, offsetY_m) {
  var displacement = ee.Image.constant(offsetX_m).rename('displacementX')
                       .addBands(ee.Image.constant(offsetY_m).rename('displacementY'));
  return image.displace(displacement);
}

// Returns the volume-to-area coefficients for the given lake name,
// using custom values if the user has enabled that option
function getCoefficientsForLake(lakeName) {
  if (c.customCoeff.checkbox.getValue()) {
    var aVal = parseFloat(c.customCoeff.a.getValue());
    var bVal = parseFloat(c.customCoeff.b.getValue());
    var cVal = parseFloat(c.customCoeff.c.getValue());
    if (isNaN(aVal) || isNaN(bVal) || isNaN(cVal)) {
      ui.alert('Invalid coefficients. Please enter valid numbers for a, b, and c.');
      throw new Error('Invalid custom coefficients');
    }
    return {a: aVal, b: bVal, c: cVal};
  }
  var coeffs = lakeCoefficients[lakeName];
  if (!coeffs) {
    ui.alert('Coefficients for "' + lakeName + '" not found. Consider using custom coefficients.');
    throw new Error('Missing lake coefficients');
  }
  return coeffs;
}


// ---- SENTINEL-1 TRAINING ----

var collectionT = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filterMetadata('resolution_meters', 'equals', 10)
  .filterBounds(roi_training)
  .select('VV', 'VH');

var imagesT = collectionT.filterDate('2023-01-01', '2024-01-01').mosaic();
var imagesT_filtered = imagesT.focal_mean(30, 'circle', 'meters');
var collectionT_combined = ee.Image.cat(imagesT_filtered).clip(roi_training);

// Merge water and land training samples
var newfc = agua_training.merge(terra_training);

var bands = ['VV', 'VH'];

var training = collectionT_combined.select(bands).sampleRegions({
  collection: newfc,
  properties: ['landcover'],
  scale: 30
});

// Clean training data: keep only valid classes and non-null values
var trainingClean = training
  .filter(ee.Filter.inList('landcover', ee.List([1, 2])))
  .filter(ee.Filter.notNull(['landcover']));


// ---- CLASSIFIER TRAINING (SVM — RBF kernel) ----

var classifierSVM = ee.Classifier.libsvm({
  kernelType: 'RBF',
  cost: 1,
  gamma: 0.01
}).train({
  features: trainingClean,
  classProperty: 'landcover',
  inputProperties: bands
});

var smoothingRadius = 30;


// ---- CLASSIFICATION FUNCTIONS ----

// Classifies a single image within the given AOI
function classifyImage(image, bands, classifierSVM, smoothingRadius, aoi) {
  var clipped  = image.clip(aoi);
  var filtered = clipped.focal_mean(smoothingRadius, 'circle', 'meters');
  var classified = filtered.select(bands).classify(classifierSVM);
  var waterMask  = classified.eq(1);
  return image
    .addBands(classified.rename('ClassificationSVM'))
    .addBands(waterMask.rename('Water'));
}

// Maps classifyImage over an entire image collection
function classifyCollection(colecao, bands, classifierSVM, smoothingRadius, aoi) {
  return colecao.map(function(image) {
    return classifyImage(image, bands, classifierSVM, smoothingRadius, aoi);
  });
}

// Computes water surface area (hectares) for a single classified image
function calculateAreaLake(image, aoi) {
  var waterMaskCleaned = image.select('WaterCleaned').clip(aoi);
  var areaPixelsSar = waterMaskCleaned.multiply(ee.Image.pixelArea())
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: aoi,
      scale: 10,
      maxPixels: 1e9
    }).get('WaterCleaned');
  var areaHectaresSar = ee.Number(areaPixelsSar).divide(10000);
  return ee.Feature(null, {
    'system:time_start': image.date().millis(),
    'data': image.date().format('YYYY-MM-dd'),
    'areaLago': areaHectaresSar
  });
}

// Maps calculateAreaLake over an entire collection to build the time series
function calculateTimeSeries(colecao, aoi) {
  return colecao.map(function(image) {
    return calculateAreaLake(image, aoi);
  });
}


// ---- OUTLIER REMOVAL AND SMOOTHING ----

// Removes global outliers (beyond `threshold` standard deviations from the mean)
function removeOutliers(featureCollection, threshold) {
  var areas  = featureCollection.aggregate_array('areaLago');
  var mean   = areas.reduce(ee.Reducer.mean());
  var stdDev = areas.reduce(ee.Reducer.stdDev());
  return featureCollection.map(function(feature) {
    var area = ee.Number(feature.get('areaLago'));
    var deviation = area.subtract(mean).abs().divide(stdDev);
    return feature.set('isOutlier', deviation.gt(threshold));
  }).filter(ee.Filter.eq('isOutlier', 0));
}

// Removes local outliers using a sliding window of `windowSize` samples
function detectAndRemoveLocalOutliers(fc, windowSize, stdDevThreshold) {
  var features = fc.sort('system:time_start').toList(fc.size());
  var outlierFlags = ee.List.sequence(0, features.size().subtract(1)).map(function(i) {
    var index   = ee.Number(i);
    var current = ee.Feature(features.get(index));
    var currentValue = ee.Number(current.get('areaLago'));
    var halfWindow = ee.Number(windowSize).divide(2).int();
    var start  = index.subtract(halfWindow).max(0);
    var end    = index.add(halfWindow).min(features.size());
    var window = ee.FeatureCollection(features.slice(start, end));
    var mean   = window.aggregate_mean('areaLago');
    var stddev = window.aggregate_total_sd('areaLago');
    var deviation = currentValue.subtract(mean).abs().divide(stddev);
    return ee.Feature(current).set('isOutlierLocal', deviation.gt(stdDevThreshold));
  });
  return ee.FeatureCollection(outlierFlags).filter(ee.Filter.eq('isOutlierLocal', 0));
}

// Applies LOWESS smoothing using a Gaussian-weighted temporal window
function lowessSmoothing(fc, windowDays, bandwidth) {
  var features = fc.sort('system:time_start').toList(fc.size());
  return ee.FeatureCollection(ee.List.sequence(0, features.size().subtract(1)).map(function(i) {
    var currentFeature = ee.Feature(features.get(i));
    var currentDate    = ee.Date(currentFeature.get('system:time_start'));
    var windowStart    = currentDate.advance(-windowDays, 'day');
    var windowEnd      = currentDate.advance(windowDays,  'day');
    var neighbors = ee.FeatureCollection(features).filter(ee.Filter.date(windowStart, windowEnd));
    var weighted  = neighbors.map(function(f) {
      var fDate  = ee.Date(f.get('system:time_start'));
      var diffDays = currentDate.difference(fDate, 'day').abs();
      var weight   = diffDays.divide(bandwidth).pow(2).multiply(-1).exp();
      var value    = ee.Number(f.get('areaLago'));
      return f.set({'weight': weight, 'weightedValue': value.multiply(weight)});
    });
    var weightSum        = weighted.aggregate_sum('weight');
    var weightedValueSum = weighted.aggregate_sum('weightedValue');
    var smoothed = ee.Number(weightedValueSum).divide(weightSum);
    return currentFeature.set('areaLago_smoothed', smoothed);
  }));
}

// Computes reservoir volume from the smoothed area time series
// using V = a*A² + b*A + c  (A in hectares, V in 10^6 m³)
function addVolumeToTimeSeries(lakeTimeSeries, coefficients) {
  return lakeTimeSeries.map(function(feature) {
    var area   = ee.Number(feature.get('areaLago_smoothed'));
    var volume = area.pow(2).multiply(coefficients.a)
                   .add(area.multiply(coefficients.b))
                   .add(coefficients.c)
                   .max(0);
    return feature.set('volume', volume);
  });
}


// ---- INCIDENT ANGLE FILTERING ----

// Adds mean incidence angle statistics to each image in the collection
function IncidentAngleAnalysis(collection, aoi) {
  return collection.map(function(image) {
    var stats = image.select('angle').reduceRegion({
      reducer: ee.Reducer.minMax()
                .combine(ee.Reducer.mean(),   '', true)
                .combine(ee.Reducer.stdDev(), '', true),
      geometry: aoi,
      scale: 10,
      maxPixels: 1e9
    });
    return image.set({
      'angle_min':    stats.get('angle_min'),
      'angle_max':    stats.get('angle_max'),
      'angle_mean':   stats.get('angle_mean'),
      'angle_stdDev': stats.get('angle_stdDev')
    });
  });
}

// Filters to images whose mean incidence angle falls in the highest-coverage
// 3-degree bin, iterating from high to low angles.
// Ensures ≥90% spatial coverage of the AOI; falls back to full collection if needed.
function PrioritizeDescendingAngleBins(collection, aoi, callback) {
  var aoiArea = aoi.area();

  var withAngle = collection.map(function(image) {
    var angle = image.select('angle');
    var stats = angle.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: aoi,
      scale: 10,
      maxPixels: 1e9
    });
    var meanAngle = ee.Number(stats.get('angle'));
    return ee.Algorithms.If(
      meanAngle,
      image.set('angle_mean', meanAngle),
      image
    );
  }).filter(ee.Filter.notNull(['angle_mean']));

  var angleList = withAngle.aggregate_array('angle_mean').distinct().sort().reverse();

  function tryNextBin(angleList, i) {
    var current = ee.Number(angleList.get(i));
    var next    = current.subtract(3);
    var binFiltered = withAngle.filter(ee.Filter.and(
      ee.Filter.gte('angle_mean', next),
      ee.Filter.lt('angle_mean', current)
    ));

    var withCoverage = binFiltered.map(function(image) {
      var mask = image.select('VV').mask();
      var coveredArea = mask.multiply(ee.Image.pixelArea())
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi, scale: 10, maxPixels: 1e9})
        .get('VV');
      var percentCovered = ee.Number(coveredArea).divide(aoiArea).multiply(100);
      return image.set('percentCovered', percentCovered);
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
    if (list.length > 1) {
      tryNextBin(ee.List(list), 0);
    } else {
      callback(withAngle.sort('system:time_start'));
    }
  });
}


// ---- TIMELAPSE ----

// Generates and displays an animated GIF timelapse of the classified water mask
function gerarTimelapse() {
  var nomeReservatorio = getActiveName();
  if (!nomeReservatorio) {
    ui.alert('Please select a reservoir first.');
    return;
  }

  var aoi = getActiveAOI();

  if (!globalCollectionSAR || !waterMaskCleaned) {
    ui.alert('Click "RUN!" first to generate the image collection.');
    return;
  }

  var offsetX_m = 18.393;
  var offsetY_m = -7.035;

  var imagens = waterMaskCleaned.map(function(img) {
    var dateStr = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');

    var sarOriginal   = img.select('VV').clip(aoi);
    var waterOriginal = img.select('WaterCleaned').clip(aoi);

    var sarShiftedImage   = shiftImage(sarOriginal,   offsetX_m, offsetY_m);
    var waterShiftedImage = shiftImage(waterOriginal, offsetX_m, offsetY_m);

    var agua = waterShiftedImage.selfMask().visualize({palette: ['0000ff'], opacity: 0.6, min: 1, max: 1});
    var fundo = sarShiftedImage.visualize({min: -24, max: 0, palette: ['black', 'white']});
    var base  = ee.ImageCollection([fundo, agua]).mosaic();

    var centro = aoi.centroid(1);
    var coords = centro.coordinates();

    var posLegenda = ee.Geometry.Point([coords.get(0), ee.Number(coords.get(1)).add(0.003)]);
    var posData    = ee.Geometry.Point([ee.Number(coords.get(0)).add(0.002), coords.get(1)]);

    var dataLayer = ee.Image().byte().paint({
      featureCollection: ee.FeatureCollection([ee.Feature(posData, {label: dateStr})]),
      color: 1
    }).visualize({palette: ['white'], opacity: 1});

    var legendaLayer = ee.Image().byte().paint({
      featureCollection: ee.FeatureCollection([
        ee.Feature(posLegenda, {label: 'SAR VV + Water Mask'})
      ]),
      color: 1
    }).visualize({palette: ['yellow'], opacity: 1});

    return ee.ImageCollection([base, dataLayer, legendaLayer]).mosaic()
      .set('system:time_start', img.get('system:time_start'));
  });

  var firstDate = ee.Date(imagens.first().get('system:time_start')).format('YYYY-MM-dd');
  var lastDate  = ee.Date(imagens.sort('system:time_start', false).first()
                    .get('system:time_start')).format('YYYY-MM-dd');

  ee.List([firstDate, lastDate]).evaluate(function(dates) {
    var gif = ui.Thumbnail({
      image: imagens.sort('system:time_start'),
      params: {region: aoi, dimensions: 512, framesPerSecond: 2, format: 'gif'},
      style: {width: '300px', border: '1px solid black'}
    });

    var infoLabel = ui.Label({
      value: nomeReservatorio + '\nFrom ' + dates[0] + ' to ' + dates[1],
      style: {fontWeight: 'bold', fontSize: '14px', whiteSpace: 'pre', padding: '8px'}
    });

    var timelapsePanel = ui.Panel({
      widgets: [infoLabel, gif],
      layout: ui.Panel.Layout.Flow('vertical'),
      style: {
        position: 'bottom-left',
        width: '300px',
        border: '1px solid black',
        backgroundColor: 'rgba(0, 0, 0, 0.9)'
      }
    });

    if (c.thumbnailPanel) c.map.remove(c.thumbnailPanel);
    c.thumbnailPanel = timelapsePanel;
    c.map.add(c.thumbnailPanel);
  });
}


// ---- GLOBAL STATE ----

var globalCollectionSAR;
var waterMaskCleaned;


// ---- MAIN PROCESSING PIPELINE ----

// Runs the full classification, time series, and charting pipeline for the selected AOI
function executeMainPipeline(collection, aoi, lakeName, coefficients) {

  // Classify the SAR collection
  var classifiedCollection = classifyCollection(
    globalCollectionSAR, bands, classifierSVM, smoothingRadius, aoi
  );

  // Extract water pixels from the classification
  var waterCollection = classifiedCollection.map(function(image) {
    var waterOnly = image.select('ClassificationSVM')
      .updateMask(image.select('ClassificationSVM').eq(1));
    return image.addBands(waterOnly.rename('Water'));
  });

  var waterMaskRefined = waterCollection;

  // Apply morphological gap-filling to close small holes in the water mask
  var waterMaskFilled = waterMaskRefined.map(function(image) {
    var waterMask    = image.select('Water');
    var waterMaskAOI = waterMask.unmask(0).clip(aoi);
    var distance     = waterMaskAOI.fastDistanceTransform(30).clip(aoi);
    var filledMask   = distance.lte(0.5).updateMask(distance.lte(0.5));
    var finalMask    = filledMask.where(waterMaskAOI, 1);
    return image.addBands(finalMask.rename('WaterFilled'));
  });

  // Retain only the largest connected water region per image
  waterMaskCleaned = waterMaskFilled.map(function(image) {
    var waterMask     = image.select('WaterFilled');
    var waterPolygons = waterMask.reduceToVectors({
      geometryType: 'polygon',
      reducer: ee.Reducer.countEvery(),
      scale: 10,
      maxPixels: 1e9,
      bestEffort: true
    });
    var waterWithArea = waterPolygons.map(function(feature) {
      return feature.set('area', feature.geometry().area({maxError: 1}));
    });
    var largestPolygon = waterWithArea.sort('area', false).first();
    var largestPolygonMask = ee.Image().paint({
      featureCollection: ee.FeatureCollection([largestPolygon]),
      color: 1
    }).rename('LargestRegionMask');
    var cleanedWaterMask = waterMask.updateMask(largestPolygonMask);
    return image.addBands(cleanedWaterMask.rename('WaterCleaned'));
  });

  // Build water surface area time series
  var lakeTimeSeries = calculateTimeSeries(waterMaskCleaned, aoi);

  // Remove outliers and apply LOWESS smoothing
  var ts1 = removeOutliers(lakeTimeSeries, 2);
  var ts2 = detectAndRemoveLocalOutliers(ts1, 5, 1.5);
  var ts3 = detectAndRemoveLocalOutliers(ts2, 5, 1.5);
  var ts4 = detectAndRemoveLocalOutliers(ts3, 10, 1.5);
  lakeTimeSeries = lowessSmoothing(ts4, 20, 7);

  // Compute volume from smoothed area
  var volumeTimeSeries = addVolumeToTimeSeries(lakeTimeSeries, coefficients);

  // Helper: load the selected image onto the map
  function loadImageOnMap(imagemSelecionada, aoi) {
    imagemSelecionada.evaluate(function(img) {
      if (img) {
        ee.Image(imagemSelecionada).date().format('YYYY-MM-dd').evaluate(function(dateStr) {
          c.dataAtual.label.setValue('Date: ' + dateStr);
        });
        c.map.layers().reset();
        var sarOriginal   = imagemSelecionada.select('VV').clip(aoi);
        var waterOriginal = imagemSelecionada.select('WaterCleaned');
        c.map.addLayer(shiftImage(sarOriginal,   18.393, -7.035), visParamSAR,   'SAR VV');
        c.map.addLayer(shiftImage(waterOriginal, 18.393, -7.035), visParamWater, 'Water');
        c.selectedImage = imagemSelecionada;
        c.exportButton.style().set('shown', true);
      } else {
        ui.alert('No image found for the selected date.');
      }
    });
  }

  // Area chart
  var areaFigure = ui.Chart.feature.byFeature(lakeTimeSeries, 'data', ['areaLago', 'areaLago_smoothed'])
    .setOptions({
      title: lakeName + ' — Water Surface Area',
      titleTextStyle: {fontSize: 16},
      hAxis: {title: 'Date', format: 'YYYY-MM-dd'},
      vAxis: {title: 'Area (ha)'},
      lineWidth: 1,
      pointSize: 4,
      series: {
        0: {color: 'blue', label: 'Original area'},
        1: {color: 'red',  label: 'Smoothed area'}
      }
    });

  areaFigure.onClick(function(xValue) {
    if (!xValue) return;
    var imagemSelecionada = waterMaskCleaned
      .filter(ee.Filter.date(ee.Date(xValue).advance(0, 'day'), ee.Date(xValue).advance(1, 'day')))
      .first();
    loadImageOnMap(imagemSelecionada, aoi);
  });

  // Volume chart
  var volumeChart = ui.Chart.feature.byFeature(volumeTimeSeries, 'data', ['volume'])
    .setOptions({
      title: lakeName + ' — Volume',
      titleTextStyle: {fontSize: 16},
      hAxis: {title: 'Date'},
      vAxis: {title: 'Volume (10⁶ m³)'},
      series: {0: {color: 'blue', label: 'Volume (10⁶ m³)'}}
    });

  volumeChart.onClick(function(xValue) {
    if (!xValue) return;
    var imagemSelecionada = waterMaskCleaned
      .filter(ee.Filter.date(ee.Date(xValue).advance(-1, 'day'), ee.Date(xValue).advance(1, 'day')))
      .first();
    loadImageOnMap(imagemSelecionada, aoi);
  });

  // Render results in the control panel
  c.graphicResults.panel.clear();
  c.graphicResults.panel.add(areaFigure);
  c.graphicResults.panel.add(c.exportButton);
  c.graphicResults.panel.add(c.exportMessage);
  c.graphicResults.panel.add(volumeChart);
  c.graphicResults.panel.add(c.graphicResults.disclaimer);

  // Show the most recent image by default
  globalCollectionSAR.size().evaluate(function(size) {
    if (size > 0) atualizarImagem(size - 1);
  });

  aoi.area().evaluate(function(area) {
    var zoomLevel = area > 5e8 ? 10 : area > 1e8 ? 13 : 14;
    c.map.centerObject(aoi, zoomLevel);
  });
}


// Triggered by RUN! — filters the Sentinel-1 collection and launches the pipeline
function atualizarMapa() {
  var isCustomActive = c.toggleCustom.getValue(); // Check the toggle state
  var aoi;
  var lakeName;

  // --- STEP 1: EXCLUSIVE GEOMETRY SELECTION ---
  if (isCustomActive) {
    // Look for drawings on the map
    var drawingLayers = c.map.drawingTools().layers();
    if (drawingLayers.length() > 0 && drawingLayers.get(0).geometries().length() > 0) {
      aoi = drawingLayers.get(0).getEeObject();
      lakeName = 'Custom_AOI'; // Generic name for custom areas
    } else {
      ui.alert('Custom Mode Active: Please draw a polygon on the map before running.');
      return; // Stop execution
    }
  } else {
    // Use the Predefined Selector (Sicilian Lakes)
    lakeName = c.selectAOI.selector.getValue();
    if (lakeName) {
      aoi = m.AOI[lakeName];
    } else {
      ui.alert('Preset Mode Active: Please select a reservoir from the list.');
      return; // Stop execution
    }
  }

  // --- STEP 2: VALIDATE DATES ---
  var startDate = c.selectDateInitial.textbox.getValue();
  var endDate   = c.selectDateFinal.textbox.getValue();

  if (!startDate || !endDate) {
    ui.alert('Please define the analysis period (Start and End dates).');
    return;
  }

  // --- STEP 3: UI CLEANUP & ZOOM ---
  if (c.thumbnailPanel) {
    c.map.remove(c.thumbnailPanel);
    c.thumbnailPanel = null;
  }

  aoi.area().evaluate(function(area) {
    var zoomLevel = area > 5e8 ? 10 : area > 1e8 ? 13 : 14;
    c.map.centerObject(aoi, zoomLevel);
  });

  // --- STEP 4: PROCEED TO PIPELINE ---
  var collectionRaw = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterDate(startDate, endDate)
    .filterBounds(aoi)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  var loadingMsg = showMapLoadingMessage('Loading SAR images', 400);

  PrioritizeDescendingAngleBins(collectionRaw, aoi, function(collectionFinal) {
    loadingMsg.stop();
    globalCollectionSAR = collectionFinal;
    
    try {
      // If custom, this will use generic/zero coefficients unless you've defined them
      var coeffs = getCoefficientsForLake(lakeName);
    } catch (e) {
      // Custom AOIs might not have coefficients in your Model, handle gracefully
      var coeffs = {a: 0, b: 0}; 
    }
    
    executeMainPipeline(globalCollectionSAR, aoi, lakeName, coeffs);
  });
}

// Displays the image at the given index from the processed collection
function atualizarImagem(indice) {
  globalCollectionSAR.size().evaluate(function(size) {
    if (size === 0) {
      ui.alert('No images available in the selected area.');
      return;
    }
    indice = Math.min(indice, size - 1);
    var imagem = ee.Image(waterMaskCleaned.toList(size).get(indice));
    imagem.date().format('YYYY-MM-dd').evaluate(function(dateStr) {
      c.dataAtual.label.setValue('Date: ' + dateStr);
    });
    var aoi = getActiveAOI();
    c.map.layers().reset();
    c.map.addLayer(imagem.select('VV').clip(aoi),       visParamSAR,   'SAR VV');
    c.map.addLayer(imagem.select('WaterCleaned').clip(aoi), visParamWater, 'Water');
  });
}


/*******************************************************************************
 * INITIALIZATION
 *
 * Sets up reservoir markers and map click interaction on app load.
 ******************************************************************************/

// Build centroid markers for all reservoirs
var reservoirMarkerList = Object.keys(m.AOI).map(function(name) {
  return ee.Feature(m.AOI[name].centroid(1), {'name': name});
});
var reservoirMarkers = ee.FeatureCollection(reservoirMarkerList);

c.map.layers().add(ui.Map.Layer(
  reservoirMarkers.style({color: 'cyan', pointSize: 6, width: 1}),
  {}, 'Reservoirs', true
));

// On map click: find the nearest reservoir within 5 km and select it
c.map.onClick(function(coords) {
  var clickedPoint = ee.Geometry.Point([coords.lon, coords.lat]);

  if (c.thumbnailPanel) {
    c.map.remove(c.thumbnailPanel);
    c.thumbnailPanel = null;
  }

  var maxDistance = 5000;

  var nearestReservoir = reservoirMarkers
    .map(function(f) {
      return f.set('distance', f.geometry().distance(clickedPoint));
    })
    .sort('distance')
    .first();

  nearestReservoir.evaluate(function(feature) {
    if (feature && feature.properties && feature.properties.name) {
      var distance = feature.properties.distance;
      if (distance < maxDistance) {
        var selectedName = feature.properties.name;
        c.selectAOI.selector.setValue(selectedName, true);

        var reservoirGeometry = m.AOI[selectedName];
        c.map.centerObject(reservoirGeometry, 12);

        if (c.aoiOutlineLayer) {
          c.map.layers().remove(c.aoiOutlineLayer);
        }
        c.aoiOutlineLayer = ui.Map.Layer(
          ee.Feature(reservoirGeometry),
          {color: 'orange', fillColor: '00000000', width: 1, opacity: 0.6},
          'Reservoir outline'
        );
        c.map.layers().add(c.aoiOutlineLayer);

        c.labelNomeReservatorio.setValue('📍 ' + selectedName);
      }
    }
  });
});
