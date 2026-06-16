#earthengine authenticate --auth_mode=notebook

"""
Submit all 48 GEE export tasks programmatically via Earth Engine Python API.
Equivalent to running exportGlobalPilot.js and clicking Run on all 48 tasks.

Usage:
  pip install earthengine-api
  earthengine authenticate          # only first time
  python analysis/submit_pilot_tasks.py
"""
import ee

try:
    ee.Initialize(project='ee-ciceromartinsjr')
except Exception:
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='ee-ciceromartinsjr')

# ── Pilot reservoir list ──────────────────────────────────────
PILOT = [
    # [RES_ID, Hylak_ID, name]
    [3345,  15647,    'Manikdoh'          ],
    [3344,  15553,    'Panam'             ],
    [3439,  15840,    'Periyar'           ],
    [3438,  179882,   'Kakki'             ],
    [3533,  15600,    'Pench_Totladoh'    ],
    [3402,  15105,    'Thein_Ranjit_Sagar'],
    [2332,  172148,   'San_Esteban_ES'   ],
    [2760,  1338571,  'Ry_de_Rome_BE'    ],
    [2376,  172818,   'Borbollon_ES'     ],
    [2405,  173730,   'Minilla_ES'       ],
    [2408,  14552,    'Gabriel_y_Galan_ES'],
    [2418,  14457,    'Ricobayo_ES'      ],
    [1240,  8630,     'Pine_River_US'    ],
    [1164,  9794,     'Benito_Juarez_MX' ],
    [1268,  105914,   'Pokegama_US'      ],
    [1244,  735,      'Winnibigoshish_US'],
    [1233,  738,      'Leech_US'         ],
    [ 550,  831,      'Elephant_Butte_US'],
    [3938,  184942,   'Maroondah_AU'     ],
    [3941,  1426822,  'OShannassy_AU'    ],
    [3936,  184949,   'Silvan_AU'        ],
    [3911,  184923,   'Upper_Coliban_AU' ],
    [4089,  184368,   'Chichester_AU'    ],
    [3942,  184943,   'Yarra_AU'         ],
]

hydroLakes = ee.FeatureCollection('projects/ee-ciceromartinsjr/assets/HydroLAKES_pilot24')
START = '2014-01-01'
END   = '2025-06-01'


def auto_training_samples(aoi):
    occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
    water90 = (occ.gte(90).selfMask()
               .sample(region=aoi, scale=30, numPixels=500, seed=42, geometries=True)
               .map(lambda f: f.set('landcover', 1)))
    water50 = (occ.gte(50).selfMask()
               .sample(region=aoi, scale=30, numPixels=500, seed=42, geometries=True)
               .map(lambda f: f.set('landcover', 1)))
    water = ee.FeatureCollection(ee.Algorithms.If(water90.size().gt(10), water90, water50))
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    land_mask  = (worldcover.neq(80)
                  .And(worldcover.neq(90))
                  .And(worldcover.neq(95))
                  .And(occ.unmask(0).eq(0))
                  .selfMask())
    land = (land_mask
            .sample(region=aoi.buffer(2000), scale=10, numPixels=500, seed=42, geometries=True)
            .map(lambda f: f.set('landcover', 2)))
    return water.merge(land)


def export_reservoir(res_id, hylak_id, name):
    feat    = hydroLakes.filter(ee.Filter.eq('Hylak_id', hylak_id)).first()
    aoi     = feat.geometry()
    area_m2 = aoi.area(1)
    perim_m = aoi.perimeter(1)
    ap_m    = area_m2.divide(perim_m)

    # ── SAR SVM ────────────────────────────────────────────────
    s1raw = (ee.ImageCollection('COPERNICUS/S1_GRD')
             .filterDate(START, END)
             .filterBounds(aoi)
             .filter(ee.Filter.eq('instrumentMode', 'IW'))
             .filter(ee.Filter.eq('resolution_meters', 10))
             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
             .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    def add_angle(img):
        mean = img.select('angle').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e6
        ).get('angle')
        return img.set('mean_angle', mean)

    with_angle  = s1raw.map(add_angle)
    valid_angle = with_angle.filter(ee.Filter.notNull(['mean_angle']))
    descending  = valid_angle.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    use_col     = ee.ImageCollection(
        ee.Algorithms.If(descending.size().gt(0), descending, valid_angle)
    )
    best       = use_col.sort('mean_angle').first()
    target_ang = best.getNumber('mean_angle')
    s1 = (use_col
          .filter(ee.Filter.And(
              ee.Filter.gte('mean_angle', target_ang.subtract(3)),
              ee.Filter.lte('mean_angle', target_ang.add(3))
          ))
          .select('VV', 'VH'))

    s1composite  = s1.median()
    sample_pts   = auto_training_samples(aoi)
    training     = s1composite.sampleRegions(
        collection=sample_pts, properties=['landcover'], scale=10, tileScale=4
    )
    classifier = (ee.Classifier.libsvm(kernelType='RBF', cost=1, gamma=0.01,
                                        decisionProcedure='Margin')
                  .train(features=training, classProperty='landcover',
                         inputProperties=['VV', 'VH']))

    def classify_img(img):
        smooth  = img.focal_mean(30, 'square', 'meters')
        water   = smooth.classify(classifier).eq(1)
        area_ha = (water.multiply(ee.Image.pixelArea())
                   .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi,
                                 scale=10, maxPixels=1e9)
                   .get('classification'))
        return ee.Feature(None, {
            'date':         img.date().format('YYYY-MM-dd'),
            'area_ha':      ee.Number(area_ha).divide(10000),
            'ap_m':         ap_m,
            'area_aoi_ha':  area_m2.divide(10000),
            'perim_aoi_m':  perim_m,
            'n_water_pts':  training.filter(ee.Filter.eq('landcover', 1)).size(),
            'n_land_pts':   training.filter(ee.Filter.eq('landcover', 2)).size(),
            'hylak_id':     hylak_id,
            'growl_res_id': res_id,
            'reservoir':    name,
        })

    classified = s1.map(classify_img)
    task_sar = ee.batch.Export.table.toDrive(
        collection=classified,
        description=f'SAR_area_{name}_{res_id}',
        folder='GROWL_SAR_pilot',
        fileFormat='CSV'
    )
    task_sar.start()
    print(f'  ✓ SAR_area_{name}_{res_id}')

    # ── JRC monthly optical area ───────────────────────────────
    def jrc_feat(img):
        wc = img.select('water')
        water_ha = (wc.eq(2).multiply(ee.Image.pixelArea())
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi,
                                  scale=30, maxPixels=1e9)
                    .get('water'))
        valid_frac = wc.gt(0).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e9
        ).get('water')
        return ee.Feature(None, {
            'date':         img.date().format('YYYY-MM-dd'),
            'jrc_area_ha':  ee.Number(water_ha).divide(10000),
            'valid_frac':   valid_frac,
            'ap_m':         ap_m,
            'area_aoi_ha':  area_m2.divide(10000),
            'hylak_id':     hylak_id,
            'growl_res_id': res_id,
            'reservoir':    name,
        })

    jrc_col = (ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
               .filterDate(START, END)
               .filterBounds(aoi)
               .map(jrc_feat))

    task_jrc = ee.batch.Export.table.toDrive(
        collection=jrc_col,
        description=f'JRC_area_{name}_{res_id}',
        folder='GROWL_SAR_pilot',
        fileFormat='CSV'
    )
    task_jrc.start()
    print(f'  ✓ JRC_area_{name}_{res_id}')


# ── Submit all ────────────────────────────────────────────────
submitted = 0
for p in PILOT:
    res_id, hylak_id, name = p
    print(f'\n[{name}]')
    export_reservoir(res_id, hylak_id, name)
    submitted += 2

print(f'\n{"="*55}')
print(f'Tasks submetidas: {submitted}  →  Google Drive / GROWL_SAR_pilot/')
print('Acompanhe o progresso em: https://code.earthengine.google.com/tasks')
