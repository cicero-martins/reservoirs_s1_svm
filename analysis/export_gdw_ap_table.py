"""
Server-side export of A/P (area-to-perimeter ratio, metres) for every GDW v1.0
reservoir polygon, computed entirely inside Earth Engine (no local download,
no GCS bucket needed) via Export.table.toAsset.

Why this exists: the GDW_RESERVOIRS_V1_0 EE asset has AREA_SKM but no
perimeter field, so gee_reservoir_monitor_app.js could not colour the
initial "Reservoirs" map layer by A/P band without computing
.geometry().perimeter() live for all 35,295 features on every app load --
too slow/likely to time out interactively. This script precomputes it once;
the app then joins against the small output table (GDW_ID, ap_m) at load
time instead.

Note: A/P here is derived from the GDW polygon itself, not the JRC
max_extent polygon used for the paper's 62-reservoir study set -- the two
sources delineate shorelines independently and can differ slightly.
"""
import truststore
truststore.inject_into_ssl()
import ee

ee.Initialize(project='ee-ciceromartinsjr')

ASSET_ID = 'projects/ee-ciceromartinsjr/assets/GDW_reservoirs_ap'

gdw = ee.FeatureCollection('projects/sat-io/open-datasets/GDW/GDW_RESERVOIRS_V1_0')


def add_ap(f):
    geom = f.geometry()
    area_m2 = geom.area(1)
    perim_m = geom.perimeter(1)
    ap_m = area_m2.divide(perim_m)
    # Export.table.toAsset requires a geometry per feature; a centroid point
    # keeps the asset small while satisfying that -- the app only needs the
    # GDW_ID -> ap_m property join, not this table's geometry.
    return ee.Feature(geom.centroid(1), {
        'GDW_ID': f.get('GDW_ID'),
        'ap_m': ap_m,
    })


ap_table = gdw.map(add_ap)

task = ee.batch.Export.table.toAsset(
    collection=ap_table,
    description='export_gdw_ap_table',
    assetId=ASSET_ID,
)
task.start()
print(f"Submitted task: {task.id}")
print(f"Target asset: {ASSET_ID}")
print("Poll status with check_gdw_ap_task.py or the EE Tasks tab.")
