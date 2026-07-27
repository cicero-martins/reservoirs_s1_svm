"""
upload_planet_complementary.py  (one-off utility, 2026-07-24)

Uploads the newly-downloaded complementary PlanetScope scenes
(raw_data/Planet_complementaryImages/<site>_lastDrawdown_psscene_analytic_sr_udm2/
PSScene/*_3B_AnalyticMS_SR_clip.tif) to a GCS staging bucket, then ingests each as
an Earth Engine image asset under projects/ee-ciceromartinsjr/assets/planet/<site>/,
matching the naming convention of the existing assets there (asset id = filename
stem, no extension).

Only the *_3B_AnalyticMS_SR_clip.tif files are uploaded (matching what's already in
the asset folders -- UDM2 masks and metadata are not assets there).
"""
import json, pathlib, sys

import truststore
truststore.inject_into_ssl()

import ee
import ee.oauth as oauth
from google.oauth2.credentials import Credentials
from google.cloud import storage

try:
    ee.Initialize(project='ee-ciceromartinsjr')
except Exception:
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='ee-ciceromartinsjr')

REPO = pathlib.Path('.')
LOCAL_ROOT = REPO / 'raw_data' / 'Planet_complementaryImages'
BUCKET = 'ee-ciceromartinsjr-planet-staging'
ASSET_ROOT = 'projects/ee-ciceromartinsjr/assets/planet'

SITE_FOLDERS = {
    'ancipa_lastDrawdown_psscene_analytic_sr_udm2': 'ancipa',
    'poma_lastDrawdown_psscene_analytic_sr_udm2': 'poma',
    'rosamarina_lastDrawdown_psscene_analytic_sr_udm2': 'rosamarina',
}


def gcs_client():
    cred_path = pathlib.Path.home() / '.config' / 'earthengine' / 'credentials'
    d = json.loads(cred_path.read_text())
    creds = Credentials(
        token=None, refresh_token=d['refresh_token'],
        token_uri='https://oauth2.googleapis.com/token',
        client_id=oauth.CLIENT_ID, client_secret=oauth.CLIENT_SECRET,
        scopes=d['scopes'],
    )
    return storage.Client(project='ee-ciceromartinsjr', credentials=creds)


def main():
    client = gcs_client()
    bucket = client.bucket(BUCKET)

    plan = []  # (local_path, gcs_uri, asset_id)
    for folder, site in SITE_FOLDERS.items():
        src_dir = LOCAL_ROOT / folder / 'PSScene'
        tifs = sorted(src_dir.glob('*_3B_AnalyticMS_SR_clip.tif'))
        for tif in tifs:
            stem = tif.stem
            gcs_uri = f'gs://{BUCKET}/{site}/{tif.name}'
            asset_id = f'{ASSET_ROOT}/{site}/{stem}'
            plan.append((tif, gcs_uri, asset_id, site))

    print(f'{len(plan)} files to upload/ingest:')
    for tif, gcs_uri, asset_id, site in plan:
        print(f'  [{site}] {tif.name} -> {asset_id}')

    print('\n--- Uploading to GCS ---')
    for tif, gcs_uri, asset_id, site in plan:
        blob_path = gcs_uri.replace(f'gs://{BUCKET}/', '')
        blob = bucket.blob(blob_path)
        if blob.exists():
            print(f'  already in GCS: {gcs_uri}')
            continue
        blob.upload_from_filename(str(tif))
        print(f'  uploaded: {gcs_uri}')

    print('\n--- Ingesting as EE assets (in-process, avoids the earthengine CLI\'s ---')
    print('--- separate Python process not inheriting the truststore SSL patch) ---')
    for tif, gcs_uri, asset_id, site in plan:
        manifest = {
            'name': asset_id,
            'tilesets': [{'sources': [{'uris': [gcs_uri]}]}],
        }
        task_id = ee.data.newTaskId()[0]
        try:
            result = ee.data.startIngestion(task_id, manifest, allow_overwrite=True)
            print(f'  [{site}] {asset_id} -> task {result.get("id", task_id)}')
        except Exception as e:
            print(f'  [{site}] {asset_id} FAILED: {e}')

    print('\nDone. Check task status with: earthengine task list')


if __name__ == '__main__':
    main()
