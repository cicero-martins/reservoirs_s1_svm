"""
download_planet_masks_drive.py  (one-off utility, 2026-07-24)

Downloads the newly-exported PlanetScope NDWI water masks (queued by
exportSicilyPlanetMasks_standalone.js -> Drive/GEE_SicilyPlanetMasks) into
raw_data/GEE_SicilyPlanetMasks/, matching the existing local naming
convention (mask_Planet_<Site>_<YYYY-MM-DD>.tif).

Reuses the earthengine OAuth credential (scopes already include
https://www.googleapis.com/auth/drive) to authenticate the Drive API --
same trick as upload_planet_complementary.py's GCS client.
"""
import json, pathlib

import truststore
truststore.inject_into_ssl()

from google.oauth2.credentials import Credentials
import ee.oauth as oauth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

REPO = pathlib.Path('.')
OUT_DIR = REPO / 'raw_data' / 'GEE_SicilyPlanetMasks'
DRIVE_FOLDER_NAME = 'GEE_SicilyPlanetMasks'

# The 21 dates from the complementary PlanetScope download batch.
NEW_DATES = {
    'Ancipa':      ['2024-11-14', '2024-11-25'],
    'Poma':        ['2026-04-08', '2026-04-30'],
    'Rosamarina':  ['2025-09-21', '2025-10-10', '2025-10-26', '2025-11-22', '2025-12-13',
                     '2026-01-01', '2026-01-13', '2026-01-28', '2026-02-06', '2026-02-13',
                     '2026-02-19', '2026-03-04', '2026-03-25', '2026-04-08', '2026-04-17',
                     '2026-04-25', '2026-05-09'],
}
EXPECTED_NAMES = {
    f'mask_Planet_{site}_{d}.tif' for site, dates in NEW_DATES.items() for d in dates
}


def drive_client():
    cred_path = pathlib.Path.home() / '.config' / 'earthengine' / 'credentials'
    d = json.loads(cred_path.read_text())
    creds = Credentials(
        token=None, refresh_token=d['refresh_token'],
        token_uri='https://oauth2.googleapis.com/token',
        client_id=oauth.CLIENT_ID, client_secret=oauth.CLIENT_SECRET,
        scopes=d['scopes'],
    )
    return build('drive', 'v3', credentials=creds)


def main():
    svc = drive_client()

    folders = svc.files().list(
        q=f"name = '{DRIVE_FOLDER_NAME}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        fields='files(id, name)',
    ).execute().get('files', [])
    if not folders:
        raise SystemExit(f"Drive folder '{DRIVE_FOLDER_NAME}' not found.")
    print(f"{len(folders)} folder(s) named '{DRIVE_FOLDER_NAME}' found "
          f"(each VS Code export run apparently created its own instead of reusing one).")

    files = []
    for folder in folders:
        page_token = None
        while True:
            resp = svc.files().list(
                q=f"'{folder['id']}' in parents and trashed = false",
                fields='nextPageToken, files(id, name, modifiedTime, size)',
                pageToken=page_token,
            ).execute()
            files.extend(resp.get('files', []))
            page_token = resp.get('nextPageToken')
            if not page_token:
                break

    print(f'{len(files)} files total across all Drive/{DRIVE_FOLDER_NAME} folders.')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    found_names = {f['name'] for f in files}
    missing = EXPECTED_NAMES - found_names
    if missing:
        print(f'\nWARNING: {len(missing)} expected files not yet in Drive:')
        for m in sorted(missing):
            print(f'  {m}')

    # De-dupe by name (keep the most recently modified copy) in case of reruns.
    by_name = {}
    for f in files:
        if f['name'] not in EXPECTED_NAMES:
            continue
        prev = by_name.get(f['name'])
        if prev is None or f['modifiedTime'] > prev['modifiedTime']:
            by_name[f['name']] = f
    to_download = list(by_name.values())
    print(f'\nDownloading {len(to_download)} matching files...')
    for f in sorted(to_download, key=lambda x: x['name']):
        dest = OUT_DIR / f['name']
        request = svc.files().get_media(fileId=f['id'])
        with open(dest, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print(f'  {f["name"]} -> {dest}  ({dest.stat().st_size} bytes)')

    print('\nDone.')


if __name__ == '__main__':
    main()
