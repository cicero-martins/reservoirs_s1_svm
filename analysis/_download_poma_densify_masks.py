"""One-off: download the 16 new Poma densification masks from Drive/GEE_SicilyMasks
into raw_data/GEE_SicilyMasks/, matching the existing mask_Poma_<date>.tif convention.
Same Drive-API-via-EE-OAuth pattern as download_planet_masks_drive.py."""
import json, pathlib

import truststore
truststore.inject_into_ssl()

from google.oauth2.credentials import Credentials
import ee.oauth as oauth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

OUT_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
DRIVE_FOLDER_NAME = 'GEE_SicilyMasks'

NEW_DATES = ['2025-12-20', '2025-12-26', '2026-01-07', '2026-01-13', '2026-01-19',
             '2026-03-02', '2026-03-08', '2026-03-14', '2026-03-20', '2026-04-01',
             '2026-04-13', '2026-04-19', '2026-04-20', '2026-04-25', '2026-05-01',
             '2026-05-07']
EXPECTED_NAMES = {f'mask_Poma_{d}.tif' for d in NEW_DATES}


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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    found_names = {f['name'] for f in files}
    missing = EXPECTED_NAMES - found_names
    if missing:
        print(f'WARNING: {len(missing)} expected files not yet in Drive:')
        for m in sorted(missing):
            print(f'  {m}')

    by_name = {}
    for f in files:
        if f['name'] not in EXPECTED_NAMES:
            continue
        prev = by_name.get(f['name'])
        if prev is None or f['modifiedTime'] > prev['modifiedTime']:
            by_name[f['name']] = f
    to_download = list(by_name.values())
    print(f'Downloading {len(to_download)}/{len(EXPECTED_NAMES)} matching files...')
    for f in sorted(to_download, key=lambda x: x['name']):
        dest = OUT_DIR / f['name']
        request = svc.files().get_media(fileId=f['id'])
        with open(dest, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print(f'  {f["name"]} -> {dest}  ({dest.stat().st_size} bytes)')

    print('Done.')


if __name__ == '__main__':
    main()
