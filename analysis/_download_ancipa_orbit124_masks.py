"""One-off: download Ancipa orbit-124 (DESCENDING) densification masks from
Drive/GEE_SicilyMasks into raw_data/GEE_SicilyMasks/. Same pattern as
_download_all_densify_masks.py, restricted to Ancipa; safe to re-run while
export tasks are still finishing (only downloads files already COMPLETED
and not yet on disk)."""
import json, pathlib, re

import truststore
truststore.inject_into_ssl()

from google.oauth2.credentials import Credentials
import ee.oauth as oauth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

OUT_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
DRIVE_FOLDER_NAME = 'GEE_SicilyMasks'
PATTERN = re.compile(r'^mask_Ancipa_\d{4}-\d{2}-\d{2}\.tif$')


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
    matching = [f for f in files if PATTERN.match(f['name'])]
    print(f'{len(matching)} Ancipa mask files found in Drive.')

    by_name = {}
    for f in matching:
        prev = by_name.get(f['name'])
        if prev is None or f['modifiedTime'] > prev['modifiedTime']:
            by_name[f['name']] = f

    to_download = [f for f in by_name.values() if not (OUT_DIR / f['name']).exists()]
    print(f'{len(to_download)} not yet on disk -- downloading...')
    for f in sorted(to_download, key=lambda x: x['name']):
        dest = OUT_DIR / f['name']
        request = svc.files().get_media(fileId=f['id'])
        with open(dest, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print(f'  {f["name"]} -> {dest}  ({dest.stat().st_size} bytes)')

    print(f'\nTotal Ancipa masks in Drive so far: {len(by_name)}/9')
    print('Done.')


if __name__ == '__main__':
    main()
