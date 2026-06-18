"""
AEGIS/CFD Sicilia — authenticated water level downloader.

Usage:
    python analysis/cfd_download.py [--list] [--element-id ID] [--station NAME]
                                    [--from 2020-01-01] [--to 2026-06-18]
                                    [--out PATH]

Quick reference — reservoir Water Level element IDs (as of 2026-06):
    Pozzillo Diga R2       58946   (Water Level, m)
    Garcia Diga R2         50637   (Idrometro Radar, m)
    Poma Diga R2           51527   (Water Level, m)
    Rosamarina Diga R2     50016   (Water Level, m)
    Ancipa Diga R2         88601   (Livello Secca, m)
    Fanaco Diga R2         51263   (Water Level, m)
    Disueri Diga R2        51524   (Water Level, m)
    Don Sturzo-Ogliastro   51546   (Water Level, m)
    Nicoletti Diga R2      51539   (Water Level, m)
    Olivo Diga R2          51533   (Water Level, m)
    Rubino Diga R2         51530   (Water Level, m)
    Lentini Diga R2        51097   (Idrometro Radar, m)
    Santa Rosalia Diga R2  51076   (Idrometro Radar, m)
    Castello Diga R2       50907   (Idrometro Radar, m)
    Cimia Diga R2          50643   (Idrometro Radar, m)
    Arancio Diga R2        50639   (Idrometro Radar, m)
"""
import sys
import argparse
import requests
import json
import uuid
import warnings
from datetime import datetime
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

HOST = 'https://www.protezionecivilesicilia.it:8443'
DATASCAPE = HOST + '/DatascapeA'
USERNAME = 'AegisUser'
PASSWORD = 'AegisUser'
CLIENT_ID = 'aegis'


def get_token(session):
    r = session.post(DATASCAPE + '/connect/token', data={
        'username': USERNAME,
        'password': PASSWORD,
        'grant_type': 'password',
        'client_id': CLIENT_ID,
        'client_instance': str(uuid.uuid4()),
    }, timeout=20)
    r.raise_for_status()
    return r.json()['access_token']


def list_elements(session, token, keyword='diga'):
    """Return DataFrame of all elements matching keyword."""
    H = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    r = session.get(DATASCAPE + '/v2/elements', headers=H, timeout=60)
    r.raise_for_status()
    elems = r.json()
    df = pd.DataFrame(elems)
    mask = df.apply(lambda row: keyword.lower() in
                    (str(row.get('stationName', '')) + str(row.get('elementName', ''))).lower(),
                    axis=1)
    return df[mask][['elementId', 'stationId', 'stationName', 'elementName',
                     'measUnit', 'time', 'value']].reset_index(drop=True)


def download_element(session, token, element_id, date_from='2014-01-01',
                     date_to=None, verbose=True):
    """Download time series for one element. Returns daily-average DataFrame."""
    if date_to is None:
        date_to = datetime.now().strftime('%Y-%m-%d')
    H = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    url = (DATASCAPE + f'/v3/data-combo/{element_id}'
           f'?from={date_from}T00:00:00&to={date_to}T23:59:59')
    if verbose:
        print(f"Fetching elementId={element_id} ({date_from} → {date_to}) ...", end=' ')
    r = session.get(url, headers=H, timeout=180)
    r.raise_for_status()
    j = r.json()

    records = (j.get('plausibleData') or []) + (j.get('extempData') or [])
    if not records:
        if verbose:
            print("NO DATA")
        return pd.DataFrame(columns=['date', 'wl_m'])

    # Records are [timestamp_str, value, flag]
    df = pd.DataFrame(records, columns=['time', 'wl_m', 'flag'])
    df['time'] = (pd.to_datetime(df['time'], utc=True)
                  .dt.tz_convert('Europe/Rome')
                  .dt.tz_localize(None))
    df['wl_m'] = pd.to_numeric(df['wl_m'], errors='coerce')
    df = df.dropna(subset=['wl_m']).sort_values('time')

    # Daily mean
    daily = (df.set_index('time')['wl_m']
               .resample('D').mean()
               .reset_index())
    daily.columns = ['date', 'wl_m']
    daily = daily.dropna().reset_index(drop=True)

    detail = j.get('elementDetail', {})
    station = detail.get('stationName', '')
    element = detail.get('elementName', '')
    unit    = detail.get('measUnit', '')

    if verbose:
        print(f"{len(daily)} daily records  "
              f"{daily.date.min().date()}–{daily.date.max().date()}  "
              f"WL {daily.wl_m.min():.3f}–{daily.wl_m.max():.3f} {unit}")
        print(f"  [{station} / {element}]")
    return daily


def make_session():
    s = requests.Session()
    s.verify = False
    s.headers['User-Agent'] = 'Mozilla/5.0'
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CFD Sicilia gauge downloader')
    parser.add_argument('--list', action='store_true',
                        help='List reservoir elements')
    parser.add_argument('--keyword', default='diga',
                        help='Keyword filter for --list (default: diga)')
    parser.add_argument('--element-id', type=int,
                        help='Element ID to download')
    parser.add_argument('--from', dest='date_from', default='2014-01-01')
    parser.add_argument('--to', dest='date_to', default=None)
    parser.add_argument('--out', default=None,
                        help='Output CSV path')
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')
    s = make_session()
    tok = get_token(s)
    print("Auth OK")

    if args.list:
        df = list_elements(s, tok, keyword=args.keyword)
        print(df.to_string())

    if args.element_id:
        daily = download_element(s, tok, args.element_id,
                                 date_from=args.date_from, date_to=args.date_to)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(out, index=False)
            print(f"Saved → {out}")
        else:
            print(daily.to_string())
