"""
gen_reservoir_ids.py

Builds a compact continent-prefixed ID for each of the 62 global-pilot reservoirs
(E1, E2... Europe; A1, A2... Africa; As1, As2... Asia; Am1, Am2... Americas;
O1, O2... Oceania), used as the point label in Figures 4/5/7/8 instead of the
full reservoir name -- full names are long and, with 47-62 points on one axis,
overlap badly even at a legible font size. The ID is looked up against the full
name in Table 1 (table_pilot.tex), which carries both.

IDs are assigned in the same static-A/P ascending order Table 1 is already
sorted by, so the ID sequence within each continent matches the table's
row order top-to-bottom.

Reads:  analysis/biome_kge.csv, analysis/global_pilot_v4_candidates.csv
Output: analysis/reservoir_ids.csv  (columns: name, country, continent, ap_m, id)
"""
import pandas as pd

COUNTRY_CONTINENT = {
    # Europe
    'France': 'Europe', 'Germany': 'Europe', 'Italy': 'Europe',
    'Portugal': 'Europe', 'Spain': 'Europe', 'United Kingdom': 'Europe',
    # Africa
    'Angola': 'Africa', 'Burkina Faso': 'Africa', 'Kenya': 'Africa',
    'Morocco': 'Africa', 'Mozambique': 'Africa', 'South Africa': 'Africa',
    'Tanzania': 'Africa',
    # Asia
    'China': 'Asia', 'India': 'Asia', 'Iran': 'Asia', 'Lebanon': 'Asia',
    'Philippines': 'Asia', 'South Korea': 'Asia', 'Sri Lanka': 'Asia',
    'Taiwan': 'Asia', 'Turkey': 'Asia',
    # Americas
    'Argentina': 'Americas', 'Brazil': 'Americas', 'Canada': 'Americas',
    'Chile': 'Americas', 'Colombia': 'Americas', 'Costa Rica': 'Americas',
    'Honduras': 'Americas', 'Mexico': 'Americas', 'USA': 'Americas',
    # Oceania
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
}
PREFIX = {'Europe': 'E', 'Africa': 'A', 'Asia': 'As', 'Americas': 'Am', 'Oceania': 'O'}

df = pd.read_csv('analysis/biome_kge.csv')[['name', 'ap_m']].copy()
cand = pd.read_csv('analysis/global_pilot_v4_candidates.csv').drop_duplicates('name').set_index('name')
df['country'] = df.name.map(cand['country'])
df['continent'] = df['country'].map(COUNTRY_CONTINENT)

missing = df[df.continent.isna()]
if not missing.empty:
    raise SystemExit(f'No continent mapping for: {missing[["name", "country"]].values.tolist()}')

df = df.sort_values('ap_m').reset_index(drop=True)
df['id'] = df.groupby('continent').cumcount() + 1
df['id'] = df['continent'].map(PREFIX) + df['id'].astype(str)

OUT = 'analysis/reservoir_ids.csv'
df[['name', 'country', 'continent', 'ap_m', 'id']].to_csv(OUT, index=False)
print(f'Saved {len(df)} reservoir IDs -> {OUT}')
print(df['continent'].value_counts())
