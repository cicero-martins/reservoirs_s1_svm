"""
Filter HydroLAKES v10 shapefile to the 24 pilot Hylak_IDs and export GeoJSON.
Run after extracting HydroLAKES_polys_v10.zip.

Usage:
  python analysis/filter_hydrolakes.py  [path/to/HydroLAKES_polys_v10.shp]

If no path given, searches common download locations automatically.
Output: validation_data/HydroLAKES_pilot24.geojson  (upload this to GEE)
"""
import sys
from pathlib import Path

# ── 24 Hylak_IDs from PILOT array ────────────────────────────
PILOT_IDS = {
    15647:   'Manikdoh',
    15553:   'Panam',
    15840:   'Periyar',
    179882:  'Kakki',
    15600:   'Pench_Totladoh',
    15105:   'Thein_Ranjit_Sagar',
    172148:  'San_Esteban_ES',
    1338571: 'Ry_de_Rome_BE',
    172818:  'Borbollon_ES',
    173730:  'Minilla_ES',
    14552:   'Gabriel_y_Galan_ES',
    14457:   'Ricobayo_ES',
    8630:    'Pine_River_US',
    9794:    'Benito_Juarez_MX',
    105914:  'Pokegama_US',
    735:     'Winnibigoshish_US',
    738:     'Leech_US',
    831:     'Elephant_Butte_US',
    184942:  'Maroondah_AU',
    1426822: 'OShannassy_AU',
    184949:  'Silvan_AU',
    184923:  'Upper_Coliban_AU',
    184368:  'Chichester_AU',
    184943:  'Yarra_AU',
}

OUT = Path('F:/reservoirs_s1_svm/validation_data/HydroLAKES_pilot24.geojson')

# ── Find shapefile ─────────────────────────────────────────────
if len(sys.argv) > 1:
    shp_path = Path(sys.argv[1])
else:
    # Search common locations
    candidates = list(Path('C:/Users').glob('*/Downloads/**/HydroLAKES_polys_v10.shp')) + \
                 list(Path('F:/').glob('**/HydroLAKES_polys_v10.shp')) + \
                 list(Path('C:/').glob('**/HydroLAKES_polys_v10.shp'))
    if not candidates:
        print("Shapefile não encontrado automaticamente.")
        print("Uso: python analysis/filter_hydrolakes.py C:/caminho/para/HydroLAKES_polys_v10.shp")
        raise SystemExit
    shp_path = candidates[0]
    print(f"Shapefile encontrado: {shp_path}")

# ── Load and filter ────────────────────────────────────────────
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

if HAS_GEOPANDAS:
    print(f"\nCarregando {shp_path} com geopandas...")
    gdf = gpd.read_file(shp_path)
    print(f"Total de lagos: {len(gdf)}")
    print(f"Colunas: {list(gdf.columns)}")

    # HydroLAKES ID field is 'Hylak_id' (int)
    id_col = next((c for c in gdf.columns if 'hylak' in c.lower()), None)
    if id_col is None:
        print("ERRO: coluna Hylak_id não encontrada.")
        print(f"Colunas disponíveis: {list(gdf.columns)}")
        raise SystemExit

    filtered = gdf[gdf[id_col].isin(PILOT_IDS.keys())].copy()
    filtered['pilot_name'] = filtered[id_col].map(PILOT_IDS)

    print(f"\nEncontrados: {len(filtered)} / {len(PILOT_IDS)} polígonos")
    missing = set(PILOT_IDS.keys()) - set(filtered[id_col].values)
    if missing:
        print(f"FALTANDO: {[PILOT_IDS[i] for i in missing]}")

    # Print summary
    for _, row in filtered.iterrows():
        area_km2 = row.geometry.area / 1e6 if row.geometry else 0
        print(f"  [{row[id_col]:>8}] {row['pilot_name']:25s}  "
              f"area≈{area_km2:.1f} km²")

    # Save GeoJSON
    filtered.to_file(OUT, driver='GeoJSON')
    print(f"\nSalvo: {OUT}")
    print(f"Tamanho: {OUT.stat().st_size / 1024:.0f} KB")

else:
    # Fallback: use pyshp (shapefile)
    try:
        import shapefile, json
    except ImportError:
        print("Instale geopandas ou pyshp:  pip install geopandas  ou  pip install pyshp")
        raise SystemExit

    print(f"Carregando com pyshp...")
    sf     = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in sf.fields[1:]]
    id_idx = next((i for i, f in enumerate(fields) if 'hylak' in f.lower()), None)
    if id_idx is None:
        print(f"ERRO: campo Hylak_id não encontrado. Campos: {fields}")
        raise SystemExit

    features = []
    found_ids = set()
    for rec in sf.iterShapeRecords():
        hid = int(rec.record[id_idx])
        if hid in PILOT_IDS:
            geom = rec.shape.__geo_interface__
            feat = {
                'type': 'Feature',
                'properties': {f: rec.record[i] for i, f in enumerate(fields)},
                'geometry': geom
            }
            feat['properties']['pilot_name'] = PILOT_IDS[hid]
            features.append(feat)
            found_ids.add(hid)

    print(f"Encontrados: {len(features)} / {len(PILOT_IDS)}")
    missing = set(PILOT_IDS.keys()) - found_ids
    if missing:
        print(f"FALTANDO: {[PILOT_IDS[i] for i in missing]}")

    geojson = {'type': 'FeatureCollection', 'features': features}
    OUT.write_text(json.dumps(geojson), encoding='utf-8')
    print(f"\nSalvo: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")

print("\n=== Próximos passos ===")
print("1. Abra code.earthengine.google.com")
print("2. Assets → New → Shape files → selecione o arquivo .geojson")
print("3. Asset ID: users/SEU_USERNAME/HydroLAKES_pilot24")
print("4. Aguarde a ingestão (ícone de loading no painel Assets)")
print("5. Informe seu username GEE para atualizar o script")
