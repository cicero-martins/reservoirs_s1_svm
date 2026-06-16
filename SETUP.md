# Environment Setup

## Requirements

- Python 3.10+
- Google Earth Engine account with project `ee-ciceromartinsjr`
- Access to Google Drive folder `GROWL_SAR_pilot` (GEE exports)

## 1. Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Earth Engine authentication

```bash
# First time only — use notebook mode if browser auth is blocked
earthengine authenticate --auth_mode=notebook
```

Or via gcloud (Windows PowerShell — quote the scopes):

```powershell
gcloud auth application-default login --scopes="https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform"
```

## 3. GEE assets required

Upload to your GEE project before running scripts:

| Asset | Source |
|---|---|
| `projects/ee-ciceromartinsjr/assets/HydroLAKES_pilot24` | `validation_data/HydroLAKES_pilot24.geojson` |

## 4. Run the global pilot export (GEE Code Editor)

1. Open `main_script/exportGlobalPilot.js` in [code.earthengine.google.com](https://code.earthengine.google.com)
2. Click **Run** → 48 tasks appear (24 SAR + 24 JRC)
3. Submit all tasks — or use the browser console trick:
   ```javascript
   $$('.run-button-label').forEach(b => b.click())
   ```
4. Downloads land to: Google Drive → `GROWL_SAR_pilot/`

Alternatively, submit programmatically (after authentication):
```bash
python analysis/submit_pilot_tasks.py
```

## 5. Download GEE exports

Download all `SAR_area_*.csv` and `JRC_area_*.csv` from Google Drive folder
`GROWL_SAR_pilot` to:
```
validation_data/GROWL_SAR_pilot/
```

## 6. Compute KGE and figures

```bash
# Main KGE analysis + AP vs KGE scatter
python analysis/compute_pilot_kge.py

# Area time series + annual KGE heatmap
python analysis/plot_pilot_timeseries.py

# Proxy median A/P diagnostic (3-panel)
python analysis/diagnose_pilot_ap.py
```

## Key file structure

```
reservoirs_s1_svm/
├── main_script/
│   ├── reservoirs_s1_svm.js      # Main GEE app (41 Sicilian reservoirs)
│   ├── entries.js                 # AOI geometries + training polygons
│   ├── exportGlobalPilot.js       # Global pilot export (24 GROWL reservoirs)
│   └── visualize_training_samples.js  # Diagnostic: sample locations in GEE
├── analysis/
│   ├── compute_pilot_kge.py       # KGE + AP vs KGE figure
│   ├── plot_pilot_timeseries.py   # Area time series + annual KGE heatmap
│   ├── diagnose_pilot_ap.py       # Proxy median A/P diagnostic
│   └── submit_pilot_tasks.py      # Submit GEE tasks via Python API
├── validation_data/
│   ├── GROWL_SAR_pilot/           # Downloaded GEE exports (gitignored)
│   ├── GROWL_pilot_sample.csv     # 24-reservoir pilot metadata
│   └── morphometric_analysis/shoreline_compactness/
│       ├── AP_all_reservoirs.csv  # A/P for all 41 Sicilian reservoirs
│       ├── pilot_kge_results.csv  # KGE results per pilot reservoir
│       └── pilot_annual_kge.csv   # Annual KGE per reservoir × year
└── requirements.txt
```

## Status (2026-06-16)

- [x] JRC area exports: complete for all 24 pilot reservoirs
- [ ] SAR area exports: need re-submission with WorldCover land sampling
  - Script ready: `main_script/exportGlobalPilot.js` (uses ESA WorldCover v200)
  - Submit for 18 valid reservoirs (6 excluded: see `EXCLUDED` dict in `compute_pilot_kge.py`)
- [ ] Paper resubmission deadline: 27/06/2026 to EnvSoft (ref: ENVSOFT-D-26-00848)
