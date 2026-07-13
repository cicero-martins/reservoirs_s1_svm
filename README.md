# Reservoirs_s1_svm: Shoreline Compactness and SAR Reservoir Monitoring

[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-JavaScript%20API-green)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Google Earth Engine tools and supplementary materials for predicting, in advance, how reliably Sentinel-1 SAR monitors reservoir surface area — and for running that monitoring globally.

## Overview

This repository contains source code, supplementary datasets, and the manuscript associated with:

> **Shoreline compactness predicts Sentinel-1 water-area monitoring reliability across reservoirs worldwide**  
> *(prepared for submission to Environmental Modelling & Software)*

The core finding is that a reservoir's shoreline area-to-perimeter ratio (A/P) — computable from global databases before any image is processed — predicts how reliably Sentinel-1 SAR tracks its surface area. The repository documents the Earth Engine pipeline behind this result (comparing a single-polarisation VV Otsu detector against a dual-polarisation SVM, both JRC auto-trained and re-estimated per scene) across 62 reference-quality-screened reservoirs on five continents plus four in Sicily validated against 3 m PlanetScope imagery, and ships an interactive app that applies the same pipeline to 35,000+ reservoirs worldwide.

## Highlights

- Shoreline A/P predicts SAR reservoir water-area accuracy a priori (Spearman ρ = 0.51)
- A simple per-scene single-polarisation detector matches the dual-pol SVM overall
- Per-scene adaptivity matters more for accuracy than the number of polarisations
- The simpler detector is also the cheapest to run — added complexity isn't justified
- An open Earth Engine app runs it over 35,000+ global reservoirs, JRC auto-trained

## Quick Start

1. **Live application (global, all reservoirs)**  
   [Reservoir SAR Monitor — global](https://ee-ciceromartinsjr.projects.earthengine.app/view/globalpilotsar)  
   Search any of 35,000+ Global Dam Watch reservoirs or the full Sicilian catalogue, pick Otsu or SVM detection, and run the pipeline over any date range. Flags each reservoir by expected reliability (A/P band) before you run anything.

2. **Live application (original, Sicily / custom AOI)**  
   [Reservoir Monitoring App — Sicily / custom AOI](https://ee-ciceromartinsjr.projects.earthengine.app/view/customaoi)  
   The earlier, manual-AOI interactive tool this project started from.

3. **Source code and implementation details**  
   See [Main Script README](./main_script/README.md)

## Repository Contents

- **`manuscript/`** — the paper itself (LaTeX source, `elsarticle` class)
  - `main.tex`, `sections/` — manuscript body
  - `supplementary.tex` — worked examples of every statistical test used (KGE, Spearman, Wilcoxon, Kruskal-Wallis)
  - `references.bib` — bibliography
  - `figures/` — all manuscript figures

- **`analysis/`** — Python and Earth Engine analysis scripts behind the paper's results
  - `gee_reservoir_monitor_app.js` — the global interactive app (see Quick Start above)
  - `exportGlobalPilotV4.js` — batch SAR export pipeline (the four detector configurations compared in the paper)
  - `compute_kge_*.py`, `plot_*.py` — accuracy metrics and figure generation
  - `export_gdw_ap_table.py` — precomputes A/P for the full Global Dam Watch catalogue

- **`main_script/`** — the original, single-reservoir Google Earth Engine JavaScript implementation
  - `reservoirs_s1_svm.js` — main application logic
  - `entries.js` — reservoir AOIs and training/sample inputs
  - `README.md` — usage notes for this script

- **`raw_data/`** — metadata and descriptions of the raw datasets used in the study
  - `planetScope_IDs.xlsx` — list of PlanetScope scenes used in the validation
  - `README.md` — description of Sentinel-1 and PlanetScope source data

- **`validation_data/`** — supplementary datasets and materials used for validation and analysis
  - `area_Planetscope_data/` — PlanetScope NDWI validation data, scripts, and reference polygons
  - `morphometric_analysis/` — shoreline compactness / morphometric analysis materials
  - `statistics/` — Matlab code and supporting files for statistical analysis
  - `volume_AdB/` — storage-volume comparison tables

- **`README.md`** — repository overview
- **`LICENSE`** — MIT license

## Documentation

- [Main Script Guide](./main_script/README.md) — instructions for using the original single-reservoir GEE application
- [Data Description](./raw_data/README.md) — dataset specifications and notes

## Requirements

- A Google Earth Engine account
- A modern web browser
- Optional: access to PlanetScope data for validation-related analyses

## Workflow Summary (global app)

1. Choose the **Global** (35,000+ Global Dam Watch reservoirs) or **Sicily** (41 named reservoirs) catalogue
2. Search by name, click a reservoir on the map, or read its A/P reliability band directly off the map before selecting anything
3. Pick a detection method — per-scene VV Otsu (recommended default) or per-scene adaptive dual-pol SVM — and an analysis period
4. Process Sentinel-1 SAR imagery in Google Earth Engine, JRC auto-trained, no manual sample delineation
5. Visualize the surface-area time series against the JRC optical reference, with per-date map inspection

## License

This repository is distributed under the **MIT License**. See the license terms for details on use, modification, and redistribution.

## Citation

If you use this repository, please cite the associated manuscript. A DOI will be added here once the manuscript is accepted; in the meantime, cite the repository directly.

```bibtex
@article{MartinsJr2026,
  title   = {Shoreline compactness predicts Sentinel-1 water-area monitoring reliability across reservoirs worldwide},
  author  = {Martins Jr., Cicero and Capodici, Fulvio and De Marchis, Mauro and Ciraolo, Giuseppe},
  year    = {2026},
  note    = {Manuscript in preparation},
  url     = {https://github.com/cicero-martins/reservoirs_s1_svm}
}
```
