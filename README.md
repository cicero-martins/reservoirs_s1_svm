# Reservoirs_s1_svm: SAR and Machine Learning for Reservoir Monitoring

[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-JavaScript%20API-green)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Google Earth Engine tools and supplementary materials for monitoring reservoir surface area and storage dynamics using Sentinel-1 SAR imagery and machine learning.

## Overview

This repository contains source code, supplementary datasets, and validation notebooks associated with the manuscript:

> **Monitoring Reservoir Surface and Storage Dynamics Using Sentinel-1 SAR and Machine Learning in Google Earth Engine**  
> *(submitted to Environmental Modelling & Software, 2026)*

The repository documents a cloud-based workflow for monitoring reservoir surface area, shoreline morphology, and storage dynamics from Sentinel-1 Synthetic Aperture Radar (SAR) imagery using Google Earth Engine. The methodology was tested across 41 reservoirs in Sicily (Italy) and includes both an interactive application and supplementary scripts/notebooks supporting the methodological assessment.

## Highlights

- A scalable cloud-processing framework tested across 41 Mediterranean reservoirs
- SVM classification trained using VV and VH backscatter achieved ~96% accuracy
- Filtering high incidence-angle SAR acquisition improved temporal consistency
- Shoreline compactness controls SAR water classification performance
- Strong agreement against PlanetScope-derived masks (R² > 0.95; KGE up to 0.96)

## Quick Start

1. **Live application**  
   Access the web application directly:  
   [Reservoir Monitoring App](https://ee-ciceromartinsjr.projects.earthengine.app/view/customaoi)

2. **Source code and implementation details**  
   See [Main Script README](./main_script/README.md)

## Video Tutorials

### Sicilian Reservoirs
[![Video tutorial - Sicilian Reservoirs](https://img.youtube.com/vi/eEvuLQMvpsc/hqdefault.jpg)](https://youtu.be/eEvuLQMvpsc)

### Custom AOI
[![Video tutorial - Custom AOI](https://img.youtube.com/vi/d-azQwtdcA8/hqdefault.jpg)](https://youtu.be/d-azQwtdcA8)

## Repository Structure

- **`main_script/`** — Core JavaScript implementation for Google Earth Engine  
  - `reservoirs_s1_svm.js` — main application logic  
  - `entries.js` — reservoir AOIs and training/sample inputs
- **`raw_data/`** — dataset descriptions and supplementary information
- **`README.md`** — repository overview

## Documentation

- [Main Script Guide](./main_script/README.md) — instructions for using the GEE application
- [Data Description](./raw_data/README.md) — dataset specifications and notes

## Requirements

- A Google Earth Engine account
- A modern web browser
- Optional: access to PlanetScope data for validation-related analyses

## Workflow Summary

1. Select one of the predefined reservoirs in Sicily or define a custom AOI
2. Specify the analysis period
3. Process Sentinel-1 SAR imagery in Google Earth Engine
4. Generate water masks, surface-area estimates, and storage time series
5. Visualize outputs through maps and time-series plots

## License

This repository is distributed under the **MIT License**. See the license terms for details on use, modification, and redistribution.

## Citation

If you use this repository, please cite the associated manuscript and/or preprint.

```bibtex
@article{MartinsJr2026,
  title   = {Monitoring Reservoir Surface and Storage Dynamics Using Sentinel-1 SAR and Machine Learning in Google Earth Engine},
  author  = {Martins Jr., Cicero and Capodici, Fulvio and De Marchis, Mauro and Ciraolo, Giuseppe},
  year    = {2026},
  note    = {Submitted to Environmental Modelling \& Software},
  doi     = {10.2139/ssrn.6403974}
}
