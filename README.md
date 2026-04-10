# Reservoirs_s1_svm: SAR & Machine Learning for Reservoir Monitoring

Google Earth Engine application to track time-series of area and volume for water reservoirs
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-JavaScript%20API-green)](https://earthengine.google.com/)

## 📋 Overview

This repository contains the source code, supplementary datasets, and validation notebooks for the manuscript: 
> **"Monitoring Reservoir Surface and Storage Dynamics Using Sentinel-1 SAR and Machine Learning in Google Earth Engine"** > *(Submitted to Environmental Modelling & Software, 2026)*.

This project provides a robust, cloud-based framework for continuously monitoring reservoir surface area, morphological complexity (Area/Perimeter ratio), and storage volume using Sentinel-1 Synthetic Aperture Radar (SAR) imagery and a Support Vector Machine (SVM) classifier. The code exemplifies the methodology for 41 reservoirs in Sicily (Italy), with the capability to draw custom Areas of Interest (AOIs) and monitor any water body globally.

**Key Features:**
- ✅ Automated SAR preprocessing (Refined Lee filtering, radiometric calibration)
- ✅ Supervised SVM classification for water detection
- ✅ Time-series storage estimation using power-law regressions
- ✅ Interactive web application for real-time monitoring
- ✅ Validated with 117 PlanetScope optical scenes

## 🚀 Quick Start

1. **Live Application**: Try the web app without setup → [Reservoir Monitoring App](https://ee-ciceromartinsjr.projects.earthengine.app/view/customaoi)
2. **Full Setup**: See [Main Script README](./main_script/README.md) for detailed instructions

[![Video tutorial - Sicilian Reservoirs](https://img.youtube.com/vi/eEvuLQMvpsc/hqdefault.jpg)](https://youtu.be/eEvuLQMvpsc)

[![Video tutorial - Custom AOI](https://img.youtube.com/vi/eEvuLQMvpsc/hqdefault.jpg)](https://youtu.be/d-azQwtdcA8)

## 📁 Repository Structure

- **`main_script/`** - Core JavaScript implementation for Google Earth Engine
  - `reservoirs_s1_svm.js` - Main application logic
  - `entries.js` - Reservoir AOIs and training samples
- **`raw_data/`** - Dataset descriptions (Sentinel-1 and PlanetScope)
- **`README.md`** - This file

## 📚 Documentation

- [Main Script Guide](./main_script/README.md) - How to use the GEE application
- [Data Description](./raw_data/README.md) - Dataset specifications
- **Associated Paper**: *Monitoring Reservoir Surface and Storage Dynamics Using Sentinel-1 SAR and Machine Learning in Google Earth Engine*

## 🛠️ Requirements

- Google Earth Engine account (free at [earthengine.google.com](https://earthengine.google.com))
- Modern web browser
- Optional: Planet API license for validation data

## 📖 How It Works

1. **Select a Reservoir** - Choose from 41 preset locations in Sicily or draw a custom AOI
2. **Define Time Period** - Set start and end dates for analysis
3. **Run Processing** - SAR data is filtered, classified, and converted to storage estimates
4. **Visualize Results** - View watermask time-series and storage plots

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests with improvements.

## 📧 Contact & Citation

For questions or to cite this work, refer to the associated manuscript.

## 📄 License

This software is released under the **MIT License**. You are free to use, modify, and distribute this code for academic and commercial purposes, provided that proper credit is given to the original authors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🖋️ Citation

If you use this code, methodology, or data in your research, please cite our paper:

```bibtex
@article{MartinsJr2026,
  title = {Monitoring Reservoir Surface and Storage Dynamics Using Sentinel-1 SAR and Machine Learning in Google Earth Engine},
  author = {Martins Jr., Cicero and Capodici, Fulvio and De Marchis, Mauro and Ciraolo, Giuseppe},
  journal = {Environmental Modelling & Software},
  year = {2026},
  doi = {10.2139/ssrn.6403974}
}
