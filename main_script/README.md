# Main Script: Reservoir Monitoring Framework

This folder contains the core JavaScript implementation of the satellite-driven reservoir monitoring framework. The code is designed to run within the **Google Earth Engine (GEE)** Code Editor environment.

## 📄 File Overview

1.  **`reservoirs_s1_svm.js`**: The main application logic. It includes the user interface (UI) components, Sentinel-1 data filtering (prioritizing IW3 sub-swath), the Support Vector Machine (SVM) classification engine, and the volumetric calculation pipeline.
2.  **`entries.js`**: A support file containing the vectorized geometries (AOIs) for the 41 Sicilian reservoirs and the samples defined for the training proccess.

## ⚙️ Core Functionalities

As detailed in **Section 2** of the main manuscript and **Appendix S4**, the script performs the following operations:

* **Mode Selection**: A toggle allows switching between the **Validated Sicilian Dataset** (preset reservoirs) and **Custom AOI Mode** (global scalability via manual drawing or asset upload).
* **SAR Preprocessing**: Automated application of the Refined Lee filter, border noise removal, and conversion to Decibels (dB).
* **Machine Learning Classification**: Implementation of a supervised SVM classifier trained on VV and VH backscatter bands for robust water detection.
* **Hydrological Analysis**: Conversion of SAR-derived watermasks into storage estimates using power-law regressions ($V = a \cdot A^b$).

## 🚀 Live Application

For a quick demonstration of the framework without setting up the environment, access the web-based app here:
🔗 **[Reservoir Monitoring App](https://ee-ciceromartinsjr.projects.earthengine.app/view/customaoi)**

## 🛠️ Usage Instructions

1.  Open the [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2.  Copy the contents of `entries.js` and `reservoirs_s1_svm.js` into a new script (or ensure the variables are reachable).
3.  Run the script.
4.  Use the **Control Panel** to:
    * Select a predefined reservoir or draw a new area.
    * Define the start and end dates for the time-series analysis.
    * Click **RUN!** to generate the watermask series and storage plots.

## 📚 References
* **Main Paper**: *Monitoring Reservoir Surface and Storage Dynamics Using Sentinel-1 SAR and Machine Learning in Google Earth Engine*.
* **Supplementary Material**: Refer to **Appendix S1** for SVM formulation and **S1.1** for SAR filtering specifics.
