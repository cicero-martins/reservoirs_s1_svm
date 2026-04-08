# PlanetScope NDWI Water Mask Validation

This repository contains the dataset and scripts used to validate water surface area extraction from **PlanetScope** imagery (3m resolution). These data serve as the ground-truth reference to evaluate the accuracy of the **SAR-based watermask time series** presented in the main paper.

## 📌 Overview

The methodology follows the adjusted NDWI filtering method described in **Sections 2.5 and 3.4**, with further technical details provided in **Appendix S2** of the Supplementary Material. 

The validation process compares high-resolution PlanetScope-derived masks (optimized by manual thresholding) against manually digitized polygons created in QGIS to ensure the highest possible precision for the study sites.

---

## 📁 Dataset Description: `NDWI_Validation_Results.xlsx`

The main spreadsheet provided in this folder is organized into the following sections:

* **`Fig7_Tab3_Selected`**: Summarizes the final validation results used to build **Table 3** and **Figure 7** of the paper.
* **`ndwiValidation`**: A comprehensive list of all PlanetScope images used, their respective manually adjusted NDWI thresholds, and the calculated areas compared against the QGIS manual reference masks.
* **Sensitivity Analysis (`0.2` to `-0.4`)**: A series of sheets illustrating the area results and associated errors obtained when applying fixed NDWI thresholds. These data are summarized in **Figure 8**.
* **Time Series Data (`Figs 9 and 10`)**: The last three sheets contain the data used for the long-term analysis shown in **Figures 9 and 10**, based on the optimal NDWI chosen for each image in the full PlanetScope dataset.

---

## 🛠️ Folder Contents

* **/polygons**: Contains the reference datasets (Shapefiles/GeoJSON) manually drawn in **QGIS**. These represent the "gold standard" used for error calculation.
* **/scripts**: Includes the Google Earth Engine (GEE) JavaScript code used to process the images and calculate the area statistics.

---

## 🌐 Interactive GEE Validation App

To ensure transparency and allow reviewers to inspect the thresholding process, an interactive web application was developed. The app allows for real-time visualization of the PlanetScope RGB imagery, the NDWI spectral response, and the resulting water masks.

🔗 **Access the App:** [PlanetScope NDWI Validation Tool](https://ee-ciceromartinsjr.projects.earthengine.app/view/ndwi-validation-planet-images)

### How to use the App:
1. Select the **Site (AOI)** from the dropdown menu.
2. Choose a specific **Image Date**.
3. Adjust the **NDWI Threshold** slider (or use the paper's recommended value).
4. Click **"Generate Results"** to view the calculated area and the mask overlay.

---

## 📖 Citation

If you use this dataset or the associated scripts, please cite:
> *Martins-Jr, C., et al. (2026). [Insert Full Paper Title Here]. Journal Name/Link.*
