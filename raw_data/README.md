# Raw Data Description: Sentinel-1 and PlanetScope

This folder contains the metadata and descriptions of the raw satellite datasets used in this study. It bridges the gap between the automated cloud processing in Google Earth Engine (GEE) and the independent validation dataset.

---

## 🛰️ 1. Sentinel-1 SAR (SAR Watermask Series)

The Sentinel-1 dataset is accessed and processed dynamically through the Google Earth Engine collection `COPERNICUS/S1_GRD`. Due to the high temporal frequency and automated workflow, individual Scene IDs are not listed manually. Instead, the reliability and reproducibility of the data are guaranteed by the filtering criteria and preprocessing steps, as detailed in **Section 2.3** of the paper and **Appendix S1.1**.

To standardize the viewing geometry and seek the highest sensitivity to the water-land interface, the following filters were applied:
* **Instrument Mode:** Interferometric Wide Swath (IW).
* **Polarization:** Dual-polarization (VV and VH) to exploit complementary sensitivity to surface roughness.
* **Incident Angle Optimization:** Selection prioritized the **IW3 sub-swath (approx. 42°–46°)**. If unavailable, IW2 or IW1 were used sequentially to maintain time-series continuity.

---

## 🌍 2. PlanetScope (Optical Validation Dataset)

The validation of the SAR-derived masks was performed using **117 cloud-free PlanetScope scenes** (approx. 3m resolution). These images were used to generate the "ground-truth" reference through scene-specific adjusted NDWI thresholds.

### 📊 Dataset: `planetscope_ids.xlsx`
The file `planetscope_ids.xlsx` contains the complete list of images used. The nomenclature follows the official Planet standard, which defines also the sensor and atmospheric correction level.

**Example Breakdown:** `20240420_095420_65_24d7_3B_AnalyticMS_SR_clip`

| Component | Example | Description |
| :--- | :--- | :--- |
| **Date** | `20240420` | Acquisition date (YYYYMMDD). |
| **Time (UTC)** | `095420` | Acquisition time (HHMMSS). |
| **Sensor ID** | `65_24d7` | Unique identifier for the Dove/SuperDove telescope. |
| **Instrument** | `3B` | Borealis (SuperDove) generation. |
| **Product Type** | `AnalyticMS` | Multispectral product for scientific analysis. |
| **Processing** | `SR` | **Surface Reflectance** (Atmospherically corrected). |
| **Geometry** | `clip` | Masked to the reservoir's Area of Interest (AOI). |

---

## 🔗 Data Access
* **Sentinel-1:** Accessible via [GEE Sentinel-1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD).
* **PlanetScope:** Scenes can be located using the IDs in `planetscope_ids.xlsx` through the [Planet Explorer](https://www.planet.com/explorer/) (requires valid license).
