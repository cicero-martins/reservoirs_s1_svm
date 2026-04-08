NDWI Water Mask Validation (PlanetScope)
This folder contains the datasets, code, and results used to validate water surface area extraction from PlanetScope imagery. These results serve as the ground-truth comparison for the SAR Watermask timeseries presented in the study.

📁 Folder Structure
NDWI_Validation_Results.xlsx: Detailed spreadsheet containing:

Fig7_Tab3_Selected: Final validation results (Reference for Table 3 and Figure 7).

ndwiValidation: List of all manually adjusted thresholds vs. areas calculated via NDWI vs. manual QGIS polygons.

Sheets (0.2 to -0.4): Sensitivity analysis with fixed thresholds used to generate Figure 8.

Final Sheets: Data supporting Figures 9 and 10 based on the full PlanetScope dataset.

polygons/: Folder containing the reference datasets (Shapefiles/GeoJSON) manually digitized in QGIS for each validation site.

scripts/:

NDWI_Validation_App.js: Source code for the interactive GEE validation app.

Bulk_Processing_Planet.js: Script used for processing the full time-series dataset.

🛠 Methodology Overview
The water masks were generated using an adjusted NDWI filtering method. While automated methods exist, this study employed a manual threshold adjustment for each image to ensure maximum precision against the high-resolution PlanetScope data (3m), as detailed in Sections 2.5 and 3.4 and Appendix S2 of the supplementary material.

Validation Workflow:
Manual Digitization: Reference polygons were drawn in QGIS for selected dates/sites.

NDWI Adjustment: For each date, the NDWI threshold was fine-tuned within the Earth Engine environment to match the physical boundaries of the reservoirs.

Area Comparison: The areas derived from the optimized NDWI were compared against the manual polygons to quantify the error margins.

🌐 Interactive Visualization
To facilitate the review process, an interactive Google Earth Engine App was developed. This tool allows users to select specific sites and dates, adjust thresholds in real-time, and view the corresponding PlanetScope RGB and NDWI masks used in this validation.

👉 Access the App here: PlanetScope NDWI Validation Tool
