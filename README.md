# Smart City Land-Use Classification
### Identifying Residential, Commercial & Industrial Zones using DBSCAN + KNN

## 📁 Project Structure
```
smart_city_project/
├── Smart_City_Dataset.csv     # Synthetic geospatial census dataset (490 areas, with null values)
├── generate_dataset.py        # Script that generated the dataset (re-runnable, seeded)
├── eda.py                      # Step 1: EDA + cleaning -> cleaned_dataset.csv
├── ml_pipeline.py               # Step 2: DBSCAN + KNN -> final_classified.csv
├── evaluate_model.py             # Step 3: Accuracy / Precision / Recall / F1 evaluation
├── app.py                       # Streamlit dashboard
├── cleaned_dataset.csv          # Output of eda.py
├── final_classified.csv         # Output of ml_pipeline.py
├── assets/                       # All generated graphs (PNG)
└── requirements.txt
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
```bash
python generate_dataset.py
```

### 3. Run EDA (cleans data + produces graphs in assets/)
```bash
python eda.py
```

### 4. Run the DBSCAN + KNN pipeline
```bash
python ml_pipeline.py
```

### 5. Evaluate model performance (Accuracy, Precision, Recall, F1)
```bash
python evaluate_model.py
```

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```
Then upload `Smart_City_Dataset.csv` in the sidebar.

---

## 📊 Dataset Description

`Smart_City_Dataset.csv` contains **490 areas** across a synthetic city grid, with:

- **Geospatial coordinates**: `Latitude`, `Longitude`
- **Demographics**: `Population`, `Household_Count`, `Avg_Household_Size`, `Literacy_Rate`, `Avg_Monthly_Income`
- **Built environment**: `Building_Density_per_sqkm`, `Avg_Building_Height_Floors`, `Road_Density_km_per_sqkm`, `Land_Price_per_sqft`
- **Economic activity**: `Num_Shops`, `Num_Factories`, `Num_Schools`, `Num_Hospitals`
- **Environment & infrastructure**: `Green_Space_Pct`, `Pollution_Index`, `Traffic_Density_Index`, `Water_Supply_Pct`, `Electricity_Access_Pct`, `Public_Transport_Access`
- **Ground truth label**: `Zone_Type` (Residential / Commercial / Industrial)
- **Non-essential columns** (intentionally included for the EDA cleaning step): `Area_ID`, `Survey_Code`, `Data_Source` (constant), `Remarks` (fully empty)
- **Missing values**: ~5-10% nulls injected into 8 numeric columns for imputation practice

## 🧠 Methodology

1. **EDA & Cleaning** (`eda.py`)
   - Drop identifier / constant / fully-empty columns
   - Impute missing values (median, grouped by zone)
   - Visualize distributions, correlations, zone profiles, geospatial scatter

2. **DBSCAN** (`ml_pipeline.py`)
   - Clusters areas by `Latitude`/`Longitude` only
   - Auto-discovers geographic districts; flags scattered "noise" points
   - Each cluster is labeled by its dominant zone type

3. **KNN Boundary Assignment** (`ml_pipeline.py`)
   - Trained on DBSCAN core points using location + socio-economic/infrastructure features
   - Assigns a zone to every area (including DBSCAN noise/boundary points)
   - Achieves ~87% agreement with ground-truth `Zone_Type`

4. **Model Evaluation** (`evaluate_model.py`)
   - Computes Accuracy, Precision, Recall, F1-Score (macro & weighted) for:
     - DBSCAN-only zone assignment (core points)
     - DBSCAN + KNN boundary assignment (all points)
   - Per-class precision/recall/F1, confusion matrices, and KNN k-sensitivity plot
   - Saves `model_evaluation_summary.csv`

   | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
   |-------|----------|--------------------|------------------|------------|
   | DBSCAN-only (core pts) | 0.866 | 0.927 | 0.838 | 0.853 |
   | DBSCAN + KNN (all pts) | 0.871 | 0.929 | 0.838 | 0.855 |

5. **Streamlit App** (`app.py`)
   - Tabs: EDA, DBSCAN, KNN Zones, **Evaluation** (Accuracy/Precision/Recall/F1), Data Table, About
   - Adjustable `eps`, `min_samples`, and `k` via sidebar sliders
