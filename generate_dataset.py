"""
Smart City Land-Use Dataset Generator
Generates a synthetic geospatial census-style dataset containing
Residential, Commercial, and Industrial zone records with realistic
feature distributions and injected null values for EDA practice.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DEFINE ZONE CLUSTER CENTERS (geospatial)
# ─────────────────────────────────────────────
# City bounding box roughly: lat 28.40-28.70, lon 77.30-77.60 (Smart City grid)
ZONE_CONFIG = {
    "Residential": {
        "centers": [(28.45, 77.35), (28.58, 77.42), (28.50, 77.55)],
        "n_per_center": [90, 80, 70],
    },
    "Commercial": {
        "centers": [(28.52, 77.40), (28.47, 77.50)],
        "n_per_center": [70, 60],
    },
    "Industrial": {
        "centers": [(28.62, 77.33), (28.40, 77.48)],
        "n_per_center": [65, 55],
    },
}

records = []
area_id = 1000

for zone, cfg in ZONE_CONFIG.items():
    for (lat_c, lon_c), n in zip(cfg["centers"], cfg["n_per_center"]):
        lat = np.random.normal(lat_c, 0.015, n)
        lon = np.random.normal(lon_c, 0.015, n)

        if zone == "Residential":
            population        = np.random.normal(8000, 2200, n).clip(500)
            household_size    = np.random.normal(4.4, 0.5, n).clip(2, 8)
            building_density  = np.random.normal(450, 90, n).clip(50)
            building_height   = np.random.normal(3.2, 1.1, n).clip(1)
            road_density      = np.random.normal(8.5, 1.8, n).clip(1)
            land_price        = np.random.normal(4200, 900, n).clip(500)
            literacy_rate     = np.random.normal(0.78, 0.08, n).clip(0.2, 1.0)
            avg_income        = np.random.normal(38000, 8000, n).clip(5000)
            num_shops         = np.random.normal(12, 5, n).clip(0)
            num_factories     = np.random.normal(0.5, 0.6, n).clip(0)
            num_schools       = np.random.normal(3, 1.3, n).clip(0)
            num_hospitals     = np.random.normal(1.2, 0.8, n).clip(0)
            green_space_pct   = np.random.normal(22, 6, n).clip(0, 60)
            pollution_index   = np.random.normal(35, 8, n).clip(5, 100)
            traffic_index     = np.random.normal(40, 10, n).clip(5, 100)
            water_supply_pct  = np.random.normal(88, 7, n).clip(40, 100)
            electricity_pct   = np.random.normal(94, 5, n).clip(50, 100)
            public_transport  = np.random.normal(60, 15, n).clip(0, 100)

        elif zone == "Commercial":
            population        = np.random.normal(4500, 1500, n).clip(200)
            household_size    = np.random.normal(3.6, 0.5, n).clip(2, 7)
            building_density  = np.random.normal(620, 110, n).clip(100)
            building_height   = np.random.normal(6.5, 2.2, n).clip(1)
            road_density      = np.random.normal(13.5, 2.5, n).clip(2)
            land_price        = np.random.normal(9800, 1800, n).clip(1000)
            literacy_rate     = np.random.normal(0.86, 0.06, n).clip(0.3, 1.0)
            avg_income        = np.random.normal(62000, 14000, n).clip(8000)
            num_shops         = np.random.normal(85, 25, n).clip(5)
            num_factories     = np.random.normal(1.0, 1.0, n).clip(0)
            num_schools       = np.random.normal(1.5, 1.0, n).clip(0)
            num_hospitals     = np.random.normal(2.0, 1.2, n).clip(0)
            green_space_pct   = np.random.normal(8, 4, n).clip(0, 30)
            pollution_index   = np.random.normal(55, 10, n).clip(10, 100)
            traffic_index     = np.random.normal(78, 12, n).clip(10, 100)
            water_supply_pct  = np.random.normal(92, 5, n).clip(50, 100)
            electricity_pct   = np.random.normal(97, 3, n).clip(60, 100)
            public_transport  = np.random.normal(82, 10, n).clip(0, 100)

        else:  # Industrial
            population        = np.random.normal(2200, 900, n).clip(100)
            household_size    = np.random.normal(4.0, 0.6, n).clip(2, 7)
            building_density  = np.random.normal(180, 50, n).clip(20)
            building_height   = np.random.normal(1.8, 0.7, n).clip(1)
            road_density      = np.random.normal(6.0, 1.5, n).clip(1)
            land_price        = np.random.normal(2600, 700, n).clip(300)
            literacy_rate     = np.random.normal(0.68, 0.10, n).clip(0.15, 1.0)
            avg_income        = np.random.normal(30000, 7000, n).clip(4000)
            num_shops         = np.random.normal(6, 3, n).clip(0)
            num_factories     = np.random.normal(14, 5, n).clip(1)
            num_schools       = np.random.normal(0.8, 0.7, n).clip(0)
            num_hospitals     = np.random.normal(0.6, 0.6, n).clip(0)
            green_space_pct   = np.random.normal(6, 3, n).clip(0, 25)
            pollution_index   = np.random.normal(78, 10, n).clip(20, 100)
            traffic_index     = np.random.normal(55, 12, n).clip(5, 100)
            water_supply_pct  = np.random.normal(75, 10, n).clip(30, 100)
            electricity_pct   = np.random.normal(90, 6, n).clip(40, 100)
            public_transport  = np.random.normal(45, 14, n).clip(0, 100)

        for i in range(n):
            records.append({
                "Area_ID": area_id,
                "Area_Name": f"{zone[:3].upper()}-{area_id}",
                "Latitude": round(lat[i], 6),
                "Longitude": round(lon[i], 6),
                "Population": int(population[i]),
                "Household_Count": int(population[i] / household_size[i]),
                "Avg_Household_Size": round(household_size[i], 2),
                "Building_Density_per_sqkm": round(building_density[i], 1),
                "Avg_Building_Height_Floors": round(building_height[i], 1),
                "Road_Density_km_per_sqkm": round(road_density[i], 2),
                "Land_Price_per_sqft": round(land_price[i], 1),
                "Literacy_Rate": round(literacy_rate[i], 3),
                "Avg_Monthly_Income": round(avg_income[i], 1),
                "Num_Shops": int(num_shops[i]),
                "Num_Factories": int(num_factories[i]),
                "Num_Schools": int(num_schools[i]),
                "Num_Hospitals": int(num_hospitals[i]),
                "Green_Space_Pct": round(green_space_pct[i], 2),
                "Pollution_Index": round(pollution_index[i], 2),
                "Traffic_Density_Index": round(traffic_index[i], 2),
                "Water_Supply_Pct": round(water_supply_pct[i], 2),
                "Electricity_Access_Pct": round(electricity_pct[i], 2),
                "Public_Transport_Access": round(public_transport[i], 2),
                "Zone_Type": zone,
            })
            area_id += 1

df = pd.DataFrame(records)

# ─────────────────────────────────────────────
# 2. SHUFFLE ROWS (so zones aren't grouped)
# ─────────────────────────────────────────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ─────────────────────────────────────────────
# 3. ADD A FEW NON-ESSENTIAL / REDUNDANT COLUMNS
#    (for the EDA "remove unnecessary columns" step)
# ─────────────────────────────────────────────
df["Survey_Code"]   = ["SC-" + str(np.random.randint(10000, 99999)) for _ in range(len(df))]
df["Data_Source"]   = "CensusDept_2024"          # constant column -> drop
df["Remarks"]       = ""                          # empty column -> drop

# ─────────────────────────────────────────────
# 4. INJECT NULL VALUES (~6-10% in several columns)
# ─────────────────────────────────────────────
null_cols = [
    "Literacy_Rate", "Avg_Monthly_Income", "Green_Space_Pct",
    "Pollution_Index", "Num_Hospitals", "Public_Transport_Access",
    "Land_Price_per_sqft", "Water_Supply_Pct"
]
rng = np.random.default_rng(7)
for col in null_cols:
    frac = rng.uniform(0.05, 0.10)
    n_null = int(len(df) * frac)
    idx = rng.choice(df.index, size=n_null, replace=False)
    df.loc[idx, col] = np.nan

# Also blank out 'Remarks' fully (all-null column -> drop in cleaning)
df["Remarks"] = np.nan

print(f"Dataset shape: {df.shape}")
print(f"Zone distribution:\n{df['Zone_Type'].value_counts()}")
print(f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum()>0]}")

df.to_csv("Smart_City_Dataset.csv", index=False)
print("\n✅ Saved Smart_City_Dataset.csv")
