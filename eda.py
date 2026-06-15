"""
Smart City Land-Use Classification
Step 1 – Exploratory Data Analysis (EDA)
Dataset: Smart_City_Dataset.csv (synthetic geospatial census data)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
ZONE_COLORS = {"Residential": "#2196F3", "Commercial": "#FF5722", "Industrial": "#4CAF50"}

OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df_raw = pd.read_csv("Smart_City_Dataset.csv")
print(f"[1] Raw shape: {df_raw.shape}")
print(f"    Columns: {list(df_raw.columns)}")

# ─────────────────────────────────────────────
# 2. INITIAL INSPECTION
# ─────────────────────────────────────────────
print(f"\n[2] Data types:\n{df_raw.dtypes}")
print(f"\n[3] Missing values:\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")
print(f"\n[4] Zone distribution:\n{df_raw['Zone_Type'].value_counts()}")

# ─────────────────────────────────────────────
# 3. COLUMN REMOVAL (non-essential columns)
# ─────────────────────────────────────────────
# - Area_ID, Survey_Code: pure identifiers, no analytical value
# - Data_Source: constant across all rows
# - Remarks: 100% missing -> drop entirely
drop_cols = []
for c in df_raw.columns:
    if df_raw[c].isnull().mean() == 1.0:
        drop_cols.append(c)        # fully empty columns
    elif df_raw[c].nunique(dropna=False) == 1:
        drop_cols.append(c)        # constant columns

id_cols = ["Area_ID", "Survey_Code"]
drop_cols = list(set(drop_cols + id_cols))
print(f"\n[5] Dropping non-essential columns: {drop_cols}")

df = df_raw.drop(columns=drop_cols)

# ─────────────────────────────────────────────
# 4. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n[6] Missing values BEFORE imputation:\n{df[num_cols].isnull().sum()[df[num_cols].isnull().sum()>0]}")

# Impute numeric columns with the median, grouped by Zone_Type so that
# zone-specific feature distributions are preserved
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df.groupby('Zone_Type')[col].transform(lambda x: x.fillna(x.median()))

print(f"\n[7] Missing values AFTER imputation: {df[num_cols].isnull().sum().sum()}")

df.to_csv("cleaned_dataset.csv", index=False)
print(f"\n[8] Cleaned shape: {df.shape}  -> saved as cleaned_dataset.csv")

# ─────────────────────────────────────────────
# 5. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print(f"\n[9] Summary statistics:\n{df.describe().T[['mean','std','min','max']].round(2).to_string()}")

# ─────────────────────────────────────────────
# 6. EDA PLOTS
# ─────────────────────────────────────────────
feature_cols = [c for c in num_cols if c not in ['Latitude', 'Longitude']]

# --- Plot 1: Zone distribution ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
vc = df['Zone_Type'].value_counts()
axes[0].bar(vc.index, vc.values, color=[ZONE_COLORS[z] for z in vc.index], edgecolor='white')
axes[0].set_title("Zone Type Distribution", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Zone"); axes[0].set_ylabel("Count")

axes[1].pie(vc.values, labels=vc.index, colors=[ZONE_COLORS[z] for z in vc.index],
            autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title("Zone Type Proportion", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_zone_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 01_zone_distribution.png")

# --- Plot 2: Geospatial scatter (Lat/Long colored by zone) ---
fig, ax = plt.subplots(figsize=(9, 8))
for zone, grp in df.groupby('Zone_Type'):
    ax.scatter(grp['Longitude'], grp['Latitude'], c=ZONE_COLORS[zone],
               label=zone, alpha=0.65, s=35, edgecolors='white', linewidths=0.4)
ax.set_xlabel("Longitude", fontsize=11); ax.set_ylabel("Latitude", fontsize=11)
ax.set_title("Geospatial Distribution of Zones (City Map)", fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_geospatial_zones.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 02_geospatial_zones.png")

# --- Plot 3: Feature distributions (histograms) ---
hist_feats = [c for c in feature_cols]
n_cols = 4
n_rows = int(np.ceil(len(hist_feats) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()
for i, col in enumerate(hist_feats):
    axes[i].hist(df[col].dropna(), bins=25, color=PALETTE[i % len(PALETTE)],
                 edgecolor='white', alpha=0.85)
    axes[i].set_title(col.replace('_', ' '), fontsize=9, fontweight='bold')
    axes[i].set_xlabel(""); axes[i].set_ylabel("Frequency")
for j in range(len(hist_feats), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Feature Distributions", fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 03_feature_distributions.png")

# --- Plot 4: Box plots by Zone Type for key features ---
key_feats = ['Population', 'Land_Price_per_sqft', 'Pollution_Index',
              'Num_Factories', 'Num_Shops', 'Green_Space_Pct',
              'Building_Density_per_sqkm', 'Literacy_Rate']
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(key_feats):
    sns.boxplot(data=df, x='Zone_Type', y=col, ax=axes[i],
                palette=ZONE_COLORS, hue='Zone_Type', legend=False)
    axes[i].set_title(col.replace('_', ' '), fontsize=10, fontweight='bold')
    axes[i].set_xlabel("")
plt.suptitle("Key Feature Distributions by Zone Type", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_boxplots_by_zone.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 04_boxplots_by_zone.png")

# --- Plot 5: Correlation heatmap ---
corr = df[feature_cols].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdYlBu_r',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 05_correlation_heatmap.png")

# --- Plot 6: Pairwise relationships for discriminative features ---
disc_feats = ['Land_Price_per_sqft', 'Num_Factories', 'Num_Shops', 'Pollution_Index', 'Zone_Type']
g = sns.pairplot(df[disc_feats], hue='Zone_Type', palette=ZONE_COLORS,
                 diag_kind='kde', plot_kws={'alpha':0.6, 's':25, 'edgecolor':'white'})
g.fig.suptitle("Pairwise Relationships of Discriminative Features", y=1.02, fontsize=14, fontweight='bold')
g.savefig(f"{OUTPUT_DIR}/06_pairplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 06_pairplot.png")

# --- Plot 7: Avg feature profile per zone (grouped bar) ---
profile_feats = ['Literacy_Rate', 'Green_Space_Pct', 'Public_Transport_Access',
                  'Water_Supply_Pct', 'Electricity_Access_Pct']
zone_means = df.groupby('Zone_Type')[profile_feats].mean()
fig, ax = plt.subplots(figsize=(11, 6))
zone_means.T.plot(kind='bar', ax=ax, color=[ZONE_COLORS[z] for z in zone_means.index], edgecolor='white')
ax.set_title("Average Infrastructure & Social Indicators by Zone", fontsize=13, fontweight='bold')
ax.set_ylabel("Value"); ax.set_xlabel("")
plt.xticks(rotation=20, ha='right')
ax.legend(title="Zone")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_zone_profile_bars.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 07_zone_profile_bars.png")

print("\n✅ EDA complete. All plots saved to assets/")
