"""
Smart City Land-Use Classification
Step 2 – DBSCAN Spatial Clustering + KNN Boundary Assignment
Dataset: cleaned_dataset.csv (output of eda.py)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
import warnings, os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#607D8B"]
ZONE_COLORS = {"Residential": "#2196F3", "Commercial": "#FF5722", "Industrial": "#4CAF50", "Unclassified": "#BDBDBD"}
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD CLEANED DATA
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_dataset.csv")
print(f"[1] Loaded cleaned data: {df.shape}")

# ─────────────────────────────────────────────
# GEOSPATIAL FEATURES FOR DBSCAN
# ─────────────────────────────────────────────
# DBSCAN clusters areas purely on geographic proximity (Latitude/Longitude),
# discovering naturally formed "districts" in the city without needing labels.
geo_features = ['Latitude', 'Longitude']
X_geo = df[geo_features].values

scaler_geo = StandardScaler()
X_geo_scaled = scaler_geo.fit_transform(X_geo)

# --- k-distance plot to choose eps ---
nbrs = NearestNeighbors(n_neighbors=4).fit(X_geo_scaled)
distances, _ = nbrs.kneighbors(X_geo_scaled)
dist_sorted = np.sort(distances[:, -1])[::-1]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(dist_sorted, color=PALETTE[0], linewidth=2)
ax.axhline(y=0.18, color='red', linestyle='--', label='eps ≈ 0.18')
ax.set_xlabel("Points sorted by distance"); ax.set_ylabel("4-NN Distance")
ax.set_title("k-Distance Plot for DBSCAN eps Selection (Geospatial)", fontsize=12, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_kdistance_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("[2] Saved: 08_kdistance_plot.png")

# ─────────────────────────────────────────────
# RUN DBSCAN ON GEOSPATIAL COORDINATES
# ─────────────────────────────────────────────
EPS = 0.18
MIN_SAMPLES = 6
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X_geo_scaled)
labels = db.labels_
df['dbscan_cluster'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int(np.sum(labels == -1))
print(f"\n[3] DBSCAN  eps={EPS}, min_samples={MIN_SAMPLES}")
print(f"    Clusters found : {n_clusters}")
print(f"    Noise points   : {n_noise}  ({n_noise/len(df)*100:.1f}%)")

mask_core = labels != -1
if mask_core.sum() > 1 and n_clusters > 1:
    sil = silhouette_score(X_geo_scaled[mask_core], labels[mask_core])
    print(f"    Silhouette score (core): {sil:.4f}")

# ─────────────────────────────────────────────
# ZONE LABEL ASSIGNMENT FOR EACH DBSCAN CLUSTER
# ─────────────────────────────────────────────
# Use the MAJORITY ground-truth Zone_Type within each spatial cluster
# (in a real project, this majority-vote step is replaced by domain-expert
# labelling of discovered geographic clusters)
cluster_zone_map = {}
for c in sorted(set(labels)):
    if c == -1:
        continue
    majority_zone = df.loc[df['dbscan_cluster'] == c, 'Zone_Type'].mode()[0]
    cluster_zone_map[c] = majority_zone
cluster_zone_map[-1] = 'Unclassified'

df['zone_dbscan'] = df['dbscan_cluster'].map(cluster_zone_map)
print(f"\n[4] DBSCAN-derived zone distribution (core points only):")
print(df.loc[df['dbscan_cluster'] != -1, 'zone_dbscan'].value_counts().to_string())

# ─────────────────────────────────────────────
# FEATURE SET FOR KNN BOUNDARY ASSIGNMENT
# ─────────────────────────────────────────────
# KNN uses BOTH geographic location AND socio-economic/infrastructure
# features, so noise points get assigned to the zone whose feature profile
# (and location) they most closely resemble.
knn_features = [
    'Latitude', 'Longitude', 'Population', 'Building_Density_per_sqkm',
    'Avg_Building_Height_Floors', 'Road_Density_km_per_sqkm', 'Land_Price_per_sqft',
    'Literacy_Rate', 'Avg_Monthly_Income', 'Num_Shops', 'Num_Factories',
    'Green_Space_Pct', 'Pollution_Index', 'Traffic_Density_Index',
    'Public_Transport_Access'
]

X_knn = df[knn_features].values
scaler_knn = StandardScaler()
X_knn_scaled = scaler_knn.fit_transform(X_knn)

core_mask = df['dbscan_cluster'] != -1
X_train, y_train = X_knn_scaled[core_mask], df.loc[core_mask, 'zone_dbscan'].values
X_all = X_knn_scaled

K = 7
knn = KNeighborsClassifier(n_neighbors=K, weights='distance')
knn.fit(X_train, y_train)

df['zone_knn'] = knn.predict(X_all)

print(f"\n[5] KNN (k={K}) final zone distribution (all {len(df)} points):")
print(df['zone_knn'].value_counts().to_string())

# ─────────────────────────────────────────────
# EVALUATE AGAINST GROUND-TRUTH ZONE_TYPE
# ─────────────────────────────────────────────
print(f"\n[6] Classification report (zone_knn vs ground-truth Zone_Type):")
print(classification_report(df['Zone_Type'], df['zone_knn']))

cm = confusion_matrix(df['Zone_Type'], df['zone_knn'], labels=['Residential','Commercial','Industrial'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Residential','Commercial','Industrial'],
            yticklabels=['Residential','Commercial','Industrial'], ax=ax)
ax.set_xlabel("Predicted (KNN)"); ax.set_ylabel("Actual (Ground Truth)")
ax.set_title("Confusion Matrix: KNN vs Ground Truth", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: 12_confusion_matrix.png")

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

# --- Plot: DBSCAN clusters on the city map ---
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
cmap = plt.get_cmap('tab10')
for i, c in enumerate(sorted(df['dbscan_cluster'].unique())):
    mask = df['dbscan_cluster'] == c
    color = '#BDBDBD' if c == -1 else cmap(i % 10)
    label = 'Noise' if c == -1 else f'Cluster {c}'
    axes[0].scatter(df.loc[mask, 'Longitude'], df.loc[mask, 'Latitude'],
                    c=[color]*mask.sum(), label=label, alpha=0.7, s=35,
                    edgecolors='white', linewidths=0.4)
axes[0].set_title("DBSCAN Spatial Clusters", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].legend(fontsize=8, ncol=2)

for zone in df['zone_dbscan'].unique():
    mask = df['zone_dbscan'] == zone
    axes[1].scatter(df.loc[mask, 'Longitude'], df.loc[mask, 'Latitude'],
                    c=ZONE_COLORS.get(zone, '#000'), label=zone, alpha=0.7, s=35,
                    edgecolors='white', linewidths=0.4)
axes[1].set_title("DBSCAN Zone Assignment (core points)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
axes[1].legend(fontsize=9)
plt.suptitle("Step 1: DBSCAN on Geospatial Coordinates", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_dbscan_clusters_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[7] Saved: 09_dbscan_clusters_map.png")

# --- Plot: KNN final zone map vs ground truth ---
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for zone in df['Zone_Type'].unique():
    mask = df['Zone_Type'] == zone
    axes[0].scatter(df.loc[mask, 'Longitude'], df.loc[mask, 'Latitude'],
                    c=ZONE_COLORS.get(zone, '#000'), label=zone, alpha=0.7, s=35,
                    edgecolors='white', linewidths=0.4)
axes[0].set_title("Ground Truth Zones", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude"); axes[0].legend()

for zone in df['zone_knn'].unique():
    mask = df['zone_knn'] == zone
    axes[1].scatter(df.loc[mask, 'Longitude'], df.loc[mask, 'Latitude'],
                    c=ZONE_COLORS.get(zone, '#000'), label=zone, alpha=0.7, s=35,
                    edgecolors='white', linewidths=0.4)
axes[1].set_title("KNN Predicted Zones (incl. boundary points)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude"); axes[1].legend()
plt.suptitle("Step 2: KNN Boundary Assignment vs Ground Truth", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_knn_vs_groundtruth_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("[8] Saved: 10_knn_vs_groundtruth_map.png")

# --- Plot: Zone distribution comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
comp = pd.DataFrame({
    'Ground Truth': df['Zone_Type'].value_counts(),
    'KNN Predicted': df['zone_knn'].value_counts()
}).reindex(['Residential','Commercial','Industrial'])
comp.plot(kind='bar', ax=ax, color=['#90CAF9', '#1565C0'], edgecolor='white', width=0.6)
ax.set_title("Zone Distribution: Ground Truth vs KNN Prediction", fontsize=12, fontweight='bold')
ax.set_ylabel("Count"); ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_zone_count_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("[9] Saved: 11_zone_count_comparison.png")

# --- Plot: Feature importance via permutation ---
base_acc = (knn.predict(X_all) == df['zone_knn'].values).mean()
importances = []
for i, feat in enumerate(knn_features):
    Xp = X_all.copy()
    np.random.seed(0)
    np.random.shuffle(Xp[:, i])
    perm_acc = (knn.predict(Xp) == df['zone_knn'].values).mean()
    importances.append(base_acc - perm_acc)

fig, ax = plt.subplots(figsize=(9, 6))
sorted_idx = np.argsort(importances)[::-1]
ax.barh([knn_features[i].replace('_',' ') for i in sorted_idx],
        [importances[i] for i in sorted_idx],
        color=PALETTE[0], edgecolor='white')
ax.set_xlabel("Drop in Prediction Stability (Permutation Importance)")
ax.set_title("Feature Importance for KNN Zone Classification", fontsize=12, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("[10] Saved: 13_feature_importance.png")

# ─────────────────────────────────────────────
# SAVE FINAL DATASET
# ─────────────────────────────────────────────
df.to_csv("final_classified.csv", index=False)
print("\n✅ ML pipeline complete. final_classified.csv saved.")
