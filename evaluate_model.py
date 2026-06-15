"""
Smart City Land-Use Classification
Step 3 – Model Evaluation
Computes Accuracy, Precision, Recall, F1-Score for:
  (a) DBSCAN-only zone assignment (core points)
  (b) DBSCAN + KNN boundary assignment (all points)
  (c) KNN with varying k (sensitivity analysis)

Run AFTER eda.py and ml_pipeline.py (uses cleaned_dataset.csv / final_classified.csv)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings, os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
ZONE_ORDER = ['Residential', 'Commercial', 'Industrial']
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_dataset.csv")
print(f"[1] Loaded cleaned data: {df.shape}")

if 'Zone_Type' not in df.columns:
    raise ValueError("Ground-truth 'Zone_Type' column not found — cannot evaluate.")

y_true = df['Zone_Type']

# ─────────────────────────────────────────────
# REBUILD PIPELINE (same settings as ml_pipeline.py)
# ─────────────────────────────────────────────
EPS, MIN_SAMPLES, K = 0.18, 6, 7

X_geo = df[['Latitude', 'Longitude']].values
X_geo_scaled = StandardScaler().fit_transform(X_geo)

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X_geo_scaled)
df['dbscan_cluster'] = db.labels_

cluster_zone_map = {}
for c in sorted(set(db.labels_)):
    if c == -1:
        continue
    cluster_zone_map[c] = df.loc[df['dbscan_cluster'] == c, 'Zone_Type'].mode()[0]
cluster_zone_map[-1] = 'Unclassified'
df['zone_dbscan'] = df['dbscan_cluster'].map(cluster_zone_map)

knn_features = [
    'Latitude', 'Longitude', 'Population', 'Building_Density_per_sqkm',
    'Avg_Building_Height_Floors', 'Road_Density_km_per_sqkm', 'Land_Price_per_sqft',
    'Literacy_Rate', 'Avg_Monthly_Income', 'Num_Shops', 'Num_Factories',
    'Green_Space_Pct', 'Pollution_Index', 'Traffic_Density_Index',
    'Public_Transport_Access'
]
X_knn = StandardScaler().fit_transform(df[knn_features].values)
core_mask = df['dbscan_cluster'] != -1

knn = KNeighborsClassifier(n_neighbors=K, weights='distance')
knn.fit(X_knn[core_mask], df.loc[core_mask, 'zone_dbscan'].values)
df['zone_knn'] = knn.predict(X_knn)

print(f"[2] Pipeline rebuilt — eps={EPS}, min_samples={MIN_SAMPLES}, k={K}")

# ─────────────────────────────────────────────
# METRIC HELPER
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name, mask=None):
    """Returns a dict of overall + per-class metrics."""
    if mask is not None:
        y_true_eval = y_true[mask]
        y_pred_eval = y_pred[mask]
    else:
        y_true_eval = y_true
        y_pred_eval = y_pred

    acc = accuracy_score(y_true_eval, y_pred_eval)
    prec_macro = precision_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
    rec_macro  = recall_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
    f1_macro   = f1_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
    prec_w = precision_score(y_true_eval, y_pred_eval, average='weighted', zero_division=0)
    rec_w  = recall_score(y_true_eval, y_pred_eval, average='weighted', zero_division=0)
    f1_w   = f1_score(y_true_eval, y_pred_eval, average='weighted', zero_division=0)

    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    print(f"  Evaluated on {len(y_true_eval)} samples")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Precision (macro)  : {prec_macro:.4f}   |  (weighted): {prec_w:.4f}")
    print(f"  Recall    (macro)  : {rec_macro:.4f}   |  (weighted): {rec_w:.4f}")
    print(f"  F1-score  (macro)  : {f1_macro:.4f}   |  (weighted): {f1_w:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true_eval, y_pred_eval, zero_division=0))

    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision (macro)': prec_macro,
        'Recall (macro)': rec_macro,
        'F1-Score (macro)': f1_macro,
        'Precision (weighted)': prec_w,
        'Recall (weighted)': rec_w,
        'F1-Score (weighted)': f1_w,
        'n_samples': len(y_true_eval),
    }, classification_report(y_true_eval, y_pred_eval, output_dict=True, zero_division=0)

# ─────────────────────────────────────────────
# (a) DBSCAN-ONLY  (evaluated on core points only — noise excluded)
# ─────────────────────────────────────────────
metrics_dbscan, report_dbscan = compute_metrics(
    y_true, df['zone_dbscan'], "DBSCAN-Only (core points, noise excluded)",
    mask=core_mask
)

# ─────────────────────────────────────────────
# (b) DBSCAN + KNN  (evaluated on ALL points)
# ─────────────────────────────────────────────
metrics_knn, report_knn = compute_metrics(
    y_true, df['zone_knn'], "DBSCAN + KNN Boundary Assignment (all points)"
)

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
summary_df = pd.DataFrame([metrics_dbscan, metrics_knn])
print(f"\n{'='*60}\n  SUMMARY COMPARISON\n{'='*60}")
print(summary_df.round(4).to_string(index=False))
summary_df.to_csv("model_evaluation_summary.csv", index=False)
print("\nSaved: model_evaluation_summary.csv")

# ─────────────────────────────────────────────
# PLOT 1: Overall metric comparison (bar chart)
# ─────────────────────────────────────────────
metrics_to_plot = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_to_plot))
width = 0.35

vals_dbscan = [metrics_dbscan[m] for m in metrics_to_plot]
vals_knn    = [metrics_knn[m] for m in metrics_to_plot]

ax.bar(x - width/2, vals_dbscan, width, label='DBSCAN-Only (core pts)', color=PALETTE[2], edgecolor='white')
ax.bar(x + width/2, vals_knn, width, label='DBSCAN + KNN (all pts)', color=PALETTE[0], edgecolor='white')

for i, (v1, v2) in enumerate(zip(vals_dbscan, vals_knn)):
    ax.text(i - width/2, v1 + 0.01, f"{v1:.2f}", ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width/2, v2 + 0.01, f"{v2:.2f}", ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_model_metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[3] Saved: 14_model_metrics_comparison.png")

# ─────────────────────────────────────────────
# PLOT 2: Per-class Precision / Recall / F1 (final KNN model)
# ─────────────────────────────────────────────
classes = [c for c in ZONE_ORDER if c in report_knn]
prf = pd.DataFrame({
    'Precision': [report_knn[c]['precision'] for c in classes],
    'Recall':    [report_knn[c]['recall'] for c in classes],
    'F1-Score':  [report_knn[c]['f1-score'] for c in classes],
}, index=classes)

fig, ax = plt.subplots(figsize=(9, 6))
prf.plot(kind='bar', ax=ax, color=PALETTE[:3], edgecolor='white', width=0.7)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)
ax.set_ylim(0, 1.15)
ax.set_title("Per-Class Precision / Recall / F1-Score — DBSCAN + KNN Model", fontsize=13, fontweight='bold')
ax.set_ylabel("Score"); ax.set_xlabel("")
plt.xticks(rotation=0)
ax.legend(title="Metric")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/15_per_class_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("[4] Saved: 15_per_class_metrics.png")

# ─────────────────────────────────────────────
# PLOT 3: Confusion matrices (DBSCAN-only vs DBSCAN+KNN)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

cm_dbscan = confusion_matrix(y_true[core_mask], df.loc[core_mask, 'zone_dbscan'], labels=ZONE_ORDER)
sns.heatmap(cm_dbscan, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=ZONE_ORDER, yticklabels=ZONE_ORDER)
axes[0].set_title("DBSCAN-Only (core points)", fontweight='bold')
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

cm_knn = confusion_matrix(y_true, df['zone_knn'], labels=ZONE_ORDER)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=ZONE_ORDER, yticklabels=ZONE_ORDER)
axes[1].set_title("DBSCAN + KNN (all points)", fontweight='bold')
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/16_confusion_matrix_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("[5] Saved: 16_confusion_matrix_comparison.png")

# ─────────────────────────────────────────────
# PLOT 4: KNN sensitivity — accuracy/F1 vs k
# ─────────────────────────────────────────────
k_values = list(range(1, 16))
acc_list, f1_list = [], []
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn_k.fit(X_knn[core_mask], df.loc[core_mask, 'zone_dbscan'].values)
    pred_k = knn_k.predict(X_knn)
    acc_list.append(accuracy_score(y_true, pred_k))
    f1_list.append(f1_score(y_true, pred_k, average='macro', zero_division=0))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_values, acc_list, marker='o', color=PALETTE[0], label='Accuracy')
ax.plot(k_values, f1_list, marker='s', color=PALETTE[2], label='F1-Score (macro)')
ax.axvline(x=K, color='gray', linestyle='--', label=f'Selected k = {K}')
ax.set_xlabel("k (number of neighbours)"); ax.set_ylabel("Score")
ax.set_title("KNN Sensitivity Analysis: Accuracy & F1 vs k", fontsize=13, fontweight='bold')
ax.set_xticks(k_values)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/17_knn_k_sensitivity.png", dpi=150, bbox_inches='tight')
plt.close()
print("[6] Saved: 17_knn_k_sensitivity.png")

print("\n✅ Evaluation complete. Metrics + plots saved to assets/ and model_evaluation_summary.csv")
