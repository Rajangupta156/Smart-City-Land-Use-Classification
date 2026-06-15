"""
Smart City Land-Use Classification Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import (
    silhouette_score, classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings

warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Smart City Land-Use Classifier",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
ZONE_COLORS = {
    "Residential": "#2196F3",
    "Commercial":  "#FF5722",
    "Industrial":  "#4CAF50",
    "Unclassified": "#9E9E9E",
}
ZONE_ICONS = {"Residential": "🏘️", "Commercial": "🏢", "Industrial": "🏭", "Unclassified": "❓"}

KNN_FEATURES = [
    'Latitude', 'Longitude', 'Population', 'Building_Density_per_sqkm',
    'Avg_Building_Height_Floors', 'Road_Density_km_per_sqkm', 'Land_Price_per_sqft',
    'Literacy_Rate', 'Avg_Monthly_Income', 'Num_Shops', 'Num_Factories',
    'Green_Space_Pct', 'Pollution_Index', 'Traffic_Density_Index',
    'Public_Transport_Access'
]

# ── Data Loading & Cleaning ────────────────────
@st.cache_data
def load_and_clean(uploaded_file):
    df_raw = pd.read_csv(uploaded_file)

    # Drop fully-empty or constant columns + ID columns
    drop_cols = []
    for c in df_raw.columns:
        if df_raw[c].isnull().mean() == 1.0:
            drop_cols.append(c)
        elif df_raw[c].nunique(dropna=False) == 1:
            drop_cols.append(c)
    for idc in ['Area_ID', 'Survey_Code']:
        if idc in df_raw.columns:
            drop_cols.append(idc)
    drop_cols = list(set(drop_cols))

    df = df_raw.drop(columns=drop_cols, errors='ignore')

    # Impute numeric NaNs with median, grouped by Zone_Type if present
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            if 'Zone_Type' in df.columns:
                df[col] = df.groupby('Zone_Type')[col].transform(lambda x: x.fillna(x.median()))
            else:
                df[col] = df[col].fillna(df[col].median())

    return df, drop_cols


@st.cache_data
def run_pipeline(_df_marker, eps_val, min_samp, k_val):
    df = st.session_state['df'].copy()

    # --- DBSCAN on geospatial coordinates ---
    X_geo = df[['Latitude', 'Longitude']].values
    scaler_geo = StandardScaler()
    X_geo_scaled = scaler_geo.fit_transform(X_geo)

    db = DBSCAN(eps=eps_val, min_samples=min_samp).fit(X_geo_scaled)
    labels = db.labels_
    df['dbscan_cluster'] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    sil = 0.0
    mask_core = labels != -1
    if mask_core.sum() > 1 and n_clusters > 1:
        sil = float(silhouette_score(X_geo_scaled[mask_core], labels[mask_core]))

    # --- Zone label per DBSCAN cluster ---
    cluster_zone_map = {}
    has_truth = 'Zone_Type' in df.columns
    for c in sorted(set(labels)):
        if c == -1:
            continue
        if has_truth:
            cluster_zone_map[c] = df.loc[df['dbscan_cluster'] == c, 'Zone_Type'].mode()[0]
        else:
            cluster_zone_map[c] = f"Zone_{c}"
    cluster_zone_map[-1] = 'Unclassified'
    df['zone_dbscan'] = df['dbscan_cluster'].map(cluster_zone_map)

    # --- KNN boundary assignment using geo + feature space ---
    feats = [f for f in KNN_FEATURES if f in df.columns]
    X_knn = df[feats].values
    scaler_knn = StandardScaler()
    X_knn_scaled = scaler_knn.fit_transform(X_knn)

    core_mask = df['dbscan_cluster'] != -1
    X_train, y_train = X_knn_scaled[core_mask], df.loc[core_mask, 'zone_dbscan'].values

    knn = KNeighborsClassifier(n_neighbors=k_val, weights='distance')
    knn.fit(X_train, y_train)
    df['zone_knn'] = knn.predict(X_knn_scaled)

    return df, X_geo_scaled, X_knn_scaled, n_clusters, n_noise, sil, knn, feats


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/smart-city.png", width=80)
    st.markdown("## ⚙️ Configuration")

    uploaded = st.file_uploader("Upload Smart City CSV", type=['csv'])
    st.caption("Use the provided `Smart_City_Dataset.csv` or your own file with the same columns.")

    st.markdown("---")
    st.markdown("### DBSCAN Parameters (Geospatial)")
    eps_val  = st.slider("eps (neighbourhood radius)", 0.05, 0.40, 0.18, 0.01)
    min_samp = st.slider("min_samples", 2, 15, 6)

    st.markdown("### KNN Parameters")
    k_val = st.slider("k (neighbours)", 3, 15, 7)

    st.markdown("---")
    run_btn = st.button("🚀 Run Classification", type="primary", use_container_width=True)

# ── Header ────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏙️ Smart City Land-Use Classification</h1>
    <p>Identifying Residential, Commercial & Industrial Zones from Geospatial Census Data</p>
    <p><em>DBSCAN Spatial Clustering + KNN Boundary Assignment</em></p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("👈 Upload **Smart_City_Dataset.csv** using the sidebar to begin.")
    st.markdown("""
    ### How it works
    1. **Upload** the smart city geospatial dataset (CSV)
    2. **EDA tab** — explore distributions, missing values, correlations
    3. **Tune** DBSCAN (spatial) and KNN parameters in the sidebar
    4. **Run** the classification pipeline
    5. **Explore** discovered zones on the city map

    ### Pipeline Overview
    - **Step 1 – DBSCAN**: clusters areas purely by their `Latitude`/`Longitude`,
      discovering natural geographic groupings (districts) without needing labels.
    - **Step 2 – Zone Labelling**: each DBSCAN cluster is labelled
      Residential / Commercial / Industrial based on dominant characteristics.
    - **Step 3 – KNN Boundary Assignment**: areas that DBSCAN marked as *noise*
      (boundary/ambiguous areas) get assigned to the nearest zone using KNN,
      based on both location and socio-economic/infrastructure features.

    ### Zone Legend
    | Zone | Icon | Typical Traits |
    |------|------|-----------------|
    | Residential | 🏘️ | High population, moderate building density, good schools |
    | Commercial  | 🏢 | High land price, many shops, high traffic |
    | Industrial  | 🏭 | Many factories, high pollution, low green space |
    """)
    st.stop()

# ── Load Data ─────────────────────────────────
df_clean, dropped = load_and_clean(uploaded)
st.session_state['df'] = df_clean

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA", "🔬 DBSCAN", "🗂️ KNN Zones", "📈 Evaluation", "📋 Data Table", "ℹ️ About"
])

# ══════════════════════════════════════════════
# TAB 1: EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Exploratory Data Analysis")

    raw_df = pd.read_csv(uploaded)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Records", len(raw_df))
    col2.metric("Raw Columns", raw_df.shape[1])
    col3.metric("Columns Dropped", len(dropped))
    col4.metric("Final Columns", df_clean.shape[1])

    st.markdown(f"**Dropped non-essential columns:** `{', '.join(dropped) if dropped else 'None'}`")

    st.markdown("---")
    st.markdown("#### Missing Values (before cleaning)")
    miss = raw_df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(miss.index, miss.values, color=PALETTE[2], edgecolor='white')
        ax.set_ylabel("Missing Count"); plt.xticks(rotation=45, ha='right')
        ax.set_title("Missing Values per Column (before imputation)", fontweight='bold')
        st.pyplot(fig); plt.close()
    else:
        st.success("No missing values found.")

    if 'Zone_Type' in df_clean.columns:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Zone Type Distribution")
            vc = df_clean['Zone_Type'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(vc.index, vc.values, color=[ZONE_COLORS.get(z, '#999') for z in vc.index], edgecolor='white')
            ax.set_ylabel("Count")
            st.pyplot(fig); plt.close()
        with c2:
            st.markdown("#### Zone Proportion")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(vc.values, labels=vc.index, colors=[ZONE_COLORS.get(z, '#999') for z in vc.index],
                   autopct='%1.1f%%', wedgeprops={'edgecolor':'white','linewidth':2})
            st.pyplot(fig); plt.close()

        st.markdown("#### Geospatial Distribution of Zones (City Map)")
        fig, ax = plt.subplots(figsize=(10, 8))
        for zone, grp in df_clean.groupby('Zone_Type'):
            ax.scatter(grp['Longitude'], grp['Latitude'], c=ZONE_COLORS.get(zone, '#999'),
                       label=zone, alpha=0.65, s=35, edgecolors='white', linewidths=0.4)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Feature Distributions")
    num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                if c not in ['Latitude', 'Longitude']]
    n_per_row = 4
    for i in range(0, len(num_cols), n_per_row):
        row_feats = num_cols[i:i+n_per_row]
        cols = st.columns(len(row_feats))
        for col, feat in zip(cols, row_feats):
            with col:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(df_clean[feat].dropna(), bins=20,
                        color=PALETTE[num_cols.index(feat) % len(PALETTE)],
                        edgecolor='white', alpha=0.85)
                ax.set_title(feat.replace('_',' '), fontsize=8, fontweight='bold')
                st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Correlation Heatmap")
    corr = df_clean[num_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlBu_r', vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, annot_kws={"size": 7},
                mask=np.triu(np.ones_like(corr, dtype=bool)))
    st.pyplot(fig); plt.close()

    if 'Zone_Type' in df_clean.columns:
        st.markdown("---")
        st.markdown("#### Key Features by Zone Type")
        key_feats = [f for f in ['Population','Land_Price_per_sqft','Pollution_Index',
                                  'Num_Factories','Num_Shops','Green_Space_Pct',
                                  'Building_Density_per_sqkm','Literacy_Rate'] if f in df_clean.columns]
        for i in range(0, len(key_feats), 4):
            row_feats = key_feats[i:i+4]
            cols = st.columns(len(row_feats))
            for col, feat in zip(cols, row_feats):
                with col:
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    sns.boxplot(data=df_clean, x='Zone_Type', y=feat, ax=ax,
                                palette=ZONE_COLORS, hue='Zone_Type', legend=False)
                    ax.set_title(feat.replace('_',' '), fontsize=8, fontweight='bold')
                    ax.set_xlabel("")
                    plt.xticks(rotation=20, fontsize=7)
                    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# TAB 2: DBSCAN
# ══════════════════════════════════════════════
with tab2:
    st.markdown("## 🔬 DBSCAN — Spatial Clustering")

    if run_btn or 'results' not in st.session_state:
        with st.spinner("Running DBSCAN + KNN pipeline…"):
            results = run_pipeline(id(df_clean), eps_val, min_samp, k_val)
            st.session_state['results'] = results

    df_res, X_geo, X_knn, n_clusters, n_noise, sil, knn, feats = st.session_state['results']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters Found", n_clusters)
    col2.metric("Noise (boundary) Points", n_noise)
    col3.metric("Silhouette Score", f"{sil:.3f}")
    col4.metric("Core Points", len(df_res) - n_noise)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### DBSCAN Clusters on City Map")
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.get_cmap('tab10')
        for i, c in enumerate(sorted(df_res['dbscan_cluster'].unique())):
            mask = df_res['dbscan_cluster'].values == c
            color = '#BDBDBD' if c == -1 else cmap(i % 10)
            label = 'Noise' if c == -1 else f'Cluster {c}'
            ax.scatter(df_res.loc[mask, 'Longitude'], df_res.loc[mask, 'Latitude'],
                       c=[color]*mask.sum(), label=label, alpha=0.7, s=40,
                       edgecolors='white', linewidths=0.4)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(fontsize=8, ncol=2)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### k-Distance Plot")
        nbrs = NearestNeighbors(n_neighbors=4).fit(X_geo)
        dist, _ = nbrs.kneighbors(X_geo)
        dsorted = np.sort(dist[:, -1])[::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(dsorted, color='#2196F3', lw=2)
        ax.axhline(y=eps_val, color='red', ls='--', label=f'eps = {eps_val}')
        ax.set_xlabel("Points"); ax.set_ylabel("4-NN Distance")
        ax.set_title("k-Distance Plot", fontweight='bold'); ax.legend()
        st.pyplot(fig); plt.close()

    st.markdown("#### DBSCAN Zone Assignment (core points only)")
    fig, ax = plt.subplots(figsize=(10, 7))
    for zone in df_res['zone_dbscan'].unique():
        mask = df_res['zone_dbscan'].values == zone
        ax.scatter(df_res.loc[mask, 'Longitude'], df_res.loc[mask, 'Latitude'],
                   c=ZONE_COLORS.get(zone, '#9E9E9E'), label=zone, alpha=0.75, s=45,
                   edgecolors='white', linewidths=0.4)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(title="Zone")
    ax.set_title("Areas Colored by DBSCAN-Derived Zone (gray = noise/boundary)", fontweight='bold')
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# TAB 3: KNN Zones
# ══════════════════════════════════════════════
with tab3:
    st.markdown("## 🗂️ KNN Final Zone Classification")
    if 'results' not in st.session_state:
        st.info("Run the pipeline first (Tab: DBSCAN).")
    else:
        df_res = st.session_state['results'][0]
        knn = st.session_state['results'][6]
        X_knn = st.session_state['results'][2]
        feats = st.session_state['results'][7]

        vc = df_res['zone_knn'].value_counts()
        cols = st.columns(len(vc))
        for col, (zone, cnt) in zip(cols, vc.items()):
            col.metric(f"{ZONE_ICONS.get(zone,'')} {zone}", cnt, f"{cnt/len(df_res)*100:.1f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Final Zone Map (after KNN boundary assignment)")
            fig, ax = plt.subplots(figsize=(7, 6))
            for zone in df_res['zone_knn'].unique():
                mask = df_res['zone_knn'].values == zone
                ax.scatter(df_res.loc[mask, 'Longitude'], df_res.loc[mask, 'Latitude'],
                           c=ZONE_COLORS.get(zone, '#9E9E9E'), label=zone, alpha=0.78, s=50,
                           edgecolors='white', linewidths=0.4)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.legend(title="Zone", fontsize=9)
            ax.set_title("KNN Predicted Zones (all areas)", fontweight='bold')
            st.pyplot(fig); plt.close()

        with c2:
            if 'Zone_Type' in df_res.columns:
                st.markdown("#### Ground Truth Zone Map (for comparison)")
                fig, ax = plt.subplots(figsize=(7, 6))
                for zone in df_res['Zone_Type'].unique():
                    mask = df_res['Zone_Type'].values == zone
                    ax.scatter(df_res.loc[mask, 'Longitude'], df_res.loc[mask, 'Latitude'],
                               c=ZONE_COLORS.get(zone, '#9E9E9E'), label=zone, alpha=0.78, s=50,
                               edgecolors='white', linewidths=0.4)
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                ax.legend(title="Zone", fontsize=9)
                ax.set_title("Ground Truth Zones", fontweight='bold')
                st.pyplot(fig); plt.close()
            else:
                st.markdown("#### Zone Distribution")
                fig, ax = plt.subplots(figsize=(7, 6))
                bars = ax.bar(vc.index, vc.values,
                              color=[ZONE_COLORS.get(z,'#9E9E9E') for z in vc.index],
                              edgecolor='white', width=0.5)
                for bar, cnt in zip(bars, vc.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                            str(cnt), ha='center', va='bottom', fontweight='bold')
                ax.set_ylabel("Count")
                st.pyplot(fig); plt.close()

        if 'Zone_Type' in df_res.columns:
            st.markdown("---")
            st.markdown("#### Model Evaluation vs Ground Truth")
            c1, c2 = st.columns([1, 1])
            with c1:
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(df_res['Zone_Type'], df_res['zone_knn'])
                st.metric("Overall Accuracy", f"{acc*100:.1f}%")
                report = classification_report(df_res['Zone_Type'], df_res['zone_knn'], output_dict=True)
                report_df = pd.DataFrame(report).T.round(3)
                st.dataframe(report_df, use_container_width=True)
            with c2:
                cm = confusion_matrix(df_res['Zone_Type'], df_res['zone_knn'],
                                       labels=['Residential','Commercial','Industrial'])
                fig, ax = plt.subplots(figsize=(5, 4.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Residential','Commercial','Industrial'],
                            yticklabels=['Residential','Commercial','Industrial'], ax=ax)
                ax.set_xlabel("Predicted (KNN)"); ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix", fontweight='bold')
                st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("#### Feature Importance (Permutation)")
        base_acc = (knn.predict(X_knn) == df_res['zone_knn'].values).mean()
        importances = []
        for i, feat in enumerate(feats):
            Xp = X_knn.copy()
            np.random.seed(0); np.random.shuffle(Xp[:, i])
            perm_acc = (knn.predict(Xp) == df_res['zone_knn'].values).mean()
            importances.append(base_acc - perm_acc)
        sorted_idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh([feats[i].replace('_',' ') for i in sorted_idx],
                [importances[i] for i in sorted_idx], color=PALETTE[0], edgecolor='white')
        ax.invert_yaxis(); ax.set_xlabel("Importance (drop in stability)")
        st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("#### Sample Areas by Zone")
        for zone in [z for z in ['Residential','Commercial','Industrial'] if z in df_res['zone_knn'].values]:
            with st.expander(f"{ZONE_ICONS.get(zone,'')} {zone} — sample areas"):
                show_cols = [c for c in ['Area_Name','Latitude','Longitude','Population',
                                          'Land_Price_per_sqft','Num_Factories','Num_Shops',
                                          'Pollution_Index'] if c in df_res.columns]
                sample = df_res[df_res['zone_knn'] == zone][show_cols].head(10)
                st.dataframe(sample, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4: Evaluation (Accuracy / Precision / Recall / F1)
# ══════════════════════════════════════════════
with tab4:
    st.markdown("## 📈 Model Evaluation")
    if 'results' not in st.session_state:
        st.info("Run the pipeline first (Tab: DBSCAN).")
    elif 'Zone_Type' not in df_clean.columns:
        st.warning("Ground-truth `Zone_Type` column not found — cannot compute evaluation metrics.")
    else:
        df_res = st.session_state['results'][0]
        n_clusters, n_noise = st.session_state['results'][3], st.session_state['results'][4]
        core_mask = df_res['dbscan_cluster'] != -1
        y_true = df_res['Zone_Type']
        ZONE_ORDER_EVAL = [z for z in ['Residential','Commercial','Industrial'] if z in y_true.unique()]

        def get_metrics(y_t, y_p):
            return {
                'Accuracy': accuracy_score(y_t, y_p),
                'Precision (macro)': precision_score(y_t, y_p, average='macro', zero_division=0),
                'Recall (macro)': recall_score(y_t, y_p, average='macro', zero_division=0),
                'F1-Score (macro)': f1_score(y_t, y_p, average='macro', zero_division=0),
                'Precision (weighted)': precision_score(y_t, y_p, average='weighted', zero_division=0),
                'Recall (weighted)': recall_score(y_t, y_p, average='weighted', zero_division=0),
                'F1-Score (weighted)': f1_score(y_t, y_p, average='weighted', zero_division=0),
            }

        m_dbscan = get_metrics(y_true[core_mask], df_res.loc[core_mask, 'zone_dbscan'])
        m_knn    = get_metrics(y_true, df_res['zone_knn'])

        st.markdown("#### Overall Metrics — DBSCAN-Only vs DBSCAN + KNN")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m_knn['Accuracy']*100:.1f}%", f"{(m_knn['Accuracy']-m_dbscan['Accuracy'])*100:+.1f}pp")
        c2.metric("Precision (macro)", f"{m_knn['Precision (macro)']*100:.1f}%", f"{(m_knn['Precision (macro)']-m_dbscan['Precision (macro)'])*100:+.1f}pp")
        c3.metric("Recall (macro)", f"{m_knn['Recall (macro)']*100:.1f}%", f"{(m_knn['Recall (macro)']-m_dbscan['Recall (macro)'])*100:+.1f}pp")
        c4.metric("F1-Score (macro)", f"{m_knn['F1-Score (macro)']*100:.1f}%", f"{(m_knn['F1-Score (macro)']-m_dbscan['F1-Score (macro)'])*100:+.1f}pp")
        st.caption("Delta shown vs DBSCAN-only (core points, noise excluded). KNN is evaluated on ALL points.")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Metric Comparison")
            metrics_to_plot = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']
            fig, ax = plt.subplots(figsize=(7, 5))
            x = np.arange(len(metrics_to_plot)); width = 0.35
            v1 = [m_dbscan[m] for m in metrics_to_plot]
            v2 = [m_knn[m] for m in metrics_to_plot]
            ax.bar(x - width/2, v1, width, label='DBSCAN-Only', color=PALETTE[2], edgecolor='white')
            ax.bar(x + width/2, v2, width, label='DBSCAN + KNN', color=PALETTE[0], edgecolor='white')
            for i, (a, b) in enumerate(zip(v1, v2)):
                ax.text(i-width/2, a+0.01, f"{a:.2f}", ha='center', fontsize=8, fontweight='bold')
                ax.text(i+width/2, b+0.01, f"{b:.2f}", ha='center', fontsize=8, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(metrics_to_plot, fontsize=8)
            ax.set_ylim(0, 1.1); ax.legend()
            st.pyplot(fig); plt.close()

        with c2:
            st.markdown("#### Confusion Matrix (DBSCAN + KNN)")
            cm = confusion_matrix(y_true, df_res['zone_knn'], labels=ZONE_ORDER_EVAL)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=ZONE_ORDER_EVAL, yticklabels=ZONE_ORDER_EVAL)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("#### Per-Class Precision / Recall / F1-Score (DBSCAN + KNN)")
        report = classification_report(y_true, df_res['zone_knn'], output_dict=True, zero_division=0)
        prf = pd.DataFrame({
            'Precision': [report[c]['precision'] for c in ZONE_ORDER_EVAL],
            'Recall':    [report[c]['recall'] for c in ZONE_ORDER_EVAL],
            'F1-Score':  [report[c]['f1-score'] for c in ZONE_ORDER_EVAL],
        }, index=ZONE_ORDER_EVAL)

        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            prf.plot(kind='bar', ax=ax, color=PALETTE[:3], edgecolor='white', width=0.7)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)
            ax.set_ylim(0, 1.15); plt.xticks(rotation=0)
            ax.legend(title="Metric")
            st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(prf.round(3), use_container_width=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        st.markdown("---")
        st.markdown("#### KNN Sensitivity: Accuracy & F1 vs k")
        st.caption("Recomputes KNN with k = 1..15 using the current DBSCAN clusters as training labels.")
        X_knn_all = st.session_state['results'][2]
        k_values = list(range(1, 16))
        acc_list, f1_list = [], []
        for k in k_values:
            knn_k = KNeighborsClassifier(n_neighbors=k, weights='distance')
            knn_k.fit(X_knn_all[core_mask], df_res.loc[core_mask, 'zone_dbscan'].values)
            pred_k = knn_k.predict(X_knn_all)
            acc_list.append(accuracy_score(y_true, pred_k))
            f1_list.append(f1_score(y_true, pred_k, average='macro', zero_division=0))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(k_values, acc_list, marker='o', color=PALETTE[0], label='Accuracy')
        ax.plot(k_values, f1_list, marker='s', color=PALETTE[2], label='F1-Score (macro)')
        ax.axvline(x=k_val, color='gray', linestyle='--', label=f'Selected k = {k_val}')
        ax.set_xlabel("k (number of neighbours)"); ax.set_ylabel("Score")
        ax.set_xticks(k_values); ax.legend()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        st.download_button(
            "⬇️ Download Evaluation Summary (CSV)",
            pd.DataFrame([
                {'Model': 'DBSCAN-Only (core points)', **m_dbscan, 'n_samples': int(core_mask.sum())},
                {'Model': 'DBSCAN + KNN (all points)', **m_knn, 'n_samples': len(df_res)},
            ]).round(4).to_csv(index=False),
            file_name="model_evaluation_summary.csv", mime="text/csv"
        )

# ══════════════════════════════════════════════
# TAB 5: Data Table
# ══════════════════════════════════════════════
with tab5:
    st.markdown("## 📋 Classified Data Table")
    if 'results' not in st.session_state:
        st.info("Run the pipeline first.")
    else:
        df_res = st.session_state['results'][0]
        zone_filter = st.multiselect("Filter by Predicted Zone", df_res['zone_knn'].unique(),
                                      default=list(df_res['zone_knn'].unique()))
        df_show = df_res[df_res['zone_knn'].isin(zone_filter)]
        st.write(f"Showing **{len(df_show)}** records")

        display_cols = [c for c in df_show.columns if c not in ['dbscan_cluster','zone_dbscan']]
        st.dataframe(df_show[display_cols].round(3), use_container_width=True, height=500)

        csv = df_show.to_csv(index=False)
        st.download_button("⬇️ Download Classified CSV", csv,
                            file_name="classified_zones.csv", mime="text/csv")

# ══════════════════════════════════════════════
# TAB 6: About
# ══════════════════════════════════════════════
with tab6:
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    ### Smart City Land-Use Classification

    This application classifies city areas into **Residential**, **Commercial**, and
    **Industrial** zones using geospatial census-style data, combining unsupervised
    spatial clustering (DBSCAN) with supervised boundary refinement (KNN).

    ---

    ### Methodology

    #### Step 1 — Data Cleaning (EDA)
    - Removed identifier columns (`Area_ID`, `Survey_Code`)
    - Removed constant columns (`Data_Source`) and fully-empty columns (`Remarks`)
    - Imputed missing numeric values (median, grouped by zone where ground truth exists)
    - Explored distributions, correlations, and zone-wise feature profiles

    #### Step 2 — DBSCAN Spatial Clustering
    - **Density-Based Spatial Clustering of Applications with Noise**
    - Clusters areas purely by `Latitude` / `Longitude`
    - Discovers natural geographic groupings (districts) without pre-defining cluster count
    - Marks scattered/boundary areas as **noise** (label = -1)
    - Each discovered cluster is labelled by its dominant zone characteristics

    #### Step 3 — KNN Boundary Assignment
    - Trained on DBSCAN **core** points using both location *and*
      socio-economic / infrastructure features
    - Predicts a zone for **every** area, including DBSCAN noise points
    - Produces smooth, complete land-use zone boundaries across the city

    ---

    ### Feature Reference

    | Feature | Description |
    |---------|-------------|
    | Population | Total residents in the area |
    | Building_Density_per_sqkm | Buildings per square km |
    | Avg_Building_Height_Floors | Average building height |
    | Road_Density_km_per_sqkm | Road length per square km |
    | Land_Price_per_sqft | Average land price |
    | Literacy_Rate | Fraction of literate population |
    | Avg_Monthly_Income | Average household income |
    | Num_Shops / Num_Factories | Commercial / industrial activity counts |
    | Green_Space_Pct | % of area under green cover |
    | Pollution_Index | Composite pollution score |
    | Traffic_Density_Index | Composite traffic congestion score |
    | Public_Transport_Access | % access to public transport |

    ---

    ### Typical Zone Signatures

    | Zone | Signature |
    |------|-----------|
    | 🏘️ Residential | High population, moderate density, good schools & green space |
    | 🏢 Commercial | High land price, many shops, high traffic, tall buildings |
    | 🏭 Industrial | Many factories, high pollution, low green space, low literacy |

    ---

    *Built for an AIML Project — Smart City Geospatial Land-Use Analysis*
    """)
