import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

REF_CSV = "Sales Dataset.csv"
pipeline_path = "pipeline.pkl"
model_path = "kmeans_model.pkl"

if not os.path.exists(REF_CSV):
    st.error(f"File referensi '{REF_CSV}' tidak ditemukan. Letakkan file tersebut di direktori yang sama.")
    st.stop()

df_ref = pd.read_csv(REF_CSV)
if "Order Date" in df_ref.columns:
    df_ref["Order Date"] = pd.to_datetime(df_ref["Order Date"], errors="coerce")

def build_pipeline_and_model(df):
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month

    for col in ["City", "Sub-Category"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0.0)

    num_features = ["Amount", "Profit", "Quantity"]
    cat_features = ["Category", "PaymentMode"]
    freq_features = ["City_freq", "Sub-Category_freq"]
    time_features = ["Year", "Month"]

    all_features = num_features + cat_features + freq_features + time_features

    preprocess = ColumnTransformer([
        ("num", RobustScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ("freq", "passthrough", freq_features),
        ("time", "passthrough", time_features),
    ])

    pipeline = Pipeline([("prep", preprocess)])

    X = pipeline.fit_transform(df[all_features])

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X)

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(model, model_path)

    return pipeline, model

try:
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
except Exception:
    pipeline, model = build_pipeline_and_model(df_ref)

st.set_page_config(page_title="Dashboard Segmentasi Produk (K-Means)")
st.title("Dashboard Segmentasi Produk (K-Means)")
st.sidebar.header("Input Data (Manual)")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "Amount", "Profit", "Quantity", "Category",
        "Sub-Category", "City", "PaymentMode", "Order Date"
    ])

def manual_input():
    Amount = st.sidebar.number_input("Amount", min_value=0.0, step=100.0, format="%.2f")
    Profit = st.sidebar.number_input("Profit", min_value=0.0, step=50.0, format="%.2f")
    Quantity = st.sidebar.number_input("Quantity", min_value=1, step=1)

    Category = st.sidebar.selectbox("Category", df_ref["Category"].dropna().unique())
    SubCat = st.sidebar.selectbox("Sub-Category", df_ref["Sub-Category"].dropna().unique())
    City = st.sidebar.selectbox("City", df_ref["City"].dropna().unique())
    Payment = st.sidebar.selectbox("Payment Mode", df_ref["PaymentMode"].dropna().unique())
    OrderDate = st.sidebar.date_input("Order Date")

    return pd.DataFrame([{
        "Amount": Amount,
        "Profit": Profit,
        "Quantity": Quantity,
        "Category": Category,
        "Sub-Category": SubCat,
        "City": City,
        "PaymentMode": Payment,
        "Order Date": pd.to_datetime(OrderDate)
    }])

df_input = manual_input()
if st.sidebar.button("Input data"):
    st.session_state.data = pd.concat([st.session_state.data, df_input], ignore_index=True)
    st.success("Data berhasil ditambahkan ke session.")

st.subheader("Data Saat Ini")
st.dataframe(st.session_state.data)

def preprocess_for_model(data):
    data = data.copy()
    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    data["Year"] = data["Order Date"].dt.year
    data["Month"] = data["Order Date"].dt.month
    data["City_freq"] = data["City"].map(df_ref["City"].value_counts(normalize=True)).fillna(0.0)
    data["Sub-Category_freq"] = data["Sub-Category"].map(df_ref["Sub-Category"].value_counts(normalize=True)).fillna(0.0)
    return data

try:
    df_single = preprocess_for_model(df_input.copy())
    feature_cols = ["Amount","Profit","Quantity","Category","PaymentMode","City_freq","Sub-Category_freq","Year","Month"]
    X_single = pipeline.transform(df_single[feature_cols])
    df_single["cluster"] = model.predict(X_single)
    st.subheader("Prediksi untuk Input Terbaru")
    st.write(df_single)
except Exception:
    pass

if st.button("Proses K-Means"):
    if len(st.session_state.data) < 2:
        st.warning("Minimal 2 data untuk clustering!")
    else:
        df_cluster = preprocess_for_model(st.session_state.data.copy())
        feature_cols = ["Amount","Profit","Quantity","Category","PaymentMode","City_freq","Sub-Category_freq","Year","Month"]
        X = pipeline.transform(df_cluster[feature_cols])
        df_cluster["Cluster"] = model.predict(X)

        cluster_labels = {i: f"Cluster {i}" for i in range(getattr(model, "n_clusters", 3))}
        df_cluster["Label"] = df_cluster["Cluster"].map(cluster_labels)

        st.success("Clustering berhasil dilakukan!")
        st.dataframe(df_cluster)

        df_cluster["Quantity"] = pd.to_numeric(df_cluster["Quantity"], errors="coerce").fillna(1)
        df_plot = df_cluster.dropna(subset=["Amount", "Profit"]).copy()

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df_plot["Amount"],
            df_plot["Profit"],
            s=(df_plot["Quantity"] + 1) * 15,
            c=df_plot["Cluster"].astype(int),
            alpha=0.8
        )

        ax.set_xlabel("Amount")
        ax.set_ylabel("Profit")
        ax.set_title("Visualisasi K-Means")

        handles, _ = scatter.legend_elements()
        labels = [cluster_labels.get(i, f"Cluster {i}") for i in range(len(handles))]
        ax.legend(handles, labels, title="Cluster")

        st.pyplot(fig)

        if st.button("Simpan hasil clustering ke CSV"): 
            out_path = "clustered_output.csv"
            df_cluster.to_csv(out_path, index=False)
            st.success(f"Clustered dataset disimpan ke {out_path}")

df = pd.read_csv(REF_CSV)
df_raw = df.copy()
df.drop_duplicates(inplace=True)
df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')

num_cols = ["Amount","Profit","Quantity"]
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

outlier_mask = ((df[num_cols] < (Q1 - 1.5 * IQR)) | 
                (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

st.subheader("Data Preparation Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"Jumlah Baris & Kolom: {df.shape}")
with col2:
    st.metric("Missing Values", df.isnull().sum().sum())
with col3:
    st.metric("Jumlah Outlier", outlier_mask.sum())

st.subheader("Distribusi Category")
st.write(df["Category"].value_counts())

fig, ax = plt.subplots()
df["Order Date"].dt.month.value_counts().sort_index().plot(kind='bar', ax=ax)
ax.set_title("Distribusi Bulan")
st.pyplot(fig)

df_clean = df[~outlier_mask].copy()
df_clean["Year"] = df_clean["Order Date"].dt.year
df_clean["Month"] = df_clean["Order Date"].dt.month

for col in ["City", "Sub-Category"]:
    freq = df_clean[col].value_counts(normalize=True)
    df_clean[f"{col}_freq"] = df_clean[col].map(freq).fillna(0.0)

num_features = ["Amount", "Profit", "Quantity"]
cat_features = ["Category", "PaymentMode"]
freq_features = ["City_freq", "Sub-Category_freq"]
time_features = ["Year", "Month"]

all_features = num_features + cat_features + freq_features + time_features

X_clean = pipeline.transform(df_clean[all_features])
df_clean["cluster"] = model.predict(X_clean)

st.subheader("Clustering pada Data Referensi")
st.dataframe(df_clean[["Amount", "Profit", "Quantity", "cluster"]].head())

num_features_plot = ["Amount", "Profit", "Quantity"]
cluster_mean = df_clean.groupby("cluster")[num_features_plot].mean()

fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(cluster_mean.values, cmap="viridis")
plt.colorbar(im)
ax.set_xticks(range(len(num_features_plot)))
ax.set_xticklabels(num_features_plot)
ax.set_yticks([0,1,2])
ax.set_yticklabels([f"Cluster {i}" for i in range(3)])
ax.set_title("Heatmap Mean Features per Cluster")
st.pyplot(fig)

silhouette_avg = silhouette_score(X_clean, df_clean["cluster"])
st.metric("Silhouette Score", f"{silhouette_avg:.3f}")