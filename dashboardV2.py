import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pipeline = joblib.load("pipeline.pkl")
model = joblib.load("kmeans_model.pkl")
df_ref = pd.read_csv("Sales Dataset.csv")
df_ref["Order Date"] = pd.to_datetime(df_ref["Order Date"], errors="coerce")

st.title("📊 Dashboard Segmentasi Produk (K-Means)")

st.sidebar.header("Input Data")

# ===========================
# Inisialisasi session state
# ===========================
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "Amount", "Profit", "Quantity", "Category",
        "Sub-Category", "City", "PaymentMode", "Order Date"
    ])

# ===========================
# Fungsi Input Manual
# ===========================
def manual_input():
    Amount = st.sidebar.number_input("Amount", min_value=0.0, step=100.0)
    Profit = st.sidebar.number_input("Profit", min_value=0.0, step=50.0)
    Quantity = st.sidebar.number_input("Quantity", min_value=1, step=1)

    Category = st.sidebar.selectbox("Category", df_ref["Category"].unique())
    SubCat = st.sidebar.selectbox("Sub-Category", df_ref["Sub-Category"].unique())
    City = st.sidebar.selectbox("City", df_ref["City"].unique())
    Payment = st.sidebar.selectbox("Payment Mode", df_ref["PaymentMode"].unique())
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


# ===========================
# PILIHAN MANUAL / UPLOAD CSV
# ===========================
choice = st.radio("Mode Input:", ["Manual", "Upload CSV"])

if choice == "Manual":
    df_input = manual_input()
    if st.sidebar.button("Input data"):
        st.session_state.data = pd.concat([st.session_state.data, df_input], ignore_index=True)
        st.success("Data berhasil ditambahkan!")
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)

        if "Order Date" in df_input.columns:
            df_input["Order Date"] = pd.to_datetime(df_input["Order Date"], errors="coerce")

        if df_input["Order Date"].isna().any():
            st.warning("Beberapa tanggal tidak dapat diparse. Pastikan format YYYY-MM-DD atau DD/MM/YYYY.")
    else:
        st.stop()

# ===========================
# Ekstraksi Year & Month
# ===========================
st.session_state.data["Year"] = pd.to_datetime(st.session_state.data["Order Date"]).dt.year
st.session_state.data["Month"] = pd.to_datetime(st.session_state.data["Order Date"]).dt.month

try:
    df_input["Year"] = df_input["Order Date"].dt.year
    df_input["Month"] = df_input["Order Date"].dt.month
except:
    st.error("Error: Kolom 'Order Date' bukan datetime. Periksa CSV Anda.")
    st.stop()

# ===========================
# FITUR UNTUK MODEL
# ===========================
features = [
    "Amount","Profit","Quantity","Sub-Category_freq","City_freq",
    "Category","PaymentMode","Year","Month"
]

# ===========================
# Tampilkan Data Saat Ini
# ===========================
st.subheader("Data Saat Ini")
st.dataframe(st.session_state.data)

# ===========================
# PREPROCESSING sebelum K-Means
# ===========================
def preprocess(data):
    data = data.copy()
    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    data["Year"] = data["Order Date"].dt.year
    data["Month"] = data["Order Date"].dt.month
    data["City_freq"] = data["City"].map(df_ref["City"].value_counts(normalize=True)).fillna(0.01)
    data["Sub-Category_freq"] = data["Sub-Category"].map(df_ref["Sub-Category"].value_counts(normalize=True)).fillna(0.01)
    return data

df_input = preprocess(df_input.copy())
X = pipeline.transform(df_input[features])
df_input["cluster"] = model.predict(X)
cluster_labels = {
    0: "Electronics Customers",
    1: "Office Supplies Customers",
    2: "Furniture Customers"
}

df_input["Label"] = df_input["cluster"].map(cluster_labels)

st.write(df_input)

# ===========================
# Proses K-Means
# ===========================
if st.button("Proses K-Means"):
    if len(st.session_state.data) < 2:
        st.warning("Minimal 2 data untuk clustering!")
    else:
        df_cluster = preprocess(st.session_state.data.copy())

        X = pipeline.transform(df_cluster[[
            "Category", "PaymentMode",
            "Amount", "Profit", "Quantity",
            "City_freq", "Sub-Category_freq",
            "Year", "Month"
        ]])
        df_cluster["Cluster"] = model.predict(X)

        cluster_labels = {i: f"Cluster {i}" for i in range(getattr(model, "n_clusters", 3))}
        df_cluster["Label"] = df_cluster["Cluster"].map(cluster_labels)

        st.success("Clustering berhasil dilakukan!")
        st.dataframe(df_cluster)
        df_cluster["Quantity"] = pd.to_numeric(df_cluster["Quantity"], errors="coerce").fillna(1)
        df_plot = df_cluster.dropna(subset=["Amount", "Profit"])

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
