import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("pipeline.pkl")
model = joblib.load("kmeans_model.pkl")
df_ref = pd.read_csv("Sales Dataset.csv")

st.title("📊 Dashboard Segmentasi Produk (K-Means)")

st.sidebar.header("Input Data")

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

else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)

        # --- PERBAIKAN: parse tanggal ---
        if "Order Date" in df_input.columns:
            df_input["Order Date"] = pd.to_datetime(df_input["Order Date"], errors="coerce")

        # Jika masih ada NaT (tanggal korup)
        if df_input["Order Date"].isna().any():
            st.warning("Beberapa tanggal tidak dapat diparse. Pastikan format YYYY-MM-DD atau DD/MM/YYYY.")
    else:
        st.stop()


# ===========================
# Ekstraksi Year & Month aman
# ===========================
try:
    df_input["Year"] = df_input["Order Date"].dt.year
    df_input["Month"] = df_input["Order Date"].dt.month
except:
    st.error("Error: Kolom 'Order Date' bukan datetime. Periksa CSV Anda.")
    st.stop()


# ===========================
# FREQUENCY ENCODING
# ===========================
for col in ["City", "Sub-Category"]:
    freq = df_ref[col].value_counts(normalize=True)
    df_input[f"{col}_freq"] = df_input[col].map(freq).fillna(0.01)

# ===========================
# FITUR UNTUK MODEL
# ===========================
features = [
    "Amount","Profit","Quantity","City_freq","Sub-Category_freq",
    "Category","PaymentMode","Year","Month"
    ]


X = pipeline.transform(df_input[features])
df_input["cluster"] = model.predict(X)
df_input["label"] = df_input["cluster"].map({
    0:"Electronics-Oriented",
    1:"Office/Furniture-Oriented"
})

st.write(df_input)