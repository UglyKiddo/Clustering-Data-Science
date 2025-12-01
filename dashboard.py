import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("pipeline.pkl")
model = joblib.load("kmeans_model.pkl")
df_ref = pd.read_csv("Sales Dataset.csv")

st.title("📊 Dashboard Segmentasi Produk (K-Means)")

st.sidebar.header("Input Data")

def manual_input():
    Amount = st.sidebar.number_input("Amount", min_value=0.0, step=100.0)
    Profit = st.sidebar.number_input("Profit", min_value=0.0, step=50.0)
    Quantity = st.sidebar.number_input("Quantity", min_value=1, step=1)
    Category = st.sidebar.selectbox("Category", df_ref["Category"].unique())
    SubCat = st.sidebar.text_input("Sub-Category")
    City = st.sidebar.text_input("City")
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

choice = st.radio("Mode Input:", ["Manual", "Upload CSV"])

if choice == "Manual":
    df_input = manual_input()
else:
    uploaded = st.file_uploader("Upload CSV")
    if uploaded:
        df_input = pd.read_csv(uploaded)
    else:
        st.stop()

df_input["Year"] = df_input["Order Date"].dt.year
df_input["Month"] = df_input["Order Date"].dt.month

# Frequency encoding
for col in ["City", "Sub-Category"]:
    freq = df_ref[col].value_counts(normalize=True)
    df_input[f"{col}_freq"] = df_input[col].map(freq).fillna(0.01)

features = [
    "Amount","Profit","Quantity","Category","PaymentMode",
    "City_freq","Sub-Category_freq","Year","Month"
]

X = pipeline.transform(df_input[features])
df_input["cluster"] = model.predict(X)
df_input["label"] = df_input["cluster"].map({0:"Electronics-Oriented",1:"Office/Furniture-Oriented"})

st.write(df_input)