from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model & pipeline
model = joblib.load("kmeans_model.pkl")
pipeline = joblib.load("pipeline.pkl")
df_ref = pd.read_csv("Sales Dataset.csv")

app = FastAPI(title="Clustering API", description="API untuk K-Means Segmentasi Produk")

class InputData(BaseModel):
    Amount: float
    Profit: float
    Quantity: float
    Category: str
    Sub_Category: str
    City: str
    PaymentMode: str
    OrderDate: str

@app.post("/predict")
def predict(data: InputData):

    # DataFrame input baru
    inp = pd.DataFrame([{
        "Amount": data.Amount,
        "Profit": data.Profit,
        "Quantity": data.Quantity,
        "Category": data.Category,
        "Sub-Category": data.Sub_Category,
        "City": data.City,
        "PaymentMode": data.PaymentMode,
        "Order Date": pd.to_datetime(data.OrderDate)
    }])

    # Feature engineering
    inp["Year"] = inp["Order Date"].dt.year
    inp["Month"] = inp["Order Date"].dt.month

    # Frequency encoding
    for col in ["City", "Sub-Category"]:
        freq = df_ref[col].value_counts(normalize=True)
        inp[f"{col}_freq"] = inp[col].map(freq).fillna(0.01)

    # Select relevant features
    features = [
        "Amount","Profit","Quantity","Category","PaymentMode",
        "City_freq","Sub-Category_freq","Year","Month"
    ]

    X = pipeline.transform(inp[features])
    cluster = int(model.predict(X)[0])
    label = "Electronics-Oriented" if cluster == 0 else "Office/Furniture-Oriented"

    return {"cluster": cluster, "label": label}