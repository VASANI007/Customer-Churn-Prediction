from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("../models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")

def predict(data: list):
    data = np.array(data).reshape(1,-1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }