<!-- 🌌 Header -->
<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=220&section=header&text=Customer%20Churn%20Prediction%20AI&fontSize=45&fontColor=ffffff&animation=fadeIn"/>
</p>

<!-- Typing Animation -->
<p align="center">
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=00F7FF&center=true&vCenter=true&width=750&lines=Customer+Churn+Prediction+System;Machine+Learning+AI+Dashboard;FastAPI+%2B+Streamlit+%2B+Python"/>
</p>

---

# 📌 Project Overview

**Customer Churn Prediction AI System** is an advanced machine learning project designed to predict whether a customer will leave a service.

This project demonstrates an **end-to-end data science pipeline** including:

- Data Analysis
- Feature Engineering
- Model Training
- Model Explainability
- API Deployment
- Interactive AI Dashboard

The system predicts churn risk and also explains **why a customer might churn** using **SHAP explainability**.

---
## Dataset

Dataset used in this project:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

This dataset contains telecom customer information including demographics, services, billing details, and a **Churn** column indicating whether the customer left the company.
---

# 🧠 How The System Works

The project follows a full **Machine Learning pipeline**.

```
Dataset
   ↓
EDA (Exploratory Data Analysis)
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Explainability (SHAP)
   ↓
Saved Model (.pkl)
   ↓
FastAPI Prediction API
   ↓
Streamlit AI Dashboard
```

---

# 🏗 Project Architecture

```
                ┌───────────────┐
                │  Dataset      │
                │  Churn.csv    │
                └──────┬────────┘
                       │
                       ▼
              Data Analysis (EDA)
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
                ML Model Training
                       │
                       ▼
                Saved Model (.pkl)
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
   FastAPI Prediction API     Streamlit Dashboard
         │                           │
         ▼                           ▼
     JSON Predictions           Interactive AI UI
```

---

# 📂 Project Structure

```
CUSTOMER-CHURN-PREDICTION-ADVANCED

api/
 └── fastapi_app.py

app/
 └── streamlit_app.py

dashboard/

dataset/
 └── Churn.csv

models/
 ├── churn_model.pkl
 ├── processed_data.pkl
 └── 04_model_explainability.ipynb

notebooks/
 ├── 01_EDA.ipynb
 ├── 02_feature_engineering.ipynb
 ├── 03_model_training.ipynb
 └── 04_model_explainability.ipynb
```

---

# 📊 Dataset

Dataset contains customer information such as:

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Online Security
- Tech Support

These features are used to predict whether a customer will **stay or churn**.

---

# 📘 Notebook Explanation

## 1️⃣ 01_EDA.ipynb

Exploratory Data Analysis.

Includes:

- Data distribution
- Missing value analysis
- Feature correlation
- Customer churn patterns
- Visualization of churn behavior

Purpose:
Understand the dataset before modeling.

---

## 2️⃣ 02_feature_engineering.ipynb

Feature preparation.

Includes:

- Encoding categorical features
- Feature scaling
- Data preprocessing
- Feature transformation

Output:

```
processed_data.pkl
```

---

## 3️⃣ 03_model_training.ipynb

Machine Learning model training.

Steps:

- Train/test split
- Model selection
- Model training
- Evaluation

Model metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Output model:

```
churn_model.pkl
```

---

## 4️⃣ 04_model_explainability.ipynb

Model interpretability using **SHAP**.

Purpose:

Explain which features influence churn predictions.

Visualization:

- Feature importance
- SHAP plots
- Waterfall explanations

---

# 🤖 FastAPI Prediction API

API allows external systems to get churn predictions.

File:

```
api/fastapi_app.py
```

Example endpoint:

```
POST /predict
```

Input:

```
[tenure, monthly_charges, total_charges, ...]
```

Output:

```
{
 "churn_prediction": 1,
 "churn_probability": 0.78
}
```

Code loads trained model:

```python
model = joblib.load("../models/churn_model.pkl")
```

API then returns prediction and probability. :contentReference[oaicite:0]{index=0}

---

# 🖥 Streamlit AI Dashboard

File:

```
app/streamlit_app.py
```

Features:

### 🔮 Prediction Panel

User inputs:

- Tenure
- Monthly charges
- Contract
- Internet service
- Payment method

System outputs:

- Churn probability
- Risk gauge chart
- Retention strategy

---

### 📊 Customer Segmentation

Scatter plot of:

```
Tenure vs MonthlyCharges
```

Used for customer behavior visualization.

---

### 📈 Model Analytics

Displays model metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

### 🤖 AI Churn Assistant

Interactive assistant that answers questions like:

- "How to reduce churn?"
- "What causes churn?"

This dashboard loads the trained model and generates predictions and SHAP explanations. :contentReference[oaicite:1]{index=1}

---

# 📦 Required Python Libraries

```
pandas
numpy
scikit-learn
joblib
streamlit
fastapi
uvicorn
plotly
matplotlib
shap
```

---

# ⚙ Installation

Clone repository

```
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-advanced.git
```

Enter project folder

```
cd customer-churn-prediction-advanced
```

Install dependencies

```
pip install -r requirements.txt
```

---

# 🚀 Running the Streamlit Dashboard

```
streamlit run app/streamlit_app.py
```

Open browser:

```
http://localhost:8501
```

---

# 🚀 Running the FastAPI Server

```
uvicorn api.fastapi_app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

Interactive API documentation will appear.

---

# 📊 Model Performance

Example performance:

| Metric | Score |
|------|------|
| Accuracy | 0.86 |
| Precision | 0.82 |
| Recall | 0.79 |
| F1 Score | 0.80 |

---

# 🚀 Future Improvements

- Real-time churn monitoring
- Cloud deployment
- Customer retention recommendation system
- Deep learning churn model
- Automated retraining pipeline

---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.

---

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=170&section=footer&text=Thanks%20for%20Visiting%20My%20Profile!&fontSize=28&fontColor=ffffff&animation=twinkling&fontAlignY=65"/>
</p>
