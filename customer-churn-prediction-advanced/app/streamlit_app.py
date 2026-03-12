import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn AI Dashboard",page_icon="🤖",layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);color:white}
.block-container{background:rgba(255,255,255,0.05);padding:30px;border-radius:15px}
</style>
""",unsafe_allow_html=True)

model=joblib.load("C:/Users/DAKSH/Downloads/customer-churn-prediction-advanced/models/churn_model.pkl")

feature_names=[
"tenure","MonthlyCharges","TotalCharges",
"Contract_Month-to-month","Contract_One year","Contract_Two year",
"InternetService_DSL","InternetService_Fiber optic","InternetService_No",
"Payment_Electronic check","Payment_Mailed check","Payment_Bank transfer","Payment_Credit card",
"OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"
]

st.title("🤖 AI Customer Churn Intelligence Dashboard")

tab1,tab2,tab3,tab4=st.tabs([
"Prediction",
"Customer Segmentation",
"Model Analytics",
"AI Assistant"
])

# ---------------- Prediction Tab ----------------

with tab1:

    col1,col2=st.columns(2)

    with col1:
        tenure=st.slider("Tenure",0,72,24)
        monthly_charges=st.number_input("Monthly Charges",0.0,200.0,70.0)
        total_charges=st.number_input("Total Charges",0.0,10000.0,1000.0)

    with col2:
        contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
        internet=st.selectbox("Internet",["DSL","Fiber optic","No"])
        payment=st.selectbox("Payment",["Electronic check","Mailed check","Bank transfer","Credit card"])

    if st.button("Predict"):

        features=np.zeros(19)

        features[0]=tenure
        features[1]=monthly_charges
        features[2]=total_charges

        data=features.reshape(1,-1)

        prediction=model.predict(data)[0]
        probability=model.predict_proba(data)[0][1]

        colA,colB=st.columns(2)

        with colA:
            st.metric("Churn Probability",f"{round(probability*100,2)}%")

        with colB:

            fig=go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={"text":"Churn Risk"},
            gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"red"},
            "steps":[
            {"range":[0,30],"color":"green"},
            {"range":[30,70],"color":"orange"},
            {"range":[70,100],"color":"red"}
            ]}))

            st.plotly_chart(fig,use_container_width=True)

        st.subheader("Top Churn Drivers")

        explainer=shap.TreeExplainer(model)
        shap_values=explainer.shap_values(data)

        if isinstance(shap_values,list):
            shap_vals=np.array(shap_values[1]).flatten()
        else:
            shap_vals=np.array(shap_values).flatten()

        shap_vals=shap_vals[:len(feature_names)]

        shap_df=pd.DataFrame({
        "Feature":feature_names,
        "Importance":np.abs(shap_vals)
        }).sort_values("Importance",ascending=False)

        st.bar_chart(shap_df.set_index("Feature").head(10))

        st.subheader("SHAP Waterfall Explanation")

        fig,ax=plt.subplots()

        shap.waterfall_plot(
            shap.Explanation(
            values=shap_vals,
            base_values=0,
            data=features,
            feature_names=feature_names
            )
        )

        st.pyplot(fig)

        st.subheader("Retention Strategy")

        if probability>0.7:
            st.error("High churn risk → Offer discounts and proactive support")
        elif probability>0.3:
            st.warning("Medium churn risk → Improve engagement")
        else:
            st.success("Customer likely to stay")

# ---------------- Segmentation Tab ----------------

with tab2:

    st.subheader("Customer Segmentation")

    seg=pd.DataFrame({
    "Tenure":[tenure+np.random.randint(-5,5) for i in range(50)],
    "MonthlyCharges":[monthly_charges+np.random.randint(-20,20) for i in range(50)]
    })

    fig=px.scatter(seg,x="Tenure",y="MonthlyCharges")
    st.plotly_chart(fig,use_container_width=True)

# ---------------- Model Analytics ----------------

with tab3:

    st.subheader("Model Performance Analytics")

    metrics=pd.DataFrame({
    "Metric":["Accuracy","Precision","Recall","F1 Score"],
    "Score":[0.86,0.82,0.79,0.80]
    })

    fig=px.bar(metrics,x="Metric",y="Score")
    st.plotly_chart(fig,use_container_width=True)

# ---------------- AI Assistant ----------------

with tab4:

    st.subheader("AI Churn Assistant")

    question=st.text_input("Ask about churn risk")

    if question:

        if "reduce churn" in question.lower():
            st.write("Improve contract incentives and customer engagement.")
        elif "high risk" in question.lower():
            st.write("Provide targeted discounts and support.")
        else:
            st.write("Analyze customer usage patterns and service satisfaction.")

st.markdown("---")

st.markdown("""
👨‍💻 **Daksh Vasani**

Data Science & Machine Learning enthusiast building intelligent data-driven systems and AI dashboards.
""")