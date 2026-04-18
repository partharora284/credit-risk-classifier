import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# 1. Load the Model
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'credit_risk_model.pkl')
    return joblib.load(model_path)

pipeline = load_model()

# 2. UI Layout
st.title("Credit Risk Explainability Dashboard")
st.sidebar.header("Applicant Details")

# INPUT SECTION: 
# Copy the exact column names from your X_train.columns list
# Example below: Add your features here
cc_util = st.sidebar.number_input("CC Utilization", 0.0, 1.0, 0.1)
pl_enq = st.sidebar.number_input("PL Enquiries", 0, 10, 0)

# Create input dataframe (Order must match training data)
input_data = pd.DataFrame({
    'CC_utilization': [cc_util],
    'PL_enq': [pl_enq]
    # ADD ALL OTHER FEATURES HERE
})

# 3. Predict & Explain
if st.button("Analyze Risk"):
    # Inference
    prediction = pipeline.predict(input_data)
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.subheader(f"Prediction: {risk}")

    # SHAP Visualization
    # Accessing your specific pipeline steps
    preprocessor = pipeline.named_steps['columntransformer']
    selector = pipeline.named_steps['selectkbest']
    model = pipeline.named_steps['xgbclassifier']

    # Transform data
    transformed_data = preprocessor.transform(input_data)
    transformed_data = selector.transform(transformed_data)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(transformed_data)
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    st.pyplot(fig)