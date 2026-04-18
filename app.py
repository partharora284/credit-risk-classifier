import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load the Pipeline (Cached for performance)
@st.cache_resource
def load_model():
    # Ensure 'credit_risk_model.pkl' is in the same folder
    return joblib.load('credit_risk_model.pkl')

pipeline = load_model()

st.title("Credit Risk Explainability Dashboard")
st.write("This app predicts credit risk and provides AI-driven insights.")

# 2. User Inputs 
# CRITICAL: These must match the column names in your original training data
st.sidebar.header("Applicant Details")
cc_utilization = st.sidebar.number_input("CC Utilization", 0.0, 1.0, 0.5)
pl_utilization = st.sidebar.number_input("PL Utilization", 0.0, 1.0, 0.5)
enq_l6m = st.sidebar.number_input("Enquiries (Last 6m)", 0, 20, 0)
# Add ALL other features from your model here using st.number_input or st.slider

# 3. Create DataFrame
# The keys must match the exact column names the model was trained on
input_data = pd.DataFrame({
    'CC_utilization': [cc_utilization],
    'PL_utilization': [pl_utilization],
    'enq_L6m': [enq_l6m],
    # Add all other features here...
})

# 4. Predict and Explain
if st.button("Predict Risk"):
    # The pipeline handles scaling and selection automatically
    prediction = pipeline.predict(input_data)
    risk_label = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.subheader(f"Prediction: {risk_label}")

    # Prepare data for SHAP (Must pass through the pipeline's preprocessing steps)
    preprocessor = pipeline.named_steps['columntransformer']
    selector = pipeline.named_steps['selectkbest']
    model = pipeline.named_steps['xgbclassifier']

    # Transform input data through the pipeline steps
    transformed_data = preprocessor.transform(input_data)
    # If SelectKBest was used, we transform the output of the preprocessor
    transformed_data = selector.transform(transformed_data)

    # Calculate SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(transformed_data)

    # Plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)