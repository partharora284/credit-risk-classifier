import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# 1. LOAD THE PIPELINE
# Ensure credit_risk_model.pkl is in the same folder as app.py
pipeline = joblib.load('credit_risk_model.pkl')

# 2. DEFINE ALL FEATURES (The 80+ columns your pipeline expects)
all_features = [
    'Total_TL', 'Tot_Closed_TL', 'Tot_Active_TL', 'Total_TL_opened_L6M', 'Tot_TL_closed_L6M', 
    'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_active_tl', 'pct_closed_tl', 'Total_TL_opened_L12M', 
    'Tot_TL_closed_L12M', 'pct_tl_open_L12M', 'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 'Auto_TL', 
    'CC_TL', 'Consumer_TL', 'Gold_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL', 
    'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'time_since_first_deliquency', 
    'time_since_recent_delinquency', 'num_times_delinquent', 'max_delinquency_level', 
    'max_recent_level_of_deliq', 'num_deliq_6mts', 'num_deliq_12mts', 'num_deliq_6_12mts', 
    'max_deliq_6mts', 'max_deliq_12mts', 'num_times_30p_dpd', 'num_times_60p_dpd', 'num_std', 
    'num_std_6mts', 'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt', 
    'num_dbt_6mts', 'num_dbt_12mts', 'num_lss', 'num_lss_6mts', 'num_lss_12mts', 'recent_level_of_deliq', 
    'tot_enq', 'CC_enq', 'CC_enq_L6m', 'CC_enq_L12m', 'PL_enq', 'PL_enq_L6m', 'PL_enq_L12m', 
    'time_since_recent_enq', 'enq_L12m', 'enq_L6m', 'enq_L3m', 'MARITALSTATUS', 'EDUCATION', 'AGE', 
    'GENDER', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'pct_of_active_TLs_ever', 
    'pct_opened_TLs_L6m_of_L12m', 'pct_currentBal_all_TL', 'CC_utilization', 'CC_Flag', 
    'PL_utilization', 'PL_Flag', 'pct_PL_enq_L6m_of_L12m', 'pct_CC_enq_L6m_of_L12m', 
    'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever', 'max_unsec_exposure_inPct', 'HL_Flag', 
    'GL_Flag', 'last_prod_enq2', 'first_prod_enq2', 'Credit_Score'
]

# 3. YOUR 10 SELECTED FEATURES (The ones to show in UI)
top_10 = ['AGE', 'Credit_Score', 'pct_currentBal_all_TL', 'pct_opened_TLs_L6m_of_L12m', 
          'CC_utilization', 'PL_utilization', 'CC_enq', 'PL_enq', 'enq_L3m', 'enq_L6m']

# 4. STREAMLIT UI
st.title("Credit Risk Explainability Dashboard")
st.sidebar.header("Enter Applicant Details")

# Collect inputs only for the 10 features
user_input_dict = {}
for feat in top_10:
    user_input_dict[feat] = st.sidebar.number_input(f"{feat}", value=0.0)

# 5. CREATE THE FULL DATAFRAME
# Start with a dictionary of all features set to 0
final_data = {feat: 0 for feat in all_features}

# Update with user values
for feat, val in user_input_dict.items():
    final_data[feat] = val

input_df = pd.DataFrame([final_data])

# 6. PREDICTION & SHAP
if st.button("Predict"):
    # Pipeline expects all 80+ features here, and it now gets them
    prediction = pipeline.predict(input_df)
    st.write(f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")

    # Access pipeline components
    # Update these names if your notebook prints something different for pipeline.named_steps.keys()
    preprocessor = pipeline.named_steps['columntransformer']
    selector = pipeline.named_steps['selectkbest']
    model = pipeline.named_steps['xgbclassifier']

    # Transform
    processed = preprocessor.transform(input_df)
    selected = selector.transform(processed)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(selected)
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)