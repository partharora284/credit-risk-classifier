import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. LOAD THE PIPELINE
pipeline = joblib.load('credit_risk_pipeline.pkl')

# 2. DEFINE ALL FEATURES
all_features = [
    'Total_TL', 'Tot_Closed_TL', 'Tot_Active_TL', 'Total_TL_opened_L6M', 'Tot_TL_closed_L6M',
    'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_active_tl', 'pct_closed_tl', 'Total_TL_opened_L12M',
    'Tot_TL_closed_L12M', 'pct_tl_open_L12M', 'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 'Auto_TL',
    'CC_TL', 'Consumer_TL', 'Gold_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL',
    'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'time_since_first_deliquency',
    'time_since_recent_deliquency', 'num_times_delinquent', 'max_delinquency_level',
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

# Categorical columns that must stay as strings
categorical_cols = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
categorical_defaults = {
    'MARITALSTATUS': 'Married',
    'EDUCATION': 'Graduate',
    'GENDER': 'M',
    'last_prod_enq2': 'PL',
    'first_prod_enq2': 'PL'
}

# 3. FRIENDLY LABELS FOR TOP 10 FEATURES
feature_config = {
    'AGE': {
        'label': 'Age (years)',
        'help': 'Applicant age in years',
        'type': 'int',
        'min': 18, 'max': 100, 'default': 30
    },
    'Credit_Score': {
        'label': 'Credit Score',
        'help': 'Applicant credit score (300 - 900)',
        'type': 'int',
        'min': 300, 'max': 900, 'default': 600
    },
    'pct_currentBal_all_TL': {
        'label': 'Current Balance % of All Trade Lines',
        'help': 'Percentage of current balance across all trade lines (0 - 100)',
        'type': 'float',
        'min': 0.0, 'max': 100.0, 'default': 0.0
    },
    'pct_opened_TLs_L6m_of_L12m': {
        'label': 'Trade Lines Opened in Last 6M vs 12M (%)',
        'help': 'Percentage of trade lines opened in last 6 months out of last 12 months (0 - 100)',
        'type': 'float',
        'min': 0.0, 'max': 100.0, 'default': 0.0
    },
    'CC_utilization': {
        'label': 'Credit Card Utilization (%)',
        'help': 'Credit card utilization percentage (0 - 100)',
        'type': 'float',
        'min': 0.0, 'max': 100.0, 'default': 0.0
    },
    'PL_utilization': {
        'label': 'Personal Loan Utilization (%)',
        'help': 'Personal loan utilization percentage (0 - 100)',
        'type': 'float',
        'min': 0.0, 'max': 100.0, 'default': 0.0
    },
    'CC_enq': {
        'label': 'Credit Card Enquiries (Total)',
        'help': 'Total number of credit card enquiries',
        'type': 'int',
        'min': 0, 'max': 100, 'default': 0
    },
    'PL_enq': {
        'label': 'Personal Loan Enquiries (Total)',
        'help': 'Total number of personal loan enquiries',
        'type': 'int',
        'min': 0, 'max': 100, 'default': 0
    },
    'enq_L3m': {
        'label': 'Enquiries in Last 3 Months',
        'help': 'Number of credit enquiries in the last 3 months',
        'type': 'int',
        'min': 0, 'max': 50, 'default': 0
    },
    'enq_L6m': {
        'label': 'Enquiries in Last 6 Months',
        'help': 'Number of credit enquiries in the last 6 months',
        'type': 'int',
        'min': 0, 'max': 50, 'default': 0
    },
}

# 4. STREAMLIT UI
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Explainability Dashboard")
st.markdown("Fill in the applicant details on the left and click **Predict** to assess credit risk.")

st.sidebar.header("📋 Enter Applicant Details")

# 5. COLLECT INPUTS WITH PROPER TYPES
user_input_dict = {}
for feat, config in feature_config.items():
    if config['type'] == 'int':
        user_input_dict[feat] = st.sidebar.number_input(
            config['label'],
            min_value=config['min'],
            max_value=config['max'],
            value=config['default'],
            step=1,
            help=config['help']
        )
    else:
        user_input_dict[feat] = st.sidebar.number_input(
            config['label'],
            min_value=config['min'],
            max_value=config['max'],
            value=config['default'],
            step=0.1,
            format="%.1f",
            help=config['help']
        )

# 6. BUILD FULL DATAFRAME with correct types
final_data = {}
for feat in all_features:
    if feat in categorical_cols:
        final_data[feat] = categorical_defaults[feat]
    else:
        final_data[feat] = 0.0

# Override with user inputs
for feat, val in user_input_dict.items():
    final_data[feat] = float(val)

input_df = pd.DataFrame([final_data])

# Cast numeric columns to float explicitly
for col in input_df.columns:
    if col not in categorical_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

# 7. PREDICTION & SHAP
if st.button("🔍 Predict Credit Risk"):
    try:
        prediction = pipeline.predict(input_df)
        proba = pipeline.predict_proba(input_df)

        risk_label = 'High Risk 🔴' if prediction[0] == 1 else 'Low Risk 🟢'
        confidence = proba[0][prediction[0]] * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", risk_label)
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        st.divider()

        # SHAP Explanation
        st.subheader("📊 Explanation (SHAP)")

        preprocessor = pipeline.named_steps['columntransformer']
        selector = pipeline.named_steps['selectkbest']
        model = pipeline.named_steps['xgbclassifier']

        processed = preprocessor.transform(input_df)
        selected = selector.transform(processed)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(selected)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0, 0], show=False)
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Prediction failed: {e}")
