import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD THE PIPELINE
pipeline = joblib.load('credit_risk_pipeline.pkl')

# 2. DEFINE ALL FEATURES WITH MEDIAN DEFAULTS
all_features_defaults = {
    'Total_TL': 2.0, 'Tot_Closed_TL': 1.0, 'Tot_Active_TL': 1.0,
    'Total_TL_opened_L6M': 0.0, 'Tot_TL_closed_L6M': 0.0,
    'pct_tl_open_L6M': 0.0, 'pct_tl_closed_L6M': 0.0,
    'pct_active_tl': 0.562, 'pct_closed_tl': 0.438,
    'Total_TL_opened_L12M': 1.0, 'Tot_TL_closed_L12M': 0.0,
    'pct_tl_open_L12M': 0.333, 'pct_tl_closed_L12M': 0.0,
    'Tot_Missed_Pmnt': 0.0, 'Auto_TL': 0.0, 'CC_TL': 0.0,
    'Consumer_TL': 0.0, 'Gold_TL': 0.0, 'Home_TL': 0.0,
    'PL_TL': 0.0, 'Secured_TL': 1.0, 'Unsecured_TL': 1.0, 'Other_TL': 0.0,
    'Age_Oldest_TL': 33.0, 'Age_Newest_TL': 8.0,
    'time_since_recent_payment': 74.0, 'time_since_first_deliquency': 21.0,
    'time_since_recent_deliquency': 10.0, 'num_times_delinquent': 0.0,
    'max_delinquency_level': 32.0, 'max_recent_level_of_deliq': 0.0,
    'num_deliq_6mts': 0.0, 'num_deliq_12mts': 0.0, 'num_deliq_6_12mts': 0.0,
    'max_deliq_6mts': 0.0, 'max_deliq_12mts': 0.0,
    'num_times_30p_dpd': 0.0, 'num_times_60p_dpd': 0.0,
    'num_std': 0.0, 'num_std_6mts': 0.0, 'num_std_12mts': 0.0,
    'num_sub': 0.0, 'num_sub_6mts': 0.0, 'num_sub_12mts': 0.0,
    'num_dbt': 0.0, 'num_dbt_6mts': 0.0, 'num_dbt_12mts': 0.0,
    'num_lss': 0.0, 'num_lss_6mts': 0.0, 'num_lss_12mts': 0.0,
    'recent_level_of_deliq': 0.0, 'tot_enq': 3.0,
    'CC_enq': 0.0, 'CC_enq_L6m': 0.0, 'CC_enq_L12m': 0.0,
    'PL_enq': 0.0, 'PL_enq_L6m': 0.0, 'PL_enq_L12m': 0.0,
    'time_since_recent_enq': 74.0, 'enq_L12m': 2.0, 'enq_L6m': 1.0,
    'enq_L3m': 1.0, 'MARITALSTATUS': 'Married', 'EDUCATION': 'Graduate',
    'AGE': 32.0, 'GENDER': 'M', 'NETMONTHLYINCOME': 23000.0,
    'Time_With_Curr_Empr': 94.0, 'pct_of_active_TLs_ever': 0.562,
    'pct_opened_TLs_L6m_of_L12m': 0.0, 'pct_currentBal_all_TL': 0.6225,
    'CC_utilization': 0.738, 'CC_Flag': 0.0, 'PL_utilization': 0.838,
    'PL_Flag': 0.0, 'pct_PL_enq_L6m_of_L12m': 0.0, 'pct_CC_enq_L6m_of_L12m': 0.0,
    'pct_PL_enq_L6m_of_ever': 0.0, 'pct_CC_enq_L6m_of_ever': 0.0,
    'max_unsec_exposure_inPct': 1.823, 'HL_Flag': 0.0, 'GL_Flag': 0.0,
    'last_prod_enq2': 'PL', 'first_prod_enq2': 'PL', 'Credit_Score': 680.0
}

categorical_cols = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

# 3. THE ACTUAL FEATURES SELECTKBEST USES (the real important ones!)
feature_config = {
    'Credit_Score': {
        'label': 'Credit Score',
        'help': 'Applicant credit score (300–900). Median: 680',
        'type': 'int', 'min': 300, 'max': 900, 'default': 680
    },
    'enq_L3m': {
        'label': 'Credit Enquiries in Last 3 Months',
        'help': 'Number of credit enquiries in last 3 months. Median: 1',
        'type': 'int', 'min': 0, 'max': 50, 'default': 1
    },
    'enq_L6m': {
        'label': 'Credit Enquiries in Last 6 Months',
        'help': 'Number of credit enquiries in last 6 months. Median: 1',
        'type': 'int', 'min': 0, 'max': 50, 'default': 1
    },
    'enq_L12m': {
        'label': 'Credit Enquiries in Last 12 Months',
        'help': 'Number of credit enquiries in last 12 months. Median: 2',
        'type': 'int', 'min': 0, 'max': 100, 'default': 2
    },
    'Age_Oldest_TL': {
        'label': 'Age of Oldest Trade Line (months)',
        'help': 'How long ago (in months) was the oldest loan/credit opened. Median: 33',
        'type': 'int', 'min': 0, 'max': 600, 'default': 33
    },
    'num_std': {
        'label': 'Number of Standard Accounts',
        'help': 'Total accounts in standard/good standing. Median: 0',
        'type': 'int', 'min': 0, 'max': 50, 'default': 0
    },
    'num_std_6mts': {
        'label': 'Standard Accounts (Last 6 Months)',
        'help': 'Accounts in standard standing in last 6 months. Median: 0',
        'type': 'int', 'min': 0, 'max': 50, 'default': 0
    },
    'num_std_12mts': {
        'label': 'Standard Accounts (Last 12 Months)',
        'help': 'Accounts in standard standing in last 12 months. Median: 0',
        'type': 'int', 'min': 0, 'max': 50, 'default': 0
    },
    'pct_PL_enq_L6m_of_L12m': {
        'label': 'Personal Loan Enquiries: 6M vs 12M (%)',
        'help': 'What % of last 12M PL enquiries happened in last 6M. Median: 0',
        'type': 'float', 'min': 0.0, 'max': 100.0, 'default': 0.0
    },
    'pct_PL_enq_L6m_of_ever': {
        'label': 'Personal Loan Enquiries: 6M vs Ever (%)',
        'help': 'What % of all-time PL enquiries happened in last 6M. Median: 0',
        'type': 'float', 'min': 0.0, 'max': 100.0, 'default': 0.0
    },
}

# 4. STREAMLIT UI
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Explainability Dashboard")
st.markdown("Fill in the applicant details on the left and click **Predict** to assess credit risk.")

st.sidebar.header("📋 Enter Applicant Details")

# 5. COLLECT INPUTS
user_input_dict = {}
for feat, config in feature_config.items():
    if config['type'] == 'int':
        user_input_dict[feat] = st.sidebar.number_input(
            config['label'], min_value=config['min'], max_value=config['max'],
            value=config['default'], step=1, help=config['help']
        )
    else:
        user_input_dict[feat] = st.sidebar.number_input(
            config['label'], min_value=config['min'], max_value=config['max'],
            value=config['default'], step=0.1, format="%.1f", help=config['help']
        )

# 6. BUILD FULL DATAFRAME using medians as defaults for non-input features
final_data = dict(all_features_defaults)  # start with all medians
for feat, val in user_input_dict.items():
    final_data[feat] = float(val)         # override with user inputs

input_df = pd.DataFrame([final_data])

# Cast numeric columns to float
for col in input_df.columns:
    if col not in categorical_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

# 7. PREDICTION & SHAP
if st.button("🔍 Predict Credit Risk"):
    try:
        prediction = pipeline.predict(input_df)
        proba = pipeline.predict_proba(input_df)

        # Map class index to label
        class_labels = {0: 'P1 - Very Low Risk 🟢', 1: 'P2 - Low Risk 🟡',
                        2: 'P3 - High Risk 🟠', 3: 'P4 - Very High Risk 🔴'}
        pred_class = int(prediction[0])
        risk_label = class_labels.get(pred_class, f'Class {pred_class}')
        confidence = float(np.max(proba[0])) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", risk_label)
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        st.divider()
        st.subheader("📊 Feature Importance (SHAP)")

        preprocessor = pipeline.named_steps['columntransformer']
        selector = pipeline.named_steps['selectkbest']
        model = pipeline.named_steps['xgbclassifier']

        processed = preprocessor.transform(input_df)
        selected = selector.transform(processed)

        # Get real feature names
        try:
            feature_names = selector.get_feature_names_out()
            # Clean up prefixes like 'standardscaler__', 'pipeline-1__'
            clean_names = []
            for name in feature_names:
                if '__' in name:
                    clean_names.append(name.split('__')[-1])
                else:
                    clean_names.append(name)
        except Exception:
            clean_names = [f"Feature {i}" for i in range(selected.shape[1])]

        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(selected)

        # Handle multi-class output (4 classes: P1, P2, P3, P4)
        if isinstance(shap_values_raw, list):
            sv = shap_values_raw[pred_class][0]
            expected_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            sv = shap_values_raw[0, :, pred_class]
            expected_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2:
            sv = shap_values_raw[0]
            expected_val = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        else:
            sv = shap_values_raw[0]
            expected_val = explainer.expected_value

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=float(expected_val),
                data=selected[0],
                feature_names=clean_names
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Prediction failed: {e}")
