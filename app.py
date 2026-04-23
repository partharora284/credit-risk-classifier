import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD THE PIPELINE
pipeline = joblib.load('credit_risk_pipeline.pkl')

# 2. ALL FEATURES WITH MEDIAN DEFAULTS (Credit Score removed)
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
    'last_prod_enq2': 'PL', 'first_prod_enq2': 'PL'
}

categorical_cols = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

# 3. THE 10 FEATURES SELECTKBEST ACTUALLY PICKED
feature_config = {
    'Age_Oldest_TL': {
        'label': 'Age of Oldest Trade Line (months)',
        'help': 'How many months ago was your oldest loan/credit account opened. Median: 33',
        'type': 'int', 'min': 0, 'max': 600, 'default': 33
    },
    'enq_L3m': {
        'label': 'Credit Enquiries in Last 3 Months',
        'help': 'Number of times lenders checked your credit in last 3 months. Median: 1',
        'type': 'int', 'min': 0, 'max': 50, 'default': 1
    },
    'enq_L6m': {
        'label': 'Credit Enquiries in Last 6 Months',
        'help': 'Number of times lenders checked your credit in last 6 months. Median: 1',
        'type': 'int', 'min': 0, 'max': 50, 'default': 1
    },
    'enq_L12m': {
        'label': 'Credit Enquiries in Last 12 Months',
        'help': 'Number of times lenders checked your credit in last 12 months. Median: 2',
        'type': 'int', 'min': 0, 'max': 100, 'default': 2
    },
    'PL_enq_L6m': {
        'label': 'Personal Loan Enquiries in Last 6 Months',
        'help': 'Number of personal loan enquiries in last 6 months. Median: 0',
        'type': 'int', 'min': 0, 'max': 50, 'default': 0
    },
    'num_std': {
        'label': 'Number of Standard Accounts (Total)',
        'help': 'Total accounts currently in good/standard standing. Median: 0',
        'type': 'int', 'min': 0, 'max': 50, 'default': 0
    },
    'num_std_6mts': {
        'label': 'Standard Accounts in Last 6 Months',
        'help': 'Accounts in good standing in the last 6 months. Median: 0',
        'type': 'int', 'min': 0, 'max': 50, 'default': 0
    },
    'num_std_12mts': {
        'label': 'Standard Accounts in Last 12 Months',
        'help': 'Accounts in good standing in the last 12 months. Median: 0',
        'type': 'int', 'min': 0, 'max': 50, 'default': 0
    },
    'pct_PL_enq_L6m_of_L12m': {
        'label': 'PL Enquiries: Last 6M as % of Last 12M',
        'help': 'What percentage of last 12 month personal loan enquiries happened in last 6 months (0-100). Median: 0',
        'type': 'float', 'min': 0.0, 'max': 100.0, 'default': 0.0
    },
    'pct_PL_enq_L6m_of_ever': {
        'label': 'PL Enquiries: Last 6M as % of All Time',
        'help': 'What percentage of all-time personal loan enquiries happened in last 6 months (0-100). Median: 0',
        'type': 'float', 'min': 0.0, 'max': 100.0, 'default': 0.0
    },
}

# 4. STREAMLIT UI
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Explainability Dashboard")
st.markdown("Fill in the applicant details on the left and click **Predict** to assess credit risk.")

st.sidebar.header("📋 Enter Applicant Details")
st.sidebar.markdown("---")

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

# 6. BUILD FULL DATAFRAME using medians as defaults
final_data = dict(all_features_defaults)
for feat, val in user_input_dict.items():
    final_data[feat] = float(val)

input_df = pd.DataFrame([final_data])
for col in input_df.columns:
    if col not in categorical_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

# 7. RISK LABEL MAPPING
risk_labels = {
    0: ('P1 — Very Low Risk', '🟢', '#28a745'),
    1: ('P2 — Low Risk', '🟡', '#ffc107'),
    2: ('P3 — High Risk', '🟠', '#fd7e14'),
    3: ('P4 — Very High Risk', '🔴', '#dc3545'),
}

# 8. PREDICTION & SHAP
if st.button("🔍 Predict Credit Risk", use_container_width=True):
    try:
        prediction = pipeline.predict(input_df)
        proba = pipeline.predict_proba(input_df)

        pred_class = int(prediction[0])
        label, emoji, color = risk_labels.get(pred_class, (f'Class {pred_class}', '⚪', '#6c757d'))
        confidence = float(np.max(proba[0])) * 100

        # Result display
        st.markdown(f"""
        <div style='background-color:{color}22; border-left: 5px solid {color};
             padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
            <h2 style='color:{color}; margin:0'>{emoji} {label}</h2>
            <p style='margin:5px 0 0 0; font-size:18px'>Confidence: <strong>{confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.subheader("📈 Probability Breakdown")
        prob_df = pd.DataFrame({
            'Risk Category': [f'{emoji} {risk_labels[i][0]}' for i, emoji in enumerate([r[1] for r in risk_labels.values()])],
            'Probability (%)': [f"{p*100:.1f}%" for p in proba[0]]
        })
        st.table(prob_df)

        st.divider()

        # SHAP
        st.subheader("📊 Feature Importance (SHAP)")
        st.caption("Shows which features pushed the prediction toward the predicted class")

        preprocessor = pipeline.named_steps['columntransformer']
        selector = pipeline.named_steps['selectkbest']
        model = pipeline.named_steps['xgbclassifier']

        processed = preprocessor.transform(input_df)
        selected = selector.transform(processed)

        # Get clean feature names
        try:
            raw_names = selector.get_feature_names_out()
            clean_names = [n.split('__')[-1] for n in raw_names]
        except Exception:
            clean_names = [f"Feature {i}" for i in range(selected.shape[1])]

        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(selected)

        # Handle multi-class (4 classes)
        if isinstance(shap_values_raw, list):
            sv = shap_values_raw[pred_class][0]
            expected_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            sv = shap_values_raw[0, :, pred_class]
            expected_val = explainer.expected_value[pred_class] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        else:
            sv = shap_values_raw[0]
            expected_val = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value

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
