# 💳 Credit Risk Classifier with Explainability

A machine learning web application that predicts the credit risk category of a loan applicant and provides explainable AI (XAI) insights using SHAP values. Built with XGBoost, scikit-learn, and deployed on Streamlit Cloud.

---

## 🌐 Live Demo

👉 [credit-risk-classifier-with-explainability.streamlit.app](https://credit-risk-classifier-with-explainability.streamlit.app)

---

## 📌 Project Overview

This project was developed as part of a Machine Learning & Deep Learning course (Semester 2). The goal is to build a credit risk classification system that not only predicts the risk category of a loan applicant but also **explains why** that prediction was made — making it interpretable for non-technical stakeholders like bank officers.

### Problem Statement
Banks receive thousands of loan applications every day. Manually assessing credit risk is time-consuming and prone to human bias. This project automates the risk assessment process using historical credit bureau data and behavioural financial data, classifying applicants into one of four risk categories:

| Category | Description |
|----------|-------------|
| **P1** | Very Low Risk — Highly likely to repay |
| **P2** | Low Risk — Likely to repay |
| **P3** | High Risk — May default |
| **P4** | Very High Risk — Likely to default |

---

## 📂 Dataset

The model was trained on two merged datasets:

| File | Description |
|------|-------------|
| `External_Cibil_Dataset.xlsx` | Credit bureau data including trade lines, delinquencies, enquiries, and utilization metrics |
| `Internal_Bank_Dataset.xlsx` | Internal bank data including income, employment, age, marital status, and education |

- Both datasets are merged on `PROSPECTID`
- Target variable: `Approved_Flag` (P1, P2, P3, P4)
- Total features after merging: **84 columns**
- Missing values handled via `SimpleImputer` and `IterativeImputer`

---

## 🧠 Model Architecture

### Pipeline
The model uses a scikit-learn `Pipeline` with three stages:

```
Input Data (84 features)
        ↓
ColumnTransformer (Preprocessing)
        ↓
SelectKBest (Feature Selection — top 10)
        ↓
XGBClassifier (Prediction)
```

### 1. Preprocessing — `ColumnTransformer`
Different strategies are applied to different column types:

- **Zero-imputed columns**: Missing values filled with 0, then StandardScaled
- **Model-imputed columns**: Missing values filled using `IterativeImputer`, then StandardScaled
- **Categorical columns**: Encoded using `OneHotEncoder`
- **Remaining numerical columns**: StandardScaled directly

### 2. Feature Selection — `SelectKBest`
- Uses `f_classif` (ANOVA F-score) to select the **top 10 most statistically significant features**
- Reduces dimensionality from 84 → 10

### Top 10 Selected Features

| Feature | Description |
|---------|-------------|
| `Age_Oldest_TL` | Age of the oldest trade line (months) |
| `enq_L3m` | Credit enquiries in last 3 months |
| `enq_L6m` | Credit enquiries in last 6 months |
| `enq_L12m` | Credit enquiries in last 12 months |
| `PL_enq_L6m` | Personal loan enquiries in last 6 months |
| `num_std` | Number of standard/good-standing accounts |
| `num_std_6mts` | Standard accounts in last 6 months |
| `num_std_12mts` | Standard accounts in last 12 months |
| `pct_PL_enq_L6m_of_L12m` | PL enquiries in last 6M as % of last 12M |
| `pct_PL_enq_L6m_of_ever` | PL enquiries in last 6M as % of all time |

### 3. Classifier — `XGBClassifier`
- Algorithm: **XGBoost (Extreme Gradient Boosting)**
- Multi-class classification (4 classes)
- Hyperparameters tuned using `GridSearchCV`
- **Model Accuracy: ~74%** (without Credit Score to avoid data leakage)

---

## 📊 Explainability — SHAP

The app uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

- `shap.TreeExplainer` is used for XGBoost
- A **waterfall plot** shows how each feature pushed the prediction above or below the baseline
- Features in **red** increase risk, features in **blue** decrease risk
- SHAP values are computed for the **predicted class**

### Why Explainability Matters
In banking and finance, regulators require that credit decisions be **explainable**. A black-box model that just says "rejected" is not sufficient — the reason must be documented. SHAP provides that explanation at the individual applicant level.

---

## 🖥️ App Features

- 📋 **10-field input form** in the sidebar (one per selected feature)
- 🎨 **Colour-coded risk result** (green → red based on severity)
- 📈 **Probability breakdown** showing likelihood of each class (P1–P4)
- 📊 **SHAP waterfall chart** explaining the prediction
- ℹ️ **Tooltip help text** on every input field
- 🔢 **Proper input validation** (integers for counts, decimals for percentages)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Programming language |
| scikit-learn | ML pipeline, preprocessing, feature selection |
| XGBoost | Classification model |
| SHAP | Model explainability |
| Streamlit | Web app framework |
| Pandas / NumPy | Data manipulation |
| Matplotlib | Plotting |
| Joblib | Model serialization |
| Google Colab | Model training environment |
| Streamlit Cloud | Deployment |

---

## 📁 Repository Structure

```
credit-risk-classifier/
│
├── app.py                      # Main Streamlit application
├── credit_risk_pipeline.pkl    # Trained ML pipeline (serialized)
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for Streamlit Cloud
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Local Setup

### Prerequisites
- Python 3.12
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/partharora284/credit-risk-classifier.git
cd credit-risk-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
joblib==1.4.2
shap
matplotlib
xgboost
scikit-learn==1.6.1
pandas==2.2.2
numpy==2.0.2
```

---

## 🔍 How to Use the App

1. Open the app in your browser
2. Fill in the **10 applicant details** in the left sidebar:
   - Age of oldest credit account
   - Number of recent credit enquiries (3M, 6M, 12M)
   - Personal loan enquiry patterns
   - Number of standard accounts
3. Click **🔍 Predict Credit Risk**
4. View the **risk category** and **confidence score**
5. Check the **probability breakdown** for all 4 classes
6. Read the **SHAP waterfall chart** to understand what drove the prediction

---

## 📉 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~74% |
| Classes | P1, P2, P3, P4 |
| Algorithm | XGBoost |
| Feature Selection | SelectKBest (k=10) |

> **Note:** An earlier version of the model included `Credit_Score` as a feature and achieved ~92% accuracy. However, since `Approved_Flag` (the target) is directly derived from credit score in real-world banking, including it caused **data leakage**. The final model excludes Credit Score to ensure the model learns from genuine behavioural features, making it more generalisable and realistic.

---

## 💡 Key Findings

1. **Enquiry patterns are strong predictors** — the more frequently a person applies for credit in a short period, the higher their risk
2. **Age of oldest trade line matters** — longer credit history correlates with lower risk
3. **Standard account counts are meaningful** — more accounts in good standing = lower risk
4. **Credit Score was removed intentionally** — to prevent data leakage and build a more interpretable behavioural model

---

## 🚀 Deployment

The app is deployed on **Streamlit Community Cloud**:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Set Python version to 3.12 in app settings
5. Deploy — Streamlit auto-installs from `requirements.txt`

---

## 👨‍💻 Author

**Parth Arora**  
Machine Learning & Deep Learning — Semester 2 Project  
GitHub: [@partharora284](https://github.com/partharora284)

---

## 📄 License

This project is for educational purposes only.
