# RetailMind AI — Customer Churn Prediction

An end-to-end machine learning project focused on predicting customer churn using the Telco Customer Churn dataset. Built incrementally over 35 days, this project explores data preprocessing, model development, evaluation, experimentation, and deployment-ready ML workflows.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave helps businesses take proactive retention measures.

RetailMind AI aims to build a machine learning pipeline that identifies potential churners and evaluates models based on their ability to detect customers who are likely to leave.

**Current Progress:** Day 1 — Baseline Model | Day 2 — Model Comparison

## Objectives

- Build an end-to-end customer churn prediction pipeline.
- Clean and preprocess real-world customer data.
- Train and evaluate classification models.
- Compare models using appropriate evaluation metrics.
- Prioritize churn detection using Recall.
- Save trained models for future predictions.
- Gradually develop the project into a more complete ML application.

## Dataset

**Telco Customer Churn Dataset**

Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information such as:

- Customer demographics
- Account information
- Subscription services
- Contract details
- Payment methods
- Monthly and total charges
- Churn status

**Target variable:** `Churn`

- `0` — No Churn
- `1` — Churn

Place the downloaded CSV file inside the `data/` directory.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and machine learning |
| Joblib | Model persistence |
| Jupyter Notebook | Experimentation and development |
| Git & GitHub | Version control |

## Project Structure

```text
RetailMind-AI/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── test.ipynb
│
├── models/
│   ├── churn_day1_baseline.joblib
│   └── churn_day2_champion.joblib
│
├── reports/
│   ├── day1_baseline_metrics.csv
│   └── day2_model_comparison.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

*Note: Model and report files are generated when the corresponding notebook sections are executed. Your actual filenames may differ.*

## Development Progress

### Day 1 — End-to-End ML Baseline

**Objective:** Build the first complete customer churn prediction pipeline.

Implemented:

- Loaded and explored the Telco Customer Churn dataset.
- Converted `TotalCharges` into a numeric column.
- Removed the `customerID` identifier.
- Encoded the target variable.
- Split data into training and testing sets using stratified sampling.
- Built a preprocessing pipeline using:
  - Median imputation
  - Most frequent categorical imputation
  - StandardScaler
  - OneHotEncoder
- Trained a Logistic Regression baseline.
- Evaluated performance using classification metrics.
- Experimented with class weighting and cross-validation-based threshold selection.
- Saved the trained pipeline and prediction threshold using Joblib.
- Created a reusable `predict_churn()` helper.

**Generated artifacts:**
- `models/churn_day1_baseline.joblib`
- `reports/day1_baseline_metrics.csv`

### Day 2 — Model Comparison and Champion Selection

**Objective:** Compare multiple classification algorithms and identify the strongest candidate for churn detection.

Models evaluated:

1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVM)

Implemented:

- Created separate preprocessing and model pipelines.
- Applied stratified 3-fold cross-validation.
- Compared models using cross-validation Recall.
- Evaluated trained models on the held-out test set.
- Compared Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Selected a champion model based on cross-validation Recall.
- Saved the comparison report and champion model.

**Generated artifacts:**
- `models/churn_day2_champion.joblib`
- `reports/day2_model_comparison.csv`

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | How many predicted churners actually churn |
| Recall | How many actual churners are correctly identified |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish churners from non-churners |

**Why Recall matters:** Missing a customer who is about to leave can mean losing potential business. Therefore, identifying as many actual churners as possible is an important objective.

However, Recall should also be considered alongside Precision to understand false-positive predictions.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RetailMind-AI.git
cd RetailMind-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the Telco Customer Churn dataset from Kaggle and place the CSV file inside the `data/` directory.

### 5. Run the notebook

Open the notebook inside the `notebooks/` directory using Jupyter Notebook or VS Code.

Execute the cells sequentially, starting from Day 1.

## Future Roadmap

This is a 35-day incremental project. Planned development areas include:

- Further model optimization and hyperparameter tuning.
- Feature engineering and feature importance analysis.
- Model explainability.
- Experiment tracking and performance monitoring.
- Building a prediction interface.
- Deployment and documentation.

*Future features will be added as development progresses.*

## Author

**Amarprit Singh Dhingra**

GitHub: [Dhingra2004](https://github.com/Dhingra2004)

LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/amarpritdhingra/)

---

**RetailMind AI — Turning customer data into actionable churn predictions.**
