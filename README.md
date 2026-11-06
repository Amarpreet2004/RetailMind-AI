# RetailMind AI — Customer Churn Prediction

An end-to-end retail AI project built incrementally over 35 days. Each day adds one module to the same codebase, using the techniques and libraries introduced that day. By Day 35 this becomes one real, deployed product.

**Current progress:** Day 5 — Deep Learning (Keras ANN) with a head-to-head benchmark against the classical champion.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave helps businesses take proactive retention measures.

RetailMind AI builds a churn-prediction pipeline that identifies potential churners and evaluates models primarily on their ability to detect customers who are likely to leave (Recall), while keeping Precision, F1 and ROC-AUC visible so that recall is not pursued blindly.

## Objectives

- Build an end-to-end customer churn prediction pipeline.
- Clean and preprocess real-world customer data.
- Train and evaluate classification models.
- Compare models using appropriate evaluation metrics.
- Prioritize churn detection using Recall.
- Save trained models for future predictions.
- Introduce deep learning where it can be benchmarked honestly against classical ML.
- Gradually develop the project into a complete ML application.

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

Place the downloaded CSV file inside the `data/` directory. The raw CSV is intentionally excluded from version control.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, classical models, metrics |
| TensorFlow / Keras | Day 5 neural networks (ANN and MNIST warm-up) |
| Joblib | Model persistence |
| Matplotlib | Training curves and ROC comparison plots |
| Jupyter Notebook | Experimentation and development |
| Git & GitHub | Version control |

## Project Structure

```text
RetailMind/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # local only (gitignored)
│
├── notebooks/
│   └── test.ipynb                              # Day 1 → Day 5 workflow
│
├── models/
│   ├── churn_day1_baseline.joblib
│   ├── churn_day2_champion.joblib
│   ├── churn_day3_tuned.joblib
│   ├── churn_day4_champion.joblib
│   ├── churn_day5_ann.keras
│   ├── churn_day5_preprocessor.joblib
│   └── churn_day5_metadata.joblib
│
├── reports/
│   ├── day1_baseline_metrics.csv
│   ├── day2_model_comparison.csv
│   ├── day3_cv_tuning_results.csv
│   ├── day3_learning_curve.csv
│   ├── day3_learning_curve.png
│   ├── day4_ensemble_comparison.csv
│   ├── day4_vs_day3_comparison.csv
│   ├── day5_ann_experiments.csv
│   ├── day5_ann_comparison.csv
│   ├── day5_ann_training_history.png
│   └── day5_roc_comparison.png
│
├── src/
│   └── train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

*Model and report files are generated when the notebook sections are executed.*

## Development Progress

### Day 1 — End-to-End ML Baseline

**Objective:** Build the first complete customer churn prediction pipeline.

Implemented:

- Loaded and explored the Telco Customer Churn dataset.
- Converted `TotalCharges` into a numeric column.
- Removed the `customerID` identifier.
- Encoded the target variable.
- Split data into training and testing sets using stratified sampling.
- Built a preprocessing pipeline using median/most-frequent imputation, `StandardScaler` and `OneHotEncoder`.
- Trained a Logistic Regression baseline.
- Experimented with class weighting and cross-validation-based threshold selection.
- Saved the trained pipeline, threshold and a reusable `predict_churn()` helper.

**Generated artifacts:** `models/churn_day1_baseline.joblib`, `reports/day1_baseline_metrics.csv`

### Day 2 — Model Comparison and Champion Selection

**Objective:** Compare multiple classification algorithms and identify the strongest candidate for churn detection.

Models evaluated: Decision Tree, Random Forest and SVM.

Implemented:

- Separate preprocessing and model pipelines.
- Stratified 3-fold cross-validation.
- Comparison using cross-validation Recall.
- Held-out test evaluation with Accuracy, Precision, Recall, F1 and ROC-AUC.
- Champion selection based on cross-validation Recall.

**Result:** SVM was selected as the Day 2 champion.

**Generated artifacts:** `models/churn_day2_champion.joblib`, `reports/day2_model_comparison.csv`

### Day 3 — Hyperparameter Tuning and Generalisation

**Objective:** Tune the candidate models and study whether more data or better parameters improve generalisation.

Implemented:

- Grid-search spaces for Decision Tree, Random Forest and SVM.
- Cross-validated tuning using Recall.
- Learning-curve analysis relating training-set size to training and cross-validation Recall.
- Comparison of the tuned champion against earlier results.

**Result:** SVM with `C=0.5` and `gamma="auto"` achieved the best cross-validation Recall (`0.8174`) and test Recall (`0.8048`), with ROC-AUC `0.8330`.

**Generated artifacts:** `models/churn_day3_tuned.joblib`, `reports/day3_cv_tuning_results.csv`, `reports/day3_learning_curve.csv`, `reports/day3_learning_curve.png`

### Day 4 — Feature Engineering and Ensembles

**Objective:** Add domain-informed features and compare ensemble models against the tuned SVM.

Feature engineering:

- `average_charge_per_month` — `TotalCharges / tenure`.
- `is_month_to_month` — flag for month-to-month contracts.
- `uses_electronic_check` — flag for electronic-check payment.
- `tenure_group` — binned tenure (`0-12`, `13-24`, `25-48`, `49-72`).

Modelling:

- Applied `SelectKBest(k=30)` after encoding.
- Compared a balanced Random Forest against XGBoost.
- Selected the champion by cross-validation Recall, consistent with the project objective.

**Result:** Random Forest was chosen as the Day 4 champion (`CV Recall 0.7184`). On the held-out test set it achieved Recall `0.6711`, F1 `0.5906` and ROC-AUC `0.8238`. XGBoost had higher accuracy and AUC but materially lower cross-validation recall, so it was not selected.

**Generated artifacts:** `models/churn_day4_champion.joblib`, `reports/day4_ensemble_comparison.csv`, `reports/day4_vs_day3_comparison.csv`

### Day 5 — Deep Learning: Keras ANN vs Classical Champion

**Objective:** Introduce deep learning and test whether neural-network depth adds real value on this tabular churn problem.

Implemented:

- **MNIST warm-up** with Keras: flatten → dense/ReLU → dropout → softmax, trained with Adam and `sparse_categorical_crossentropy`.
- **Feed-forward ANN for churn:** normalised inputs → dense/ReLU layers with dropout → one sigmoid output, trained with `binary_crossentropy`.
- **Early stopping** on validation loss with best-weight restoration, plus `ReduceLROnPlateau`.
- **Balanced class weights** to stay consistent with the recall-first objective.
- **Fair benchmark:** the ANN uses the exact same train/test split, the same Day 4 engineered features and the same 30 `SelectKBest` features as the classical champion. Only the estimator family changes.
- **Threshold selection** on a validation split by maximising F1; the test set is never used for tuning.
- Saved model, preprocessing object and metadata; added a reusable `predict_churn_ann()` helper.

**Depth experiment (validation split):**

| Architecture | Hidden layers | Validation AUC | Validation F1 |
|---|---:|---:|---:|
| ANN — Small (1 × 32) | 1 | **0.8469** | 0.6543 |
| ANN — Deep (3 × 128-64-32) | 3 | 0.8461 | **0.6656** |
| ANN — Medium (2 × 64-32) | 2 | 0.8455 | 0.6622 |

**Head-to-head (untouched test set; ANN threshold 0.6694):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Day 4 classical champion | 0.7530 | 0.5273 | **0.6711** | 0.5906 | 0.8238 |
| Day 5 ANN (1 × 32) | **0.7828** | **0.5895** | 0.5989 | **0.5942** | **0.8376** |

**Honest finding:** depth did **not** help. The single-hidden-layer ANN generalised best. The ANN improved ROC-AUC by `+0.0137` and F1 by `+0.0036`, with better precision and accuracy, but the classical champion retained a `+0.0722` Recall advantage at its operating point. On ~5.6k tabular training rows, the ANN is a competitive challenger, not a categorical upgrade.

**Generated artifacts:** `models/churn_day5_ann.keras`, `models/churn_day5_preprocessor.joblib`, `models/churn_day5_metadata.joblib`, `reports/day5_ann_experiments.csv`, `reports/day5_ann_comparison.csv`, `reports/day5_ann_training_history.png`, `reports/day5_roc_comparison.png`

## When to Use Which — Day 5 Decision Rule

| Situation | Prefer |
|---|---|
| Small/medium tabular data, fast iteration, explainability | Classical ML: Random Forest / gradient-boosted trees / SVM |
| Recall-first retention campaign with a fixed operating threshold | Classical champion with a tuned threshold |
| Ranking customers to allocate a limited retention budget | ANN or a calibrated tree ensemble; compare ROC-AUC and lift |
| Large, high-cardinality or unstructured data where representation learning matters | Deep learning |
| Need per-feature auditability and stakeholder trust | Classical ML with feature importance (and later, SHAP) |
| Starting a brand-new tabular problem | Build the classical baseline first; only then try an ANN |

**Carry-forward rule:** on this dataset, keep the Day 4 classical champion as the production default and the ANN as a benchmarked challenger. Deep learning must earn its place through measured gains, not novelty. Revisit the choice when the data grows substantially, when richer behavioural/sequential features arrive, or when the product needs a learned representation rather than hand-engineered features.

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | How many predicted churners actually churn |
| Recall | How many actual churners are correctly identified |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish churners from non-churners across thresholds |

**Why Recall matters:** missing a customer who is about to leave can mean losing potential business. However, Recall is always read alongside Precision, F1 and ROC-AUC so that false positives and operating costs stay visible.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Dhingra2004/RetailMind.git
cd RetailMind
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

Day 5 uses TensorFlow/Keras. The MNIST warm-up downloads the dataset automatically on first run (about 11 MB) and caches it locally.

### 4. Add the dataset

Download the Telco Customer Churn dataset from Kaggle and place the CSV file inside the `data/` directory as:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5. Run the notebook

Open `notebooks/test.ipynb` in Jupyter Notebook, JupyterLab or VS Code, then execute the cells sequentially from Day 1 through Day 5.

## Reproducibility Notes

- `RANDOM_STATE = 42` is used for the train/test split, cross-validation, preprocessing and model seeding.
- The Day 5 ANN sets Python, NumPy and Keras random seeds.
- The test set is used only for final evaluation; early stopping and threshold selection use a validation split carved out of training data.
- TensorFlow may produce minor floating-point differences across CPU/GPU hardware.

## Future Roadmap

This is a 35-day incremental project. Planned development areas include:

- Model explainability and feature-attribution analysis.
- Experiment tracking and performance monitoring.
- A prediction interface / API.
- Deployment, CI and documentation.
- Further deep-learning work where the data justifies it (sequence or text features), with the same benchmark-first discipline.

*Future features will be added as development progresses.*

## Author

**Amarprit Singh Dhingra**

GitHub: [Dhingra2004](https://github.com/Dhingra2004)

LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/amarpritdhingra/)

---

**RetailMind AI — Turning customer data into actionable churn predictions.**
