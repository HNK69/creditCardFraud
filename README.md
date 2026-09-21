# 💳 Credit Card Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit-card transactions under extreme class imbalance, with a production-oriented MLOps workflow covering model development, explainability, experiment tracking, data versioning, reproducibility, testing, Docker, CI/CD, cloud deployment, and monitoring.

## 🚀 Live Demo

**Production App:** https://creditcardfraud-avt4.onrender.com

> The application is deployed as a Dockerized Streamlit service on Render.

---

## 📌 Problem Statement

Credit-card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small fraction of all transactions.

The objective of this project is to build a supervised classification system that estimates whether a transaction is fraudulent while:

- Handling severe class imbalance
- Avoiding data leakage
- Optimizing the classification threshold
- Evaluating models using fraud-appropriate metrics
- Providing model explanations
- Packaging the model for inference
- Tracking experiments and model versions
- Testing the inference pipeline automatically
- Containerizing and deploying the application
- Establishing a monitoring and retraining strategy

---

## 🧠 Dataset

This project uses the **ULB / Worldline Credit Card Fraud Detection dataset**.

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset characteristics

- 284,807 transactions originally
- 492 fraudulent transactions
- 31 columns
- 30 input features
- `Class` is the target
- `Class = 1` → Fraud
- `Class = 0` → Legitimate
- Fraud rate ≈ **0.172%**
- `V1`–`V28` are anonymized PCA-derived features
- `Time` and `Amount` are non-transformed features

### Important limitation

The dataset represents transactions collected over a limited historical period and contains anonymized features. Therefore, results should not be interpreted as direct evidence of performance on a modern production banking dataset.

---

# 🔬 Machine Learning Workflow

## 1. Data Understanding

Initial analysis covered:

- Dataset dimensions
- Data types
- Missing values
- Class distribution
- Duplicate records
- Feature characteristics

After removing exact duplicate rows:

```text
Dataset shape: (283726, 31)
Features:      30
Target:        Class
```

No missing values were found.

---

## 2. Exploratory Data Analysis

EDA investigated:

- Severe class imbalance
- Transaction amount distribution
- Transaction time patterns
- Feature distributions
- Feature-target correlations
- Duplicate groups
- Duplicate groups with conflicting labels

The dataset was found to be extremely imbalanced, making accuracy unsuitable as the primary evaluation metric.

---

## 3. Preprocessing

### Duplicate handling

Exact duplicate rows were removed.

### Train/Test Split

A stratified split was used:

```text
Training set: 226,980 transactions
Test set:       56,746 transactions
```

### Scaling

`StandardScaler` was fitted only on training data and then applied to the test data.

This prevents information from the test set leaking into preprocessing.

The scaler was serialized as:

```text
models/standard_scaler.joblib
```

---

# 🤖 Model Development

Three major classifiers were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Because fraud detection is highly imbalanced, the evaluation emphasized:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC / Average Precision
- Confusion Matrix

## Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0564 | 0.8737 | 0.1060 | 0.9657 | 0.6738 |
| Random Forest | 0.9714 | 0.7158 | 0.8242 | 0.9447 | 0.8078 |
| XGBoost | **0.9048** | **0.8000** | **0.8492** | **0.9761** | **0.8248** |

---

# 🏆 Final Model

The final candidate is a class-weighted **XGBoost** classifier using `scale_pos_weight` to account for class imbalance.

### Configuration

```text
n_estimators:      200
max_depth:         6
learning_rate:     0.1
subsample:         0.8
colsample_bytree:  0.8
scale_pos_weight:  599.4762
random_state:      42
```

The classification threshold was optimized using a validation set rather than the final test set.

### Final threshold

```text
0.2325
```

This threshold is intentionally lower than the default 0.5 to reflect the fraud-detection precision/recall trade-off.

---

# 📊 Final Test Results

| Metric | Value |
|---|---:|
| Precision | **0.9048** |
| Recall | **0.8000** |
| F1-score | **0.8492** |
| ROC-AUC | **0.9761** |
| PR-AUC | **0.8248** |
| Decision threshold | **0.2325** |

### Confusion Matrix

```text
[[56643,     8],
 [   19,    76]]
```

The final test evaluation was performed only after model and threshold decisions were completed.

---

# ⚖️ Imbalanced Learning

Two approaches were compared:

### Class-weighted XGBoost

Used:

```python
scale_pos_weight
```

Cross-validation mean PR-AUC:

```text
0.8499
```

### SMOTE + XGBoost

SMOTE was applied only inside the training folds using an imbalanced-learn pipeline.

Cross-validation mean PR-AUC:

```text
0.8459
```

The class-weighted approach was retained for the final model based on the cross-validation comparison.

---

# 🎯 Threshold Optimization

The default classification threshold of `0.5` was not assumed to be optimal.

A validation subset was used to select a threshold based on F1-score.

Selected threshold:

```text
0.2325
```

Validation result at the selected threshold:

```text
Precision: 0.9565
Recall:    0.8684
F1:        0.9103
```

The test set remained untouched during threshold selection.

---

# 🔍 Explainability with SHAP

SHAP was used to understand how the XGBoost model makes predictions.

### Global feature importance

The most influential features included:

```text
V14
V4
V12
V10
V3
V11
V8
V1
V26
```

Important note:

> Feature importance does not imply causation, and the anonymized PCA-derived features cannot be interpreted as direct banking variables.

### Local explanations

SHAP waterfall explanations were also generated for individual transactions to show how individual feature values pushed predictions toward fraud or legitimate classes.

---

# 🏭 Production Inference Pipeline

The production inference flow is:

```text
Raw transaction
       ↓
Input validation
       ↓
Feature validation
       ↓
Missing/infinite value checks
       ↓
StandardScaler
       ↓
XGBoost
       ↓
Fraud probability
       ↓
Threshold = 0.2325
       ↓
Fraud / Legitimate
```

Inference implementation:

```text
src/inference/predict.py
```

Production artifacts:

```text
models/
├── final_xgb_model.joblib
├── standard_scaler.joblib
└── fraud_threshold.joblib
```

---

# 🧪 Testing

The inference pipeline contains automated tests covering:

- Valid prediction structure
- Valid probability range
- Missing features
- Extra features
- Non-numeric features
- Missing values

Current test suite:

```text
6 tests passed
```

Tests are executed locally with `pytest` and automatically through GitHub Actions.

---

# 📈 Experiment Tracking — MLflow

MLflow is used to track:

- Model parameters
- Evaluation metrics
- Experiment runs
- Model artifacts
- Registered model versions

Tracked final metrics include:

```text
Precision: 0.9048
Recall:    0.8000
F1:        0.8492
ROC-AUC:   0.9761
PR-AUC:    0.8248
```

---

# 🗂️ Model Registry

The final XGBoost model is registered in the MLflow Model Registry as:

```text
CreditCardFraud-XGBoost
```

Model aliases were used for deployment management:

```text
candidate
production
```

The production alias points to the validated model version.

A rollback exercise was also performed by moving the production alias between registered model versions.

---

# 📦 Data Versioning — DVC

DVC is used to version the large training dataset instead of storing it directly in Git.

The dataset is represented by:

```text
data/raw/creditcard.csv.dvc
```

A local DVC remote was used for the project to demonstrate:

- Dataset versioning
- Dataset updates
- Dataset rollback
- Reproducible dataset retrieval

The large CSV itself remains outside normal Git version control.

---

# 🔁 Reproducibility

Model configuration is stored in:

```text
config/model_config.yaml
```

It contains:

- Model hyperparameters
- Random seed
- Class-weight configuration
- Classification threshold
- Test split configuration

Reproducibility is based on maintaining:

```text
Dataset version
+
Code version
+
Preprocessing
+
Model configuration
+
Random seeds
+
Threshold
+
Environment/dependencies
```

---

# 🐳 Docker

The application is containerized using Docker.

Build:

```bash
docker build -t credit-card-fraud .
```

Run:

```bash
docker run -d -p 8501:8501 --name credit-card-fraud-app credit-card-fraud:latest
```

Application:

```text
http://localhost:8501
```

The production Docker image contains the inference code, Streamlit application, configuration, and model artifacts, but does not package the large training dataset.

---

# 🔄 CI — Continuous Integration

GitHub Actions automatically runs the test suite when changes are pushed to `main` or submitted through a pull request.

Workflow:

```text
Git push
   ↓
GitHub Actions
   ↓
Install dependencies
   ↓
Run pytest
   ↓
6 tests
   ↓
Pass / Fail
```

Workflow file:

```text
.github/workflows/tests.yml
```

The CI tests use lightweight test-time model artifacts so that CI does not depend on local production model files.

---

# 🚀 CD — Continuous Deployment

The application is deployed using Render.

Deployment flow:

```text
GitHub
   ↓
Render detects new commit
   ↓
Docker build
   ↓
Container deployment
   ↓
Streamlit
   ↓
Public application
```

Production deployment:

https://creditcardfraud-avt4.onrender.com

The deployed application was tested with synthetic transactions to verify the complete inference path.

> Synthetic transactions validate deployment and inference functionality; they are not evidence of real-world model accuracy.

---

# 📡 Monitoring Strategy

The project defines three monitoring layers.

## 1. Data Drift

The Kolmogorov-Smirnov test was demonstrated to compare feature distributions.

The reference comparison showed statistically detectable but very small differences for a few features.

In real production, the comparison should be:

```text
Training/reference distribution
             vs
New production transaction batches
```

rather than relying on the original test set indefinitely.

## 2. Prediction Drift

The reference prediction baseline was measured:

```text
Mean fraud probability ≈ 0.153%
Predicted fraud rate    ≈ 0.148%
```

Large sustained changes in these values can indicate changing transaction behavior or operational issues requiring investigation.

## 3. Model Performance

When ground-truth fraud labels become available, production predictions can be evaluated using:

- Precision
- Recall
- F1-score
- PR-AUC

Current reference baseline:

```text
Precision: 0.9048
Recall:    0.8000
F1:        0.8492
PR-AUC:    0.8248
```

---

# 🔁 Retraining Strategy

Retraining should not happen merely because a single statistical drift test is significant.

A retraining investigation can be triggered by:

- Sustained meaningful data drift
- Significant prediction-distribution changes
- Degradation in production PR-AUC
- Degradation in recall/precision
- Changes in business requirements
- Availability of sufficiently large, labeled new data

Proposed lifecycle:

```text
Production Data
      ↓
Monitoring
      ↓
Drift / Performance Issue
      ↓
Collect & Validate New Labeled Data
      ↓
Retrain
      ↓
Cross-validation
      ↓
Compare against Production Model
      ↓
MLflow Registry
      ↓
Candidate Model
      ↓
Validation
      ↓
Production Alias
      ↓
Deployment
      ↓
Monitoring
```

---

# 🏗️ Project Structure

```text
creditCardFraud/
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── creditcard.csv.dvc
│   │   └── creditcard.csv
│   └── processed/
│
├── models/
│   ├── final_xgb_model.joblib
│   ├── standard_scaler.joblib
│   └── fraud_threshold.joblib
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_finalization.ipynb
│   ├── 06_inference_pipeline.ipynb
│   ├── 07_experiment_tracking.ipynb
│   ├── 08_reproducibility.ipynb
│   └── 09_monitoring.ipynb
│
├── src/
│   └── inference/
│       ├── __init__.py
│       └── predict.py
│
├── tests/
│   └── test_inference.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP

### MLOps

- MLflow
- DVC
- Git
- GitHub Actions
- Docker

### Deployment

- Streamlit
- Render

### Development

- Jupyter
- Pytest

---

# ▶️ Run Locally

## 1. Clone

```bash
git clone https://github.com/HNK69/creditCardFraud.git
cd creditCardFraud
```

## 2. Create environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run tests

```bash
$env:PYTHONPATH="."
pytest -q
```

## 5. Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 📚 Key Engineering Decisions

### Why PR-AUC?

Fraud is extremely rare. PR-AUC focuses directly on the precision/recall behavior of the positive class and is more informative than accuracy for this problem.

### Why not simply use 0.5 as the threshold?

The optimal threshold depends on the operational trade-off between false positives and false negatives. A validation-set threshold of `0.2325` was selected instead.

### Why not use SMOTE as the final approach?

SMOTE was evaluated through a leakage-safe cross-validation pipeline. The class-weighted XGBoost approach achieved a slightly higher mean CV PR-AUC in this experiment.

### Why keep the test set untouched during threshold selection?

Using the test set to choose the threshold would leak information from the final evaluation set and produce an overly optimistic estimate.

### Why use DVC?

The dataset is large and should not be managed directly through normal Git history. DVC tracks dataset versions while keeping the large data artifact separate.

### Why use MLflow Model Registry?

The registry separates model training from deployment and allows model versions and aliases such as `candidate` and `production` to be managed explicitly.

---

# ⚠️ Limitations

This project is an engineering and learning implementation rather than a production banking fraud-detection system.

Important limitations include:

- The dataset covers a limited historical period.
- V1–V28 are anonymized PCA-derived features.
- No real-time transaction stream is available.
- No real production fraud labels are available.
- Monitoring is demonstrated offline.
- Production fraud detection requires continuously updated labeled data.
- The deployed Streamlit application is a demonstration system, not a regulated payment-processing system.
- The public deployment may sleep when inactive depending on the hosting plan.

---

# 🔮 Future Improvements

Possible extensions include:

- Real-time transaction ingestion API
- Kafka-based streaming
- Feature store
- Online feature engineering
- Automated drift monitoring
- Evidently-based monitoring dashboards
- Production data logging
- Delayed-label performance monitoring
- Automated retraining pipelines
- Model approval gates
- Canary/shadow deployment
- Cloud object storage for DVC
- Remote MLflow tracking server
- Authentication and access control
- Alerting through email/Slack
- Model calibration
- Cost-sensitive threshold optimization

---

# 📄 License

This project is intended for educational, portfolio, and experimentation purposes.

---

## 👤 Author

**H N Krupaal**

GitHub: https://github.com/HNK69

Repository: https://github.com/HNK69/creditCardFraud
