# Customer Churn Prediction

End-to-end machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset.

The project covers the complete machine learning workflow:

- Data understanding
- Data cleaning
- Exploratory data analysis
- Preprocessing
- Baseline comparison
- Model comparison
- Hyperparameter tuning
- Out-of-fold threshold selection
- Final test evaluation
- Train vs test diagnostics
- Model interpretation
- Error analysis
- Input validation
- Model serialization
- Automated testing

---

## Project Objective

Customer churn means that a customer stops using a company's service.

The goal of this project is to predict whether a customer is likely to churn based on customer, service, contract, and billing information.

The target variable is:

- `Churn = 1` → Customer churned
- `Churn = 0` → Customer did not churn

---

## Dataset

This project uses the IBM Telco Customer Churn dataset.

The dataset contains:

- 7,043 customers
- 21 original columns
- Customer information
- Service information
- Contract information
- Billing information
- Churn target

The `customerID` column is excluded from modeling because it is an identifier rather than a meaningful predictive feature.

### Dataset Source and Attribution

This project uses the IBM Telco Customer Churn sample dataset.

The dataset represents a fictional telecommunications company and is intended for analytics and machine learning practice.

Sources:

- [IBM Telco Customer Churn sample](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn)
- [Kaggle mirror: Telco Customer Churn by BlastChar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains 7,043 customer records and 21 columns.

The Kaggle dataset page attributes the data to the original authors and lists the data files as © Original Authors. This project does not claim ownership of the dataset.

---

## Main Features

### Customer Information

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

### Customer Relationship

- `tenure`

`tenure` represents the number of months the customer has stayed with the company.

### Phone Services

- `PhoneService`
- `MultipleLines`

### Internet Services

- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

### Contract and Billing

- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

---

## Data Cleaning

The `TotalCharges` column contains blank string values even though it represents numeric information.

It is converted using:

```python
pd.to_numeric(..., errors="coerce")
```

The affected rows correspond to customers with:

```text
tenure = 0
```

For these customers:

```text
TotalCharges = 0
```

Duplicate checks showed:

- No duplicated rows
- No duplicated customer IDs

---

## Exploratory Data Analysis

The dataset is moderately imbalanced:

- No Churn: approximately 73.5%
- Churn: approximately 26.5%

### Contract Type

Observed churn rates:

- Month-to-month: approximately 42.7%
- One year: approximately 11.3%
- Two year: approximately 2.8%

### Internet Service

Observed churn rates:

- Fiber optic: approximately 41.9%
- DSL: approximately 19.0%
- No internet: approximately 7.4%

### Important Observed Patterns

Customers who churned tended to have:

- Shorter tenure
- Higher monthly charges
- Month-to-month contracts
- Fiber optic internet
- No technical support
- No online security

The combination of month-to-month contracts and fiber optic service showed the highest observed churn rate among the analyzed Contract × InternetService combinations.

These findings describe statistical associations and should not be interpreted as causal relationships.

---

## Train-Test Split

The dataset is split using a stratified train/test split:

```text
80% training data
20% test data
```

Stratification preserves the churn class distribution.

The test set is kept untouched during model comparison, hyperparameter tuning, and threshold selection.

---

## Preprocessing Pipeline

All preprocessing is placed inside a scikit-learn `Pipeline` to reduce the risk of data leakage.

### Numeric Features

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Processing:

1. Median imputation
2. Standard scaling

### Categorical Features

Processing:

1. Most-frequent imputation
2. One-hot encoding

Unknown categories during inference are handled using:

```python
OneHotEncoder(handle_unknown="ignore")
```

`SeniorCitizen` is stored numerically as `0/1`, but it is treated as a categorical variable.

---

## Baseline Model

A `DummyClassifier` is used as a simple baseline.

Because approximately 73.5% of customers belong to the No Churn class, a model that always predicts the majority class can achieve relatively high accuracy while completely failing to identify churn customers.

This demonstrates why accuracy alone is not sufficient.

---

## Model Comparison

The following models were compared using 5-fold stratified cross-validation on the training data:

- Logistic Regression
- Gradient Boosting
- Random Forest
- Decision Tree

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8019 | 0.6521 | 0.5438 | 0.5924 | 0.8461 |
| Gradient Boosting | 0.8033 | 0.6622 | 0.5291 | 0.5880 | 0.8482 |
| Random Forest | 0.7867 | 0.6273 | 0.4856 | 0.5471 | 0.8202 |
| Decision Tree | 0.7322 | 0.4957 | 0.5003 | 0.4975 | 0.6582 |

These initial results were used as a screening stage before hyperparameter tuning.

Logistic Regression and Gradient Boosting were selected for tuning because they were the two strongest candidates in the initial cross-validation screening, while Logistic Regression also offered better interpretability for the final portfolio model.

---

## Hyperparameter Tuning

### Logistic Regression

Best parameters:

```text
C = 0.1
class_weight = balanced
```

Best cross-validated F1:

```text
0.6288
```

### Gradient Boosting

Best parameters:

```text
learning_rate = 0.1
max_depth = 3
n_estimators = 100
```

Best cross-validated F1:

```text
0.5880
```

---

## Final Model Selection

Logistic Regression was selected as the final model because it achieved the stronger tuned cross-validated F1 score.

It also provides relatively interpretable coefficients compared with more complex ensemble models.

The tuned F1 comparison was performed at the default 0.5 classification threshold, while the final deployed threshold was selected separately using out-of-fold training predictions. Gradient Boosting did not use class weighting, and the screening ROC-AUC scores for Logistic Regression and Gradient Boosting were very close. Therefore, the selection should be interpreted as a practical portfolio-model choice based on tuned F1 and interpretability rather than proof that Logistic Regression is universally superior.

---

## Threshold Selection

The classification threshold was selected using out-of-fold probabilities generated from the training data only.

The test set was not used for threshold tuning.

The selected threshold was:

```text
0.54
```

OOF performance at this threshold:

| Metric | Score |
|---|---:|
| Accuracy | 0.7666 |
| Precision | 0.5422 |
| Recall | 0.7732 |
| F1 | 0.6374 |

F1 was used as the threshold-selection objective because no real business cost matrix was available.

In a production environment, the threshold should ideally reflect the business cost of false negatives and false positives.

---

## Final Test Evaluation

After model selection, hyperparameter tuning, and threshold selection were complete, the final model was trained using the full training dataset.

The holdout test set was then used for the official final evaluation.

### Confusion Matrix

```text
TN = 780
FP = 255
FN = 91
TP = 283
```

### Final Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.7544 |
| Precision | 0.5260 |
| Recall | 0.7567 |
| F1 Score | 0.6206 |
| ROC-AUC | 0.8408 |
| PR-AUC | 0.6330 |

The model identifies approximately 75.7% of the actual churn customers.

---

## Train vs Test Diagnostic

| Metric | Train | Test |
|---|---:|---:|
| Accuracy | 0.7677 | 0.7544 |
| Precision | 0.5435 | 0.5260 |
| Recall | 0.7779 | 0.7567 |
| F1 | 0.6399 | 0.6206 |
| ROC-AUC | 0.8485 | 0.8408 |
| PR-AUC | 0.6640 | 0.6330 |

The relatively small differences between training and test performance do not indicate strong overfitting.

---

## Model Interpretation

The Logistic Regression coefficients were inspected after training.

Features associated with higher predicted churn included:

- Month-to-month contract
- Fiber optic internet
- Electronic check payment
- No online security
- No technical support

Features associated with lower predicted churn included:

- Longer tenure
- Two-year contract
- DSL internet
- Online security
- Technical support

Odds ratios are calculated using:

```python
np.exp(coefficient)
```

Because one-hot encoding keeps all category levels and the model uses L2 regularization, coefficients should be interpreted as regularized model associations rather than causal effects.

---

## Error Analysis

The final model produced:

```text
91 false negatives
255 false positives
```

### False Negatives

False negatives were disproportionately associated with:

- One-year contracts
- Two-year contracts
- DSL internet
- No internet service
- Technical support enabled

This suggests that the model has more difficulty identifying churn customers whose profiles look less like the common high-risk churn pattern.

### False Positives

False positives were strongly associated with:

- Month-to-month contracts
- Fiber optic internet
- No technical support

This suggests that the model sometimes predicts churn for customers who look similar to common high-risk profiles but ultimately remain with the company.

Error analysis is used to understand model behavior and is not used to retune the model using the test set.

---

## Input Validation

The prediction module validates incoming customer data before inference.

Validation includes:

- Missing features
- Unexpected features
- Invalid categorical values
- Invalid numeric types
- Boolean values supplied for numeric features
- Non-finite numeric values such as `NaN` and infinity
- Negative numeric values
- Internet-service consistency
- Phone-service consistency

Example:

If:

```text
InternetService = No
```

internet-dependent features must contain:

```text
No internet service
```

---

## Model Serialization

The trained model and selected threshold are saved together using `joblib`.

```python
{
    "model": final_model,
    "threshold": best_threshold
}
```

Saved model:

```text
models/churn_model.joblib
```

The prediction module loads the saved threshold automatically instead of hardcoding it.

The model bundle is loaded lazily on the first prediction request and cached for subsequent predictions, avoiding unnecessary model loading during module imports.

---

## Automated Tests

The project uses `pytest` for automated testing.

Current tests cover:

- `TotalCharges` numeric conversion
- Zero-tenure cleaning behavior
- Valid customer input
- Missing input features
- Unexpected input features
- Negative numeric values
- `NaN` rejection
- Infinite numeric-value rejection
- Boolean rejection for numeric fields
- Invalid categorical values
- Internet-service consistency
- Phone-service consistency
- `predict_churn` model/threshold behavior
- Missing model-file handling
- Invalid model-bundle handling
- Training metric calculation
- Final model selection
- Exact out-of-fold threshold selection

Run all tests with:

```bash
pytest
```

Current test result:

```text
19 passed
```

---

## Project Structure

```text
customer-churn-ml/
│
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_predict.py
│   └── test_train.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hazem-elsayed1/customer-churn-ml.git
```

Enter the project:

```bash
cd customer-churn-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

`src/train.py` is the canonical source of truth for the reusable training pipeline. Notebook 03 mirrors the same methodology for experimentation and portfolio presentation.

## Training

Run the complete training pipeline:

```bash
python -m src.train
```

The training workflow performs:

```text
Load data
↓
Clean data
↓
Train/Test split
↓
Cross-validation model comparison
↓
Hyperparameter tuning
↓
Final model selection
↓
OOF threshold selection
↓
Train final model
↓
Final test evaluation
↓
Train/Test diagnostic
↓
Save model and threshold
```

---

## Prediction

Run:

```bash
python -m src.predict
```

Example output:

```text
==================================================
CUSTOMER CHURN PREDICTION
==================================================
Prediction: 1
Label: Churn
Probability: 0.9105
Threshold: 0.54
```

Where:

```text
1 = Churn
0 = No Churn
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Pytest
- Jupyter Notebook
- Git
- GitHub
- GitHub Actions

---

## Key Learning Outcomes

This project demonstrates:

- Data cleaning
- Exploratory data analysis
- Class imbalance awareness
- Feature preprocessing
- Scikit-learn pipelines
- Leakage prevention
- Dummy baselines
- Cross-validation
- Hyperparameter tuning
- Out-of-fold predictions
- Threshold optimization
- Holdout evaluation
- Model interpretation
- Error analysis
- Input validation
- Automated testing
- Model serialization
- Reusable Python modules
- Git version control
- Continuous integration

---

## Limitations

This project is an offline machine learning portfolio project and does not use live company data.

Current limitations include:

- No real business cost matrix
- No external validation dataset
- No production monitoring
- No probability calibration analysis
- No confidence intervals for final metrics

The selected threshold should therefore not be interpreted as a universal business-optimal threshold.

---

## Continuous Integration

GitHub Actions is configured to run the automated test suite on pushes and pull requests to the `main` branch.

The CI workflow:

- Sets up Python
- Installs project dependencies
- Runs the full `pytest` test suite

This helps catch regressions automatically before changes are merged.

---

## Future Improvements

Possible future improvements include:

- Probability calibration
- Bootstrap confidence intervals
- Cost-sensitive threshold selection
- FastAPI REST API
- Docker deployment
- Model versioning
- Data drift monitoring
- Model performance monitoring
