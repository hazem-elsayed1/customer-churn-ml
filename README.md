# Customer Churn Prediction

End-to-end machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset.

The project covers the full ML workflow, including data understanding, cleaning, exploratory data analysis, preprocessing, model comparison, hyperparameter tuning, threshold tuning, model interpretation, error analysis, training, and prediction.

---

## Project Overview

Customer churn means that a customer stops using a company's service.

The goal of this project is to build a machine learning model that predicts whether a customer is likely to churn based on information such as:

- Contract type
- Tenure
- Internet service
- Technical support
- Monthly charges
- Payment method
- Other customer and service-related features

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
- Contract and billing information
- Churn target

---

## Main Features

### Customer Information

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

### Customer Relationship

- `tenure`

`tenure` represents how many months the customer has stayed with the company.

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

### Target

- `Churn`

---

## Data Cleaning

During data analysis, the `TotalCharges` column was found to contain blank string values even though standard missing-value checks initially showed no missing values.

The column was converted to numeric using:

```python
pd.to_numeric(..., errors="coerce")