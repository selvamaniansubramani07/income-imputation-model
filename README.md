# income-imputation-model
Income Imputation Model
Project Overview

This project demonstrates how I built a machine learning model to impute customer income when it is missing or unreliable. Using loan disbursal data, demographics, and bureau scores, I applied data cleaning, feature engineering, and regression models to predict income. The work connects directly to credit risk use-cases such as underwriting, pricing, and fraud detection.

Project Objective

The goal was to impute (predict) customer income using loan disbursal data, demographics, and bureau scores.
By doing this, financial institutions can:

Reduce reliance on incomplete or misreported income fields.

Strengthen credit underwriting and risk-based pricing.

Improve early fraud detection in lending portfolios.

Approach & Methodology

Data Cleaning & Preprocessing

Handled missing values, duplicates, and outliers.

Applied encoding and scaling for model readiness.

Feature Engineering

Created derived features such as loan-to-TPV ratio, age bands, risk bands, and geographic tiers from pincode.

Model Development

Experimented with multiple ML models:

Linear Regression

Random Forest

Gradient Boosting

Evaluated models with RMSE, MAE, and R².

Interpretability

Analyzed feature importance to understand key income drivers such as disbursed loan amount, bureau score, and margin %.

Key Deliverables

Business Report – methodology, insights, and results.

Jupyter Notebook – full data cleaning, feature engineering, and model training pipeline.

Python Script – standalone code to train and test the imputer.

Supporting CSVs – feature importance and final dataset with imputed incomes.

Outcome

Delivered a reusable ML pipeline for income prediction.

Validated performance with cross-validation and error metrics.

Demonstrated how proxy income can support risk modeling, pricing, and regulatory reporting in retail credit.

Tools & Skills Used

Python: pandas, scikit-learn

Machine Learning: regression models, feature importance

Credit Risk Knowledge: PD, LGD, EAD context, income as a driver

Other: Jupyter Notebook, GitHub versioning

Next Steps

Extend to large-scale data using PySpark pipelines.

Deploy as an API for batch and real-time scoring.

Add model monitoring for regulatory audit readiness.

This project reflects my learning journey in credit risk analytics and machine learning, where I combine domain knowledge with practical implementation.
Machine Learning model to impute customer income using loan disbursal data, demographics, and bureau scores.
