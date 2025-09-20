# Step 1: Load & Explore Data
from dbm import error

import pandas as pd
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import label_ranking_average_precision_score

df = pd.read_csv("Income_Imputation_Base_Data.csv", low_memory=False)

print("Shape:",df.shape)                                                        #Rows & Columns
print("Columns:",df.columns)                                                    #Show initial column names
print(df.head())                                                                #Preview Of Rows
print(df.isna().mean().sort_values(ascending=False).head(12).round(3))

# Step 2: Cleaning
# Drop useless columns
# Tier & Margin are 100% missing, safe to drop

import numpy as np

df = df.drop(columns=["Tier","Margin"])

if "loan_application_no" in df.columns:                             #removing duplicate records
    before = df.shape[0]
    df     = df.drop_duplicates(subset=["loan_application_no"])
    after  = df.shape[0]
    print(f"Dropped {before-after} Duplicate Rows!")

df["disbursed_date"]   = pd.to_datetime(df["disbursed_date"], errors="coerce")
df["disbursed_amount"] = pd.to_numeric(df["disbursed_amount"], errors="coerce")
df["score"]            = pd.to_numeric(df["score"], errors="coerce")
df["age"]              = pd.to_numeric(df["age"], errors="coerce")

# cap extreme outliers
def winsorize(s):
    q01, q99 = s.quantile([0.01,0.99])
    return s.clip(q01,q99)

for col in ["disbursed_amount", "final_tpv"]:
    df[col] = winsorize(df[col])

# filling categorical nulls with 'Unknown'
df.columns = df.columns.str.lower()
for col in ["gender", "industryy", "status", "city", "state", "source_entity_name"]:
    df[col] = df[col].fillna("unknown")

# filling numeric nulls with median
for col in ["score", "age", "disbursed_amount", "final_tpv"]:
    df[col] = df[col].fillna(df[col].median())

print("After cleaning:", df.shape)
print(df.isna().mean().sort_values(ascending=False).head(10))

# Step 3: Feature Engineering
df["disbursed_year"]    = df["disbursed_date"].dt.year
df["disbursed_month"]   = df["disbursed_date"].dt.month
df["disbursed_quarter"] = df["disbursed_date"].dt.quarter

# Risk bands from score
df["score_band"] = pd.cut(
    df["score"],
    bins=[-np.inf,600,700,750, np.inf],
    labels=["<600","600-699","700-749",">750"]
).astype(str)

# Risk bands from age
df["age_band"] = pd.cut(
    df["age"],
    bins=[-np.inf,21,25,30,35,45,60,np.inf],
    labels=["<=21","22-25","26-30","31-35","36-45","46-60","60+"]
).astype(str)


# Metro flag

metros = {"mumbai","delhi","new delhi","chennai","bengaluru","bangalore","hyderabad","kolkata","pune","ahmedabad"}
df["city_clean"] = df ["city"].astype(str).str.strip().str.lower()
df["is_metro_city"] = df["city_clean"].isin(metros).astype(int)

#Ratios
df["loan_to_tpv"]     = df["disbursed_amount"] / (df["final_tpv"].replace(0, np.nan))
df["amount_per_age"]  = df["disbursed_amount"] / df["age"].replace(0, np.nan)
df["amount_x_score"]  = df["disbursed_amount"] * df["score"]

print("After feature engineering:", df.shape)
print(df.head())

#Model Development
from sklearn.model_selection import train_test_split

y = df["final_tpv"]

# drop columns that are IDs or directly leak target
X = df.drop(columns=[
    "final_tpv",                                                                            # target itself
    "loan_application_no",                                                                  # ID, not useful
    "disbursed_date",                                                               # already extracted year/month
    "city", "city_clean"                                                            # metro flag already made
])

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# convert categorical to numeric
X = pd.get_dummies(X, drop_first=True)

print("Any missing values left?", X.isna().sum().sum())

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Baseline Model: Linear Regression
# Linear regression is our baseline model ---> quick, interpretable, easy to compare.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# train model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# predictions
y_pred_lr = lin_reg.predict(X_test)

# metrics
mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

#Random Forest model

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("MAE :", mae_rf)
print("RMSE:", rmse_rf)
print("R²  :", r2_rf)

# Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# predictions
y_pred_gbr = gbr.predict(X_test)

# metrics
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
r2_gbr = r2_score(y_test, y_pred_gbr)

print("\nGradient Boosting Results:")
print("MAE :", mae_gbr)
print("RMSE:", rmse_gbr)
print("R²  :", r2_gbr)

#Cross-validation,validating RF isn’t overfitting(5-fold on Random Forest)

from sklearn.model_selection import KFold, cross_val_score
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# scikit-learn returns negative errors for loss metrics; flip sign and take sqrt for RMSE
cv_mae  = -cross_val_score(rf, X, y, scoring="neg_mean_absolute_error", cv=kf)
cv_mse  = -cross_val_score(rf, X, y, scoring="neg_mean_squared_error", cv=kf)
cv_rmse = np.sqrt(cv_mse)
cv_r2   =  cross_val_score(rf, X, y, scoring="r2", cv=kf)

print("\nCross-validation (Random Forest, 5 folds):")
print(f"MAE  : {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
print(f"RMSE : {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
print(f"R²   : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

#importance of feature what drives predicted income
import pandas as pd

fi = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 features:")
print(fi.head(15))

#Score every row + save all deliverables

import json
from joblib import dump

# 1) predict income for each row (using best model → RF or GB)
df["predicted_income"] = rf.predict(X)   # you can switch rf → gbr if GB is better

# 2) save cleaned + predictions
df.to_csv("cleaned_with_predicted_income.csv", index=False)

# 3) save feature importance (from RF)
fi.to_csv("feature_importance.csv", index=False)

# 4) save metrics (LR, RF, GB, + CV)
metrics = {
    "linear_regression": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    },
    "random_forest": {
        "MAE": float(mae_rf),
        "RMSE": float(rmse_rf),
        "R2": float(r2_rf)
    },
    "gradient_boosting": {
        "MAE": float(mae_gb),
        "RMSE": float(rmse_gb),
        "R2": float(r2_gb)
    },
    "cv_random_forest": {
        "MAE_mean": float(cv_mae.mean()), "MAE_std": float(cv_mae.std()),
        "RMSE_mean": float(cv_rmse.mean()), "RMSE_std": float(cv_rmse.std()),
        "R2_mean": float(cv_r2.mean()), "R2_std": float(cv_r2.std())
    }
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 5) save the trained model artifact (choose best one — here RF)
dump(rf, "best_income_imputer.joblib")
# If GB is better, you can also save it like this:
# dump(gbr, "best_income_imputer.joblib")

print("\nArtifacts written:")
print(" - cleaned_with_predicted_income.csv")
print(" - feature_importance.csv")
print(" - model_metrics.json")
print(" - best_income_imputer.joblib")
print(" - feature_importance_top15.png")









