import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# Load data
data = pd.read_csv('data.txt', sep="\s+", index_col=0)
X = data.drop(columns=["$E_a$"])
y = data["$E_a$"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom RMSE scorer
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# --- Hyperparameter grids ---
param_grids = {
    "RFR": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "ETR": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "SVR": {
        "C": [1, 10, 100],
        "gamma": ['scale', 0.01, 0.1, 1],
        "kernel": ['rbf', 'poly']
    },
    "XGBR": {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    },
    "RR": {
        "alpha": [0.1, 1.0, 10.0, 100.0]
    },
    "GBR": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ['uniform', 'distance'],
        "p": [1, 2]  # Manhattan, Euclidean
    },
    "KRR": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "kernel": ['linear', 'rbf', 'polynomial'],
        "gamma": [0.01, 0.1, 1.0]
    }
}

# Base models (without params)
base_models = {
    "RR": Ridge(),
    "XGBR": XGBRegressor(random_state=42, verbosity=0),
    "RFR": RandomForestRegressor(random_state=42),
    "ETR": ExtraTreesRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "GBR": GradientBoostingRegressor(random_state=42),
    "KRR": KernelRidge()
}

# Store results
results = []

# --- Hyperparameter optimization ---
for name, model in base_models.items():
    print(f"\n Optimizing {name} ...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        scoring=rmse_scorer,
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    print(f"Best params for {name}: {grid.best_params_}")

    # Predict
    y_pred_test = best_model.predict(X_test_scaled)
    y_pred_train = best_model.predict(X_train_scaled)

    # Metrics
    rmse_test = rmse(y_test, y_pred_test)
    rmse_train = rmse(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    # Cross-validation (with best model)
    rmse_cv = -cross_val_score(best_model, X_train_scaled, y_train, scoring=rmse_scorer, cv=5)
    r2_cv = cross_val_score(best_model, X_train_scaled, y_train, scoring='r2', cv=5)

    results.append({
        "Model": name,
        "Best_Params": grid.best_params_,
        "RMSE_Train": rmse_train,
        "RMSE_Test": rmse_test,
        "R2_Train": r2_train,
        "R2_Test": r2_test,
        "RMSE_CV_Mean": rmse_cv.mean(),
        "RMSE_CV_STD": rmse_cv.std(),
        "R2_CV_Mean": r2_cv.mean(),
        "R2_CV_STD": r2_cv.std()
    })

results_df = pd.DataFrame(results)

# --- PLOTS ---
sns.set(style="whitegrid", font_scale=1.1)

# R² Comparison
plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
results_df_sorted = results_df.sort_values("R2_Test", ascending=False)
plt.barh(results_df_sorted["Model"], results_df_sorted["R2_Test"], color='seagreen', label="Test")
plt.barh(results_df_sorted["Model"], results_df_sorted["R2_Train"], color='lightgreen', alpha=0.6, label="Train")
plt.xlabel("R² Score")
plt.title("R² Score Comparison (Train vs Test)")
plt.legend()
plt.tight_layout()
plt.show()

# RMSE Comparison
plt.figure(figsize=(10, 6))
results_df_sorted = results_df.sort_values("RMSE_Test")
plt.barh(results_df_sorted["Model"], results_df_sorted["RMSE_Test"], color='firebrick', label="Test")
plt.barh(results_df_sorted["Model"], results_df_sorted["RMSE_Train"], color='salmon', alpha=0.6, label="Train")
plt.xlabel("RMSE")
plt.title("RMSE Comparison (Train vs Test)")
plt.legend()
plt.tight_layout()
plt.show()

# Cross-Validated R²
plt.figure(figsize=(10, 6))
plt.errorbar(results_df["Model"], results_df["R2_CV_Mean"], yerr=results_df["R2_CV_STD"], fmt='o', color='mediumblue', capsize=5)
plt.xticks(rotation=45)
plt.ylabel("Cross-Validated R²")
plt.title("Cross-Validated R² with ±1 Std Dev")
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.tight_layout()
plt.show()

# Cross-Validated RMSE
plt.figure(figsize=(10, 6))
plt.errorbar(results_df["Model"], results_df["RMSE_CV_Mean"], yerr=results_df["RMSE_CV_STD"], fmt='o', color='darkred', capsize=5)
plt.xticks(rotation=45)
plt.ylabel("Cross-Validated RMSE")
plt.title("Cross-Validated RMSE with ±1 Std Dev")
plt.tight_layout()
plt.show()

# Print best model
best_model = results_df.sort_values("RMSE_Test").iloc[0]
print(f"\n Best model: {best_model['Model']} with params {best_model['Best_Params']}")
