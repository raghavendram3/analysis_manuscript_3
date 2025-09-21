# Machine Learning Model Comparison for Activation Energy Prediction

A comprehensive comparison of 8 different machine learning regression models (regression.py) for predicting activation energy ($E_a$) with hyperparameter optimization and cross-validation.

The data used in this work is published here: **Meena R,** Purcell MJ, Kluijtmans W, Zuilhof H, Bitter JH, Ouyang R, et al. Activity Descriptors of Mo2C-based Catalysts for C-OH Bond Activation. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-pg52l

## 📊 Overview

This project implements and compares multiple machine learning algorithms to predict activation energy values, featuring automated hyperparameter tuning, cross-validation, and comprehensive performance visualization.

## 🔧 Models Implemented

- **Random Forest Regressor (RFR)**
- **Extra Trees Regressor (ETR)**
- **Support Vector Regressor (SVR)**
- **XGBoost Regressor (XGBR)**
- **Ridge Regression (RR)**
- **Gradient Boosting Regressor (GBR)**
- **K-Nearest Neighbors (KNN)**
- **Kernel Ridge Regression (KRR)**

## 📋 Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## 📁 Data Format

The script expects a data file `data.txt` with:
- Space-separated values
- First column as index
- Target variable column named `$E_a$`
- All other columns as features

## 🚀 Features

### Automated Hyperparameter Optimization
- GridSearchCV with 5-fold cross-validation
- Custom RMSE scorer for model evaluation
- Comprehensive parameter grids for each model

### Performance Metrics
- **RMSE** (Root Mean Square Error) - Train and Test
- **R²** (Coefficient of Determination) - Train and Test
- **Cross-Validation** scores with standard deviation

### Visualization
Four comprehensive plots:
1. **R² Score Comparison** (Train vs Test)
2. **RMSE Comparison** (Train vs Test)
3. **Cross-Validated R²** with error bars
4. **Cross-Validated RMSE** with error bars

## 📊 Model Parameters

### Random Forest & Extra Trees
- `n_estimators`: [100, 300, 500]
- `max_depth`: [None, 5, 10, 20]
- `min_samples_split`: [2, 5, 10]

### Support Vector Regressor
- `C`: [1, 10, 100]
- `gamma`: ['scale', 0.01, 0.1, 1]
- `kernel`: ['rbf', 'poly']

### XGBoost
- `n_estimators`: [100, 300, 500]
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.01, 0.1, 0.2]
- `subsample`: [0.8, 1.0]

### Ridge Regression
- `alpha`: [0.1, 1.0, 10.0, 100.0]

### Gradient Boosting
- `n_estimators`: [100, 300, 500]
- `learning_rate`: [0.01, 0.1, 0.2]
- `max_depth`: [3, 5, 7]

### K-Nearest Neighbors
- `n_neighbors`: [3, 5, 7, 9]
- `weights`: ['uniform', 'distance']
- `p`: [1, 2] (Manhattan, Euclidean)

### Kernel Ridge
- `alpha`: [0.01, 0.1, 1.0, 10.0]
- `kernel`: ['linear', 'rbf', 'polynomial']
- `gamma`: [0.01, 0.1, 1.0]

## 🔄 Workflow

1. **Data Loading**: Reads space-separated data file
2. **Preprocessing**: 
   - Train/test split (85%/15%)
   - StandardScaler normalization
3. **Model Training**: 
   - GridSearchCV optimization for each model
   - 5-fold cross-validation
4. **Evaluation**: 
   - Multiple performance metrics
   - Cross-validation with error estimation
5. **Visualization**: 
   - Publication-ready plots with Times New Roman font
   - Comparative performance charts

## 📈 Output

The script provides:
- Best hyperparameters for each model
- Performance metrics table
- Visual comparisons of model performance
- Identification of the best performing model

## 🎯 Usage

```bash
python model_comparison.py
```

Ensure your data file `data.txt` is in the same directory with the correct format.

## 📊 Results Format

Results include:
- Model name and optimized parameters
- Training and testing RMSE/R² scores  
- Cross-validation means and standard deviations
- Best model recommendation based on test RMSE

## 🎨 Plotting Features

- Professional styling with seaborn
- Times New Roman font (size 20)
- Error bars for cross-validation uncertainty
- Color-coded performance metrics
- Publication-ready figure quality

## 🔍 Model Selection

The best model is automatically identified based on the lowest test RMSE, providing both the model name and its optimized hyperparameters.

## 📝 Notes

- All ensemble models use `random_state=42` for reproducibility
- XGBoost verbosity is set to 0 to reduce console output
- Parallel processing enabled with `n_jobs=-1` for faster computation
- Figures are automatically displayed and can be saved if needed

## 🤝 Contributing

Feel free to contribute by:
- Adding new models
- Expanding hyperparameter grids
- Improving visualization
- Optimizing performance

---
