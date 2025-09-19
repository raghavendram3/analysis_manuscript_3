"""
Machine Learning Model Comparison and Hyperparameter Optimization

This module compares multiple regression models for predicting activation energy ($E_a$)
and performs hyperparameter optimization using GridSearchCV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Store model evaluation results."""
    model_name: str
    best_params: Dict[str, Any]
    rmse_train: float
    rmse_test: float
    r2_train: float
    r2_test: float
    rmse_cv_mean: float
    rmse_cv_std: float
    r2_cv_mean: float
    r2_cv_std: float


class ModelEvaluator:
    """Handles model training, evaluation, and comparison."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results: List[ModelResults] = []
        
    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data from file."""
        try:
            data = pd.read_csv(filepath, sep=r"\s+", index_col=0)
            X = data.drop(columns=["$E_a$"])
            y = data["$E_a$"]
            logger.info(f"Data loaded successfully. Shape: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                     test_size: float = 0.15) -> Tuple[np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray]:
        """Split and scale the data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def get_models(self) -> Dict[str, BaseEstimator]:
        """Initialize all models to be evaluated."""
        return {
            "Ridge": Ridge(),
            "XGBoost": XGBRegressor(random_state=self.random_state, verbosity=0),
            "Random Forest": RandomForestRegressor(random_state=self.random_state),
            "Extra Trees": ExtraTreesRegressor(random_state=self.random_state),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
            "Kernel Ridge": KernelRidge()
        }
    
    def get_param_grids(self) -> Dict[str, Dict[str, List[Any]]]:
        """Define hyperparameter grids for each model."""
        return {
            "Random Forest": {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "Extra Trees": {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "SVR": {
                "C": [1, 10, 100],
                "gamma": ['scale', 0.01, 0.1, 1],
                "kernel": ['rbf', 'poly']
            },
            "XGBoost": {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0]
            },
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            },
            "Gradient Boosting": {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "KNN": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ['uniform', 'distance'],
                "p": [1, 2]
            },
            "Kernel Ridge": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "kernel": ['linear', 'rbf', 'polynomial'],
                "gamma": [0.01, 0.1, 1.0]
            }
        }
    
    def optimize_model(self, model: BaseEstimator, param_grid: Dict[str, List[Any]], 
                      X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
        """Perform hyperparameter optimization for a single model."""
        rmse_scorer = make_scorer(self.rmse, greater_is_better=False)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=rmse_scorer,
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_model(self, model: BaseEstimator, X_train: np.ndarray, 
                      X_test: np.ndarray, y_train: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a model and return metrics."""
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        rmse_train = self.rmse(y_train, y_pred_train)
        rmse_test = self.rmse(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        rmse_scorer = make_scorer(self.rmse, greater_is_better=False)
        rmse_cv = -cross_val_score(model, X_train, y_train, scoring=rmse_scorer, cv=5)
        r2_cv = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
        
        return {
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "rmse_cv_mean": rmse_cv.mean(),
            "rmse_cv_std": rmse_cv.std(),
            "r2_cv_mean": r2_cv.mean(),
            "r2_cv_std": r2_cv.std()
        }
    
    def run_experiment(self, X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Run the complete experiment for all models."""
        models = self.get_models()
        param_grids = self.get_param_grids()
        
        for name, model in models.items():
            logger.info(f"Optimizing {name}...")
            
            # Hyperparameter optimization
            best_model, best_params = self.optimize_model(
                model, param_grids[name], X_train, y_train
            )
            logger.info(f"Best params for {name}: {best_params}")
            
            # Evaluation
            metrics = self.evaluate_model(best_model, X_train, X_test, y_train, y_test)
            
            # Store results
            result = ModelResults(
                model_name=name,
                best_params=best_params,
                **metrics
            )
            self.results.append(result)
        
        return self._results_to_dataframe()
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([
            {
                "Model": r.model_name,
                "Best_Params": r.best_params,
                "RMSE_Train": r.rmse_train,
                "RMSE_Test": r.rmse_test,
                "R2_Train": r.r2_train,
                "R2_Test": r.r2_test,
                "RMSE_CV_Mean": r.rmse_cv_mean,
                "RMSE_CV_STD": r.rmse_cv_std,
                "R2_CV_Mean": r.r2_cv_mean,
                "R2_CV_STD": r.r2_cv_std
            }
            for r in self.results
        ])


class ModelVisualizer:
    """Handle visualization of model results."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Set up plotting style."""
        sns.set(style="whitegrid", font_scale=1.1)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
    
    def plot_r2_comparison(self, results_df: pd.DataFrame):
        """Plot R² comparison between train and test sets."""
        plt.figure(figsize=self.figsize)
        results_sorted = results_df.sort_values("R2_Test", ascending=False)
        
        plt.barh(results_sorted["Model"], results_sorted["R2_Test"], 
                color='seagreen', label="Test")
        plt.barh(results_sorted["Model"], results_sorted["R2_Train"], 
                color='lightgreen', alpha=0.6, label="Train")
        
        plt.xlabel("R² Score")
        plt.title("R² Score Comparison (Train vs Test)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_rmse_comparison(self, results_df: pd.DataFrame):
        """Plot RMSE comparison between train and test sets."""
        plt.figure(figsize=self.figsize)
        results_sorted = results_df.sort_values("RMSE_Test")
        
        plt.barh(results_sorted["Model"], results_sorted["RMSE_Test"], 
                color='firebrick', label="Test")
        plt.barh(results_sorted["Model"], results_sorted["RMSE_Train"], 
                color='salmon', alpha=0.6, label="Train")
        
        plt.xlabel("RMSE")
        plt.title("RMSE Comparison (Train vs Test)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_cv_r2(self, results_df: pd.DataFrame):
        """Plot cross-validated R² with error bars."""
        plt.figure(figsize=self.figsize)
        plt.errorbar(results_df["Model"], results_df["R2_CV_Mean"], 
                    yerr=results_df["R2_CV_STD"], fmt='o', 
                    color='mediumblue', capsize=5)
        
        plt.xticks(rotation=45)
        plt.ylabel("Cross-Validated R²")
        plt.title("Cross-Validated R² with ±1 Std Dev")
        plt.axhline(0, linestyle='--', color='gray', linewidth=1)
        plt.tight_layout()
        plt.show()
    
    def plot_cv_rmse(self, results_df: pd.DataFrame):
        """Plot cross-validated RMSE with error bars."""
        plt.figure(figsize=self.figsize)
        plt.errorbar(results_df["Model"], results_df["RMSE_CV_Mean"], 
                    yerr=results_df["RMSE_CV_STD"], fmt='o', 
                    color='darkred', capsize=5)
        
        plt.xticks(rotation=45)
        plt.ylabel("Cross-Validated RMSE")
        plt.title("Cross-Validated RMSE with ±1 Std Dev")
        plt.tight_layout()
        plt.show()
    
    def plot_all(self, results_df: pd.DataFrame):
        """Generate all plots."""
        self.plot_r2_comparison(results_df)
        self.plot_rmse_comparison(results_df)
        self.plot_cv_r2(results_df)
        self.plot_cv_rmse(results_df)


def main():
    """Main execution function."""
    # Initialize components
    evaluator = ModelEvaluator(random_state=42)
    visualizer = ModelVisualizer(figsize=(10, 6))
    
    # Load and prepare data
    X, y = evaluator.load_data('data.txt')
    X_train_scaled, X_test_scaled, y_train, y_test = evaluator.prepare_data(X, y)
    
    # Run experiment
    results_df = evaluator.run_experiment(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Visualize results
    visualizer.plot_all(results_df)
    
    # Print best model
    best_model = results_df.sort_values("RMSE_Test").iloc[0]
    logger.info(f"\nBest model: {best_model['Model']} with params {best_model['Best_Params']}")
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    logger.info("Results saved to 'model_comparison_results.csv'")
    
    return results_df


if __name__ == "__main__":
    results = main()
