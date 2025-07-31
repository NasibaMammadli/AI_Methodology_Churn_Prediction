"""
Model training script for RetailGenius churn prediction project.

This script handles:
- Model training with multiple algorithms
- Hyperparameter tuning
- Model evaluation and validation
- MLflow experiment tracking
- Model saving and versioning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training churn prediction models."""
    
    def __init__(self, experiment_name: str = "retailgenius_churn_prediction"):
        """
        Initialize the ModelTrainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
    def load_data(self, data_path: str = "data/features.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the processed features data.
        
        Args:
            data_path: Path to the features data
            
        Returns:
            Tuple of features and target
        """
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Features data not found at {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'churn']
        X = df[feature_columns]
        y = df['churn']
        
        self.feature_names = feature_columns
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Testing set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model names and instances
        """
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        
        return models
    
    def get_hyperparameters(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for each model.
        
        Returns:
            Dictionary of hyperparameter grids
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return param_grids
    
    def train_model(self, model_name: str, model: Any, X_train: pd.DataFrame, 
                   y_train: pd.Series, param_grid: Optional[Dict] = None) -> Any:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid for tuning
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(nested=True):
            # Log model parameters
            mlflow.log_param("model_name", model_name)
            
            if param_grid:
                # Perform hyperparameter tuning
                logger.info(f"Performing hyperparameter tuning for {model_name}...")
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                # Log best parameters and score
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_score", best_score)
                
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best CV score: {best_score:.4f}")
                
            else:
                # Train without hyperparameter tuning
                best_model = model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())
            
            # Log model
            mlflow.sklearn.log_model(best_model, f"{model_name}_model")
            
            logger.info(f"{model_name} training completed")
            return best_model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                      y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics to MLflow
        with mlflow.start_run(nested=True):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_metric(f"{model_name}_tn", cm[0, 0])
            mlflow.log_metric(f"{model_name}_fp", cm[0, 1])
            mlflow.log_metric(f"{model_name}_fn", cm[1, 0])
            mlflow.log_metric(f"{model_name}_tp", cm[1, 1])
        
        logger.info(f"{model_name} evaluation completed")
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation results for each model
        """
        models = self.get_models()
        param_grids = self.get_hyperparameters()
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training and evaluating {model_name}")
            logger.info(f"{'='*50}")
            
            # Train model
            param_grid = param_grids.get(model_name)
            trained_model = self.train_model(model_name, model, X_train, y_train, param_grid)
            
            # Evaluate model
            metrics = self.evaluate_model(trained_model, X_test, y_test, model_name)
            
            # Store results
            self.models[model_name] = trained_model
            results[model_name] = metrics
            
            logger.info(f"{model_name} results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict], 
                         metric: str = 'f1') -> Tuple[str, Any]:
        """
        Select the best model based on a metric.
        
        Args:
            results: Dictionary of evaluation results
            metric: Metric to use for selection
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        best_model_name = max(results.keys(), key=lambda k: results[k][metric])
        best_model = self.models[best_model_name]
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        logger.info(f"\nBest model based on {metric}: {best_model_name}")
        logger.info(f"Best {metric} score: {results[best_model_name][metric]:.4f}")
        
        return best_model_name, best_model
    
    def save_model(self, model: Any, model_name: str, 
                  output_path: str = "models/") -> None:
        """
        Save a trained model.
        
        Args:
            model: Trained model to save
            model_name: Name of the model
            output_path: Directory to save the model
        """
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_dir / f"{model_name}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_file}")
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_artifact(str(model_file))
    
    def save_best_model(self, output_path: str = "models/best_model.pkl") -> None:
        """
        Save the best model.
        
        Args:
            output_path: Path to save the best model
        """
        if self.best_model is None:
            raise ValueError("No best model to save. Run select_best_model() first.")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(output_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        logger.info(f"Best model saved to {output_path}")
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_artifact(output_path)
            mlflow.log_param("best_model_name", self.best_model_name)
    
    def create_evaluation_report(self, results: Dict[str, Dict], 
                               output_path: str = "docs/model_evaluation_report.txt") -> None:
        """
        Create a detailed evaluation report.
        
        Args:
            results: Dictionary of evaluation results
            output_path: Path to save the report
        """
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("RetailGenius Churn Prediction - Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Model Performance Summary:\n")
            f.write("-" * 30 + "\n")
            
            # Create comparison table
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            f.write(f"{'Model':<20}")
            for metric in metrics:
                f.write(f"{metric.upper():<12}")
            f.write("\n")
            
            for model_name, model_results in results.items():
                f.write(f"{model_name:<20}")
                for metric in metrics:
                    f.write(f"{model_results[metric]:<12.4f}")
                f.write("\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"Best F1 Score: {results[self.best_model_name]['f1']:.4f}\n")
            f.write(f"Best ROC AUC: {results[self.best_model_name]['roc_auc']:.4f}\n")
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def plot_model_comparison(self, results: Dict[str, Dict], 
                            output_path: str = "docs/model_comparison.png") -> None:
        """
        Create a visualization comparing model performance.
        
        Args:
            results: Dictionary of evaluation results
            output_path: Path to save the plot
        """
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(results.keys())
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric.upper()} Score')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
        
        # Remove extra subplot
        axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {output_path}")


def main():
    """Main function to run model training."""
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Load data
    X, y = trainer.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train and evaluate all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_model_name, best_model = trainer.select_best_model(results)
    
    # Save best model
    trainer.save_best_model()
    
    # Save all models
    for model_name, model in trainer.models.items():
        trainer.save_model(model, model_name)
    
    # Create evaluation report
    trainer.create_evaluation_report(results)
    
    # Create comparison plot
    trainer.plot_model_comparison(results)
    
    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    main() 