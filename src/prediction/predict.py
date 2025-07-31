"""
Prediction script for RetailGenius churn prediction project.

This script handles:
- Loading trained models
- Making predictions on new data
- Probability scoring
- Batch prediction processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import pickle
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Class for making churn predictions using trained models."""
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Initialize the ChurnPredictor.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            data: Raw customer data
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Preparing features for prediction...")
        
        # Import feature engineering functions
        import sys
        sys.path.append('src')
        from feature_engineering.create_features import FeatureEngineer
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Create features
        df_with_features = feature_engineer.create_features(data)
        
        # Encode categorical features
        categorical_columns = df_with_features.select_dtypes(include=['object']).columns.tolist()
        df_encoded = feature_engineer.encode_categorical_features(df_with_features, categorical_columns)
        
        # For prediction, we need to use the same features as training
        # Based on the training features file
        main_features = [
            'tenure', 'monthly_charges', 'senior_citizen', 'tenure_monthly_interaction',
            'avg_monthly_charge', 'charge_per_service', 'contract_risk', 'payment_risk',
            'tenure_risk', 'overall_risk_score', 'high_value_customer', 'contract_type_encoded',
            'payment_method_encoded', 'streaming_movies_encoded', 'contract_payment_interaction_encoded'
        ]
        
        # Select only features that exist in the dataframe
        available_features = [f for f in main_features if f in df_encoded.columns]
        df_selected = df_encoded[available_features]
        
        # For now, let's skip scaling to avoid the scaler issue
        # We'll use the raw features
        df_final = df_selected.copy()
        
        # Remove target column if present
        if 'churn' in df_final.columns:
            df_final = df_final.drop('churn', axis=1)
        
        self.feature_names = df_final.columns.tolist()
        logger.info(f"Features prepared. Shape: {df_final.shape}")
        
        return df_final
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make churn predictions.
        
        Args:
            data: Customer data
            
        Returns:
            Array of predictions (0: no churn, 1: churn)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Make predictions
        predictions = self.model.predict(features)
        
        logger.info(f"Made predictions for {len(predictions)} customers")
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get churn probabilities.
        
        Args:
            data: Customer data
            
        Returns:
            Array of churn probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features)
        
        logger.info(f"Calculated probabilities for {len(probabilities)} customers")
        return probabilities
    
    def predict_with_confidence(self, data: pd.DataFrame, 
                              threshold: float = 0.5) -> pd.DataFrame:
        """
        Make predictions with confidence scores.
        
        Args:
            data: Customer data
            threshold: Probability threshold for churn prediction
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        # Get probabilities
        probabilities = self.predict_proba(data)
        churn_probs = probabilities[:, 1]
        
        # Make predictions based on threshold
        predictions = (churn_probs >= threshold).astype(int)
        
        # Calculate confidence (distance from threshold)
        confidence = np.abs(churn_probs - threshold)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': data.get('customer_id', range(len(predictions))),
            'churn_prediction': predictions,
            'churn_probability': churn_probs,
            'confidence': confidence,
            'risk_level': self._get_risk_level(churn_probs)
        })
        
        logger.info(f"Predictions with confidence completed for {len(results)} customers")
        return results
    
    def _get_risk_level(self, probabilities: np.ndarray) -> List[str]:
        """
        Assign risk levels based on churn probability.
        
        Args:
            probabilities: Churn probabilities
            
        Returns:
            List of risk levels
        """
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append('Low')
            elif prob < 0.7:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        
        return risk_levels
    
    def batch_predict(self, data_path: str, 
                     output_path: str = "predictions/batch_predictions.csv") -> None:
        """
        Make batch predictions from a data file.
        
        Args:
            data_path: Path to the input data file
            output_path: Path to save predictions
        """
        logger.info(f"Starting batch prediction from {data_path}")
        
        # Load data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Make predictions
        results = self.predict_with_confidence(data)
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results.to_csv(output_path, index=False)
        logger.info(f"Batch predictions saved to {output_path}")
        
        # Print summary
        self._print_prediction_summary(results)
    
    def _print_prediction_summary(self, results: pd.DataFrame) -> None:
        """
        Print a summary of predictions.
        
        Args:
            results: Prediction results DataFrame
        """
        total_customers = len(results)
        churn_predictions = results['churn_prediction'].sum()
        churn_rate = churn_predictions / total_customers
        
        risk_counts = results['risk_level'].value_counts()
        
        logger.info("\n" + "="*50)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total customers: {total_customers}")
        logger.info(f"Predicted churn: {churn_predictions} ({churn_rate:.2%})")
        logger.info(f"Average churn probability: {results['churn_probability'].mean():.3f}")
        logger.info(f"Average confidence: {results['confidence'].mean():.3f}")
        logger.info("\nRisk Level Distribution:")
        for risk, count in risk_counts.items():
            logger.info(f"  {risk}: {count} ({count/total_customers:.2%})")
        logger.info("="*50)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df


def create_sample_customer_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample customer data for testing predictions.
    
    Args:
        n_samples: Number of sample customers
        
    Returns:
        DataFrame with sample customer data
    """
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure': np.random.randint(1, 73, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'unlimited_data': np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    }
    
    return pd.DataFrame(data)


def main():
    """Main function to run predictions."""
    # Check if model exists
    model_path = "models/best_model.pkl"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Please train a model first.")
        return
    
    # Initialize predictor
    predictor = ChurnPredictor(model_path)
    
    # Create sample data for testing
    sample_data = create_sample_customer_data(50)
    
    # Make predictions
    results = predictor.predict_with_confidence(sample_data)
    
    # Save predictions
    output_path = "predictions/sample_predictions.csv"
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    # Print summary
    predictor._print_prediction_summary(results)
    
    # Get feature importance
    importance_df = predictor.get_feature_importance()
    logger.info("\nTop 10 most important features:")
    logger.info(importance_df.head(10))
    
    logger.info("Prediction completed successfully!")


if __name__ == "__main__":
    main() 