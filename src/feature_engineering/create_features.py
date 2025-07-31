"""
Feature engineering script for RetailGenius churn prediction project.

This script handles:
- Feature creation and transformation
- Feature selection
- Encoding categorical variables
- Scaling numerical features
- Saving feature engineering pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import pickle
import mlflow
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for feature engineering in churn prediction."""
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.feature_pipeline = None
        self.selected_features = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from the raw data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            DataFrame with new features
        """
        logger.info("Starting feature creation...")
        df = data.copy()
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Create ratio features
        df = self._create_ratio_features(df)
        
        # Create categorical aggregations
        df = self._create_categorical_features(df)
        
        # Create time-based features
        df = self._create_time_features(df)
        
        # Create behavioral features
        df = self._create_behavioral_features(df)
        
        logger.info(f"Feature creation completed. New shape: {df.shape}")
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        # Contract type and payment method interaction
        if 'contract_type' in df.columns and 'payment_method' in df.columns:
            df['contract_payment_interaction'] = (
                df['contract_type'].astype(str) + '_' + 
                df['payment_method'].astype(str)
            )
        
        # Internet service and streaming interaction
        if 'internet_service' in df.columns and 'streaming_tv' in df.columns:
            df['internet_streaming_interaction'] = (
                df['internet_service'].astype(str) + '_' + 
                df['streaming_tv'].astype(str)
            )
        
        # Tenure and monthly charges interaction
        if 'tenure' in df.columns and 'monthly_charges' in df.columns:
            df['tenure_monthly_interaction'] = df['tenure'] * df['monthly_charges']
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features."""
        # Average monthly charge per month of tenure
        if 'total_charges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_charge'] = df['total_charges'] / (df['tenure'] + 1)
        
        # Charge per service ratio
        if 'monthly_charges' in df.columns:
            service_count = 0
            service_columns = [
                'phone_service', 'internet_service', 'online_security',
                'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies'
            ]
            
            for col in service_columns:
                if col in df.columns:
                    service_count += (df[col] != 'No').astype(int)
                    service_count += (df[col] != 'No internet service').astype(int)
            
            df['services_count'] = service_count
            df['charge_per_service'] = df['monthly_charges'] / (service_count + 1)
        
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical features and aggregations."""
        # Contract type risk (month-to-month is higher risk)
        if 'contract_type' in df.columns:
            contract_risk = {
                'Month-to-month': 3,
                'One year': 2,
                'Two year': 1
            }
            df['contract_risk'] = df['contract_type'].map(contract_risk)
        
        # Payment method risk (electronic check is higher risk)
        if 'payment_method' in df.columns:
            payment_risk = {
                'Electronic check': 3,
                'Mailed check': 2,
                'Bank transfer': 1,
                'Credit card': 1
            }
            df['payment_risk'] = df['payment_method'].map(payment_risk)
        
        # Internet service type risk
        if 'internet_service' in df.columns:
            internet_risk = {
                'Fiber optic': 3,
                'DSL': 2,
                'No': 1
            }
            df['internet_risk'] = df['internet_service'].map(internet_risk)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        # Tenure categories
        if 'tenure' in df.columns:
            df['tenure_category'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, float('inf')],
                labels=['New', 'Established', 'Loyal', 'Very Loyal']
            )
            
            # Tenure risk (newer customers are higher risk)
            df['tenure_risk'] = pd.cut(
                df['tenure'],
                bins=[0, 6, 12, 24, float('inf')],
                labels=[4, 3, 2, 1]
            ).astype(int)
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features."""
        # Risk score based on multiple factors
        risk_factors = []
        
        if 'contract_risk' in df.columns:
            risk_factors.append(df['contract_risk'])
        
        if 'payment_risk' in df.columns:
            risk_factors.append(df['payment_risk'])
        
        if 'internet_risk' in df.columns:
            risk_factors.append(df['internet_risk'])
        
        if 'tenure_risk' in df.columns:
            risk_factors.append(df['tenure_risk'])
        
        if risk_factors:
            df['overall_risk_score'] = np.mean(risk_factors, axis=0)
        
        # High-value customer flag
        if 'monthly_charges' in df.columns:
            high_value_threshold = df['monthly_charges'].quantile(0.75)
            df['high_value_customer'] = (df['monthly_charges'] > high_value_threshold).astype(int)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with categorical features
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                # Create label encoder
                le = LabelEncoder()
                
                # Handle missing values
                col_data = df[col].fillna('Unknown')
                
                # Fit and transform
                df_encoded[col + '_encoded'] = le.fit_transform(col_data)
                
                # Store encoder for later use
                self.label_encoders[col] = le
                
                logger.info(f"Encoded column: {col}")
        
        return df_encoded
    
    def select_features(self, df: pd.DataFrame, target_column: str = 'churn',
                       n_features: int = 20, is_prediction: bool = False) -> pd.DataFrame:
        """
        Select the most important features.
        
        Args:
            df: DataFrame with features
            target_column: Name of the target column
            n_features: Number of features to select
            is_prediction: Whether this is for prediction (no target column)
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting top {n_features} features...")
        
        if is_prediction or target_column not in df.columns:
            # For prediction, use the same features as training
            # This should be the same set of features used during training
            expected_features = [
                'tenure', 'monthly_charges', 'total_charges', 'contract_type_encoded',
                'payment_method_encoded', 'paperless_billing_encoded', 'gender_encoded',
                'senior_citizen', 'partner_encoded', 'dependents_encoded', 'phone_service_encoded',
                'multiple_lines_encoded', 'internet_service_encoded', 'online_security_encoded',
                'online_backup_encoded', 'device_protection_encoded', 'tech_support_encoded',
                'streaming_tv_encoded', 'streaming_movies_encoded', 'unlimited_data_encoded',
                'contract_payment_interaction_encoded', 'internet_streaming_interaction_encoded',
                'tenure_months', 'monthly_charges_per_tenure', 'total_charges_per_tenure'
            ]
            
            # Select only the features that exist in the dataframe
            available_features = [f for f in expected_features if f in df.columns]
            if len(available_features) >= n_features:
                selected_features = available_features[:n_features]
            else:
                selected_features = available_features
            
            df_selected = df[selected_features]
            self.selected_features = selected_features
            self.feature_names = selected_features
            
            logger.info(f"Selected features for prediction: {selected_features}")
            return df_selected
        else:
            # For training, use statistical feature selection
            # Separate features and target
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]
            
            # Remove non-numeric columns for feature selection
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_columns]
            
            # Use SelectKBest for feature selection
            selector = SelectKBest(score_func=f_classif, k=min(n_features, len(numeric_columns)))
            X_selected = selector.fit_transform(X_numeric, y)
            
            # Get selected feature names
            selected_feature_names = numeric_columns[selector.get_support()].tolist()
            
            # Create DataFrame with selected features
            df_selected = df[selected_feature_names + [target_column]]
            
            self.selected_features = selected_feature_names
            self.feature_names = selected_feature_names
            
            logger.info(f"Selected features: {selected_feature_names}")
            
            return df_selected
    
    def scale_features(self, df: pd.DataFrame, target_column: str = 'churn', is_prediction: bool = False) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: DataFrame with features
            target_column: Name of the target column
            is_prediction: Whether this is for prediction (no target column)
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling numerical features...")
        
        if is_prediction or target_column not in df.columns:
            # For prediction, scale all features
            X = df
            X_scaled = self.scaler.transform(X)
            
            # Create DataFrame with scaled features
            df_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
            
            logger.info("Feature scaling completed for prediction")
            return df_scaled
        else:
            # For training, separate features and target
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create DataFrame with scaled features
            df_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)
            df_scaled[target_column] = y
            
            logger.info("Feature scaling completed")
            
            return df_scaled
    
    def create_feature_pipeline(self, categorical_columns: List[str],
                              numerical_columns: List[str]) -> Pipeline:
        """
        Create a feature engineering pipeline.
        
        Args:
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            Feature engineering pipeline
        """
        logger.info("Creating feature engineering pipeline...")
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', LabelEncoder(), categorical_columns)
            ]
        )
        
        # Create pipeline
        self.feature_pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        logger.info("Feature engineering pipeline created")
        return self.feature_pipeline
    
    def save_feature_pipeline(self, output_path: str = "models/feature_pipeline.pkl") -> None:
        """
        Save the feature engineering pipeline.
        
        Args:
            output_path: Path to save the pipeline
        """
        if self.feature_pipeline is None:
            raise ValueError("No feature pipeline to save. Run create_feature_pipeline() first.")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        with open(output_path, 'wb') as f:
            pickle.dump(self.feature_pipeline, f)
        
        logger.info(f"Feature pipeline saved to {output_path}")
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_artifact(output_path)
    
    def save_processed_features(self, df: pd.DataFrame, 
                              output_path: str = "data/features.csv") -> None:
        """
        Save the processed features.
        
        Args:
            df: DataFrame with processed features
            output_path: Path to save the features
        """
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        df.to_csv(output_path, index=False)
        logger.info(f"Processed features saved to {output_path}")
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_artifact(output_path)
            mlflow.log_metric("feature_count", len(df.columns) - 1)  # Exclude target
            mlflow.log_metric("sample_count", len(df))
    
    def get_feature_importance(self, df: pd.DataFrame, target_column: str = 'churn') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            df: DataFrame with features
            target_column: Name of the target column
            
        Returns:
            DataFrame with feature importance scores
        """
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        # Calculate feature importance using f_classif
        f_scores, p_values = f_classif(X, y)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'f_score': f_scores,
            'p_value': p_values
        })
        
        # Sort by f_score
        importance_df = importance_df.sort_values('f_score', ascending=False)
        
        return importance_df


def main():
    """Main function to run feature engineering."""
    # Load processed data
    data_path = "data/processed_data.csv"
    
    if not Path(data_path).exists():
        logger.error(f"Processed data not found at {data_path}. Please run data preparation first.")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create features
    df_with_features = feature_engineer.create_features(df)
    
    # Encode categorical features
    categorical_columns = df_with_features.select_dtypes(include=['object']).columns.tolist()
    df_encoded = feature_engineer.encode_categorical_features(df_with_features, categorical_columns)
    
    # Select features
    df_selected = feature_engineer.select_features(df_encoded, n_features=15)
    
    # Scale features
    df_scaled = feature_engineer.scale_features(df_selected)
    
    # Save processed features
    feature_engineer.save_processed_features(df_scaled)
    
    # Get and display feature importance
    importance_df = feature_engineer.get_feature_importance(df_scaled)
    logger.info("Top 10 most important features:")
    logger.info(importance_df.head(10))
    
    logger.info("Feature engineering completed successfully")


if __name__ == "__main__":
    main() 