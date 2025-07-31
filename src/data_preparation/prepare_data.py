"""
Data preparation script for RetailGenius churn prediction project.

This script handles:
- Loading the churn dataset
- Basic data cleaning and preprocessing
- Data validation
- Saving processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
import mlflow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Class for preparing and preprocessing churn prediction data."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the DataPreparator.
        
        Args:
            data_path: Path to the raw data file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the churn dataset.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        if data_path is None and self.data_path is None:
            # Create sample data for demonstration
            logger.info("Creating sample churn dataset...")
            return self._create_sample_data()
        
        file_path = data_path or self.data_path
        
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            logger.info(f"Successfully loaded data from {file_path}")
            logger.info(f"Data shape: {data.shape}")
            
            self.raw_data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample churn dataset for demonstration.
        
        Returns:
            Sample DataFrame with churn prediction features
        """
        np.random.seed(42)
        n_samples = 10000
        
        # Generate sample data
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
        
        df = pd.DataFrame(data)
        
        # Create target variable (churn) based on some business logic
        churn_prob = (
            (df['tenure'] < 12) * 0.3 +
            (df['monthly_charges'] > 80) * 0.2 +
            (df['contract_type'] == 'Month-to-month') * 0.3 +
            (df['payment_method'] == 'Electronic check') * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        df['churn'] = (churn_prob > 0.5).astype(int)
        
        logger.info(f"Created sample dataset with {n_samples} records")
        return df
    
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy() if data is not None else self.raw_data.copy()
        
        logger.info("Starting data cleaning process...")
        
        # Handle missing values
        logger.info("Handling missing values...")
        df = self._handle_missing_values(df)
        
        # Convert data types
        logger.info("Converting data types...")
        df = self._convert_data_types(df)
        
        # Remove duplicates
        logger.info("Removing duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Basic validation
        logger.info("Performing data validation...")
        self._validate_data(df)
        
        self.processed_data = df
        logger.info("Data cleaning completed successfully")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types to appropriate formats."""
        # Convert boolean-like columns
        boolean_cols = ['senior_citizen', 'churn']
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Ensure customer_id is string
        if 'customer_id' in df.columns:
            df['customer_id'] = df['customer_id'].astype(str)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate the cleaned data."""
        # Check for remaining missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Remaining missing values: {missing_counts[missing_counts > 0]}")
        
        # Check data ranges
        if 'tenure' in df.columns:
            if (df['tenure'] < 0).any():
                logger.warning("Found negative tenure values")
        
        if 'monthly_charges' in df.columns:
            if (df['monthly_charges'] < 0).any():
                logger.warning("Found negative monthly charges")
        
        logger.info("Data validation completed")
    
    def save_processed_data(self, data: Optional[pd.DataFrame] = None, 
                          output_path: str = "data/processed_data.csv") -> None:
        """
        Save the processed data.
        
        Args:
            data: DataFrame to save
            output_path: Path to save the data
        """
        df = data if data is not None else self.processed_data
        
        if df is None:
            raise ValueError("No data to save. Please run clean_data() first.")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        # Log data info to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_artifact(output_path)
            mlflow.log_metric("processed_rows", len(df))
            mlflow.log_metric("processed_columns", len(df.columns))
    
    def get_data_summary(self, data: Optional[pd.DataFrame] = None) -> dict:
        """
        Get a summary of the data.
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with data summary
        """
        df = data if data is not None else self.processed_data
        
        if df is None:
            raise ValueError("No data to summarize. Please run clean_data() first.")
        
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            'categorical_summary': {}
        }
        
        # Add categorical column summaries
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = df[col].value_counts().to_dict()
        
        return summary


def main():
    """Main function to run data preparation."""
    # Initialize data preparator
    preparator = DataPreparator()
    
    # Load data
    data = preparator.load_data()
    
    # Clean data
    cleaned_data = preparator.clean_data(data)
    
    # Save processed data
    preparator.save_processed_data(cleaned_data)
    
    # Print summary
    summary = preparator.get_data_summary(cleaned_data)
    logger.info(f"Data preparation completed. Final shape: {summary['shape']}")


if __name__ == "__main__":
    main() 