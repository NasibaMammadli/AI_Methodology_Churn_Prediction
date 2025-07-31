#!/usr/bin/env python3
"""
Main pipeline script for RetailGenius churn prediction project.

This script runs the complete ML pipeline:
1. Data preparation
2. Feature engineering
3. Model training
4. Prediction
5. SHAP analysis (Part 3)

Usage:
    python run_pipeline.py [--skip-data] [--skip-features] [--skip-training] [--skip-prediction] [--skip-shap]
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """
    Run a command and handle errors.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'mlflow', 'shap', 
        'fastapi', 'uvicorn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data', 'models', 'docs', 'predictions', 'notebooks', 'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("✅ Directories created")


def run_data_preparation() -> bool:
    """Run data preparation step."""
    return run_command(
        "python src/data_preparation/prepare_data.py",
        "Data preparation"
    )


def run_feature_engineering() -> bool:
    """Run feature engineering step."""
    return run_command(
        "python src/feature_engineering/create_features.py",
        "Feature engineering"
    )


def run_model_training() -> bool:
    """Run model training step."""
    return run_command(
        "python src/model_training/train_model.py",
        "Model training"
    )


def run_prediction() -> bool:
    """Run prediction step."""
    return run_command(
        "python src/prediction/predict.py",
        "Prediction"
    )


def run_shap_analysis() -> bool:
    """Run SHAP analysis step."""
    return run_command(
        "python src/explainable_ai/shap_analysis.py",
        "SHAP analysis"
    )


def start_mlflow_ui():
    """Start MLflow UI in the background."""
    logger.info("Starting MLflow UI...")
    try:
        subprocess.Popen(
            ["mlflow", "ui", "--port", "5000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("✅ MLflow UI started at http://localhost:5000")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start MLflow UI: {e}")
        return False


def start_api_server():
    """Start the FastAPI server in the background."""
    logger.info("Starting API server...")
    try:
        subprocess.Popen(
            ["python", "src/prediction/api.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("✅ API server started at http://localhost:8000")
        logger.info("📚 API documentation available at http://localhost:8000/docs")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")
        return False


def print_summary():
    """Print a summary of the project."""
    logger.info("\n" + "="*60)
    logger.info("🎉 RETAILGENIUS CHURN PREDICTION PROJECT COMPLETED!")
    logger.info("="*60)
    
    logger.info("\n📁 Project Structure:")
    logger.info("├── data/                    # Processed data files")
    logger.info("├── models/                  # Trained models")
    logger.info("├── docs/                    # Documentation and reports")
    logger.info("├── predictions/             # Prediction results")
    logger.info("├── src/                     # Source code")
    logger.info("│   ├── data_preparation/    # Data preprocessing")
    logger.info("│   ├── feature_engineering/ # Feature creation")
    logger.info("│   ├── model_training/      # Model training")
    logger.info("│   ├── prediction/          # Prediction scripts")
    logger.info("│   └── explainable_ai/      # SHAP analysis")
    logger.info("└── notebooks/               # Jupyter notebooks")
    
    logger.info("\n🌐 Available Services:")
    logger.info("• MLflow UI: http://localhost:5000")
    logger.info("• API Server: http://localhost:8000")
    logger.info("• API Docs: http://localhost:8000/docs")
    
    logger.info("\n📊 Generated Files:")
    
    # Check for generated files
    files_to_check = [
        ("data/processed_data.csv", "Processed data"),
        ("data/features.csv", "Engineered features"),
        ("models/best_model.pkl", "Best trained model"),
        ("docs/model_evaluation_report.txt", "Model evaluation report"),
        ("docs/shap_explanation_report.txt", "SHAP explanation report"),
        ("predictions/sample_predictions.csv", "Sample predictions")
    ]
    
    for file_path, description in files_to_check:
        if Path(file_path).exists():
            logger.info(f"✅ {description}: {file_path}")
        else:
            logger.info(f"❌ {description}: {file_path} (not found)")
    
    logger.info("\n🚀 Next Steps:")
    logger.info("1. View experiment results in MLflow UI")
    logger.info("2. Test the API with sample data")
    logger.info("3. Explore SHAP visualizations in docs/")
    logger.info("4. Check model performance reports")
    logger.info("5. Deploy to cloud infrastructure")
    
    logger.info("\n" + "="*60)


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Run RetailGenius churn prediction pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip prediction")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--start-services", action="store_true", help="Start MLflow UI and API server")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting RetailGenius Churn Prediction Pipeline")
    logger.info("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Track overall success
    success = True
    
    # Run pipeline steps
    steps = [
        ("Data Preparation", run_data_preparation, args.skip_data),
        ("Feature Engineering", run_feature_engineering, args.skip_features),
        ("Model Training", run_model_training, args.skip_training),
        ("Prediction", run_prediction, args.skip_prediction),
        ("SHAP Analysis", run_shap_analysis, args.skip_shap)
    ]
    
    for step_name, step_func, skip in steps:
        if skip:
            logger.info(f"⏭️  Skipping {step_name}")
            continue
        
        if not step_func():
            logger.error(f"❌ Pipeline failed at {step_name}")
            success = False
            break
        
        logger.info(f"✅ {step_name} completed")
        time.sleep(1)  # Small delay between steps
    
    # Start services if requested
    if args.start_services and success:
        start_mlflow_ui()
        time.sleep(2)
        start_api_server()
    
    # Print summary
    print_summary()
    
    if success:
        logger.info("🎉 Pipeline completed successfully!")
        return 0
    else:
        logger.error("❌ Pipeline failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 