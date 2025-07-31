"""
SHAP Analysis for RetailGenius churn prediction project.

This script implements explainable AI using SHAP (SHapley Additive exPlanations)
to explain model predictions and provide insights into feature importance.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Tuple, List
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Class for SHAP analysis of churn prediction models."""
    
    def __init__(self, model_path: str = "models/best_model.pkl", 
                 data_path: str = "data/features.csv"):
        """
        Initialize the SHAP Analyzer.
        
        Args:
            model_path: Path to the trained model
            data_path: Path to the feature data
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """Load the trained model and data."""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {self.model_path}")
            
            # Load data
            self.data = pd.read_csv(self.data_path)
            # Remove target column if present
            if 'churn' in self.data.columns:
                self.data = self.data.drop('churn', axis=1)
            
            self.feature_names = self.data.columns.tolist()
            logger.info(f"Data loaded from {self.data_path}. Shape: {self.data.shape}")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model or data: {e}")
            raise
    
    def create_explainer(self):
        """Create SHAP explainer for the model."""
        logger.info("Creating SHAP explainer...")
        
        # Choose explainer based on model type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer for tree-based model")
        elif hasattr(self.model, 'coef_'):
            # Linear models (Logistic Regression)
            self.explainer = shap.LinearExplainer(self.model, self.data)
            logger.info("Using LinearExplainer for linear model")
        else:
            # Generic explainer
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.data.iloc[:100])
            logger.info("Using KernelExplainer for generic model")
    
    def compute_shap_values(self, sample_size: Optional[int] = None):
        """
        Compute SHAP values for the data.
        
        Args:
            sample_size: Number of samples to use (None for all data)
        """
        if self.explainer is None:
            self.create_explainer()
        
        logger.info("Computing SHAP values...")
        
        # Sample data if specified
        if sample_size and sample_size < len(self.data):
            sample_data = self.data.sample(n=sample_size, random_state=42)
        else:
            sample_data = self.data
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(sample_data)
        
        # Handle different output formats
        if isinstance(self.shap_values, list):
            # For binary classification, use the positive class (churn)
            self.shap_values = self.shap_values[1]
        
        logger.info(f"SHAP values computed for {len(sample_data)} samples")
    
    def plot_summary(self, output_path: str = "docs/shap_summary.png"):
        """
        Create SHAP summary plot.
        
        Args:
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info("Creating SHAP summary plot...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle different SHAP value formats
        if len(self.shap_values.shape) == 3:
            shap_values_2d = self.shap_values[:, :, 1]
        else:
            shap_values_2d = self.shap_values
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.abs(shap_values_2d).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_shap
        }).sort_values('mean_shap', ascending=True)  # Sort ascending for horizontal bar plot
        
        # Create custom summary plot with guaranteed visibility
        n_features = len(self.feature_names)
        fig_height = max(10, n_features * 0.5)  # 0.5 inches per feature
        
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_importance))
        bars = ax.barh(y_pos, feature_importance['mean_shap'], 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add feature names as labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_importance['feature'], fontsize=10)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, feature_importance['mean_shap'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', va='center', fontsize=9)
        
        # Customize the plot
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        ax.set_title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Ensure all features are visible
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(left=0.3)  # Make room for feature names
        
        # Save with high quality
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Summary plot saved to {output_path}")
    
    def plot_waterfall(self, sample_idx: int = 0, 
                      output_path: str = "docs/shap_waterfall.png"):
        """
        Create SHAP waterfall plot for a specific sample.
        
        Args:
            sample_idx: Index of the sample to explain
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                data=self.data.iloc[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Waterfall plot saved to {output_path}")
    
    def plot_force(self, sample_idx: int = 0, 
                  output_path: str = "docs/shap_force.png"):
        """
        Create SHAP force plot for a specific sample.
        
        Args:
            sample_idx: Index of the sample to explain
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info(f"Creating SHAP force plot for sample {sample_idx}...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create force plot
        plt.figure(figsize=(12, 6))
        shap.plots.force(
            self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            self.shap_values[sample_idx],
            self.data.iloc[sample_idx],
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Force plot saved to {output_path}")
    
    def plot_beeswarm(self, output_path: str = "docs/shap_beeswarm.png"):
        """
        Create SHAP beeswarm plot.
        
        Args:
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info("Creating SHAP beeswarm plot...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(
            shap.Explanation(
                values=self.shap_values,
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                data=self.data.iloc[:len(self.shap_values)],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Beeswarm plot saved to {output_path}")
    
    def plot_dependence(self, feature_name: str, 
                       output_path: str = "docs/shap_dependence.png"):
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            feature_name: Name of the feature to analyze
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in data")
        
        logger.info(f"Creating SHAP dependence plot for {feature_name}...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dependence plot
        plt.figure(figsize=(10, 6))
        shap.plots.scatter(
            self.shap_values[:, self.feature_names.index(feature_name)],
            color=self.shap_values,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dependence plot saved to {output_path}")
    
    def plot_mean_shap(self, output_path: str = "docs/shap_mean.png"):
        """
        Create mean SHAP values plot.
        
        Args:
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info("Creating mean SHAP values plot...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate mean absolute SHAP values
        # Handle different SHAP value formats
        if len(self.shap_values.shape) == 3:
            # For multi-output models, use the positive class (churn)
            shap_values_2d = self.shap_values[:, :, 1]
        else:
            shap_values_2d = self.shap_values
            
        mean_shap = np.abs(shap_values_2d).mean(axis=0)
        
        # Create DataFrame for plotting
        mean_shap_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_shap
        }).sort_values('mean_shap', ascending=True)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(mean_shap_df)), mean_shap_df['mean_shap'])
        plt.yticks(range(len(mean_shap_df)), mean_shap_df['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Mean Absolute SHAP Values by Feature')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Mean SHAP plot saved to {output_path}")
    
    def plot_simple_waterfall(self, sample_idx: int = 0, 
                             output_path: str = "docs/shap_simple_waterfall.png"):
        """
        Create a simple waterfall plot for a specific sample.
        
        Args:
            sample_idx: Index of the sample to explain
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info(f"Creating simple SHAP waterfall plot for sample {sample_idx}...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get SHAP values for the sample
        if len(self.shap_values.shape) == 3:
            # For multi-output models, use the positive class (churn)
            sample_shap = self.shap_values[sample_idx, :, 1]
        else:
            sample_shap = self.shap_values[sample_idx]
        
        # Create a simple bar plot
        plt.figure(figsize=(12, 8))
        feature_importance = np.abs(sample_shap)
        sorted_idx = np.argsort(feature_importance)
        
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
        plt.xlabel('|SHAP Value|')
        plt.title(f'SHAP Values for Sample {sample_idx}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Simple waterfall plot saved to {output_path}")
    
    def plot_simple_force(self, sample_idx: int = 0, 
                         output_path: str = "docs/shap_simple_force.png"):
        """
        Create a simple force plot for a specific sample.
        
        Args:
            sample_idx: Index of the sample to explain
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info(f"Creating simple SHAP force plot for sample {sample_idx}...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get SHAP values for the sample
        if len(self.shap_values.shape) == 3:
            # For multi-output models, use the positive class (churn)
            sample_shap = self.shap_values[sample_idx, :, 1]
        else:
            sample_shap = self.shap_values[sample_idx]
        sample_data = self.data.iloc[sample_idx]
        
        # Create a simple horizontal bar plot showing feature contributions
        plt.figure(figsize=(12, 8))
        
        # Sort features by absolute SHAP value
        feature_importance = np.abs(sample_shap)
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        # Create horizontal bar plot
        colors = ['red' if x < 0 else 'blue' for x in sample_shap[sorted_idx]]
        plt.barh(range(len(sorted_idx)), sample_shap[sorted_idx], color=colors)
        plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
        plt.xlabel('SHAP Value')
        plt.title(f'Feature Contributions for Sample {sample_idx}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Simple force plot saved to {output_path}")
    
    def plot_interaction(self, output_path: str = "docs/shap_interaction.png"):
        """
        Create SHAP interaction plot for top features.
        
        Args:
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info("Creating SHAP interaction plot...")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get top 4 features for interaction plot
        if len(self.shap_values.shape) == 3:
            shap_values_2d = self.shap_values[:, :, 1]
        else:
            shap_values_2d = self.shap_values
            
        mean_shap = np.abs(shap_values_2d).mean(axis=0)
        top_features_idx = np.argsort(mean_shap)[-4:]  # Top 4 features
        top_features = [self.feature_names[i] for i in top_features_idx]
        
        # Create interaction plot with proper sizing
        plt.figure(figsize=(16, 12))  # Large figure for interaction matrix
        
        try:
            # Create interaction values
            interaction_values = self.explainer.shap_interaction_values(
                self.data.iloc[:min(1000, len(self.data))]  # Limit samples for performance
            )
            
            # For binary classification, use positive class
            if len(interaction_values.shape) == 4:
                interaction_values = interaction_values[:, :, :, 1]
            
            # Create interaction plot for top features
            feature_indices = [self.feature_names.index(f) for f in top_features]
            interaction_subset = interaction_values[:, feature_indices][:, :, feature_indices]
            
            # Create a custom interaction plot using matplotlib
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('SHAP Feature Interactions', fontsize=16, fontweight='bold')
            
            # Create interaction plot for top 2 features only to avoid indexing issues
            top_2_features = top_features[:2]
            top_2_indices = feature_indices[:2]
            
            # Plot positions
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            
            for idx, (i, j) in enumerate(positions):
                ax = axes[i, j]
                
                if idx == 0:  # Top-left: Feature 1 main effect
                    feat_idx = top_2_indices[0]
                    feat_name = top_2_features[0]
                    ax.scatter(self.data.iloc[:1000, feat_idx], 
                             interaction_values[:1000, feat_idx, feat_idx],
                             alpha=0.6, s=20, color='blue')
                    ax.set_title(f'{feat_name} Main Effect')
                    ax.set_xlabel(feat_name)
                    ax.set_ylabel('SHAP Value')
                    
                elif idx == 3:  # Bottom-right: Feature 2 main effect
                    feat_idx = top_2_indices[1]
                    feat_name = top_2_features[1]
                    ax.scatter(self.data.iloc[:1000, feat_idx], 
                             interaction_values[:1000, feat_idx, feat_idx],
                             alpha=0.6, s=20, color='red')
                    ax.set_title(f'{feat_name} Main Effect')
                    ax.set_xlabel(feat_name)
                    ax.set_ylabel('SHAP Value')
                    
                else:  # Interaction effects
                    feat1_idx = top_2_indices[0]
                    feat2_idx = top_2_indices[1]
                    feat1_name = top_2_features[0]
                    feat2_name = top_2_features[1]
                    
                    if idx == 1:  # Top-right: Feature 1 vs Feature 2
                        ax.scatter(self.data.iloc[:1000, feat1_idx], 
                                 interaction_values[:1000, feat1_idx, feat2_idx],
                                 alpha=0.6, s=20, color='green')
                        ax.set_title(f'{feat1_name} vs {feat2_name}')
                        ax.set_xlabel(feat1_name)
                        ax.set_ylabel(f'SHAP Interaction with {feat2_name}')
                    else:  # Bottom-left: Feature 2 vs Feature 1
                        ax.scatter(self.data.iloc[:1000, feat2_idx], 
                                 interaction_values[:1000, feat2_idx, feat1_idx],
                                 alpha=0.6, s=20, color='orange')
                        ax.set_title(f'{feat2_name} vs {feat1_name}')
                        ax.set_xlabel(feat2_name)
                        ax.set_ylabel(f'SHAP Interaction with {feat1_name}')
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout(pad=3.0)  # Extra padding for interaction plot
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Interaction plot saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not create interaction plot: {e}")
            # Create a fallback plot showing feature correlations
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.data[top_features].corr()
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(top_features)), top_features, rotation=45)
            plt.yticks(range(len(top_features)), top_features)
            plt.title('Feature Correlation Matrix (Fallback)')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Fallback correlation plot saved to {output_path}")
    
    def explain_prediction(self, sample_idx: int = 0) -> dict:
        """
        Explain a specific prediction.
        
        Args:
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Get SHAP values for the sample
        if len(self.shap_values.shape) == 3:
            # For multi-output models, use the positive class (churn)
            sample_shap = self.shap_values[sample_idx, :, 1]
        else:
            sample_shap = self.shap_values[sample_idx]
        sample_data = self.data.iloc[sample_idx]
        
        # Create explanation dictionary
        explanation = {
            'sample_index': sample_idx,
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            'prediction': self.model.predict_proba(sample_data.values.reshape(1, -1))[0][1],
            'feature_contributions': {}
        }
        
        # Add feature contributions
        for i, feature in enumerate(self.feature_names):
            explanation['feature_contributions'][feature] = {
                'value': sample_data[feature],
                'shap_value': sample_shap[i],
                'contribution': sample_shap[i]
            }
        
        # Sort features by absolute contribution
        sorted_features = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )
        
        explanation['top_features'] = sorted_features[:10]
        
        return explanation
    
    def create_all_plots(self, sample_idx: int = 0):
        """Create all SHAP plots."""
        logger.info("Creating all SHAP plots...")
        
        # Create plots
        self.plot_summary()
        self.plot_mean_shap()
        
        # Create simplified plots for demonstration
        self.plot_simple_waterfall(sample_idx)
        self.plot_simple_force(sample_idx)
        
        # Create interaction plot for top features
        self.plot_interaction()
        
        logger.info("All SHAP plots created successfully!")
    
    def save_explanation_report(self, output_path: str = "docs/shap_explanation_report.txt"):
        """
        Save a detailed explanation report.
        
        Args:
            output_path: Path to save the report
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("RetailGenius Churn Prediction - SHAP Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Model Information:\n")
            f.write(f"Model type: {type(self.model).__name__}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")
            f.write(f"Number of samples: {len(self.data)}\n\n")
            
            f.write("Feature Importance (Mean Absolute SHAP Values):\n")
            f.write("-" * 50 + "\n")
            
            # Calculate and sort mean SHAP values
            # Handle different SHAP value formats
            if len(self.shap_values.shape) == 3:
                # For multi-output models, use the positive class (churn)
                shap_values_2d = self.shap_values[:, :, 1]
            else:
                shap_values_2d = self.shap_values
                
            mean_shap = np.abs(shap_values_2d).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'mean_shap': mean_shap
            }).sort_values('mean_shap', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
                f.write(f"{i:2d}. {row['feature']:<30} {row['mean_shap']:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Sample Prediction Explanation:\n")
            f.write("-" * 40 + "\n")
            
            # Explain a sample prediction
            explanation = self.explain_prediction(0)
            f.write(f"Sample Index: {explanation['sample_index']}\n")
            base_value = explanation['base_value']
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            f.write(f"Base Value: {base_value:.4f}\n")
            f.write(f"Predicted Probability: {explanation['prediction']:.4f}\n\n")
            
            f.write("Top 10 Feature Contributions:\n")
            for feature, details in explanation['top_features']:
                f.write(f"  {feature:<25} {details['shap_value']:>8.4f} (value: {details['value']})\n")
        
        logger.info(f"Explanation report saved to {output_path}")


def main():
    """Main function to run SHAP analysis."""
    # Check if model and data exist
    model_path = "models/best_model.pkl"
    data_path = "data/features.csv"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Please train a model first.")
        return
    
    if not Path(data_path).exists():
        logger.error(f"Data not found at {data_path}. Please run feature engineering first.")
        return
    
    # Initialize SHAP analyzer
    analyzer = SHAPAnalyzer(model_path, data_path)
    
    # Create all plots
    analyzer.create_all_plots()
    
    # Save explanation report
    analyzer.save_explanation_report()
    
    # Print sample explanation
    explanation = analyzer.explain_prediction(0)
    logger.info("\nSample Prediction Explanation:")
    logger.info(f"Predicted Probability: {explanation['prediction']:.4f}")
    logger.info("Top 5 Feature Contributions:")
    for feature, details in explanation['top_features'][:5]:
        logger.info(f"  {feature}: {details['shap_value']:.4f}")
    
    logger.info("SHAP analysis completed successfully!")


if __name__ == "__main__":
    main() 