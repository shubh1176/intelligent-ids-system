import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from typing import Tuple, List, Optional
import logging
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class NetworkDataCleaner:
    def __init__(self, log_level: str = 'INFO', n_bins: int = 10, 
                 output_dir: str = 'cleaning_artifacts'):
        # Set up output directory for visualizations and logs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger(log_level)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        self.feature_names = None
        
    def _setup_logger(self, log_level: str) -> logging.Logger:
        # Set up both file and console logging
        logger = logging.getLogger('NetworkDataCleaner')
        logger.setLevel(log_level)
        
        if not logger.handlers:
            # Save detailed logs to file
            fh = logging.FileHandler(self.output_dir / f'cleaning_{datetime.now():%Y%m%d_%H%M%S}.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
            
            # Show simplified logs in console
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(ch)
            
        return logger
    
    def visualize_distributions(self, df: pd.DataFrame, filename: str):
        # Create histograms for each numeric feature
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        
        if n_cols == 0:
            return
            
        n_rows = (n_cols - 1) // 3 + 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            sns.histplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            
        for i in range(n_cols, len(axes)):
            axes[i].remove()
            
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}_distributions.png')
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, filename: str):
        # Generate and save a heatmap of feature correlations
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{filename}_correlation.png')
            plt.close()
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize column names for consistency
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill missing values using appropriate strategies for each data type
        self.logger.info(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Use median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
            
        # Use mode for categorical columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        self.logger.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")
        return df
    
    def handle_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                       threshold: float = 3.0) -> pd.DataFrame:
        # Remove outliers using the z-score method
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        outliers_summary = {}
        for column in columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers_mask = z_scores > threshold
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                outliers_summary[column] = outliers_count
                df[column] = df[column].mask(outliers_mask, df[column].median())
        
        if outliers_summary:
            self.logger.info("Outliers removed per column:")
            for col, count in outliers_summary.items():
                self.logger.info(f"{col}: {count} outliers")
                
        return df
    
    def process_dataset(self, file_path: str, discretize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Main pipeline for processing network traffic data
        filename = Path(file_path).stem
        self.logger.info(f"Processing dataset: {file_path}")
        
        df = pd.read_csv(file_path, low_memory=False)
        initial_shape = df.shape
        self.logger.info(f"Initial shape: {initial_shape}")
        
        # Save visualizations before cleaning
        self.visualize_distributions(df, f"{filename}_initial")
        
        # Clean and process the data
        df = self.clean_column_names(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        
        # Save visualizations after cleaning
        self.visualize_distributions(df, f"{filename}_processed")
        self.plot_correlation_matrix(df, filename)
        
        # Split features and labels
        label_col = next((col for col in df.columns if 'label' in col), None)
        if label_col is None:
            raise ValueError("Could not find label column in dataset")
            
        features = df.drop(columns=[label_col])
        labels = df[label_col]
        
        # Normalize and encode
        features_normalized = self.scaler.fit_transform(features)
        if discretize:
            features_normalized = self.discretizer.fit_transform(features_normalized)
        
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Log final processing summary
        self.logger.info(f"Final processed features shape: {features_normalized.shape}")
        self.logger.info(f"Number of unique classes: {len(np.unique(labels_encoded))}")
        self.logger.info(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        return features_normalized, labels_encoded
