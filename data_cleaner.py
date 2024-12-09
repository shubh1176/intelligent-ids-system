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
from typing import Tuple
from pathlib import Path
from tqdm import tqdm

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
            
        self.logger.info(f"Creating distribution plots for {n_cols} numeric features...")
        n_rows = (n_cols - 1) // 3 + 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, col in tqdm(enumerate(numeric_cols), total=n_cols, desc="Plotting distributions"):
            sns.histplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            
        for i in range(n_cols, len(axes)):
            axes[i].remove()
            
        plt.tight_layout()
        self.logger.info(f"Saving distribution plots to {filename}_distributions.png")
        plt.savefig(self.output_dir / f'{filename}_distributions.png')
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, filename: str):
        # Generate and save a heatmap of feature correlations
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            self.logger.info("Calculating correlation matrix...")
            plt.figure(figsize=(12, 8))
            
            self.logger.info("Creating correlation heatmap...")
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            self.logger.info(f"Saving correlation matrix to {filename}_correlation.png")
            plt.savefig(self.output_dir / f'{filename}_correlation.png')
            plt.close()
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize column names for consistency
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing and infinite values"""
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Calculate missing percentages
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_percentages[missing_percentages > 30].index
        
        # Drop columns with more than 30% missing values
        if len(high_missing) > 0:
            self.logger.info(f"Dropping columns with >30% missing values: {high_missing.tolist()}")
            df = df.drop(columns=high_missing)
        
        # Fill remaining missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip outliers to bounds
            df[column] = df[column].clip(lower_bound, upper_bound)
        
        return df
    
    def remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns based on content"""
        duplicates = df.T.duplicated(keep='first')
        if duplicates.any():
            dropped_cols = df.columns[duplicates].tolist()
            self.logger.info(f"Removing duplicate columns: {dropped_cols}")
            df = df.T[~duplicates].T
        return df
    
    def remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features that have constant values"""
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        if constant_columns:
            self.logger.info(f"Removing constant columns: {constant_columns}")
            df = df.drop(columns=constant_columns)
        return df
    
    def process_dataset(self, file_path: str, discretize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        filename = Path(file_path).stem
        self.logger.info(f"Processing dataset: {file_path}")
        
        self.logger.info("Reading CSV file...")
        df = pd.read_csv(file_path, low_memory=False)
        initial_shape = df.shape
        self.logger.info(f"Initial shape: {initial_shape}")
        
        self.logger.info("Cleaning column names...")
        df = self.clean_column_names(df)
        
        self.logger.info("Handling missing values...")
        df = self.handle_missing_values(df)
        
        self.logger.info("Handling outliers...")
        df = self.handle_outliers(df)
        
        # Split features and labels
        label_col = next((col for col in df.columns if 'label' in col), None)
        if label_col is None:
            raise ValueError("Could not find label column in dataset")
        
        features = df.drop(columns=[label_col])
        labels = df[label_col]
        
        # Normalize and encode
        self.logger.info("Normalizing features...")
        features_normalized = self.scaler.fit_transform(features)
        if discretize:
            features_normalized = self.discretizer.fit_transform(features_normalized)
        
        self.logger.info("Encoding labels...")
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        self.logger.info(f"Final processed shape - Features: {features_normalized.shape}, Labels: {labels_encoded.shape}")
        return features_normalized, labels_encoded
