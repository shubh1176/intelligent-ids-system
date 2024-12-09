import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_array
from typing import Tuple
from imblearn.over_sampling import SMOTE
from data_cleaner import NetworkDataCleaner
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

def load_and_preprocess_data(data_path):
    # Load data
    data = pd.read_csv(data_path, low_memory=False)
    
    # Initialize cleaner
    cleaner = NetworkDataCleaner()
    
    # Basic preprocessing
    data = cleaner.clean_column_names(data)
    data = cleaner.remove_constant_features(data)
    
    # Find label column
    label_col = next((col for col in data.columns if col.lower().strip() == 'label'), None)
    if label_col is None:
        raise ValueError(f"Could not find label column in {data_path}")
    
    # Separate features and labels
    features = data.drop(columns=[label_col])
    labels = data[label_col]
    
    # Clean features
    features = cleaner.handle_missing_values(features)
    features = cleaner.handle_outliers(features)
    
    # Normalize features
    features_normalized = cleaner.scaler.fit_transform(features)
    
    # Encode labels
    labels_encoded = cleaner.label_encoder.fit_transform(labels)
    
    return features_normalized, labels_encoded, cleaner.label_encoder

def create_sliding_windows(data, window_size=100, stride=50):
    """
    Create sliding windows for temporal analysis of network traffic
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data of shape (n_samples, n_features)
    window_size : int
        Size of each window
    stride : int
        Number of steps to move forward for each window
    
    Returns:
    --------
    numpy.ndarray
        Windowed data of shape (n_windows, window_size, n_features)
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def combine_datasets(file_paths):
    """
    Combine multiple CSV files into a single preprocessed dataset
    
    Parameters:
    -----------
    file_paths : list
        List of paths to CSV files
    
    Returns:
    --------
    tuple
        (combined_features, combined_labels, label_encoder)
    """
    all_features = []
    all_labels = []
    label_encoder = None
    
    for file_path in file_paths:
        features, labels, le = load_and_preprocess_data(file_path)
        all_features.append(features)
        all_labels.append(labels)
        if label_encoder is None:
            label_encoder = le
    
    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    return combined_features, combined_labels, label_encoder

def select_features(features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove low variance and highly correlated features"""
    # Remove constant and quasi-constant features
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(features)
    features = features.iloc[:, selector.get_support(indices=True)]
    
    # Remove highly correlated features
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return features.drop(columns=to_drop)

def balance_classes(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Balance classes using SMOTE"""
    smote = SMOTE(random_state=42)
    features_balanced, labels_balanced = smote.fit_resample(features, labels)
    return features_balanced, labels_balanced

def select_important_features(features: np.ndarray, labels: np.ndarray, n_features: int = 20) -> np.ndarray:
    """Select most important features using Random Forest feature importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    
    # Get feature importance scores
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top n features
    selected_indices = indices[:n_features]
    return features[:, selected_indices]