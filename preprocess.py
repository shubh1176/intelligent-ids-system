import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(data_path):
    # Load network traffic data
    data = pd.read_csv(data_path)
    
    # Print column names to debug
    print(f"Available columns in {data_path}:")
    print(data.columns.tolist())
    
    print("Column names with repr:", [repr(col) for col in data.columns])
    
    # Find the label column case-insensitively
    label_col = next((col for col in data.columns if col.lower().strip() == 'label'), None)
    if label_col is None:
        raise ValueError(f"Could not find label column in {data_path}. Available columns are: {data.columns.tolist()}")
    
    # Extract features (excluding the label column)
    features = data.drop([label_col], axis=1)
    
    # Handle missing values and infinities
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    # Convert all features to numeric, handling any non-numeric columns
    for column in features.columns:
        if features[column].dtype == 'object':
            features[column] = pd.to_numeric(features[column], errors='coerce')
    
    # Clip extremely large values to a reasonable range
    # You might need to adjust these values based on your data
    clip_value = 1e9  # Adjust this threshold as needed
    features = features.clip(-clip_value, clip_value)
    
    # Extract labels and encode them
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data[label_col])
    
    # Add debug information
    print("\nData statistics before normalization:")
    print(features.describe())
    print("\nChecking for remaining infinities:", np.isinf(features.values).sum())
    print("Checking for remaining NaNs:", np.isnan(features.values).sum())
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized, labels, label_encoder

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