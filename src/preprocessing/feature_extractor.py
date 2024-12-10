import pandas as pd
import numpy as np
import torch
import time

class FeatureExtractor:
    def __init__(self, model_input_dims):
        self.input_dims = model_input_dims
        
    def extract(self, packet):
        # Extract more detailed features
        features = {
            'packet_size': len(packet),
            'protocol': packet.proto if hasattr(packet, 'proto') else 0,
            'sport': packet.sport if hasattr(packet, 'sport') else 0,
            'dport': packet.dport if hasattr(packet, 'dport') else 0,
            'time': packet.time if hasattr(packet, 'time') else 0,
            'flags': int(packet.flags) if hasattr(packet, 'flags') else 0
        }
        
        # Add attack-specific features
        is_common_port = features['sport'] in [80, 443, 53, 22, 21, 25]
        has_multiple_flags = bin(features['flags']).count('1') > 2
        is_suspicious_port = features['sport'] > 10000 or features['dport'] < 1024
        
        # Normalize features
        normalized_features = [
            features['packet_size'] / 1500.0,  # Typical MTU size
            features['protocol'] / 255.0,      # Max protocol number
            features['sport'] / 65535.0,       # Max port number
            features['dport'] / 65535.0,       # Max port number
            (features['time'] - time.time()) / 3600.0,  # Normalize to hours
            features['flags'] / 63.0,          # Max possible flags
            float(is_common_port),             # Common port indicator
            float(has_multiple_flags),         # Multiple flags indicator
            float(is_suspicious_port)          # Suspicious port indicator
        ]
        
        # Convert to tensor
        feature_vector = torch.tensor(normalized_features, dtype=torch.float32)
        
        # Pad to match model input dimensions
        if len(feature_vector) < self.input_dims:
            feature_vector = torch.nn.functional.pad(
                feature_vector, 
                (0, self.input_dims - len(feature_vector))
            )
        else:
            feature_vector = feature_vector[:self.input_dims]
            
        return feature_vector.unsqueeze(0).unsqueeze(-1)