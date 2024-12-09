import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveChannelCalibration(nn.Module):
    """Dynamically calibrates channel importance based on current input patterns"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        # Combine global information
        global_info = torch.cat([avg_pool, max_pool], dim=1)
        channel_weights = self.mlp(global_info).view(b, c, 1)
        
        return x * channel_weights

class DynamicFeatureFusion(nn.Module):
    """Adaptively fuses features based on their discriminative power"""
    def __init__(self, channels):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(channels * 3, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features at different scales
        x1 = F.avg_pool1d(x, 3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, 5, stride=1, padding=2)
        x3 = x
        
        # Concatenate multi-scale features
        concat = torch.cat([x1, x2, x3], dim=1)
        
        # Generate fusion weights
        weights = self.gate(concat)
        
        # Fuse features
        fused = self.conv_1x1(concat)
        return fused * weights

class NetworkSignalModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        
        # Initial feature extraction
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Feature processing blocks
        self.block1 = nn.Sequential(
            DynamicFeatureFusion(64),
            AdaptiveChannelCalibration(64)
        )
        
        self.block2 = nn.Sequential(
            DynamicFeatureFusion(64),
            AdaptiveChannelCalibration(64)
        )
        
        # Classification head with feature aggregation
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # Dynamic feature processing with stronger residual connections
        identity = x
        x = self.block1(x)
        x = x + identity  # Skip connection
        x = F.relu(x)  # Added activation after residual
        
        identity = x
        x = self.block2(x)
        x = x + identity  # Skip connection
        x = F.relu(x)  # Added activation after residual
        
        # Classification
        return self.classifier(x) 