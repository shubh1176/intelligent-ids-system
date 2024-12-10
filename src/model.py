import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, C, L = x.size()
        
        q = self.query(x).view(batch, -1, L).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, L)
        v = self.value(x).view(batch, -1, L)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, C, L)
        
        return x + self.gamma * out

class ImprovedNetworkModel(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        
        # Initial feature extraction
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Deep feature extraction
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(128),
                AttentionModule(128),
                nn.Dropout(0.2)
            ) for _ in range(3)
        ])
        
        # Multi-scale processing
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(128, 64, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # Deep feature extraction with residual connections and attention
        for layer in self.feature_extractor:
            x = layer(x) + x
        
        # Multi-scale feature extraction
        multi_scale_features = [conv(x) for conv in self.multi_scale]
        x = torch.cat(multi_scale_features, dim=1)
        
        # Feature fusion
        x = self.fusion(x)
        
        # Classification
        return self.classifier(x) 