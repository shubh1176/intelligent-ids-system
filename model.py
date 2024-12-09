import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Helps the model focus on important features by adding channel attention"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Processes features while maintaining signal integrity through skip connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        # Skip connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return self.dropout(out)

class ImprovedSignalTransformerBlock(nn.Module):
    """Processes temporal relationships in network signals using self-attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1, max_seq_length=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Position bias for better temporal understanding
        self.max_seq_length = max_seq_length
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_seq_length - 1, heads))
        
    def forward(self, x, rel_pos=None):
        if rel_pos is not None:
            rel_pos = rel_pos.squeeze()
            if rel_pos.dim() > 2:
                rel_pos = None
                
        att_out, _ = self.attention(x, x, x, rel_pos)
        att_out = self.dropout(att_out)
        out1 = self.norm1(x + att_out)
        
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_out)
        
        return out2
    
    def _get_rel_pos(self, length):
        pos = torch.arange(length, device=self.rel_pos_bias.device)
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        rel_pos += self.max_seq_length - 1
        rel_pos = torch.clamp(rel_pos, min=0, max=2 * self.max_seq_length - 2)
        return F.embedding(rel_pos, self.rel_pos_bias)

class EnhancedNetworkSignalModel(nn.Module):
    """Main model combining CNN, Transformer, and multi-scale processing for network traffic analysis"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        
        # Convert input features to initial representation
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Extract hierarchical features
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512)
        )
        
        # Process temporal patterns
        self.transformer_blocks = nn.ModuleList([
            ImprovedSignalTransformerBlock(dim=512) for _ in range(4)
        ])
        
        # Capture patterns at different scales
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(512, 512, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.input_projection(x)
        x = self.res_blocks(x)
        
        multi_scale_features = [x]
        for conv in self.multi_scale:
            multi_scale_features.append(conv(x))
        
        x = torch.cat(multi_scale_features, dim=1)
        
        x = x.transpose(1, 2)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.transpose(1, 2)
        
        x = self.classifier(x)
        return x 