import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
import copy

class FederatedSignalModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        
        # Feature Extractor (Local)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Global Model (Server)
        self.global_model = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.global_model(features)
        return output

class FederatedLearningSystem:
    def __init__(self, num_clients, input_dim, num_classes=2, device='cuda'):
        self.num_clients = num_clients
        self.device = device
        
        # Initialize global model
        self.global_model = FederatedSignalModel(input_dim, num_classes).to(device)
        
        # Initialize client models
        self.client_models = [
            FederatedSignalModel(input_dim, num_classes).to(device) 
            for _ in range(num_clients)
        ]
        
        # Initialize optimizers for each client
        self.client_optimizers = [
            optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            for model in self.client_models
        ]
        
        self.criterion = nn.CrossEntropyLoss()

    def train_client(self, client_id, dataloader, epochs=5):
        model = self.client_models[client_id]
        optimizer = self.client_optimizers[client_id]
        
        model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(dataloader)
        
        return total_loss / epochs

    def aggregate_models(self, weights=None):
        if weights is None:
            weights = [1/self.num_clients] * self.num_clients
            
        global_state = self.global_model.state_dict()
        
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            for client_id, client_model in enumerate(self.client_models):
                global_state[key] += weights[client_id] * client_model.state_dict()[key]
        
        # Update global model
        self.global_model.load_state_dict(global_state)
        
        # Distribute updated model to clients
        for client_model in self.client_models:
            client_model.load_state_dict(copy.deepcopy(global_state))

    def evaluate_global(self, dataloader):
        self.global_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy 