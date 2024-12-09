import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from model import NetworkSignalModel
from utils import VisualizationUtils
from tqdm import tqdm
import logging
from data_cleaner import NetworkDataCleaner
import time
from datetime import datetime
from preprocess import load_and_preprocess_data, create_sliding_windows, combine_datasets
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=100, device='cuda', patience=5, accuracy_threshold=99.99):
    model = model.to(device)
    best_val_loss = float('inf')
    best_accuracy = 0
    patience_counter = 0
    perfect_accuracy_counter = 0
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    accuracies = []
    
    logging.info(f"Starting training on device: {device}")
    logging.info(f"Model architecture:\n{model}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Step the scheduler
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        epoch_time = time.time() - epoch_start_time
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        
        # Log metrics
        logging.info(f'\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        logging.info(f'Train Loss: {train_loss:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f}')
        logging.info(f'Accuracy: {accuracy:.2f}%')
        
        # Check if this is the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy
            }, 'best_model.pth')
            
            logging.info(f"New best model saved with accuracy: {accuracy:.2f}%")
        else:
            patience_counter += 1
        
        # Check for perfect or near-perfect accuracy
        if accuracy >= accuracy_threshold:
            perfect_accuracy_counter += 1
            if perfect_accuracy_counter >= 3:  # Require 3 consecutive epochs of near-perfect accuracy
                logging.info(f"\nAchieved sustained accuracy of {accuracy:.2f}% for 3 consecutive epochs")
                logging.info("Early stopping triggered - Model has achieved optimal performance")
                return train_losses, val_losses, accuracies
        else:
            perfect_accuracy_counter = 0
        
        # Early stopping check
        if patience_counter >= patience:
            logging.info(f"\nEarly stopping triggered - No improvement for {patience} epochs")
            return train_losses, val_losses, accuracies
    
    # Add final visualization after training is complete
    VisualizationUtils.plot_training_history(
        train_losses, val_losses, accuracies,
        save_path='final_training_progress.png'
    )
    
    return train_losses, val_losses, accuracies

def main():
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the data cleaner
    cleaner = NetworkDataCleaner()
    
    # Load and preprocess all CSV files from the dataset folder
    dataset_path = 'dataset'
    csv_files = [
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv'
    ]
    
    # Initialize lists to store combined data
    all_features = []
    all_labels = []
    
    for csv_file in csv_files:
        file_path = f"{dataset_path}/{csv_file}"
        # Use the cleaner to process each dataset
        features, labels = cleaner.process_dataset(file_path)
        all_features.append(features)
        all_labels.append(labels)
    
    # Combine all data
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create sliding windows for sequence data
    window_size = 100
    stride = 50
    
    X_train = create_sliding_windows(X_train, window_size, stride)
    X_val = create_sliding_windows(X_val, window_size, stride)
    
    # Adjust labels to match the windowed data size
    y_train = y_train[window_size-1::stride][:len(X_train)]
    y_val = y_val[window_size-1::stride][:len(X_val)]
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).transpose(1, 2),  # Shape: [batch, features, sequence]
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).transpose(1, 2),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model with correct input dimensions
    model = NetworkSignalModel(
        input_dim=X_train.shape[2],  # Number of features
        num_classes=len(np.unique(y))  # Number of unique classes
    )
    
    # Use Label Smoothing loss with reduced smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced from 0.1
    
    # Modified optimizer with lower initial learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # Modified scheduler parameters
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Reduced from 10
        T_mult=2
    )
    
    # Train model with modified parameters
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        patience=10,  # Increased from 5
        accuracy_threshold=99.5  # Reduced from 99.99
    )

    unique_classes = np.unique(y)
    logging.info(f"Dataset contains {len(unique_classes)} classes: {unique_classes}")

if __name__ == '__main__':
    main() 