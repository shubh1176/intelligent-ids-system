import torch
from torch.utils.data import random_split
from federated_model import FederatedLearningSystem
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'federated_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def simulate_federated_training(fed_system, client_dataloaders, val_loader, 
                              num_rounds=50, local_epochs=5):
    setup_logging()
    best_accuracy = 0
    
    for round_num in tqdm(range(num_rounds), desc="Federated Training Rounds"):
        # Train each client
        client_losses = []
        for client_id, dataloader in enumerate(client_dataloaders):
            loss = fed_system.train_client(
                client_id, dataloader, epochs=local_epochs
            )
            client_losses.append(loss)
            logging.info(f"Round {round_num + 1}, Client {client_id + 1} Loss: {loss:.4f}")
        
        # Aggregate models
        fed_system.aggregate_models()
        
        # Evaluate global model
        val_loss, accuracy = fed_system.evaluate_global(val_loader)
        logging.info(f"Round {round_num + 1}")
        logging.info(f"Average Client Loss: {np.mean(client_losses):.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}")
        logging.info(f"Validation Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'round': round_num,
                'model_state_dict': fed_system.global_model.state_dict(),
                'accuracy': accuracy
            }, 'best_federated_model.pth')
            logging.info(f"Saved new best model with accuracy: {accuracy:.2f}%")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess your data here
    # X = your_signal_data
    # y = your_labels
    
    # Create federated learning system
    num_clients = 5  # Number of clients
    fed_system = FederatedLearningSystem(
        num_clients=num_clients,
        input_dim=X.shape[1],
        device=device
    )
    
    # Split data for clients (simulate distributed data)
    total_samples = len(X)
    samples_per_client = total_samples // num_clients
    client_datasets = random_split(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        [samples_per_client] * (num_clients - 1) + [total_samples - samples_per_client * (num_clients - 1)]
    )
    
    # Create dataloaders for each client
    client_dataloaders = [
        DataLoader(dataset, batch_size=32, shuffle=True)
        for dataset in client_datasets[:-1]  # Last split is validation
    ]
    
    # Validation dataloader
    val_loader = DataLoader(client_datasets[-1], batch_size=32, shuffle=False)
    
    # Train federated model
    simulate_federated_training(fed_system, client_dataloaders, val_loader)

if __name__ == '__main__':
    main() 