import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class VisualizationUtils:
    @staticmethod
    def plot_training_history(train_losses, val_losses, accuracies, save_path='training_history.png'):
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_signal_examples(signals, predictions, true_labels, save_path='signal_examples.png'):
        plt.figure(figsize=(15, 10))
        num_examples = min(5, len(signals))
        
        for i in range(num_examples):
            plt.subplot(num_examples, 1, i+1)
            plt.plot(signals[i].T)
            plt.title(f'True: {true_labels[i]}, Predicted: {predictions[i]}')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 