import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the transformer model
from jamba import jamba_model_transformer
from jamba.model_config import ModelConfig

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
dataset_sizes = [45000, 70000, 90000, 120000]
num_epochs = 3
batch_size = 128
input_dim = 20  # as defined in our models

# Loss function: BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()


def generate_synthetic_data(num_samples, input_dim):
    # Generate random features and binary targets
    X = torch.randn(num_samples, input_dim)
    # Targets: 0 or 1 with equal probability
    y = (torch.rand(num_samples) > 0.5).long()
    return X, y


def compute_metrics(outputs, targets):
    # outputs: raw logits (shape [n,1])
    preds = (torch.sigmoid(outputs) > 0.5).long().view(-1)
    targets = targets.view(-1)
    accuracy = (preds == targets).float().mean().item()
    
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return accuracy, f1


def train_and_evaluate(model, dataloader, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)  # shape: [batch, 1]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_outputs.append(outputs.cpu())
            all_targets.append(y_batch)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    accuracy, f1 = compute_metrics(all_outputs, all_targets)
    return accuracy, f1, training_time


if __name__ == "__main__":
    results = []
    sns.set(style='whitegrid')

    for size in dataset_sizes:
        print(f"\nTraining on dataset of size: {size}")
        # Generate synthetic data
        X, y = generate_synthetic_data(size, input_dim)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create transformer model configuration
        config = ModelConfig(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=1,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=batch_size,
            device=str(device),
            feature_layers=0  # not used by transformer
        )
        # Add transformer-specific parameters
        setattr(config, 'num_heads', 4)
        setattr(config, 'transformer_layers', 2)
        
        model = jamba_model_transformer.JambaThreatTransformerModel(config)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        print(f"Training Transformer model for {num_epochs} epochs...")
        acc, f1, train_time = train_and_evaluate(model, dataloader, optimizer, criterion, num_epochs, device)
        print(f"Transformer model - Dataset size: {size} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Training Time: {train_time:.2f} sec")
        
        results.append({
            'dataset_size': size,
            'accuracy': acc,
            'f1': f1,
            'training_time': train_time
        })
        
    # Plotting the results
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=results_df, x='dataset_size', y='accuracy', marker='o')
    sns.lineplot(data=results_df, x='dataset_size', y='f1', marker='s', linestyle='--')
    plt.title('Transformer Model Performance vs Dataset Size')
    plt.xlabel('Dataset Size')
    plt.ylabel('Score')
    plt.legend(['Accuracy', 'F1 Score'])
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=results_df, x='dataset_size', y='training_time', marker='o')
    plt.title('Transformer Training Time vs Dataset Size')
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (sec)')
    
    plt.tight_layout()
    plt.show() 