import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our models
from jamba import jamba_model_ff
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
        # Print epoch loss
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


# To store results
results = []

# We'll compare two model types: feed-forward and transformer
model_types = ['FeedForward', 'Transformer']

for size in dataset_sizes:
    print(f"\nTraining on dataset of size: {size}")
    # Generate synthetic data
    X, y = generate_synthetic_data(size, input_dim)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        # Create a ModelConfig instance. Assume version field exists in ModelConfig
        config = ModelConfig(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=1,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=batch_size,
            device=str(device),
            feature_layers=2  # for feedforward model; ignored by transformer
        )
        
        if model_type == 'FeedForward':
            model = jamba_model_ff.JambaThreatModel(config)
        else:
            # For transformer model, add transformer-specific config
            # Ensure our ModelConfig accepts num_heads and transformer_layers
            # If not, they might be extra fields
            setattr(config, 'num_heads', 4)
            setattr(config, 'transformer_layers', 2)
            model = jamba_model_transformer.JambaThreatTransformerModel(config)
        
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        print(f"Training {model_type} model for {num_epochs} epochs...")
        acc, f1, train_time = train_and_evaluate(model, dataloader, optimizer, criterion, num_epochs, device)
        print(f"{model_type} model - Dataset size: {size} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Training Time: {train_time:.2f} sec")
        
        results.append({
            'model': model_type,
            'dataset_size': size,
            'accuracy': acc,
            'f1': f1,
            'training_time': train_time
        })

# Plotting the results
import pandas as pd
results_df = pd.DataFrame(results)

sns.set(style='whitegrid')

# Plot Accuracy and F1 score
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.lineplot(data=results_df, x='dataset_size', y='accuracy', hue='model', marker='o', label='Accuracy')
sns.lineplot(data=results_df, x='dataset_size', y='f1', hue='model', marker='s', linestyle='--', label='F1 Score')
plt.title('Model Performance vs Dataset Size')
plt.xlabel('Dataset Size')
plt.ylabel('Score')
plt.legend()

# Plot Training Time
plt.subplot(1, 2, 2)
sns.lineplot(data=results_df, x='dataset_size', y='training_time', hue='model', marker='o')
plt.title('Training Time vs Dataset Size')
plt.xlabel('Dataset Size')
plt.ylabel('Training Time (sec)')
plt.legend()

plt.tight_layout()
plt.savefig('comparison.png')
plt.close() 