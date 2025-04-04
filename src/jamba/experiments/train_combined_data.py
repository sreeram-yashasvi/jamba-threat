#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import logging
import wandb
from sklearn.model_selection import train_test_split

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.jamba_model_transformer import JambaThreatTransformerModel, ThreatDataset
from jamba.model_config import ModelConfig
from jamba.utils.checkpoint_manager import CheckpointManager, save_pretrained

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(num_samples=150000):
    """Generate synthetic incident data."""
    np.random.seed(42)
    
    # Generate normal incidents (60% of data)
    normal_size = int(num_samples * 0.6)
    normal_data = {
        'Severity': np.random.choice(['low', 'medium'], size=normal_size, p=[0.7, 0.3]),
        'Investigation state': 'Queued',
        'Categories': 'InitialAccess',
        'Active alerts': np.random.randint(1, 3, size=normal_size),
        'Service sources': np.random.choice(['Office 365', 'Endpoint'], size=normal_size, p=[0.8, 0.2]),
        'Detection sources': 'MDO',
        'Status': 'Active',
        'Tags': '-',
        'Policy name': np.nan,
        'Classification': 'Not set',
        'Determination': 'Not set'
    }
    
    # Generate high-risk incidents (40% of data)
    high_risk_size = num_samples - normal_size
    high_risk_data = {
        'Severity': np.random.choice(['high'], size=high_risk_size),
        'Investigation state': np.random.choice(['Queued', '2 investigation states'], size=high_risk_size, p=[0.7, 0.3]),
        'Categories': np.random.choice(['InitialAccess', 'InitialAccess, Suspicious activity'], size=high_risk_size, p=[0.8, 0.2]),
        'Active alerts': np.random.randint(2, 5, size=high_risk_size),
        'Service sources': np.random.choice(['Office 365', 'Endpoint'], size=high_risk_size, p=[0.6, 0.4]),
        'Detection sources': np.random.choice(['MDO', 'Custom TI'], size=high_risk_size, p=[0.7, 0.3]),
        'Status': 'Active',
        'Tags': np.random.choice(['-', 'Credential Phish'], size=high_risk_size, p=[0.8, 0.2]),
        'Policy name': np.random.choice([np.nan, 'High Risk Policy'], size=high_risk_size, p=[0.9, 0.1]),
        'Classification': 'Not set',
        'Determination': 'Not set'
    }
    
    # Combine and shuffle data
    normal_df = pd.DataFrame(normal_data)
    high_risk_df = pd.DataFrame(high_risk_data)
    synthetic_df = pd.concat([normal_df, high_risk_df], ignore_index=True)
    synthetic_df = synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return synthetic_df

def preprocess_data(df):
    """Preprocess the incidents data."""
    features = {
        'severity_score': df['Severity'].map({'low': 0, 'medium': 1, 'high': 2, 'informational': -1}).fillna(0),
        'is_initial_access': df['Categories'].str.contains('InitialAccess', na=False).astype(float),
        'is_suspicious_activity': df['Categories'].str.contains('SuspiciousActivity', na=False).astype(float),
        'active_alerts': pd.to_numeric(df['Active alerts'].fillna(0)),
        'has_multiple_users': df['Categories'].str.count(',').gt(1).astype(float),
        'investigation_state_score': df['Investigation state'].map({
            'Queued': 0,
            'Unsupported alert type': 1,
            '2 investigation states': 2
        }).fillna(0),
        'is_active': (df['Status'] == 'Active').astype(float),
        'is_office365': (df['Service sources'] == 'Office 365').astype(float),
        'is_endpoint': (df['Service sources'] == 'Endpoint').astype(float),
        'is_mdo': (df['Detection sources'] == 'MDO').astype(float),
        'is_custom_ti': df['Detection sources'].str.contains('Custom TI', na=False).astype(float),
        'has_tags': (df['Tags'] != '-').astype(float),
        'has_policy': df['Policy name'].notna().astype(float),
        'has_classification': (df['Classification'] != 'Not set').astype(float),
        'has_determination': (df['Determination'] != 'Not set').astype(float)
    }
    
    features_df = pd.DataFrame(features)
    
    # Create target variable (1 for high-risk incidents)
    y = ((df['Severity'] == 'high') | 
         (df['Investigation state'] == '2 investigation states') |
         (features['has_multiple_users'] > 0)).astype(float)
    
    # Normalize numerical features
    numerical_cols = ['severity_score', 'active_alerts', 'investigation_state_score']
    for col in numerical_cols:
        features_df[col] = (features_df[col] - features_df[col].mean()) / (features_df[col].std() + 1e-8)
    
    # Fill any remaining NaN values with 0
    features_df = features_df.fillna(0)
    
    # Ensure we have exactly 20 features
    current_features = features_df.shape[1]
    if current_features < 20:
        for i in range(20 - current_features):
            features_df[f'padding_{i}'] = 0.0
    elif current_features > 20:
        features_df = features_df.iloc[:, :20]
    
    return features_df.values.astype(np.float32), y.values.astype(np.float32)

class IncidentDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_model():
    # Initialize wandb
    wandb.init(
        project="jamba-threat-detection",
        config={
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 64,
            "hidden_dim": 256,
            "n_heads": 8,
            "feature_layers": 4,
            "dropout_rate": 0.3
        }
    )
    
    # Load and combine data
    logger.info("Loading and preprocessing data...")
    real_df = pd.read_csv('incidents-queue-20250402.csv')
    synthetic_df = generate_synthetic_data(150000)
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    logger.info(f"Combined dataset size: {len(combined_df)} rows")
    
    # Preprocess data
    X, y = preprocess_data(combined_df)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = IncidentDataset(X_train, y_train)
    val_dataset = IncidentDataset(X_val, y_val)
    test_dataset = IncidentDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=False
    )
    
    # Initialize model
    config = ModelConfig(
        version='1.0.0',
        input_dim=X.shape[1],
        hidden_dim=wandb.config.hidden_dim,
        output_dim=1,
        dropout_rate=wandb.config.dropout_rate,
        n_heads=wandb.config.n_heads,
        feature_layers=wandb.config.feature_layers,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = JambaThreatTransformerModel(config)
    model = model.to(config.device)
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        base_dir=f'checkpoints/combined_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        max_checkpoints=5
    )
    
    # Loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(wandb.config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (output.squeeze() > 0).float()
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                val_loss += criterion(output.squeeze(), target).item()
                predicted = (output.squeeze() > 0).float()
                val_correct += (predicted == target).sum().item()
                val_total += target.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch
        })
        
        logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            },
            is_best=is_best,
            name=f"epoch_{epoch}"
        )
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Test best model
    logger.info("Loading best model for testing...")
    best_checkpoint = ckpt_manager.get_best_checkpoint()
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            test_loss += criterion(output.squeeze(), target).item()
            predicted = (output.squeeze() > 0).float()
            test_correct += (predicted == target).sum().item()
            test_total += target.size(0)
    
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    
    logger.info(f'Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc
    })
    
    # Save final model as pretrained
    logger.info("Saving final model...")
    save_pretrained(
        model=model,
        save_dir=f'pretrained/combined_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        optimizer=optimizer,
        epoch=best_checkpoint['epoch'],
        metrics={
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_val_loss': best_val_loss
        }
    )
    
    wandb.finish()

if __name__ == '__main__':
    train_model() 