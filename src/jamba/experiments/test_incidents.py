#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import logging

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.jamba_model_transformer import JambaThreatTransformerModel, ThreatDataset
from jamba.model_config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_incidents(csv_path):
    """Preprocess the incidents queue data."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract relevant features
    features = {
        'severity_score': df['Severity'].map({'low': 0, 'medium': 1, 'high': 2, 'informational': -1}).fillna(0),
        'is_initial_access': df['Categories'].str.contains('InitialAccess', na=False).astype(float),
        'is_suspicious_activity': df['Categories'].str.contains('SuspiciousActivity', na=False).astype(float),
        'active_alerts': pd.to_numeric(df['Active alerts'].fillna(0)),
        'has_multiple_users': df['Impacted assets'].str.count(',').gt(1).astype(float),
        'investigation_state_score': df['Investigation state'].map({
            'Queued': 0,
            'Unsupported alert type': 1,
            '2 investigation states': 2
        }).fillna(0),
        'is_active': (df['Status'] == 'Active').astype(float),
        'is_office365': (df['Service sources'] == 'Office 365').astype(float),
        'is_endpoint': (df['Service sources'] == 'Endpoint').astype(float),
        'is_mdo': (df['Detection sources'] == 'MDO').astype(float),
        'is_custom_ti': df['Detection sources'].str.contains('Custom TI', na=False).astype(float)
    }
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    # Create target variable (1 for high-risk incidents)
    # High severity OR multiple investigation states OR multiple impacted users
    y = ((df['Severity'] == 'high') | 
         (df['Investigation state'] == '2 investigation states') |
         (features['has_multiple_users'] > 0)).astype(float)
    
    # Normalize numerical features
    numerical_cols = ['severity_score', 'active_alerts', 'investigation_state_score']
    for col in numerical_cols:
        features_df[col] = (features_df[col] - features_df[col].mean()) / (features_df[col].std() + 1e-8)
    
    # Fill any remaining NaN values with 0
    features_df = features_df.fillna(0)
    
    # Ensure we have exactly 20 features (pad if necessary)
    current_features = features_df.shape[1]
    if current_features < 20:
        for i in range(20 - current_features):
            features_df[f'padding_{i}'] = 0.0
    elif current_features > 20:
        features_df = features_df.iloc[:, :20]
    
    return features_df.values.astype(np.float32), y.values.astype(np.float32)

def test_model_on_incidents():
    # Load the incidents data
    logger.info("Loading and preprocessing incidents data...")
    X, y = preprocess_incidents('incidents-queue-20250402.csv')
    
    # Load the trained model
    model_dir = 'models'
    model_files = list(Path(model_dir).glob('transformer_large_*/best_model.pt'))
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model from {latest_model}")
    
    checkpoint = torch.load(latest_model, map_location='cpu')
    config = ModelConfig(**checkpoint['config'])
    
    # Initialize model and load weights
    model = JambaThreatTransformerModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset and dataloader
    dataset = ThreatDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Test the model
    logger.info("Testing model on incidents data...")
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            outputs = model(data)
            predicted = (outputs.squeeze() > 0).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            predictions.extend(predicted.numpy())
            true_labels.extend(target.numpy())
    
    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Calculate additional metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Print detailed analysis
    logger.info("\nDetailed Analysis:")
    logger.info(f"Total incidents: {len(y)}")
    logger.info(f"High-risk incidents detected: {sum(predictions)}")
    logger.info(f"True high-risk incidents: {sum(true_labels)}")
    logger.info(f"False positives: {fp}")
    logger.info(f"False negatives: {fn}")

if __name__ == '__main__':
    test_model_on_incidents() 