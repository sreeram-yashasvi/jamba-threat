import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generate synthetic network traffic data."""
    
    def __init__(self, n_features=20, random_state=None):
        """Initialize the dataset generator."""
        self.n_features = n_features
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def generate_normal_traffic(self, n_samples):
        """Generate normal network traffic data."""
        # Generate base features
        data = self.rng.normal(loc=0.5, scale=0.15, size=(n_samples, self.n_features))
        
        # Ensure values are within [0, 1]
        data = np.clip(data, 0, 1)
        
        return data
    
    def generate_threat_traffic(self, n_samples):
        """Generate threat network traffic data."""
        # Generate base features with different distribution
        data = self.rng.normal(loc=0.7, scale=0.2, size=(n_samples, self.n_features))
        
        # Add some anomalous patterns
        for i in range(n_samples):
            # Randomly select features to make anomalous
            n_anomalous = self.rng.randint(2, 5)
            anomalous_features = self.rng.choice(self.n_features, n_anomalous, replace=False)
            
            # Make selected features more extreme
            data[i, anomalous_features] = self.rng.uniform(0.8, 1.0, n_anomalous)
        
        # Ensure values are within [0, 1]
        data = np.clip(data, 0, 1)
        
        return data
    
    def save_dataset(self, X, y, output_dir='data', prefix='balanced'):
        """Save dataset to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for easier saving
        feature_cols = [f'feature_{i+1}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_cols)
        y_df = pd.DataFrame(y, columns=['threat'])
        
        # Save in chunks to handle large datasets
        chunk_size = 10000
        for i in range(0, len(X_df), chunk_size):
            mode = 'w' if i == 0 else 'a'
            header = True if i == 0 else False
            
            chunk_X = X_df.iloc[i:i+chunk_size]
            chunk_y = y_df.iloc[i:i+chunk_size]
            
            chunk_X.to_csv(os.path.join(output_dir, f"{prefix}_features.csv"), 
                          mode=mode, header=header, index=False)
            chunk_y.to_csv(os.path.join(output_dir, f"{prefix}_labels.csv"), 
                          mode=mode, header=header, index=False)
        
        # Log dataset statistics
        logger.info(f"Dataset shape: {X.shape}")
        class_dist = pd.Series(y).value_counts(normalize=True)
        logger.info("Class distribution:\nthreat\n" + class_dist.to_string())

def generate_balanced_dataset(n_normal_samples, n_threat_samples, n_features=20, random_state=42):
    """
    Generate a balanced synthetic dataset for threat detection.
    
    Args:
        n_normal_samples (int): Number of normal (non-threat) samples
        n_threat_samples (int): Number of threat samples
        n_features (int): Number of features per sample
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (features, labels) as numpy arrays
    """
    np.random.seed(random_state)
    
    # Generate normal samples
    normal_samples = np.random.randn(n_normal_samples, n_features)
    normal_labels = np.zeros(n_normal_samples)
    
    # Generate threat samples with a different distribution
    threat_samples = np.random.randn(n_threat_samples, n_features) * 1.5 + 0.5
    threat_labels = np.ones(n_threat_samples)
    
    # Combine samples
    X = np.vstack([normal_samples, threat_samples])
    y = np.hstack([normal_labels, threat_labels])
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    logger.info(f"Generated balanced dataset with {len(X)} samples "
               f"({n_normal_samples} normal, {n_threat_samples} threat)")
    
    return X, y 