import os
import pandas as pd
import numpy as np
from model_training import ThreatModelTrainer
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_threat_data(num_samples=1000, save_path='data/synthetic_threat_data.csv'):
    """Generate synthetic threat intelligence data for model training.
    
    Args:
        num_samples: Number of samples to generate
        save_path: Path to save the generated dataset
        
    Returns:
        DataFrame containing synthetic data
    """
    logger.info(f"Generating {num_samples} synthetic threat data samples")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    
    # Feature generation
    data = {
        # Temporal features
        'timestamp': pd.date_range(start='2023-01-01', periods=num_samples, freq='10T'),
        'hour_of_day': np.random.randint(0, 24, num_samples),
        'day_of_week': np.random.randint(0, 7, num_samples),
        
        # Network features
        'source_port': np.random.randint(1024, 65535, num_samples),
        'destination_port': np.random.randint(1, 1024, num_samples),
        'packet_size': np.random.normal(500, 200, num_samples),
        'packet_count': np.random.poisson(10, num_samples),
        'connection_duration': np.random.exponential(60, num_samples),
        
        # Behavioral features
        'connection_attempts': np.random.poisson(3, num_samples),
        'data_exfiltration_volume': np.random.exponential(100, num_samples),
        'encryption_detected': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'unusual_pattern_score': np.random.normal(0.3, 0.2, num_samples),
        
        # Threat indicators
        'known_malware_pattern': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'blacklisted_ip': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'suspicious_ua_string': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        'anomaly_score': np.random.normal(0.2, 0.15, num_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable based on features (synthetic rule)
    # A combination of features that indicate threat
    df['is_threat'] = (
        (df['anomaly_score'] > 0.5) | 
        (df['blacklisted_ip'] == 1) | 
        ((df['unusual_pattern_score'] > 0.6) & (df['encryption_detected'] == 1)) |
        ((df['known_malware_pattern'] == 1) & (df['suspicious_ua_string'] == 1))
    ).astype(float)
    
    # Add some noise to make it more realistic
    noise_mask = np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    df['is_threat'] = np.abs(df['is_threat'] - noise_mask)
    
    # Convert timestamp to features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.weekday
    
    # Drop timestamp as it's not directly usable for ML
    df = df.drop('timestamp', axis=1)
    
    # Save to file
    df.to_csv(save_path, index=False)
    logger.info(f"Synthetic dataset saved to {save_path}")
    
    # Print class distribution
    threat_count = df['is_threat'].sum()
    logger.info(f"Class distribution - Threats: {threat_count} ({threat_count/num_samples:.2%}), " 
              f"Benign: {num_samples - threat_count} ({1 - threat_count/num_samples:.2%})")
    
    return df

def preprocess_data(df, save_path='data/processed_threat_data.csv'):
    """Preprocess the data for training.
    
    Args:
        df: DataFrame containing raw data
        save_path: Path to save the processed dataset
        
    Returns:
        DataFrame containing processed data
    """
    logger.info("Preprocessing data")
    
    # Handle categorical variables if needed
    # For this synthetic dataset, all features are already numerical
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = [
        'source_port', 'destination_port', 'packet_size', 'packet_count',
        'connection_duration', 'connection_attempts', 'data_exfiltration_volume',
        'unusual_pattern_score', 'anomaly_score'
    ]
    
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save processed data
    df.to_csv(save_path, index=False)
    logger.info(f"Processed dataset saved to {save_path}")
    
    return df

def main():
    # Generate synthetic data
    df = generate_synthetic_threat_data(num_samples=5000, save_path='data/synthetic_threat_data.csv')
    
    # Preprocess the data
    processed_df = preprocess_data(df, save_path='data/processed_threat_data.csv')
    
    # Initialize the trainer
    trainer = ThreatModelTrainer(model_save_dir='models')
    
    # Train the model
    model, history = trainer.train_model(
        data_path='data/processed_threat_data.csv',
        target_column='is_threat',
        epochs=30,
        learning_rate=0.001,
        batch_size=64
    )
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 