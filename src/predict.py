import argparse
import pandas as pd
import numpy as np
import torch
import os
import logging
from model_training import ThreatModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JambaThreatPredictor:
    def __init__(self, model_path):
        """Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.trainer = ThreatModelTrainer()
        self.model = self.trainer.load_model(model_path)
        self.model.eval()
        self.device = self.trainer.device
        
    def predict(self, data_path, output_path=None):
        """Make predictions on new data.
        
        Args:
            data_path: Path to the data file (.csv or .parquet)
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data based on file extension
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        # Check if target column exists in the data
        has_target = 'is_threat' in df.columns
        
        # Prepare input features
        X = df.drop(['is_threat'], axis=1) if has_target else df.copy()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            raw_predictions = self.model(X_tensor).cpu().numpy()
        
        # Convert to binary predictions
        binary_predictions = (raw_predictions > 0.5).astype(int)
        
        # Add predictions to the dataframe
        df['threat_probability'] = raw_predictions
        df['is_threat_predicted'] = binary_predictions
        
        # Calculate accuracy if target is available
        if has_target:
            accuracy = (df['is_threat'] == df['is_threat_predicted']).mean()
            logger.info(f"Prediction accuracy: {accuracy:.4f}")
        
        # Save predictions if output path is provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        return df
    
    def predict_single(self, features_dict):
        """Make prediction on a single sample.
        
        Args:
            features_dict: Dictionary with feature names and values
            
        Returns:
            Probability of threat and binary prediction
        """
        # Convert dictionary to DataFrame
        sample = pd.DataFrame([features_dict])
        
        # Convert to tensor
        sample_tensor = torch.FloatTensor(sample.values).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            probability = self.model(sample_tensor).cpu().numpy()[0][0]
        
        # Binary prediction
        is_threat = probability > 0.5
        
        return {"probability": float(probability), "is_threat": bool(is_threat)}

def main():
    parser = argparse.ArgumentParser(description='Make predictions with Jamba Threat Model')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--data', required=True, help='Path to data file for prediction')
    parser.add_argument('--output', help='Path to save prediction results')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = JambaThreatPredictor(args.model)
    
    # Make predictions
    predictions = predictor.predict(args.data, args.output)
    
    # Display sample predictions
    pd.set_option('display.max_columns', None)
    logger.info("\nSample predictions:")
    logger.info(predictions[['threat_probability', 'is_threat_predicted']].head())

if __name__ == "__main__":
    main() 