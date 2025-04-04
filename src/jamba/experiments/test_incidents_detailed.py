#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import logging
from sklearn.metrics import classification_report, confusion_matrix

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.jamba_model_transformer import JambaThreatTransformerModel
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
        'is_custom_ti': df['Detection sources'].str.contains('Custom TI', na=False).astype(float),
        'has_tags': (df['Tags'] != '-').astype(float),
        'has_policy': df['Policy name'].notna().astype(float),
        'has_classification': (df['Classification'] != 'Not set').astype(float),
        'has_determination': (df['Determination'] != 'Not set').astype(float)
    }
    
    # Convert to DataFrame
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
    
    # Ensure we have exactly 20 features (pad if necessary)
    current_features = features_df.shape[1]
    if current_features < 20:
        for i in range(20 - current_features):
            features_df[f'padding_{i}'] = 0.0
    elif current_features > 20:
        features_df = features_df.iloc[:, :20]
    
    return features_df.values.astype(np.float32), y.values.astype(np.float32), df

def analyze_predictions(model, X, y, df):
    """Analyze model predictions with detailed insights."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = model(X_tensor)
        predictions = (outputs.squeeze() > 0).float().numpy()
        probabilities = torch.sigmoid(outputs.squeeze()).numpy()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Incident_Name': df['Incident name'],
        'True_Risk': y,
        'Predicted_Risk': predictions,
        'Risk_Probability': probabilities,
        'Severity': df['Severity'],
        'Categories': df['Categories'],
        'Investigation_State': df['Investigation state'],
        'Active_Alerts': df['Active alerts'],
        'Service_Sources': df['Service sources'],
        'Detection_Sources': df['Detection sources']
    })
    
    # Add prediction correctness
    results['Prediction_Status'] = 'Correct'
    mask = results['True_Risk'] != results['Predicted_Risk']
    results.loc[mask & (results['True_Risk'] == 1), 'Prediction_Status'] = 'False Negative'
    results.loc[mask & (results['True_Risk'] == 0), 'Prediction_Status'] = 'False Positive'
    
    return results

def test_model():
    # Load and preprocess data
    logger.info("Loading and preprocessing incidents data...")
    X, y, df = preprocess_incidents('incidents-queue-20250402.csv')
    
    # Load the trained model
    model_dir = 'models'
    model_files = list(Path(model_dir).glob('incidents_model_*/best_model.pt'))
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model from {latest_model}")
    
    checkpoint = torch.load(latest_model, map_location='cpu')
    config = ModelConfig(**checkpoint['config'])
    
    model = JambaThreatTransformerModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Analyze predictions
    results = analyze_predictions(model, X, y, df)
    
    # Print summary statistics
    logger.info("\n=== Model Performance Summary ===")
    logger.info(f"Total incidents analyzed: {len(results)}")
    logger.info(f"True high-risk incidents: {sum(y)}")
    logger.info(f"Predicted high-risk incidents: {sum(results['Predicted_Risk'])}")
    
    # Classification report
    logger.info("\n=== Classification Report ===")
    logger.info(classification_report(y, results['Predicted_Risk'], target_names=['Low Risk', 'High Risk']))
    
    # Confusion matrix
    cm = confusion_matrix(y, results['Predicted_Risk'])
    logger.info("\n=== Confusion Matrix ===")
    logger.info("                  Predicted Low  Predicted High")
    logger.info(f"Actual Low       {cm[0][0]:12d}  {cm[0][1]:13d}")
    logger.info(f"Actual High      {cm[1][0]:12d}  {cm[1][1]:13d}")
    
    # Analysis by category
    logger.info("\n=== Analysis by Category ===")
    category_analysis = results.groupby('Categories')['Prediction_Status'].value_counts().unstack(fill_value=0)
    logger.info("\nPrediction accuracy by incident category:")
    logger.info(category_analysis)
    
    # High confidence misclassifications
    logger.info("\n=== High Confidence Misclassifications (probability > 0.9) ===")
    high_conf_errors = results[
        ((results['Prediction_Status'] != 'Correct') & (results['Risk_Probability'] > 0.9)) |
        ((results['Prediction_Status'] != 'Correct') & (results['Risk_Probability'] < 0.1))
    ]
    if len(high_conf_errors) > 0:
        logger.info(high_conf_errors[['Incident_Name', 'True_Risk', 'Risk_Probability', 'Severity', 'Categories']])
    else:
        logger.info("No high confidence misclassifications found.")
    
    # Save detailed results
    output_file = f'analysis/incident_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
    logger.info(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    test_model() 