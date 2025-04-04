#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from jamba.jamba_model_transformer import JambaThreatTransformerModel
from jamba.model_config import ModelConfig
from jamba.utils.checkpoint_manager import load_pretrained

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def analyze_predictions(model, X, y, df):
    """Analyze model predictions with detailed insights."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = model(X_tensor)
        probabilities = torch.sigmoid(outputs.squeeze()).numpy()
        predictions = (probabilities > 0.5).astype(float)
    
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

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def test_model():
    # Create output directory
    output_dir = Path(f'analysis/model_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing incidents data...")
    df = pd.read_csv('incidents-queue-20250402.csv')
    X, y = preprocess_data(df)
    
    # Load the trained model
    logger.info("Loading trained model...")
    model_dir = 'pretrained'
    model_dirs = list(Path(model_dir).glob('combined_model_*'))
    if not model_dirs:
        raise FileNotFoundError("No trained model found!")
    
    latest_model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using model from: {latest_model_dir}")
    
    model, checkpoint = load_pretrained(latest_model_dir)
    model.eval()
    
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
    
    # Plot confusion matrix
    plot_confusion_matrix(y, results['Predicted_Risk'], output_dir / 'confusion_matrix.png')
    
    # Plot ROC curve
    plot_roc_curve(y, results['Risk_Probability'], output_dir / 'roc_curve.png')
    
    # Analysis by category
    logger.info("\n=== Analysis by Category ===")
    category_analysis = results.groupby('Categories')['Prediction_Status'].value_counts().unstack(fill_value=0)
    logger.info("\nPrediction accuracy by incident category:")
    logger.info(category_analysis)
    
    # Analysis by severity
    logger.info("\n=== Analysis by Severity ===")
    severity_analysis = results.groupby('Severity')['Prediction_Status'].value_counts().unstack(fill_value=0)
    logger.info("\nPrediction accuracy by severity level:")
    logger.info(severity_analysis)
    
    # High confidence misclassifications
    logger.info("\n=== High Confidence Misclassifications (probability > 0.9 or < 0.1) ===")
    high_conf_errors = results[
        ((results['Prediction_Status'] != 'Correct') & (results['Risk_Probability'] > 0.9)) |
        ((results['Prediction_Status'] != 'Correct') & (results['Risk_Probability'] < 0.1))
    ]
    if len(high_conf_errors) > 0:
        logger.info(high_conf_errors[['Incident_Name', 'True_Risk', 'Risk_Probability', 'Severity', 'Categories']])
    else:
        logger.info("No high confidence misclassifications found.")
    
    # Save detailed results
    results_file = output_dir / 'detailed_predictions.csv'
    results.to_csv(results_file, index=False)
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Save model metrics
    metrics = {
        'total_incidents': len(results),
        'true_high_risk': int(sum(y)),
        'predicted_high_risk': int(sum(results['Predicted_Risk'])),
        'accuracy': float((results['True_Risk'] == results['Predicted_Risk']).mean()),
        'confusion_matrix': cm.tolist(),
        'high_conf_errors': len(high_conf_errors)
    }
    
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to: {output_dir}")

if __name__ == '__main__':
    test_model() 