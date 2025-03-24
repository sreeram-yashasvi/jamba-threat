import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import logging

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jamba.model_config import ModelConfig
from jamba.train import ModelTrainer
from jamba.data.sample_data import generate_sample_data
from jamba.jamba_model import JambaThreatModel, create_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_predictions(model, data_loader, device):
    """Get model predictions and true labels for a dataset."""
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())
    return np.array(predictions), np.array(true_labels)

def analyze_feature_importance(model, data_loader, device, feature_names):
    """Analyze feature importance using gradient-based approach."""
    model.eval()
    feature_importance = torch.zeros(len(feature_names)).to(device)
    n_samples = 0
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        inputs.requires_grad = True
        
        outputs = model(inputs)
        outputs.sum().backward()
        
        feature_importance += torch.abs(inputs.grad).mean(0)
        n_samples += 1
    
    feature_importance = feature_importance.cpu().numpy() / n_samples
    return dict(zip(feature_names, feature_importance))

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path='plots/roc_curve.png'):
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

def find_optimal_threshold(y_true, y_prob):
    """Find the optimal decision threshold using F1 score."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True)
        f1 = report.get('1', {}).get('f1-score', 0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.close()

def analyze_predictions():
    # Initialize model with config
    config = ModelConfig(
        version='1.0.0',
        input_dim=9,
        hidden_dim=128,
        output_dim=1,
        dropout_rate=0.3,
        n_heads=4,
        feature_layers=3,
        use_mixed_precision=False,
        batch_size=64,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        epochs=20
    )
    
    model = JambaThreatModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    checkpoint = torch.load('models/model_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate test data and create data loader
    data = generate_sample_data()
    X_test, y_test = data['test']
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test.values),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Get predictions
    predictions, true_labels = get_predictions(model, test_loader, device)
    
    # Calculate raw prediction scores for PR curve
    raw_scores = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            raw_scores.extend(scores.cpu().numpy())
    raw_scores = np.array(raw_scores)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    # Calculate and print F1 score
    f1 = f1_score(true_labels, predictions, average='binary')
    print(f"\nF1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, classes=['Not Threat', 'Threat'])
    
    # Plot precision-recall curve
    plot_precision_recall_curve(true_labels, raw_scores)
    
    # Print additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

if __name__ == '__main__':
    analyze_predictions() 