import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, Any, List
from pathlib import Path

class TrainingTracker:
    """Track and visualize training runs with different parameters."""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.log_dir / "training_runs.csv"
        self.metrics_dir = self.log_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir = self.log_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize or load runs DataFrame
        if self.runs_file.exists():
            self.runs_df = pd.read_csv(self.runs_file)
        else:
            self.runs_df = pd.DataFrame(columns=[
                'run_id', 'timestamp', 'config', 'final_accuracy', 
                'final_f1', 'best_val_loss', 'epochs_trained',
                'training_time', 'early_stopped'
            ])
    
    def log_run(self, 
                config: Dict[str, Any],
                metrics: Dict[str, float],
                training_history: Dict[str, List[float]],
                run_id: str = None) -> str:
        """Log a training run with its configuration and results."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Save run metrics
        run_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'config': json.dumps(config),
            'final_accuracy': metrics.get('accuracy', 0.0),
            'final_f1': metrics.get('f1_score', 0.0),
            'best_val_loss': metrics.get('best_val_loss', float('inf')),
            'epochs_trained': metrics.get('epochs_trained', 0),
            'training_time': metrics.get('training_time', 0.0),
            'early_stopped': metrics.get('early_stopped', False)
        }
        
        # Append to DataFrame
        self.runs_df = pd.concat([self.runs_df, pd.DataFrame([run_data])], ignore_index=True)
        self.runs_df.to_csv(self.runs_file, index=False)
        
        # Save detailed metrics
        metrics_file = self.metrics_dir / f"{run_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'config': config,
                'metrics': metrics,
                'history': training_history
            }, f, indent=2)
            
        return run_id
    
    def plot_run_comparison(self, metric: str = 'accuracy', save: bool = True) -> None:
        """Plot comparison of different runs based on specified metric."""
        plt.figure(figsize=(12, 6))
        
        # Extract configurations for comparison
        self.runs_df['config_dict'] = self.runs_df['config'].apply(json.loads)
        key_params = ['learning_rate', 'batch_size', 'n_heads', 'feature_layers']
        
        # Create parameter combination string
        def get_param_str(config):
            return "_".join(f"{k}={config.get(k, 'default')}" 
                          for k in key_params)
        
        self.runs_df['params'] = self.runs_df['config_dict'].apply(get_param_str)
        
        # Plot
        sns.barplot(data=self.runs_df, x='params', y=f'final_{metric}')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Comparison of {metric.capitalize()} Across Different Configurations')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / f"{metric}_comparison.png")
        else:
            plt.show()
            
    def plot_learning_curves(self, run_id: str, save: bool = True) -> None:
        """Plot learning curves for a specific run."""
        metrics_file = self.metrics_dir / f"{run_id}_metrics.json"
        if not metrics_file.exists():
            raise ValueError(f"No metrics found for run {run_id}")
            
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        history = data['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(history['train_acc'], label='Training Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / f"{run_id}_learning_curves.png")
        else:
            plt.show()
            
    def get_best_runs(self, metric: str = 'accuracy', top_n: int = 5) -> pd.DataFrame:
        """Get the top N runs based on specified metric."""
        return self.runs_df.nlargest(top_n, f'final_{metric}')
    
    def analyze_parameter_impact(self) -> pd.DataFrame:
        """Analyze the impact of different parameters on model performance."""
        self.runs_df['config_dict'] = self.runs_df['config'].apply(json.loads)
        
        # Extract key parameters
        params = ['learning_rate', 'batch_size', 'n_heads', 'feature_layers', 
                 'dropout_rate', 'hidden_dim']
        
        results = []
        for param in params:
            param_values = self.runs_df['config_dict'].apply(lambda x: x.get(param, None))
            if param_values.nunique() > 1:  # Only analyze if parameter varies
                avg_acc = self.runs_df.groupby(param_values)['final_accuracy'].mean()
                std_acc = self.runs_df.groupby(param_values)['final_accuracy'].std()
                
                results.append({
                    'parameter': param,
                    'best_value': avg_acc.idxmax(),
                    'best_accuracy': avg_acc.max(),
                    'impact_std': std_acc.mean(),
                    'n_trials': len(param_values.unique())
                })
                
        return pd.DataFrame(results) 