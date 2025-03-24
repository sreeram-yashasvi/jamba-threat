import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(size):
    """Load metrics for a specific dataset size."""
    metrics_file = os.path.join('models', f'size_{size}', 'metrics.json')
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_training_comparison():
    """Create comparison plots for different training runs."""
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Results Comparison Across Dataset Sizes', fontsize=16)
    
    # Dataset sizes
    sizes = [45000, 70000, 90000]
    
    # Load metrics for each size
    for size in sizes:
        metrics = load_metrics(size)
        
        # Plot training loss
        ax1.plot(metrics['train_loss'], label=f'{size:,} samples')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot validation loss
        ax2.plot(metrics['val_loss'], label=f'{size:,} samples')
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot validation accuracy
        ax3.plot(metrics['val_accuracy'], label=f'{size:,} samples')
        ax3.set_title('Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True)
        
        # Plot validation F1 score
        ax4.plot(metrics['val_f1'], label=f'{size:,} samples')
        ax4.set_title('Validation F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final metrics
    print("\nFinal Metrics for Each Dataset Size:")
    print("-" * 60)
    print(f"{'Size':>10} {'Accuracy':>12} {'F1 Score':>12} {'Time (s)':>12}")
    print("-" * 60)
    
    # Load all metrics
    with open('models/all_metrics.json', 'r') as f:
        all_metrics = json.load(f)
    
    for size in sorted(map(int, all_metrics.keys())):
        metrics = all_metrics[str(size)]
        print(f"{size:>10,} {metrics['accuracy']:>11.2f}% {metrics['f1']:>11.4f} {metrics['training_time']:>11.2f}")

def main():
    """Generate training comparison plots."""
    plot_training_comparison()

if __name__ == "__main__":
    main() 