# Jamba Threat Detection Model

A deep learning-based threat detection system using advanced neural network architectures and robust data processing pipelines.

## Overview

The Jamba Threat Detection Model is designed to identify security threats using a combination of system metrics and network behavior patterns. The system employs a sophisticated neural network architecture with self-attention mechanisms and provides comprehensive tooling for training, evaluation, and deployment.

## Project Structure

### Core Components

#### Model and Training (`src/jamba/`)
- **`jamba_model.py`**
  - Purpose: Defines the core threat detection model architecture
  - Dependencies: `torch`, `numpy`, `logging`
  - Key features: Multi-head attention, feature processing layers, classification head

- **`model_config.py`**
  - Purpose: Manages model configuration and parameter validation
  - Dependencies: `dataclasses`, `typing`
  - Key features: Configuration validation, version compatibility checks

- **`train.py`**
  - Purpose: Main training script with command-line interface
  - Dependencies: `torch`, `argparse`, `pandas`
  - Features: Training loop, early stopping, checkpointing

### Data Processing

- **`data_preprocessing.py`**
  - Purpose: Data preprocessing and feature engineering
  - Dependencies: `pandas`, `numpy`, `sklearn.preprocessing`
  - Features: 
    - StandardScaler for numerical features
    - LabelEncoder for categorical features
    - Missing value handling
    - Data validation

### Utilities (`src/jamba/utils/`)

- **`dataset_generator.py`**
  - Purpose: Generates balanced datasets for training
  - Dependencies: `numpy`, `pandas`
  - Features: 
    - Configurable dataset size
    - Balanced class distribution
    - Feature engineering

- **`training_tracker.py`**
  - Purpose: Tracks and visualizes training progress
  - Dependencies: `pandas`, `matplotlib`, `seaborn`
  - Features:
    - Training metrics logging
    - Performance visualization
    - Parameter impact analysis

- **`runpod_utils.py`**
  - Purpose: RunPod integration utilities
  - Dependencies: `subprocess`, `sys`, `os`
  - Features:
    - RunPod installation
    - Environment setup
    - Path management

### Analysis Tools (`src/jamba/analysis/`)

- **`analyze_codebase.py`**
  - Purpose: Analyzes code complexity metrics
  - Dependencies: `os`, `json`
  - Features:
    - Code complexity analysis
    - Metrics generation
    - Report creation

- **`code_complexity.py`**
  - Purpose: Calculates code complexity metrics
  - Dependencies: `ast`, `typing`
  - Features:
    - Halstead metrics
    - Cyclomatic complexity
    - Code analysis utilities

- **`fault_tree.py`**
  - Purpose: Implements fault tree analysis
  - Dependencies: `networkx`, `pandas`
  - Features:
    - Fault tree construction
    - Risk analysis
    - Report generation

### Experiments

- **`run_experiments.py`**
  - Purpose: Runs training experiments with different configurations
  - Dependencies: `torch`, `pandas`, `logging`
  - Features:
    - Multiple configuration testing
    - Performance comparison
    - Results logging

### Command Line Interface

- **`cli.py`**
  - Purpose: Command-line interface for the system
  - Dependencies: `argparse`
  - Features:
    - Training command
    - Model evaluation
    - Configuration management

## Dependencies

### Core Dependencies
- Python 3.8+
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0

### Cloud and Storage Dependencies
- Azure Storage Blob >= 12.0.0
- Requests >= 2.26.0
- RunPod >= 0.10.0 (for cloud training)

### Visualization Dependencies
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

### Analysis Dependencies
- NetworkX >= 2.6.0 (for fault tree analysis)
- AST (built-in, for code analysis)

### Development Dependencies
- Black (code formatting)
- Pytest (testing)
- Mypy (type checking)
- Jupyter (notebooks and visualization)

### Optional Dependencies
- CUDA >= 11.0 (for GPU acceleration)
- TensorBoard >= 2.6.0 (for training visualization)
- Ray >= 1.0.0 (for distributed training)

## Features

### 1. Model Architecture
- Multi-head self-attention mechanism for capturing complex feature interactions
- Progressive dimension reduction through feature processing layers
- Batch normalization and adaptive dropout for training stability
- SiLU (Swish) activation functions for improved gradient flow
- Configurable architecture parameters (hidden dimensions, number of heads, layers)

### 2. Dataset Generation
The system includes a sophisticated dataset generator that creates balanced, realistic threat detection datasets:

- **Balanced Class Distribution**: Configurable threat ratio (default 50-50 split)
- **Feature Engineering**:
  - Temporal correlation through exponential moving averages
  - Controlled noise injection for robustness
  - Distinct patterns for normal and threat samples
  - Automatic feature scaling and normalization
- **Data Characteristics**:
  - Configurable dataset size (default 35,000 samples)
  - 20 engineered features with meaningful patterns
  - Automatic train/validation/test splitting (70/15/15)

### 3. Training Pipeline
- **Experiment Tracking**:
  - Comprehensive logging of training metrics
  - Visualization of learning curves
  - Parameter impact analysis
  - Automated early stopping
  
- **Training Features**:
  - Mixed precision training support
  - Configurable batch sizes and learning rates
  - Multi-head attention optimization
  - Progressive learning rate scheduling

### 4. Performance Monitoring
The system includes a sophisticated training tracker that provides:
- Real-time visualization of training progress
- Comparative analysis of different model configurations
- Parameter impact assessment
- Best model selection based on multiple metrics

## Model Configurations

### Default Configuration
```python
{
    'version': '1.0.0',
    'input_dim': 20,
    'hidden_dim': 64,
    'output_dim': 1,
    'dropout_rate': 0.3,
    'n_heads': 4,
    'feature_layers': 2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 30
}
```

### Performance Metrics
Based on extensive experiments with different configurations:
- Validation Accuracy: 99.9%+ across configurations
- Early convergence (typically within 15-25 epochs)
- Robust performance across different batch sizes (32-256)
- Stable training with various learning rates (0.0001-0.001)

## Usage

### 1. Dataset Generation
```python
from jamba.utils.dataset_generator import generate_balanced_dataset

# Generate a balanced dataset
X, y = generate_balanced_dataset(
    n_samples=35000,
    n_features=20,
    threat_ratio=0.5,
    random_state=42
)
```

### 2. Training Experiments
```python
from jamba.utils.training_tracker import TrainingTracker
from jamba.model import JambaThreatModel
from jamba.config import ModelConfig

# Initialize training tracker
tracker = TrainingTracker(log_dir="experiments/training_logs")

# Create and train model
config = ModelConfig(
    input_dim=20,
    hidden_dim=64,
    output_dim=1,
    dropout_rate=0.3,
    epochs=30
)

model = JambaThreatModel(config)
metrics, history = train_model(model, train_loader, val_loader, config)

# Log and visualize results
tracker.log_run(config=config.__dict__, metrics=metrics, history=history)
tracker.plot_run_comparison(metric='accuracy')
tracker.plot_learning_curves(run_id)
```

### 3. Analysis and Visualization
```python
# Get parameter impact analysis
impact_analysis = tracker.analyze_parameter_impact()

# Get best performing configurations
best_runs = tracker.get_best_runs(metric='accuracy', top_n=3)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jamba-threat.git
cd jamba-threat
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install optional dependencies (if needed):
```bash
pip install -r requirements-optional.txt
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Testing

### Test Suite Structure

#### Unit Tests (`src/jamba/tests/`)
- **`test_regression.py`**
  - Tests model training and prediction
  - Validates loss calculation
  - Checks gradient flow
  - Verifies model checkpointing

- **`test_fault_tree.py`**
  - Tests fault tree construction
  - Validates risk calculations
  - Checks report generation
  - Verifies tree traversal

#### Integration Tests
- **`test_gpu.py`**
  - Validates GPU compatibility
  - Tests CUDA operations
  - Checks memory management
  - Verifies mixed precision training

- **`runpod_startup.py`**
  - Tests RunPod environment setup
  - Validates model imports
  - Checks CUDA availability
  - Tests model initialization

### Running Tests

1. Run all tests:
```bash
python -m pytest src/jamba/tests/
```

2. Run specific test files:
```bash
python -m pytest src/jamba/tests/test_regression.py
python -m pytest src/jamba/tests/test_fault_tree.py
```

3. Run GPU tests:
```bash
python src/test_gpu.py
```

4. Run RunPod environment tests:
```bash
python src/runpod_startup.py --check-environment
```

### Test Coverage

To generate a test coverage report:
```bash
coverage run -m pytest src/jamba/tests/
coverage report
coverage html  # Generates HTML report
```

### Continuous Integration

The project uses GitHub Actions for CI/CD:
- Runs test suite on each push
- Validates code formatting
- Checks type hints
- Generates coverage reports

## Data Structure

### Directory Layout
```
jamba-threat/
├── data/
│   ├── raw/                 # Raw input data
│   ├── processed/           # Preprocessed datasets
│   ├── balanced/           # Balanced datasets
│   └── experiments/        # Experimental results
├── models/                 # Saved model checkpoints
├── logs/                  # Training and error logs
└── experiments/
    ├── training_logs/     # Experiment tracking
    ├── metrics/          # Performance metrics
    └── plots/            # Visualization plots
```

### Data Formats

#### Input Data
- **Raw Data Format**: CSV or Parquet files with the following structure:
```python
  {
      'timestamp': datetime64[ns],
      'feature_1': float64,
      'feature_2': float64,
      ...,
      'feature_n': float64,
      'is_threat': int32  # Target variable (0 or 1)
  }
  ```

#### Processed Data
- **Training Data**: Preprocessed and scaled features
- **Validation Data**: Held-out data for model validation
- **Test Data**: Separate data for final evaluation
- File Format: `.pt` (PyTorch tensors) or `.npz` (NumPy arrays)

#### Model Checkpoints
- Format: `.pt` files containing:
  ```python
  {
      'epoch': int,
      'model_state': dict,
      'optimizer_state': dict,
      'scheduler_state': dict,
      'config': dict,
      'metrics': dict
  }
  ```

#### Experiment Logs
- **Training Logs**: CSV files with metrics per epoch
- **Configuration**: JSON files with experiment parameters
- **Visualizations**: PNG/PDF files with performance plots

### Data Processing Pipeline

1. **Raw Data**
   - Load from CSV/Parquet
   - Validate schema
   - Check for missing values

2. **Preprocessing**
   - Scale numerical features
   - Encode categorical variables
   - Handle missing data
   - Apply feature engineering

3. **Dataset Creation**
   - Generate balanced datasets
   - Split into train/val/test
   - Create PyTorch datasets
   - Apply data augmentation

4. **Experiment Tracking**
   - Log training metrics
   - Save model checkpoints
   - Generate visualizations
   - Track parameter impacts 

## GGUF Model Support

The Jamba Threat Detection Model now supports conversion to GGUF format for efficient deployment and inference. GGUF (GPT-Generated Unified Format) provides several advantages:

- Reduced model size through quantization
- Faster inference speed
- Lower memory usage
- Better compatibility with deployment platforms

### Converting Models to GGUF

To convert a trained PyTorch model to GGUF format, use the provided conversion script:

```bash
python -m src.jamba.scripts.convert_model \
    --input-model /path/to/model.pt \
    --model-name my_model \
    --quantization q4_k_m
```

Available quantization options:

- `q4_k_m`: 4-bit quantization with K-means clustering (highest compression)
- `q5_k_m`: 5-bit quantization with K-means clustering (balanced)
- `q8_0`: 8-bit linear quantization (highest quality)

To view detailed information about quantization types:

```bash
python -m src.jamba.scripts.convert_model --show-info
```

### Using GGUF Models

GGUF models can be loaded and used for inference using the `ModelConverter` class:

```python
from jamba.utils.model_converter import ModelConverter

converter = ModelConverter()
model = converter.load_gguf_model("/path/to/model.gguf")
predictions = model(input_data)
```

### Directory Structure

GGUF models are stored in the following directory structure:

```
models/
├── gguf/
│   ├── q4_k_m/
│   │   └── model_name.gguf
│   ├── q5_k_m/
│   │   └── model_name.gguf
│   └── q8_0/
│       └── model_name.gguf
└── pytorch/
    └── model_name.pt
``` 