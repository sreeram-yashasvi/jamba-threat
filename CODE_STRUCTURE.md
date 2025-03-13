# Jamba Threat Detection - Code Structure

This document outlines the structure and organization of the Jamba Threat Detection codebase.

## Directory Structure

```
jamba-threat/
├── models/                 # Directory for storing trained models
├── logs/                   # Log files directory
├── README.md               # Project documentation
├── Dockerfile              # Container definition for RunPod deployment
├── src/                    # Main source code directory
│   ├── jamba_model.py      # Core model definition and dataset classes
│   ├── handler.py          # RunPod serverless handler implementation
│   ├── train_runpod.py     # RunPod training implementation
│   ├── run_training.py     # CLI for RunPod training
│   ├── README_RUNPOD_TRAINING.md # Documentation for RunPod training
│   ├── runpod_entry.sh     # Container startup script for RunPod
│   ├── runpod_health_check.py # Health check utility for RunPod deployments
│   ├── runpod_verify.py    # Verification script for RunPod deployments
│   ├── runpod_client.py    # Client for interacting with RunPod endpoints
│   ├── train_with_runpod.py # Legacy script for training models using RunPod
│   ├── train_jamba.py      # Local training script
│   ├── model_training.py   # Model training utilities
│   └── utils/              # Shared utility modules
│       ├── __init__.py     # Package initialization
│       ├── environment.py  # Environment setup and configuration
│       ├── validation.py   # Model and endpoint validation utilities
│       ├── cli.py          # Command-line interface utilities
│       └── shell.py        # Shell and subprocess utilities
```

## Core Components

### Model Implementation

The core threat detection model is implemented in `jamba_model.py`, which defines:

- `JambaThreatModel`: PyTorch neural network for threat detection
- `ThreatDataset`: Custom dataset class for handling threat data

### RunPod Integration

The following files handle the RunPod serverless integration:

- `handler.py`: Implements the RunPod serverless handler for inference and training
- `runpod_entry.sh`: Entry point script for the RunPod container
- `runpod_client.py`: Client for interacting with RunPod endpoints

### RunPod Training System

The new RunPod training system consists of:

- `train_runpod.py`: Core implementation of the `RunPodTrainer` class that handles:
  - Data preparation and chunking for large datasets
  - Training job submission to RunPod endpoints
  - Job status monitoring and result retrieval
  - Model saving and metrics tracking

- `run_training.py`: User-friendly command-line interface for RunPod training with:
  - Simplified parameter configuration
  - Support for configuration files
  - Progress reporting and error handling

- `README_RUNPOD_TRAINING.md`: Detailed documentation for using the RunPod training system

### Training Scripts

- `train_runpod.py`: Primary script for training models using RunPod GPUs
- `train_with_runpod.py`: Legacy script for basic RunPod training
- `train_jamba.py`: Script for local training
- `model_training.py`: Utilities for model training

### Utility Modules

The `utils` package contains shared functionality:

- `environment.py`: Environment setup, paths, and configuration management
- `validation.py`: Model verification, health checks, and validation
- `cli.py`: Command-line interface utilities
- `shell.py`: Shell command execution utilities

### Diagnostic Tools

- `runpod_health_check.py`: Script to verify RunPod endpoint health
- `runpod_verify.py`: Script to verify model compatibility with RunPod deployment

## Architecture Overview

The Jamba Threat Detection system follows a modular architecture:

1. **Model Layer**: The core neural network implementation (`JambaThreatModel`)
2. **Serving Layer**: RunPod serverless handler and container configuration
3. **Training Layer**: RunPod training system with local and remote options
4. **Client Layer**: RunPod client for remote interaction
5. **Utility Layer**: Shared functions for environment setup and validation

### Data Flow

1. The model is trained either locally or on RunPod using training scripts
2. For RunPod training:
   - Local data is prepared and potentially chunked
   - Data is sent to RunPod for GPU training
   - Training progress is monitored with status checks
   - Trained model is retrieved and saved locally
3. Trained models are stored in the `models` directory
4. The RunPod handler loads the model to serve predictions
5. Clients interact with the deployed model via the RunPod endpoint

## RunPod Training Flow

The RunPod training system follows this process flow:

1. **Data Preparation**: Local training data is read and processed
2. **Size Assessment**: System determines if data exceeds RunPod's size limits
3. **Data Chunking**: Large datasets are split into manageable chunks
4. **Job Submission**: Training job is sent to the RunPod endpoint
5. **Status Monitoring**: System periodically checks job status
6. **Result Retrieval**: Trained model and metrics are downloaded when complete
7. **Model Saving**: Model is saved locally in the specified format

## Environment Configuration

The system relies on several environment variables:

- `RUNPOD_API_KEY`: API key for accessing RunPod services
- `RUNPOD_ENDPOINT_ID`: ID of the RunPod endpoint
- `MODEL_DIR`: Directory for storing models (default: `/app/models`)
- `LOGS_DIR`: Directory for storing logs (default: `/app/logs`)
- `DEBUG_MODE`: Enable debug logging (default: `false`)

## Verification and Health Checks

The system includes comprehensive verification and health check capabilities:

1. **Model Structure Verification**: Ensures the model can be initialized
2. **Serialization Testing**: Verifies model can be properly serialized/deserialized
3. **Endpoint Health Checks**: Confirms the RunPod endpoint is operational
4. **Environment Validation**: Ensures all required variables are set

## Best Practices

When working with this codebase:

1. Use the utility modules for environment setup and validation
2. Follow the established error handling patterns
3. Add proper logging for all operations
4. Run verification scripts before deployment
5. For large training datasets, use the chunking capabilities of `RunPodTrainer`
6. Keep the model implementation consistent across all files 