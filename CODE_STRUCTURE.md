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
│   ├── runpod_entry.sh     # Container startup script for RunPod
│   ├── runpod_health_check.py # Health check utility for RunPod deployments
│   ├── runpod_verify.py    # Verification script for RunPod deployments
│   ├── runpod_client.py    # Client for interacting with RunPod endpoints
│   ├── train_with_runpod.py # Script for training models using RunPod
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

### Training Scripts

- `train_with_runpod.py`: Script for training models using RunPod
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
3. **Client Layer**: RunPod client for remote interaction
4. **Utility Layer**: Shared functions for environment setup and validation

### Data Flow

1. The model is trained either locally or on RunPod using training scripts
2. Trained models are stored in the `models` directory
3. The RunPod handler loads the model to serve predictions
4. Clients interact with the deployed model via the RunPod endpoint

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
5. Keep the model implementation consistent across all files 