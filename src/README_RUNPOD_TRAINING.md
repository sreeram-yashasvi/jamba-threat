# Jamba Threat Model Training with RunPod

This README provides instructions on how to train the Jamba Threat model using RunPod's GPU infrastructure.

## Overview

The Jamba Threat Detection model can be trained using RunPod's GPU infrastructure, which provides scalable and cost-effective access to high-performance GPUs. The training process is handled by sending a training job to a RunPod serverless endpoint, which processes the data and returns the trained model.

## Prerequisites

Before you can train the model using RunPod, you need:

1. A RunPod account with API access
2. A deployed serverless endpoint running the Jamba Threat handler
3. Your RunPod API key
4. Your endpoint ID

## Setup

### Environment Variables

Set up your environment variables:

```bash
export RUNPOD_API_KEY="your_api_key_here"
export RUNPOD_ENDPOINT_ID="your_endpoint_id_here"
```

Alternatively, you can provide these values as command-line arguments when running the training script.

### Data Preparation

Prepare your training data in CSV or Parquet format. The data should include the features used for training and a target column indicating whether each sample is a threat.

## Running Training

### Using the Command-Line Interface

The simplest way to train the model is using the `run_training.py` script:

```bash
python src/run_training.py --data-path path/to/your/data.csv --output-model models/jamba_model.pth
```

### Command-Line Arguments

- `--data-path`: Path to your training data (CSV or Parquet)
- `--output-model`: Path where the trained model will be saved
- `--config-file`: (Optional) Path to a JSON file with training parameters
- `--epochs`: Number of training epochs (default: 30)
- `--learning-rate`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 128)
- `--api-key`: (Optional) RunPod API key (defaults to environment variable)
- `--endpoint-id`: (Optional) RunPod endpoint ID (defaults to environment variable)

### Using a Configuration File

You can also provide training parameters in a JSON configuration file:

```json
{
  "epochs": 50,
  "learning_rate": 0.0005,
  "batch_size": 256,
  "target_column": "is_threat"
}
```

And run the training with:

```bash
python src/run_training.py --data-path path/to/data.csv --config-file path/to/config.json
```

## Advanced Usage

### Using the RunPodTrainer Directly

You can also use the `RunPodTrainer` class directly in your Python code:

```python
from src.train_runpod import RunPodTrainer
import pandas as pd

# Initialize the trainer
trainer = RunPodTrainer(api_key="your_api_key", endpoint_id="your_endpoint_id")

# Prepare the data
data = trainer.prepare_data("path/to/data.csv")

# Define training parameters
params = {
    "epochs": 30,
    "learning_rate": 0.001,
    "batch_size": 128
}

# Submit the training job
job_result = trainer.submit_training_job(data, params)
job_id = job_result["id"]

# Wait for the job to complete and get results
result = trainer.wait_for_completion(job_id)

# Save the model
trainer.save_model(result, "models/jamba_model.pth")
```

## Monitoring Training

The training script will log progress updates including:

1. Job submission confirmation
2. Regular status checks
3. Completion notification with training metrics

If the job completes successfully, the model will be saved to the specified output path, along with a JSON file containing training metrics.

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure your API key is correct and has not expired
2. **Endpoint Not Found**: Verify your endpoint ID and make sure the endpoint is running
3. **Timeout Errors**: For large datasets, increase the timeout value using the `--timeout` parameter

### Logs

Review the logs for detailed information about the training process. If training fails, the logs will include error messages that can help identify the issue.

## Additional Resources

- [RunPod Documentation](https://docs.runpod.io/docs)
- [Jamba Threat Model Documentation](link_to_your_model_docs) 