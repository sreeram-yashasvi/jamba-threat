# Jamba Threat Model Training with RunPod

This guide provides detailed instructions on how to train the Jamba Threat model using RunPod's GPU infrastructure.

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
- `--timeout`: (Optional) Maximum time to wait for job completion in seconds (default: 3600)

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

## How it Works

The RunPod training process works through the following steps:

1. **Data Preparation**: The local data is read and preprocessed
2. **Size Assessment**: The system checks if the data exceeds RunPod's 10MiB request size limit
3. **Data Chunking**: If necessary, the data is split into manageable chunks
4. **Job Submission**: The data and training parameters are sent to the RunPod endpoint
5. **Status Monitoring**: The system periodically checks the job status
6. **Result Retrieval**: When complete, the trained model and metrics are downloaded
7. **Model Saving**: The model is saved locally in the specified format

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

### Handling Large Datasets

For datasets exceeding RunPod's request size limit:

```python
# For very large datasets, you can customize the chunking behavior
trainer = RunPodTrainer(api_key="your_api_key", endpoint_id="your_endpoint_id")
data = trainer.prepare_data("path/to/large_data.csv")

# The submission will automatically handle chunking
result = trainer.submit_training_job(data, params)
```

## Monitoring Training

The training script will log progress updates including:

1. Job submission confirmation
2. Regular status checks
3. Completion notification with training metrics

The logs will show:
- Epoch-by-epoch progress (as reported by the endpoint)
- Accuracy metrics during training
- Final model performance statistics

If the job completes successfully, the model will be saved to the specified output path, along with a JSON file containing training metrics.

## Troubleshooting

### Common Issues

1. **Authentication Errors**: 
   - Error: "Invalid API key" or "Unauthorized"
   - Solution: Verify your API key is correct and not expired

2. **Endpoint Not Found**:
   - Error: "Endpoint not found" or "Invalid endpoint ID"
   - Solution: Check your endpoint ID and ensure the endpoint is running

3. **Request Size Limit**:
   - Error: "Payload too large" or similar
   - Solution: The system should automatically handle this, but you might need to reduce your dataset size

4. **Timeout Errors**:
   - Error: "Job timed out after X seconds"
   - Solution: Increase timeout with `--timeout` parameter

5. **GPU Memory Errors**:
   - Error: "CUDA out of memory"
   - Solution: Reduce batch size or number of epochs

### Verifying Endpoint Health

You can use the included health check script to verify your endpoint is working:

```bash
python src/runpod_health_check.py
```

### Logs

Review the logs for detailed information about the training process. If training fails, the logs will include error messages that can help identify the issue.

## Best Practices

1. **Start Small**: Begin with a smaller dataset to verify everything works
2. **Batch Size**: Adjust batch size based on your GPU's memory capacity
3. **Save Configurations**: Keep track of successful training configurations
4. **Monitor Costs**: RunPod charges based on GPU usage time

## Additional Resources

- [RunPod Documentation](https://docs.runpod.io/docs)
- [Jamba Threat Model Documentation](https://github.com/yourusername/jamba-threat)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 