# Jamba Threat Detection

A machine learning system for detecting threats in cybersecurity intelligence feeds using the Jamba language model.

## Overview

This project fine-tunes AI21's Jamba model (or alternative models like DistilBERT) to classify cyber threat intelligence data. The system processes structured data from threat feeds, converts it into a format suitable for language models, and trains a classifier to identify genuine threats.

## Features

- Preprocessing of threat intelligence data
- Fine-tuning of state-of-the-art language models
- Memory-optimized training for large models
- GPU acceleration with advanced features like mixed precision and 8-bit quantization
- Comprehensive logging and evaluation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-compatible GPU (recommended for large models)
- Hugging Face account with access token (for gated models like Jamba)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/jamba-threat.git
cd jamba-threat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token (for accessing gated models):
```bash
export HF_TOKEN=your_token_here
```
or login via CLI:
```bash
huggingface-cli login
```

## Deployment on RunPod.io

RunPod.io provides a range of GPU options that are ideal for training large language models like Jamba.

### 1. Selecting a RunPod GPU

RunPod offers various GPU configurations. For this project, we recommend:

- **L4 (24GB)**: Good cost-effective option for smaller models
- **A10G (24GB)**: Good balance of performance and cost
- **A40 (48GB)** or **A5000 (24GB)**: Better for larger models with memory offloading
- **A100 (40GB/80GB)**: Optimal for the full Jamba model with fast training
- **H100 (80GB)**: Highest performance for large models (if budget allows)

### 2. Starting a RunPod Instance

1. Create an account on [RunPod.io](https://www.runpod.io/)
2. Select "Deploy" and choose a GPU type from the available options
3. Select a template (PyTorch or Tensorflow templates work well)
4. Deploy your pod and connect via SSH or JupyterLab

### 3. Automated Deployment

We provide an automated deployment script for RunPod instances:

```bash
# Clone the repository
git clone https://github.com/yourusername/jamba-threat.git
cd jamba-threat

# Make the deployment script executable
chmod +x deploy_runpod.sh

# Run the deployment script
./deploy_runpod.sh
```

The script will:
- Auto-detect your RunPod GPU type
- Install appropriate CUDA optimizations
- Configure the environment for best performance
- Install all required dependencies
- Configure Hugging Face authentication
- Provide recommended training commands for your specific GPU

### 4. Running Training on RunPod

Our deployment script will automatically suggest the optimal configuration for your specific GPU. Here are some examples:

#### For RunPod A100 (40/80GB):
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model ai21labs/AI21-Jamba-1.5-Mini \
  --batch-size 24 \
  --bf16 \
  --log-memory
```

#### For RunPod H100:
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model ai21labs/AI21-Jamba-1.5-Mini \
  --batch-size 32 \
  --bf16 \
  --log-memory
```

#### For RunPod L4:
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model distilbert-base-uncased \
  --batch-size 64 \
  --fp16 \
  --log-memory
```

#### For Jamba on RunPod L4:
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model ai21labs/AI21-Jamba-1.5-Mini \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --fp16 \
  --log-memory
```

### 5. RunPod-Specific Optimizations

- **Shared Memory**: The deployment script increases shared memory allocation for better dataloader performance
- **Flash Attention**: Automatically installs flash-attention for compatible GPUs
- **CUDA Version Detection**: Optimizes Triton installation based on your specific CUDA version
- **TensorBoard Access**: Use the `--bind_all` flag with TensorBoard to access through RunPod's port forwarding

### 6. Saving Your Work on RunPod

RunPod instances are ephemeral. To persist your work:

1. Save your trained models to the `/workspace` directory (which is persistent)
2. Or use RunPod's volume feature to attach persistent storage
3. Push your trained models to Hugging Face Hub:
```bash
# Push your model to Hugging Face Hub
python src/push_to_hub.py --model-path models/jamba-threat/jamba-threat-final-TIMESTAMP --repo-name your-username/jamba-threat
```

## Deployment on Hetzner GPU Servers

### 1. Selecting a Hetzner GPU Server

Hetzner offers several GPU server configurations. For this project, we recommend:

- **CCX13** (RTX A4000): Good balance of cost and performance
- **CCX33** (RTX A5000): Better for larger models with memory offloading
- **CCX63** (RTX A6000): Recommended for the full Jamba model without compromises

### 2. Server Setup

After provisioning your Hetzner server, connect via SSH and set up the environment:

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv git nvidia-driver-535 nvidia-utils-535

# Verify NVIDIA installation
nvidia-smi
```

### 3. Automated Deployment

We provide an automated deployment script for Hetzner GPU servers:

```bash
# Clone the repository
git clone https://github.com/yourusername/jamba-threat.git
cd jamba-threat

# Make the deployment script executable
chmod +x deploy_hetzner.sh

# Run the deployment script
./deploy_hetzner.sh
```

The script will:
- Verify your system has GPU drivers
- Set up a Python virtual environment
- Install all required dependencies
- Configure Hugging Face authentication
- Create necessary directories
- Provide example commands for training

### 4. Running the Training

The script provides examples, but here are the recommended configurations for different Hetzner GPU models:

#### For RTX A4000 (16GB VRAM):
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model distilbert-base-uncased \
  --batch-size 32 \
  --fp16 \
  --log-memory
```

#### For RTX A5000 (24GB VRAM):
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model ai21labs/AI21-Jamba-1.5-Mini \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --fp16 \
  --log-memory
```

#### For RTX A6000 (48GB VRAM):
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model ai21labs/AI21-Jamba-1.5-Mini \
  --batch-size 16 \
  --fp16 \
  --log-memory
```

#### For A100 (80GB VRAM):
```bash
python src/train_jamba.py \
  --data data/threat_intel_feed.csv \
  --model ai21labs/AI21-Jamba-1.5-Mini \
  --batch-size 24 \
  --bf16 \
  --log-memory
```

### 5. Memory Optimization Techniques

The system supports several memory optimization techniques:

- **FP16 Precision** (`--fp16`): Uses half-precision floating point to reduce memory usage
- **BF16 Precision** (`--bf16`): Better for newer GPUs like A100
- **Gradient Accumulation** (`--gradient-accumulation-steps`): Simulates larger batch sizes
- **8-bit Quantization** (`--use-8bit`): Further reduces memory usage for very large models
- **CPU Offloading** (`--offload-to-cpu`): Moves some model layers to CPU
- **Smaller Model Fallback** (`--smaller-model-fallback`): Automatically switches to a smaller model if needed

### 6. Monitoring Training

You can monitor the training process with TensorBoard:

```bash
# In another terminal
pip install tensorboard
tensorboard --logdir models/jamba-threat/logs
```

Then access TensorBoard in your browser at `http://your-server-ip:6006`

## Using Your Trained Model

After training, your model will be saved in the `models/jamba-threat` directory. You can use it for inference with:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "models/jamba-threat/jamba-threat-final-TIMESTAMP"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example text to classify
text = "Source: ThreatPost | Threat Type: Malware | Actor: APT29 | Confidence Score: 0.95 | Description: New variant of CobaltStrike observed in the wild"

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
print(f"Prediction: {'Threat' if prediction == 1 else 'Benign'}")
```

## Troubleshooting

### Common Issues on RunPod

1. **Shared Memory Errors**: Try running with fewer dataloader workers:
```bash
python src/train_jamba.py --dataloader-num-workers 1
```

2. **Flash Attention Installation Issues**: If flash-attention installation fails, try without it:
```bash
python src/train_jamba.py --use-flash-attn false
```

3. **Port Forwarding**: Access TensorBoard by setting up port forwarding in the RunPod dashboard or use the public URL feature.

4. **Model Size Errors**: For very large models on smaller GPUs, try:
```bash
python src/train_jamba.py --batch-size 1 --gradient-accumulation-steps 32 --use-8bit --offload-to-cpu
```

5. **Persistent Storage**: If your pod restarts, mount a persistent volume to `/workspace` to keep your models.

### Common Issues on Hetzner

1. **CUDA Out of Memory**: Reduce batch size or enable memory optimizations:
```bash
python src/train_jamba.py --batch-size 4 --gradient-accumulation-steps 8 --fp16 --offload-to-cpu
```

2. **Model Too Large**: Try using 8-bit quantization:
```bash
python src/train_jamba.py --use-8bit --offload-to-cpu
```

3. **Hugging Face Authentication**: Make sure your token has access to gated models:
```bash
huggingface-cli whoami
```

4. **Slow Training**: Enable more workers for dataloading:
```bash
python src/train_jamba.py --dataloader-num-workers 4
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AI21 Labs for the Jamba model
- Hugging Face for the Transformers library
- The open-source community for tools and libraries 