#!/bin/bash
# Script to set up and deploy the Jamba threat detection model on RunPod.io GPU instances

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting deployment process for Jamba Threat Detection on RunPod.io...${NC}"

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    echo -e "${RED}Error: This script is intended for Linux instances only.${NC}"
    exit 1
fi

# Check NVIDIA GPU availability with more detailed output for RunPod
echo -e "${GREEN}Checking GPU configuration...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: NVIDIA drivers not detected. Please ensure you've selected a GPU pod on RunPod.io.${NC}"
    exit 1
fi

# Display detailed GPU information
echo -e "${GREEN}NVIDIA GPU detected:${NC}"
nvidia-smi
echo ""
echo -e "${GREEN}GPU details:${NC}"
nvidia-smi --query-gpu=name,memory.total,driver_version,gpu_uuid --format=csv,noheader

# Create and activate virtual environment
echo -e "${GREEN}Setting up Python virtual environment...${NC}"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install pip requirements
echo -e "${GREEN}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Install additional NVIDIA-specific packages optimized for RunPod
echo -e "${GREEN}Installing NVIDIA-specific packages for RunPod...${NC}"
pip install nvidia-ml-py
pip install bitsandbytes==0.41.0  # For 8-bit quantization
pip install accelerate>=0.23.0    # For better GPU utilization
pip install flash-attn --no-build-isolation  # For faster attention computation if compatible

# Check CUDA version and install specific optimizations
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo -e "${GREEN}Detected CUDA version: ${CUDA_VERSION}${NC}"

# Install specific optimizations based on CUDA version
if [[ $CUDA_VERSION == 11.* ]]; then
    echo -e "${GREEN}Installing optimizations for CUDA 11.x...${NC}"
    pip install triton==2.0.0
elif [[ $CUDA_VERSION == 12.* ]]; then
    echo -e "${GREEN}Installing optimizations for CUDA 12.x...${NC}"
    pip install triton>=2.1.0
fi

# Check HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN environment variable not set.${NC}"
    echo -e "${YELLOW}If you need to access gated models like Jamba, please set it:${NC}"
    echo -e "${YELLOW}export HF_TOKEN='your_token_here'${NC}"
    echo -e "${YELLOW}Attempting to login via huggingface-cli...${NC}"
    
    # Try huggingface-cli login
    if command -v huggingface-cli &> /dev/null; then
        echo -e "${GREEN}Running huggingface-cli login...${NC}"
        huggingface-cli login
    else
        echo -e "${YELLOW}huggingface-cli not found, installing...${NC}"
        pip install huggingface_hub
        echo -e "${GREEN}Running huggingface-cli login...${NC}"
        huggingface-cli login
    fi
fi

# Create data directory if doesn't exist
if [ ! -d data ]; then
    mkdir -p data
    echo -e "${YELLOW}Created data directory. Please place your threat_intel_feed.csv file in the data directory.${NC}"
fi

# Create models directory
if [ ! -d models ]; then
    mkdir -p models/jamba-threat/logs
    echo -e "${GREEN}Created models directory for storing trained models.${NC}"
fi

# Configure memory settings for better performance on RunPod
echo -e "${GREEN}Configuring system for optimal GPU performance...${NC}"
# Increase shared memory size for dataloader workers
sudo mount -o remount,size=50g /dev/shm || echo -e "${YELLOW}Could not increase shared memory size (non-root user). This might limit dataloader performance.${NC}"

echo -e "${GREEN}Setup complete!${NC}"

# Detect specific GPU model for targeted recommendations
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

echo -e "${GREEN}Detected GPU: ${GPU_MODEL}${NC}"
echo -e "${GREEN}Recommended training commands for your RunPod GPU:${NC}"

# Provide specific recommendations based on detected GPU
if [[ $GPU_MODEL == *"a100"* ]]; then
    echo -e "${YELLOW}# For RunPod A100 (40/80GB):${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 24 --bf16 --log-memory${NC}"
elif [[ $GPU_MODEL == *"h100"* ]]; then
    echo -e "${YELLOW}# For RunPod H100:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 32 --bf16 --log-memory${NC}"
elif [[ $GPU_MODEL == *"l4"* ]]; then
    echo -e "${YELLOW}# For RunPod L4:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model distilbert-base-uncased --batch-size 64 --fp16 --log-memory${NC}"
    echo -e "${YELLOW}# For Jamba on L4:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 8 --gradient-accumulation-steps 4 --fp16 --log-memory${NC}"
elif [[ $GPU_MODEL == *"rtx8000"* ]] || [[ $GPU_MODEL == *"rtx6000"* ]] || [[ $GPU_MODEL == *"a6000"* ]]; then
    echo -e "${YELLOW}# For RunPod RTX 6000/8000 or A6000:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 16 --fp16 --log-memory${NC}"
elif [[ $GPU_MODEL == *"a40"* ]] || [[ $GPU_MODEL == *"a5000"* ]]; then
    echo -e "${YELLOW}# For RunPod A40/A5000:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 12 --gradient-accumulation-steps 2 --fp16 --log-memory${NC}"
elif [[ $GPU_MODEL == *"a10g"* ]] || [[ $GPU_MODEL == *"a10"* ]]; then
    echo -e "${YELLOW}# For RunPod A10G:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 8 --gradient-accumulation-steps 4 --fp16 --log-memory${NC}"
elif [[ $GPU_MODEL == *"3090"* ]] || [[ $GPU_MODEL == *"4090"* ]]; then
    echo -e "${YELLOW}# For RunPod RTX 3090/4090:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 8 --gradient-accumulation-steps 4 --fp16 --log-memory${NC}"
elif [[ $GPU_MODEL == *"v100"* ]]; then
    echo -e "${YELLOW}# For RunPod V100:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 6 --gradient-accumulation-steps 4 --fp16 --log-memory${NC}"
else
    echo -e "${YELLOW}# General configuration for your GPU:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model distilbert-base-uncased --batch-size 32 --fp16 --log-memory${NC}"
    echo -e "${YELLOW}# For Jamba model:${NC}"
    echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 4 --gradient-accumulation-steps 8 --fp16 --use-8bit --offload-to-cpu --log-memory${NC}"
fi

echo -e "\n${GREEN}For extremely large models with memory constraints:${NC}"
echo -e "${YELLOW}python src/train_jamba.py --data data/threat_intel_feed.csv --model ai21labs/AI21-Jamba-1.5-Mini --batch-size 2 --gradient-accumulation-steps 16 --fp16 --use-8bit --log-memory --offload-to-cpu --smaller-model-fallback distilbert-base-uncased${NC}"

echo -e "\n${GREEN}To monitor training with TensorBoard:${NC}"
echo -e "${YELLOW}tensorboard --logdir models/jamba-threat/logs --port 6006 --bind_all${NC}"
echo -e "${GREEN}Then access http://[your-runpod-ip]:6006 in your browser${NC}"

echo -e "\n${GREEN}RunPod deployment completed successfully!${NC}" 