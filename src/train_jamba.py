#!/usr/bin/env python3
import os
import argparse
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import TrainingArguments, Trainer, TrainerState, TrainerControl
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from huggingface_hub import HfFolder, hf_hub_download, list_repo_files
from transformers.integrations import TensorBoardCallback
import gc
import psutil
import sys
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_column, label_column, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx][self.text_column]
        label = self.dataframe.iloc[idx][self.label_column]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert dict of tensors to dict of lists for the Trainer
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }

def preprocess_data_for_jamba(data_path):
    """Preprocess the threat data for Jamba model training."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create text description from available features
    df['text'] = df.apply(
        lambda row: f"Source: {row['Source']} | Threat Type: {row['Threat Type']} | "
                   f"Actor: {row['Threat Actor']} | Confidence: {row['Confidence Score']} | "
                   f"Description: {row['Description']}",
        axis=1
    )
    
    # Create binary label (1 for threat, 0 for benign)
    # Assuming threats have confidence scores above 0.7
    df['is_threat'] = (df['Confidence Score'] > 0.7).astype(int)
    
    logger.info(f"Processed data: {len(df)} rows, with {df['is_threat'].sum()} threats")
    return df

# Add NVIDIA GPU monitoring for Hetzner servers
def get_gpu_info():
    """Get GPU information for all available NVIDIA GPUs."""
    gpu_info = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info['device_count'] = device_count
        gpu_info['devices'] = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            device_info = {
                'id': i,
                'name': gpu_name,
                'total_memory_mb': memory_info.total / (1024 * 1024),
                'free_memory_mb': memory_info.free / (1024 * 1024),
                'used_memory_mb': memory_info.used / (1024 * 1024),
                'memory_utilization': memory_info.used / memory_info.total * 100
            }
            
            gpu_info['devices'].append(device_info)
        
        pynvml.nvmlShutdown()
    except Exception as e:
        gpu_info['error'] = str(e)
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['devices'] = []
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i)
                }
                if hasattr(torch.cuda, 'mem_get_info'):
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    device_info['total_memory_mb'] = total_mem / (1024 * 1024)
                    device_info['free_memory_mb'] = free_mem / (1024 * 1024)
                    device_info['used_memory_mb'] = (total_mem - free_mem) / (1024 * 1024)
                gpu_info['devices'].append(device_info)
                
    return gpu_info

# Improved function to detect available compute devices
def detect_compute_device():
    """Detect available compute devices and return the best one to use."""
    # First try CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Get detailed GPU info
        gpu_info = get_gpu_info()
        if 'devices' in gpu_info:
            for device in gpu_info['devices']:
                if 'free_memory_mb' in device:
                    logger.info(f"GPU {device['id']} ({device['name']}): {device['free_memory_mb']:.2f} MB free / {device['total_memory_mb']:.2f} MB total")
        
        return "cuda"
    # Then try MPS (Metal Performance Shaders for Mac)
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS (Metal Performance Shaders) is available for Mac")
        return "mps"
    # Finally, fall back to CPU
    else:
        logger.info("No GPU detected, using CPU")
        return "cpu"

# Function to clear up memory
def clear_memory():
    """Force garbage collection and clear CUDA cache if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Try to forcefully deallocate all tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    obj.cpu()
            except:
                pass
    
    # Log current memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage after cleanup: {mem_info.rss / (1024 * 1024):.2f} MB")

def log_memory_usage(message=""):
    """Log the current memory usage."""
    # Force garbage collection
    gc.collect()
    process = psutil.Process()
    mem_info = process.memory_info()
    memory_usage = mem_info.rss / (1024 * 1024)  # Convert to MB
    logger.info(f"Memory usage {message}: {memory_usage:.2f} MB")
    
    # Log GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if hasattr(torch.cuda, 'mem_get_info'):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                used_mem = total_mem - free_mem
                logger.info(f"GPU {i} memory usage: {used_mem / (1024 * 1024):.2f} MB / {total_mem / (1024 * 1024):.2f} MB")

# Add a function to check available system resources before loading model
def check_system_resources():
    """Check available system resources and log information."""
    # Log memory information
    mem = psutil.virtual_memory()
    logger.info(f"Total system memory: {mem.total / (1024**3):.2f} GB")
    logger.info(f"Available memory: {mem.available / (1024**3):.2f} GB")
    logger.info(f"Memory usage: {mem.percent}%")
    
    # Log CPU information
    logger.info(f"CPU count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    
    # Log system information
    logger.info(f"System: {platform.system()} {platform.release()}")
    
    # Check if CUDA is available and log GPU information
    device = detect_compute_device()
    if device == "cuda":
        gpu_info = get_gpu_info()
        if 'devices' in gpu_info:
            for device in gpu_info['devices']:
                if 'memory_utilization' in device:
                    logger.info(f"GPU {device['id']} memory utilization: {device['memory_utilization']:.2f}%")
    
    # Return True if resources seem sufficient, False otherwise
    if mem.available < 4 * (1024**3):  # Less than 4GB available
        logger.warning("Very low available memory, may encounter issues")
        return False
    return True

def train_jamba_model(
    data_path, 
    model_name="ai21labs/AI21-Jamba-1.5-Mini",
    output_dir="models/jamba-threat",
    batch_size=16,
    epochs=3,
    learning_rate=5e-5,
    max_length=256,
    gradient_accumulation_steps=4,
    fp16=False,
    bf16=False,  # Add support for bfloat16 which is better on newer GPUs
    use_8bit=False,  # Add support for 8-bit quantization
    log_memory=False,
    offload_to_cpu=False,
    smaller_model_fallback=None,
    max_memory_usage=0.9,  # Fraction of available memory to use
    device=None  # Allow explicit device specification
):
    """Fine-tune the Jamba model on threat data."""
    logger.info(f"Starting Jamba model fine-tuning with {model_name}")
    
    # Check system resources
    if log_memory:
        has_sufficient_resources = check_system_resources()
        if not has_sufficient_resources and smaller_model_fallback:
            logger.warning(f"Insufficient resources detected. Falling back to smaller model: {smaller_model_fallback}")
            model_name = smaller_model_fallback
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect the best device to use
    compute_device = device if device else detect_compute_device()
    logger.info(f"Using compute device: {compute_device}")
    
    # Set specific device configurations
    if compute_device == "cuda":
        # For Hetzner NVIDIA GPUs
        logger.info("Configuring for NVIDIA GPU")
        
        # Determine if we can use bfloat16 (better on newer GPUs like A100/H100)
        if bf16 and torch.cuda.is_bf16_supported():
            logger.info("BFloat16 is supported and enabled")
        elif bf16:
            logger.info("BFloat16 was requested but is not supported on this GPU, falling back to FP16")
            bf16 = False
            fp16 = True
        
        # Check for 8-bit quantization support
        if use_8bit:
            try:
                import bitsandbytes as bnb
                logger.info("8-bit quantization enabled with bitsandbytes")
            except ImportError:
                logger.warning("bitsandbytes is not installed, 8-bit quantization disabled")
                use_8bit = False
    
    # Process the data
    if data_path.endswith('.csv'):
        raw_df = pd.read_csv(data_path)
    else:
        raise ValueError("Only CSV files are supported for now")
    
    # Process data
    processed_df = preprocess_data_for_jamba(data_path)
    
    # Split into train and evaluation sets
    train_df, eval_df = train_test_split(processed_df, test_size=0.2, random_state=42)
    logger.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
    
    # Load tokenizer and model
    logger.info(f"Loading {model_name} tokenizer and model")
    
    # Get token from environment or files
    token = os.environ.get("HF_TOKEN")
    if token is None:
        token = HfFolder.get_token()
        
        if token is None:
            token_paths = [
                os.path.expanduser("~/.huggingface/token"),
                os.path.expanduser("~/.cache/huggingface/token"),
            ]
            for path in token_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        token = f.read().strip()
                    logger.info(f"Found token in {path}")
                    break
        
        if token is None and "ai21labs" in model_name.lower():
            logger.warning("No Hugging Face token found for gated model. This might lead to authentication errors.")
            logger.info("Please set the HF_TOKEN environment variable or login via `huggingface-cli login`")
            raise ValueError("No Hugging Face token available. Cannot access gated model.")
    
    # Prepare device map for memory-efficient loading
    device_map = None
    if compute_device == "cuda":
        if offload_to_cpu:
            # Offload some layers to CPU to save GPU memory
            device_map = "auto"
            logger.info("Using automatic device mapping to optimize memory usage")
        else:
            # Use GPU only but with memory efficient loading
            device_map = {"": 0}  # All on GPU 0
            logger.info("Loading model entirely on GPU with memory efficient loading")
    
    # For standard models, use the simpler loading approach with proper device management
    try:
        if model_name.lower() != "ai21labs/ai21-jamba-1.5-mini":
            logger.info(f"Loading {model_name} using standard sequence classification")
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                token=token
            )
            logger.info("Successfully loaded tokenizer")
            
            if log_memory:
                log_memory_usage("after loading tokenizer")
                
            # Clear memory before loading model
            clear_memory()
            
            # Prepare model loading parameters
            model_kwargs = {
                "num_labels": 2,
                "low_cpu_mem_usage": True,
            }
            
            # Add device specific parameters
            if compute_device == "cuda":
                # Handle precision options
                if fp16:
                    model_kwargs["torch_dtype"] = torch.float16
                elif bf16 and torch.cuda.is_bf16_supported():
                    model_kwargs["torch_dtype"] = torch.bfloat16
                
                # Device mapping for multi-GPU or CPU offloading
                if device_map:
                    model_kwargs["device_map"] = device_map
                
                # 8-bit quantization for memory savings
                if use_8bit:
                    model_kwargs["load_in_8bit"] = True
            
            # Load model with optimized parameters
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=token,
                **model_kwargs
            )
            
            # Move model to device if not using device_map
            if not device_map and compute_device != "cpu":
                model = model.to(compute_device)
                
            logger.info(f"Successfully loaded model with standard sequence classification")
            model_loaded = True
            
        # Special handling for Jamba models
        else:
            logger.info(f"Attempting to download and load {model_name} with custom classification head")
            
            try:
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,
                    token=token
                )
                logger.info("Successfully loaded tokenizer")
                
                if log_memory:
                    log_memory_usage("after loading tokenizer")
                    
                # Clear memory before loading model
                clear_memory()
                
                # For sequence classification with Jamba, we need to:
                # 1. Load the base model (not as AutoModelForSequenceClassification)
                # 2. Then add a classification head
                from transformers import AutoModel, AutoConfig
                
                # Get the config first
                config = AutoConfig.from_pretrained(model_name, token=token)
                
                # Loading strategies for base model
                loading_strategies = [
                    # Strategy 1: Direct loading with dtype conversion
                    {"low_cpu_mem_usage": True, "torch_dtype": torch.float16 if fp16 else None},
                    # Strategy 2: With CPU offloading
                    {"device_map": "auto", "low_cpu_mem_usage": True, "torch_dtype": torch.float16 if fp16 else None},
                    # Strategy 3: CPU only, last resort
                    {"device_map": {"": "cpu"}, "low_cpu_mem_usage": True}
                ]
                
                # Try different loading strategies
                for i, strategy in enumerate(loading_strategies):
                    try:
                        logger.info(f"Loading base model with strategy {i+1}: {strategy}")
                        clear_memory()
                        
                        # Load as base model first, then add classification head
                        base_model = AutoModel.from_pretrained(
                            model_name,
                            config=config,
                            token=token,
                            **strategy
                        )
                        
                        logger.info(f"Successfully loaded base model with strategy {i+1}")
                        
                        # Now we need to add a classification head
                        # For simplicity in this example, we'll use a custom model class
                        
                        # Get hidden size from config
                        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.d_model
                        
                        # Create custom model for classification
                        from transformers import PreTrainedModel
                        import torch.nn as nn
                        
                        class JambaForSequenceClassification(nn.Module):
                            def __init__(self, base_model, num_labels=2):
                                super().__init__()
                                self.base_model = base_model
                                self.classifier = nn.Linear(hidden_size, num_labels)
                                self.num_labels = num_labels
                                
                            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                                # Filter out keywords that are not for the base model
                                # We'll explicitly filter out known Trainer keywords
                                base_model_kwargs = {k: v for k, v in kwargs.items() 
                                                  if k not in ['num_items_in_batch', 'return_loss']}
                                
                                # Get the base model outputs
                                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **base_model_kwargs)
                                
                                # Use the last hidden state of the [CLS] token for classification
                                # For most models, this is the first token
                                sequence_output = outputs.last_hidden_state[:, 0, :]
                                
                                # Pass through the classifier
                                logits = self.classifier(sequence_output)
                                
                                loss = None
                                if labels is not None:
                                    loss_fct = nn.CrossEntropyLoss()
                                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                                    
                                return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
                            
                            # Add support for gradient checkpointing
                            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                                """Enable gradient checkpointing for the model if supported by the base model."""
                                if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                                    self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                                    logger.info("Gradient checkpointing enabled for base model")
                                else:
                                    logger.warning("Base model does not support gradient checkpointing")
                                    
                            def gradient_checkpointing_disable(self):
                                """Disable gradient checkpointing for the model if supported by the base model."""
                                if hasattr(self.base_model, 'gradient_checkpointing_disable'):
                                    self.base_model.gradient_checkpointing_disable()
                        
                        # Create our classification model
                        model = JambaForSequenceClassification(base_model, num_labels=2)
                        logger.info("Added classification head to base model")
                        
                        if log_memory:
                            log_memory_usage("after creating classification model")
                        
                        model_loaded = True
                        break
                        
                    except Exception as e:
                        logger.warning(f"Failed to load model with strategy {i+1}: {str(e)}")
                        if i == len(loading_strategies) - 1:
                            raise
            
            except Exception as e:
                # Try fallback model if specified
                if smaller_model_fallback:
                    logger.warning(f"Attempting to load smaller fallback model: {smaller_model_fallback}")
                    try:
                        # This is for models that directly support sequence classification
                        clear_memory()
                        
                        tokenizer = AutoTokenizer.from_pretrained(
                            smaller_model_fallback,
                            use_fast=True,
                            token=token
                        )
                        
                        model = AutoModelForSequenceClassification.from_pretrained(
                            smaller_model_fallback,
                            num_labels=2,
                            device_map="auto" if offload_to_cpu else None,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16 if fp16 else None,
                            token=token
                        )
                        logger.info(f"Successfully loaded smaller fallback model: {smaller_model_fallback}")
                        model_loaded = True
                    except Exception as fallback_err:
                        raise ValueError(f"Failed to load model with all strategies and fallbacks: {fallback_err}")
                else:
                    raise ValueError(f"Failed to load model: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error downloading or loading model: {str(e)}")
        logger.error("Permission issue or model incompatibility. Try specifying a different model like:")
        logger.error("--model google/flan-t5-small or --model distilbert-base-uncased")
        raise
    
    if not model_loaded or model is None or tokenizer is None:
        raise ValueError("Failed to load model or tokenizer")
    
    # Create datasets
    train_dataset = ThreatDataset(train_df, tokenizer, 'text', 'is_threat', max_length)
    eval_dataset = ThreatDataset(eval_df, tokenizer, 'text', 'is_threat', max_length)
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define training arguments with memory optimizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if the model supports gradient checkpointing
    supports_gradient_checkpointing = hasattr(model, 'gradient_checkpointing_enable')
    
    # Configure training arguments based on device
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"jamba-threat-{timestamp}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        # GPU-specific settings
        fp16=fp16 and compute_device == "cuda",
        bf16=bf16 and compute_device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16_opt_level="O1",
        dataloader_num_workers=4 if compute_device == "cuda" else 0,  # Use workers on Hetzner
        dataloader_pin_memory=compute_device == "cuda",  # Pin memory on GPU servers
        group_by_length=True,
        gradient_checkpointing=supports_gradient_checkpointing,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model with memory tracking
    logger.info("Starting training...")
    
    # Add memory tracking if requested
    if log_memory:
        log_memory_usage("before training")
        
        class MemoryCallback(TensorBoardCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 50 == 0:  # Log every 50 steps
                    log_memory_usage(f"training step {state.global_step}")
                return super().on_step_end(args, state, control, **kwargs)
        
        trainer.add_callback(MemoryCallback())
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        if log_memory:
            log_memory_usage("after training error")
        
        # Suggest solutions based on the error
        if "CUDA out of memory" in str(e) or "killed" in str(e).lower() or "MemoryError" in str(e):
            logger.error("Memory error detected. Try the following:")
            logger.error("1. Reduce batch size (--batch-size)")
            logger.error("2. Increase gradient accumulation steps (--gradient-accumulation-steps)")
            logger.error("3. Enable mixed precision training (--fp16 or --bf16)")
            logger.error("4. Use a smaller model with --model parameter")
            logger.error("5. Enable CPU offloading with --offload-to-cpu")
            logger.error("6. Try 8-bit quantization with --use-8bit")
        raise
    
    if log_memory:
        log_memory_usage("after training")
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")
    
    # Save the final model
    final_model_path = os.path.join(output_dir, f"jamba-threat-final-{timestamp}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"Model saved to {final_model_path}")
    return final_model_path, results

def main():
    parser = argparse.ArgumentParser(description='Train AI21 Jamba model for threat detection')
    parser.add_argument('--data', type=str, default='data/threat_intel_feed.csv', 
                       help='Path to the threat intel data')
    parser.add_argument('--model', type=str, default='ai21labs/AI21-Jamba-1.5-Mini',
                       help='Hugging Face model name')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, 
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='models/jamba-threat',
                       help='Output directory for the trained model')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Number of gradient accumulation steps to reduce memory usage')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training (float16)')
    parser.add_argument('--bf16', action='store_true',
                       help='Use bfloat16 precision (better for newer GPUs like A100/H100)')
    parser.add_argument('--use-8bit', action='store_true',
                       help='Use 8-bit quantization to reduce memory usage')
    parser.add_argument('--log-memory', action='store_true',
                       help='Log memory usage during training')
    parser.add_argument('--offload-to-cpu', action='store_true',
                       help='Offload model layers to CPU to save GPU memory')
    parser.add_argument('--smaller-model-fallback', type=str, 
                       help='Smaller model to use as fallback if main model is too large')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], 
                       help='Device to use for training (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Train the model
    model_path, results = train_jamba_model(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        use_8bit=args.use_8bit,
        log_memory=args.log_memory,
        offload_to_cpu=args.offload_to_cpu,
        smaller_model_fallback=args.smaller_model_fallback,
        device=args.device
    )
    
    # Print final results
    logger.info(f"Training completed. Final eval loss: {results['eval_loss']:.4f}")
    
if __name__ == "__main__":
    main() 