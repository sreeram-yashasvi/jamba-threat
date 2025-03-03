#!/usr/bin/env python3
import argparse
import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, login
from getpass import getpass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def push_model_to_hub(model_path, repo_name, private=False, token=None):
    """
    Push a trained model to the Hugging Face Hub.
    
    Args:
        model_path: Path to the trained model directory
        repo_name: Name of the repository on Hugging Face Hub (format: username/repo-name)
        private: Whether to create a private repository
        token: Hugging Face token for authentication
    """
    logger.info(f"Preparing to push model from {model_path} to {repo_name}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist!")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    # If still None, prompt the user
    if token is None:
        logger.info("No Hugging Face token found. Please enter your token:")
        token = getpass("Hugging Face token: ")
    
    # Login to Hugging Face
    try:
        login(token=token)
        logger.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        return False
    
    # Load model and tokenizer to verify they're valid
    try:
        logger.info("Loading model and tokenizer to verify integrity...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        return False
    
    # Push to hub
    try:
        logger.info(f"Pushing model to {repo_name}...")
        # Push model
        model.push_to_hub(repo_name, private=private, token=token)
        logger.info("Model pushed successfully")
        
        # Push tokenizer
        tokenizer.push_to_hub(repo_name, private=private, token=token)
        logger.info("Tokenizer pushed successfully")
        
        # Get the API to retrieve the repository URL
        api = HfApi(token=token)
        repo_url = f"https://huggingface.co/{repo_name}"
        
        logger.info(f"Model successfully pushed to {repo_url}")
        logger.info("To use this model:")
        logger.info(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
        logger.info(f"  model = AutoModelForSequenceClassification.from_pretrained('{repo_name}')")
        logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{repo_name}')")
        
        return True
    except Exception as e:
        logger.error(f"Failed to push to Hugging Face Hub: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push a trained model to Hugging Face Hub")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--repo-name", type=str, required=True, help="Name of the repository on Hugging Face Hub (format: username/repo-name)")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--token", type=str, help="Hugging Face token (optional, will use HF_TOKEN env var if not provided)")
    
    args = parser.parse_args()
    
    success = push_model_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
        token=args.token
    )
    
    if success:
        logger.info("Model upload completed successfully!")
    else:
        logger.error("Model upload failed.")
        exit(1)

if __name__ == "__main__":
    main() 