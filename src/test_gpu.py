#!/usr/bin/env python3

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test CUDA memory allocation
        try:
            x = torch.rand(1000, 1000).cuda()
            logger.info("✓ Successfully allocated CUDA tensor")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to allocate CUDA tensor: {e}")
    else:
        logger.warning("No CUDA GPUs available")

if __name__ == "__main__":
    test_gpu() 