#!/usr/bin/env python3
import unittest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import logging

from jamba.model_config import ModelConfig, DEFAULT_CPU_CONFIG
from jamba.data_preprocessing import DataPreprocessor, create_preprocessor
from jamba.jamba_model import JambaThreatModel, create_model
from jamba.train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Ensure deterministic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temporary directory for test artifacts
        cls.test_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.test_dir, 'models')
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Create synthetic dataset
        cls.n_samples = 1000
        cls.n_features = 20
        cls.create_synthetic_dataset()
    
    @classmethod
    def create_synthetic_dataset(cls):
        """Create synthetic dataset for testing"""
        # Generate features
        X = np.random.randn(cls.n_samples, cls.n_features)
        
        # Generate target (binary classification)
        w = np.random.randn(cls.n_features)
        y = (X.dot(w) + np.random.randn(cls.n_samples) * 0.1 > 0).astype(int)
        
        # Create DataFrame with features first
        feature_cols = [f'feature_{i}' for i in range(cls.n_features)]
        df_features = pd.DataFrame(X, columns=feature_cols)
        
        # Add target column
        df_features['is_threat'] = y
        cls.df = df_features
        
        # Save to temporary file
        cls.data_path = os.path.join(cls.test_dir, 'test_data.csv')
        cls.df.to_csv(cls.data_path, index=False)
    
    def test_model_initialization(self):
        """Test model initialization with different configurations"""
        # Test default configuration
        config = ModelConfig()
        model = create_model(config)
        self.assertIsInstance(model, JambaThreatModel)
        
        # Test custom configuration
        custom_config = ModelConfig(
            input_dim=self.n_features,
            hidden_dim=64,
            output_dim=1
        )
        model = create_model(custom_config)
        self.assertEqual(model.config.input_dim, self.n_features)
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline"""
        preprocessor = create_preprocessor(self.data_path, 'is_threat', self.model_dir)
        
        # Test preprocessing
        X, y = preprocessor.preprocess(self.df, 'is_threat', is_training=True)
        
        # Verify output shapes
        self.assertEqual(X.shape[0], self.n_samples)
        self.assertEqual(X.shape[1], self.n_features)
        self.assertEqual(y.shape[0], self.n_samples)
        
        # Test data validation
        is_valid, message = preprocessor.validate_data(self.df)
        self.assertTrue(is_valid, message)
    
    def test_model_training(self):
        """Test model training pipeline"""
        # Initialize trainer
        trainer = ModelTrainer(config=DEFAULT_CPU_CONFIG, save_dir=self.model_dir)
        trainer.setup_training(learning_rate=0.01)
        
        # Prepare data
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(100, self.n_features)
        y = (torch.randn(100) > 0).float()
        
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)
        
        # Test single epoch training
        train_loss = trainer.train_epoch(train_loader)
        self.assertIsInstance(train_loss, float)
        
        # Test validation
        val_loss = trainer.validate(val_loader)
        self.assertIsInstance(val_loss, float)
    
    def test_model_checkpointing(self):
        """Test model checkpointing functionality"""
        trainer = ModelTrainer(config=DEFAULT_CPU_CONFIG, save_dir=self.model_dir)
        trainer.setup_training()
        
        # Save checkpoint
        val_loss = 0.5
        trainer.save_checkpoint(val_loss, is_best=True)
        
        # Verify checkpoint files exist
        checkpoint_path = os.path.join(self.model_dir, 'checkpoint_latest.pt')
        best_model_path = os.path.join(self.model_dir, 'model_best.pt')
        
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertTrue(os.path.exists(best_model_path))
        
        # Test checkpoint loading
        trainer.load_checkpoint(checkpoint_path)
        self.assertEqual(trainer.best_val_loss, val_loss)
    
    def test_model_inference(self):
        """Test model inference"""
        # Create and train a small model
        config = ModelConfig(input_dim=self.n_features, hidden_dim=32)
        model = create_model(config)
        
        # Test inference
        with torch.no_grad():
            x = torch.randn(10, self.n_features)
            output = model(x)
            
            # Check output shape and range
            self.assertEqual(output.shape[0], 10)
            self.assertEqual(output.shape[1], 1)
    
    def test_error_handling(self):
        """Test error handling in critical components"""
        # Test invalid data loading
        with self.assertRaises(FileNotFoundError):
            create_preprocessor('nonexistent.csv', 'is_threat', self.model_dir)
        
        # Test invalid checkpoint loading
        trainer = ModelTrainer(save_dir=self.model_dir)
        with self.assertRaises(Exception):
            trainer.load_checkpoint('nonexistent.pt')
        
        # Test invalid model configuration
        with self.assertRaises(ValueError):
            ModelConfig(input_dim=-1)
    
    def test_training_resumption(self):
        """Test training resumption from checkpoint"""
        trainer = ModelTrainer(config=DEFAULT_CPU_CONFIG, save_dir=self.model_dir)
        trainer.setup_training()
        
        # Save initial checkpoint
        trainer.current_epoch = 5
        trainer.save_checkpoint(0.5)
        
        # Create new trainer and load checkpoint
        new_trainer = ModelTrainer(config=DEFAULT_CPU_CONFIG, save_dir=self.model_dir)
        new_trainer.setup_training()
        new_trainer.load_checkpoint(os.path.join(self.model_dir, 'checkpoint_latest.pt'))
        
        # Verify state restoration
        self.assertEqual(new_trainer.current_epoch, 5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts"""
        import shutil
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main() 