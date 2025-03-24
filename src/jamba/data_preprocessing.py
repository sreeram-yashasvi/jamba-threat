#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles all data preprocessing tasks with validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.target_column = None
        self.config_path = config_path
        self.column_means = {}
        
    def analyze_data(self, df: pd.DataFrame) -> None:
        """Analyze data types and identify columns for preprocessing"""
        logger.info("Analyzing data types...")
        
        # Identify numeric and categorical columns
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Found {len(self.numerical_columns)} numerical columns")
        logger.info(f"Found {len(self.categorical_columns)} categorical columns")
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate the input data"""
        try:
            # Check for missing values
            if df.isnull().any().any():
                return False, "Missing values found in data"
            
            # Check for infinite values and replace with NaN
            if np.isinf(df.select_dtypes(include=np.number).values).any():
                logger.warning("Infinite values found, replacing with NaN")
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Check data types
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                return False, "No numeric columns found"
            
            return True, "Data validation passed"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def preprocess(self, df: pd.DataFrame, target_column: str, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data"""
        logger.info("Starting data preprocessing...")
        
        try:
            # Store target column name
            self.target_column = target_column
            
            # Analyze data if in training mode
            if is_training:
                self.analyze_data(df)
            
            # Validate data
            logger.info("Validating data quality...")
            is_valid, message = self.validate_data(df)
            if not is_valid:
                raise ValueError(f"Data validation failed: {message}")
            
            # Handle infinite values
            df = df.copy()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill NaN values with mean for numeric columns
            numeric_cols = [col for col in self.numerical_columns if col != target_column]
            for col in numeric_cols:
                if is_training:
                    self.column_means[col] = df[col].mean()
                df[col].fillna(self.column_means.get(col, 0), inplace=True)
            
            # Separate features and target
            X = df[numeric_cols]
            y = df[target_column] if target_column in df.columns else None
            
            # Scale numerical features
            if is_training:
                X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            else:
                X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            
            logger.info("Data preprocessing completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise
    
    def save(self, path: str):
        """Save preprocessor state"""
        save_dict = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'target_column': self.target_column,
            'column_means': self.column_means
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(save_dict, path)
        logger.info(f"Preprocessor state saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor state"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor state file not found: {path}")
        
        save_dict = joblib.load(path)
        self.scaler = save_dict['scaler']
        self.label_encoders = save_dict['label_encoders']
        self.categorical_columns = save_dict['categorical_columns']
        self.numerical_columns = save_dict['numerical_columns']
        self.target_column = save_dict['target_column']
        self.column_means = save_dict['column_means']
        logger.info(f"Preprocessor state loaded from {path}")

def create_preprocessor(data_path: str, target_column: str, 
                       save_dir: str) -> DataPreprocessor:
    """Factory function to create and initialize preprocessor"""
    try:
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_parquet(data_path)
        
        # Preprocess data
        preprocessor.preprocess(df, target_column, is_training=True)
        
        # Save preprocessor state
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'preprocessor_state.joblib')
        preprocessor.save(save_path)
        
        return preprocessor
        
    except Exception as e:
        logger.error(f"Error creating preprocessor: {str(e)}")
        raise 