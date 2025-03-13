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
        
    def analyze_data(self, df: pd.DataFrame) -> None:
        """Analyze data types and identify columns for preprocessing"""
        logger.info("Analyzing data types...")
        
        # Identify numeric and categorical columns
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Found {len(self.numerical_columns)} numerical columns")
        logger.info(f"Found {len(self.categorical_columns)} categorical columns")
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data quality"""
        logger.info("Validating data quality...")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            return False, f"Missing values found in columns: {missing[missing > 0].to_dict()}"
        
        # Check for infinite values in numeric columns
        if not df[self.numerical_columns].replace([np.inf, -np.inf], np.nan).isnull().sum().any():
            return False, "Infinite values found in numeric columns"
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            return False, f"Constant columns found: {constant_cols}"
        
        # Check for low variance in numeric columns
        low_var_cols = [col for col in self.numerical_columns 
                       if df[col].std() < 1e-6]
        if low_var_cols:
            logger.warning(f"Low variance columns found: {low_var_cols}")
        
        return True, "Data validation passed"
    
    def preprocess(self, df: pd.DataFrame, target_column: str, 
                  is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess data with validation and proper error handling"""
        try:
            logger.info("Starting data preprocessing...")
            
            # Store target column name
            self.target_column = target_column
            
            # Analyze data if in training mode
            if is_training:
                self.analyze_data(df)
            
            # Validate data
            is_valid, message = self.validate_data(df)
            if not is_valid:
                raise ValueError(f"Data validation failed: {message}")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column] if target_column in df.columns else None
            
            # Handle categorical columns
            for col in self.categorical_columns:
                if col in X.columns:  # Skip if column is target
                    if is_training:
                        self.label_encoders[col] = LabelEncoder()
                        X[col] = self.label_encoders[col].fit_transform(X[col])
                    else:
                        if col in self.label_encoders:
                            # Handle unknown categories
                            unknown_values = ~X[col].isin(self.label_encoders[col].classes_)
                            if unknown_values.any():
                                logger.warning(f"Unknown categories in {col}: {X[col][unknown_values].unique()}")
                                # Add unknown category handling here if needed
                            X[col] = self.label_encoders[col].transform(X[col])
            
            # Scale numerical features
            numerical_features = X[self.numerical_columns]
            if is_training:
                X[self.numerical_columns] = self.scaler.fit_transform(numerical_features)
            else:
                X[self.numerical_columns] = self.scaler.transform(numerical_features)
            
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
            'target_column': self.target_column
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