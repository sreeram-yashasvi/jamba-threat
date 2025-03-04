import os
import json
import requests
import time
import logging
import base64
import pickle
import pandas as pd
import numpy as np
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodClient:
    def __init__(self, api_key, endpoint_id, max_retries=5, backoff_factor=0.5):
        """Initialize the RunPod client.
        
        Args:
            api_key: RunPod API key
            endpoint_id: ID of the endpoint to use
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retry delay
        """
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.io/v2/ub3lew01pzj80j"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Configure robust session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        logger.info(f"RunPod client initialized for endpoint {endpoint_id}")
    
    def run_sync(self, input_data, timeout=3600):
        """Run a synchronous job on RunPod.
        
        Args:
            input_data: Input data for the job
            timeout: Timeout in seconds
            
        Returns:
            Job result
        """
        try:
            logger.info("Submitting job to RunPod")
            url = f"{self.base_url}/run"
            
            response = self.session.post(
                url,
                headers=self.headers,
                json={"input": input_data},
                timeout=30  # Specific timeout for initial request
            )
            
            if response.status_code != 200:
                logger.error(f"Request failed with status {response.status_code}")
                logger.error(response.text)
                raise Exception(f"RunPod API request failed: {response.text}")
            
            job_id = response.json()['id']
            logger.info(f"Job submitted with ID: {job_id}")
            
            # Poll for results
            start_time = time.time()
            retry_count = 0
            max_polls = 100  # Safety limit on polling attempts
            poll_count = 0
            
            while time.time() - start_time < timeout and poll_count < max_polls:
                poll_count += 1
                try:
                    status_response = self._check_status(job_id)
                    status = status_response.get('status')
                    
                    if status == 'COMPLETED':
                        logger.info(f"Job completed successfully in {time.time() - start_time:.2f} seconds")
                        result = status_response.get('output', {})
                        # Deserialize any data that was serialized
                        return self._deserialize_data(result)
                    
                    elif status == 'FAILED':
                        error = status_response.get('error', 'Unknown error')
                        logger.error(f"Job failed: {error}")
                        raise Exception(f"RunPod job failed: {error}")
                    
                    # Wait before polling again, increasing delay on subsequent attempts
                    wait_time = min(5 * (1.5 ** retry_count), 60)  # Cap at 60s
                    logger.info(f"Job status: {status}. Waiting {wait_time:.1f}s before checking again.")
                    time.sleep(wait_time)
                    
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout, 
                        requests.exceptions.RequestException) as e:
                    retry_count += 1
                    wait_time = min(5 * (2 ** retry_count), 60)  # Exponential backoff, capped at 60s
                    
                    if retry_count > 10:  # Maximum network error retries
                        logger.error(f"Too many connection errors: {str(e)}")
                        raise Exception(f"Failed to connect to RunPod API after multiple attempts: {str(e)}")
                    
                    logger.warning(f"Connection error (attempt {retry_count}): {str(e)}")
                    logger.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            if poll_count >= max_polls:
                raise Exception(f"Exceeded maximum polling attempts ({max_polls}) for job {job_id}")
            
            raise TimeoutError(f"Job did not complete within {timeout} seconds")
            
        except Exception as e:
            logger.error(f"Error in run_sync: {str(e)}")
            if "Connection reset by peer" in str(e):
                logger.error("SSL handshake failed. This might be due to network issues or API limits.")
                logger.error("Try again later or check your API key and endpoint ID.")
            raise
    
    def _check_status(self, job_id):
        """Check the status of a job with retry logic.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            JSON response with job status
        """
        url = f"{self.base_url}/status/{job_id}"
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Status check failed with code {response.status_code}")
                logger.error(response.text)
                raise Exception(f"RunPod status check failed: {response.text}")
            
            return response.json()
            
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout, 
                requests.exceptions.RequestException) as e:
            logger.error(f"Error checking job status: {str(e)}")
            # Let the caller handle retries
            raise
    
    def _serialize_data(self, data):
        """Serialize data for RunPod.
        
        Converts numpy arrays and pandas DataFrames to base64 strings.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data
        """
        if data is None:
            return None
            
        if isinstance(data, dict):
            serialized = {}
            for key, value in data.items():
                serialized[key] = self._serialize_data(value)
            return serialized
            
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
            
        elif isinstance(data, pd.DataFrame):
            logger.info(f"Serializing DataFrame with shape {data.shape}")
            try:
                # Use parquet for more efficient serialization
                buffer = io.BytesIO()
                data.to_parquet(buffer, engine='pyarrow', compression='snappy')
                buffer.seek(0)
                serialized = {
                    'type': 'dataframe',
                    'format': 'parquet',
                    'data': base64.b64encode(buffer.read()).decode('utf-8')
                }
                logger.info(f"DataFrame serialized successfully, size: {len(serialized['data']) // 1024} KB")
                return serialized
            except Exception as e:
                logger.warning(f"Error serializing DataFrame with parquet: {e}")
                # Fallback to pickle
                return {
                    'type': 'dataframe',
                    'format': 'pickle',
                    'data': base64.b64encode(pickle.dumps(data)).decode('utf-8')
                }
                
        elif isinstance(data, np.ndarray):
            return {
                'type': 'ndarray',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'data': base64.b64encode(data.tobytes()).decode('utf-8')
            }
            
        elif isinstance(data, (str, int, float, bool)) or data is None:
            return data
            
        else:
            try:
                logger.warning(f"Serializing unknown type {type(data)} using pickle")
                return {
                    'type': 'pickle',
                    'data': base64.b64encode(pickle.dumps(data)).decode('utf-8')
                }
            except Exception as e:
                logger.error(f"Failed to serialize {type(data)}: {e}")
                # Convert to string as last resort
                return str(data)
    
    def _deserialize_data(self, data):
        """Deserialize data from RunPod.
        
        Converts base64 strings back to numpy arrays and pandas DataFrames.
        
        Args:
            data: Data to deserialize
            
        Returns:
            Deserialized data
        """
        if data is None:
            return None
            
        if isinstance(data, dict):
            if 'type' in data:
                if data['type'] == 'dataframe':
                    try:
                        binary_data = base64.b64decode(data['data'])
                        
                        if data.get('format') == 'parquet':
                            buffer = io.BytesIO(binary_data)
                            return pd.read_parquet(buffer)
                        else:  # pickle format
                            return pickle.loads(binary_data)
                            
                    except Exception as e:
                        logger.error(f"Failed to deserialize DataFrame: {e}")
                        return None
                        
                elif data['type'] == 'ndarray':
                    try:
                        binary_data = base64.b64decode(data['data'])
                        array = np.frombuffer(binary_data, dtype=data['dtype'])
                        return array.reshape(data['shape'])
                    except Exception as e:
                        logger.error(f"Failed to deserialize ndarray: {e}")
                        return None
                        
                elif data['type'] == 'pickle':
                    try:
                        return pickle.loads(base64.b64decode(data['data']))
                    except Exception as e:
                        logger.error(f"Failed to deserialize pickled object: {e}")
                        return None
            
            # Regular dict without type indicator
            return {key: self._deserialize_data(value) for key, value in data.items()}
            
        elif isinstance(data, list):
            return [self._deserialize_data(item) for item in data]
            
        else:
            return data
    
    def train_model(self, data_path, target_column='is_threat', epochs=50, learning_rate=0.001, batch_size=32):
        """Train a model using RunPod.
        
        Args:
            data_path: Path to the training data file (CSV or parquet)
            target_column: Name of the target column
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            
        Returns:
            Training result with model and metrics
        """
        logger.info(f"Loading data from {data_path}...")
        
        try:
            # Load the data
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            logger.info(f"Data loaded with shape {data.shape}")
            
            # Prepare input for the job
            input_data = {
                "operation": "train",
                "data": self._serialize_data({
                    "dataset": data,
                    "target_column": target_column,
                    "params": {
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size
                    }
                })
            }
            
            # Run the job
            logger.info(f"Sending training job with {epochs} epochs, batch size {batch_size}")
            result = self.run_sync(input_data)
            
            # On success, we get back a serialized model and metrics
            logger.info("Training completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, model_path, data_path):
        """Make predictions using a trained model.
        
        Args:
            model_path: Path to the saved model file (.pth)
            data_path: Path to the data for predictions
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Loading model from {model_path} and data from {data_path}")
        
        try:
            # Load the model
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            # Load the data
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            logger.info(f"Data loaded with shape {data.shape}")
            
            # Prepare input for prediction
            input_data = {
                "operation": "predict",
                "data": self._serialize_data({
                    "model": base64.b64encode(model_data).decode('utf-8'),
                    "dataset": data
                })
            }
            
            # Run the prediction
            logger.info("Sending prediction request to RunPod")
            result = self.run_sync(input_data)
            
            # Extract and return prediction results
            logger.info("Prediction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
