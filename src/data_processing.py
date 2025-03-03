import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from azure.storage.blob import BlobServiceClient
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatDataProcessor:
    def __init__(self, storage_conn_str, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(storage_conn_str)
        self.container_name = container_name
        self.scaler = StandardScaler()

    def load_data_from_blob(self, blob_prefix):
        """Load threat data from blob storage."""
        container_client = self.blob_service_client.get_container_client(self.container_name)
        threat_data = []

        for blob in container_client.list_blobs(name_starts_with=blob_prefix):
            blob_client = container_client.get_blob_client(blob.name)
            data = json.loads(blob_client.download_blob().readall())
            threat_data.append(data)

        return pd.DataFrame(threat_data)

    def preprocess_data(self, df):
        """Preprocess the threat data."""
        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Handle categorical variables
        df = pd.get_dummies(df, columns=['threat_type', 'source'])
        
        # Handle missing values
        df = df.fillna(0)
        
        return df

    def engineer_features(self, df):
        """Create features for threat detection."""
        # Aggregate features
        df['threat_count'] = df.groupby('source_ip')['timestamp'].transform('count')
        df['unique_targets'] = df.groupby('source_ip')['target_ip'].transform('nunique')
        
        # Calculate time-based features
        df['time_since_last'] = df.groupby('source_ip')['timestamp'].diff().dt.total_seconds()
        
        # Scale numerical features
        numerical_columns = ['threat_count', 'unique_targets', 'time_since_last']
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        
        return df

    def process_threat_data(self, blob_prefix):
        """Main processing pipeline."""
        try:
            # Load data
            logger.info("Loading data from blob storage...")
            df = self.load_data_from_blob(blob_prefix)
            
            # Preprocess
            logger.info("Preprocessing data...")
            df = self.preprocess_data(df)
            
            # Engineer features
            logger.info("Engineering features...")
            df = self.engineer_features(df)
            
            # Save processed data
            output_blob_name = f"processed_data/{pd.Timestamp.now():%Y%m%d_%H%M%S}_processed.parquet"
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            container_client.upload_blob(
                name=output_blob_name,
                data=df.to_parquet(),
                overwrite=True
            )
            
            logger.info(f"Processing complete. Data saved to {output_blob_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing threat data: {str(e)}")
            raise

if __name__ == "__main__":
    STORAGE_CONN_STR = "your_storage_connection_string"
    CONTAINER_NAME = "threat-data"
    
    processor = ThreatDataProcessor(STORAGE_CONN_STR, CONTAINER_NAME)
    processor.process_threat_data("threat_data/") 