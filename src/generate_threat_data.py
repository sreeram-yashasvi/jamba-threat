import csv
import random
import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_threat_intel_feed(output_file="data/threat_intel_feed.csv", num_entries=15000):
    """Generate a threat intelligence feed dataset with the specified number of entries.
    
    Args:
        output_file: Path to save the generated CSV file
        num_entries: Number of entries to generate
        
    Returns:
        Path to the generated CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sources = ["ThreatGrid", "DarkPulse", "CyberSentry", "VigilantEye", "NexThreat"]
    threat_types = ["Malware", "Phishing", "DDoS", "Botnet", "Exploit"]
    actors = ["ShadowSyndicate", "AquaPhish", "NullStorm", "RedVortex", "GhostNet"]
    descriptions = [
        "Suspicious IP linked to {0} activity.",
        "Domain hosting {0} campaign.",
        "IP observed in {0} attack.",
        "SHA-256 hash of {0} payload.",
        "Domain used in {0} operation."
    ]

    logger.info(f"Generating threat intelligence feed with {num_entries} entries")
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feed ID", "Source", "Timestamp", "Threat Type", "IOC", "Confidence Score", "Threat Actor", "Description"])
        
        for i in range(1, num_entries + 1):
            feed_id = f"TF-{i:03d}"
            source = random.choice(sources)
            start_date = datetime(2025, 1, 1)
            random_days = random.randint(0, 55)  # Up to Feb 25, 2025
            random_time = timedelta(seconds=random.randint(0, 86400))
            timestamp = (start_date + timedelta(days=random_days) + random_time).strftime("%Y-%m-%d %H:%M:%S")
            threat = random.choice(threat_types)
            if threat in ["Malware", "Botnet"]:
                ioc = random.choice([f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}", 
                                    "".join(random.choices("0123456789abcdef", k=32))])
            else:
                ioc = f"mal-{random.randint(1,1000)}.org"
            score = random.randint(50, 100)
            actor = random.choice(actors)
            desc = random.choice(descriptions).format(threat.lower())
            writer.writerow([feed_id, source, timestamp, threat, ioc, score, actor, desc])
    
    logger.info(f"Threat intelligence feed saved to {output_file}")
    return output_file

def preprocess_threat_intel(input_file, output_file="data/processed_threat_data.csv"):
    """Preprocess the threat intelligence feed for model training.
    
    Args:
        input_file: Path to the raw threat intel feed CSV
        output_file: Path to save the processed data
        
    Returns:
        Processed DataFrame ready for model training
    """
    logger.info(f"Preprocessing threat intelligence data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract temporal features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['IsWeekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Extract IOC type
    df['IsIP'] = df['IOC'].apply(lambda x: 1 if '.' in x and any(c.isdigit() for c in x) else 0)
    df['IsDomain'] = df['IOC'].apply(lambda x: 1 if '.org' in x or '.com' in x or '.net' in x else 0)
    df['IsHash'] = df['IOC'].apply(lambda x: 1 if all(c in '0123456789abcdef' for c in x) and len(x) >= 32 else 0)
    
    # Measure description severity based on keywords
    severity_keywords = ['attack', 'malicious', 'suspicious', 'critical', 'compromise']
    df['Description_Severity'] = df['Description'].apply(
        lambda x: sum(1 for keyword in severity_keywords if keyword in x.lower())
    )
    
    # Calculate threat actor prevalence
    actor_counts = df['Threat Actor'].value_counts()
    df['Actor_Prevalence'] = df['Threat Actor'].map(actor_counts) / len(df)
    
    # Create high confidence flag
    df['High_Confidence'] = df['Confidence Score'].apply(lambda x: 1 if x >= 75 else 0)
    
    # One-hot encode categorical variables
    categorical_cols = ['Source', 'Threat Type', 'Threat Actor']
    
    # Prepare encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    
    # Get the encoded column names
    encoded_cols = []
    for i, col in enumerate(categorical_cols):
        encoded_cols.extend([f"{col}_{cat}" for cat in encoder.categories_[i]])
    
    # Create a DataFrame with encoded categorical variables
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols)
    
    # Combine with original DataFrame
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    
    # Calculate synthetic target variable based on confidence score and threat type
    # Note: In a real scenario, this would be a known label
    malware_botnet = df.filter(regex='Threat Type_(Malware|Botnet)').any(axis=1)
    high_confidence = df['Confidence Score'] >= 80
    suspicious_features = (df['Description_Severity'] >= 2) | df['IsIP']
    
    # Generate the threat flag
    df['is_threat'] = ((malware_botnet & high_confidence) | 
                        (high_confidence & suspicious_features) | 
                        (df['Confidence Score'] > 90)).astype(float)
    
    # Add some noise to make it more realistic
    np.random.seed(42)
    noise_mask = np.random.choice([0, 1], len(df), p=[0.95, 0.05])
    df['is_threat'] = np.abs(df['is_threat'] - noise_mask)
    
    # Select relevant features for the model
    features = [
        'Hour', 'Day', 'Month', 'Weekday', 'IsWeekend',
        'IsIP', 'IsDomain', 'IsHash',
        'Confidence Score', 'Description_Severity',
        'Actor_Prevalence', 'High_Confidence',
        'is_threat'
    ]
    
    # Add the encoded categorical columns
    features.extend(encoded_cols)
    
    # Subset the DataFrame
    model_df = df[features]
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = [
        'Hour', 'Day', 'Month', 'Weekday',
        'Confidence Score', 'Description_Severity', 'Actor_Prevalence'
    ]
    
    model_df[numerical_features] = scaler.fit_transform(model_df[numerical_features])
    
    # Save to CSV
    model_df.to_csv(output_file, index=False)
    
    # Print class distribution
    threat_count = model_df['is_threat'].sum()
    logger.info(f"Class distribution - Threats: {threat_count} ({threat_count/len(model_df):.2%}), " 
              f"Benign: {len(model_df) - threat_count} ({1 - threat_count/len(model_df):.2%})")
    
    logger.info(f"Processed dataset saved to {output_file}")
    return model_df

def main():
    parser = argparse.ArgumentParser(description='Generate and preprocess threat intelligence data')
    parser.add_argument('--entries', type=int, default=1000, help='Number of threat intel entries to generate')
    parser.add_argument('--output-raw', default='data/threat_intel_feed.csv', help='Path to save raw threat intel data')
    parser.add_argument('--output-processed', default='data/processed_threat_data.csv', help='Path to save processed data')
    parser.add_argument('--use-existing', action='store_true', help='Use existing data instead of generating new data')
    
    args = parser.parse_args()
    
    # Generate threat intelligence feed
    if not args.use_existing or not os.path.exists(args.output_raw):
        threat_feed_path = generate_threat_intel_feed(args.output_raw, args.entries)
    else:
        threat_feed_path = args.output_raw
        logger.info(f"Using existing threat intel feed from {threat_feed_path}")
    
    # Preprocess the data for model training
    if not args.use_existing or not os.path.exists(args.output_processed):
        processed_df = preprocess_threat_intel(threat_feed_path, args.output_processed)
    else:
        processed_df = pd.read_csv(args.output_processed)
        logger.info(f"Using existing processed data from {args.output_processed}")
    
    logger.info(f"Dataset ready for training with {processed_df.shape[0]} samples and {processed_df.shape[1]} features")
    logger.info(f"Feature columns: {', '.join(processed_df.columns.drop('is_threat'))}")

if __name__ == "__main__":
    main() 