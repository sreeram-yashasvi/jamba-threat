#!/usr/bin/env python3
import os
import argparse
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datetime import datetime

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

def preprocess_data_for_threat_detection(data_path):
    """Preprocess the threat data for model training."""
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

def train_threat_model(
    data_path, 
    model_name="distilbert-base-uncased",
    output_dir="models/threat-model",
    batch_size=8,
    epochs=3,
    learning_rate=5e-5,
    max_length=256
):
    """Fine-tune a model on threat data."""
    logger.info(f"Starting threat model fine-tuning with {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the data
    processed_df = preprocess_data_for_threat_detection(data_path)
    
    # Split into train and evaluation sets
    train_df, eval_df = train_test_split(processed_df, test_size=0.2, random_state=42)
    logger.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
    
    # Load tokenizer and model - without complex memory management
    logger.info(f"Loading {model_name} tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification
    )
    
    # Create datasets
    train_dataset = ThreatDataset(train_df, tokenizer, 'text', 'is_threat', max_length)
    eval_dataset = ThreatDataset(eval_df, tokenizer, 'text', 'is_threat', max_length)
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"threat-model-{timestamp}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",  # Disable TensorBoard to simplify
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        dataloader_num_workers=0,  # Avoid potential multi-processing issues
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
    
    # Train the model
    logger.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")
    
    # Save the final model
    final_model_path = os.path.join(output_dir, f"threat-model-final-{timestamp}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"Model saved to {final_model_path}")
    return final_model_path, results

def main():
    parser = argparse.ArgumentParser(description='Train a model for threat detection')
    parser.add_argument('--data', type=str, default='data/threat_intel_feed.csv', 
                       help='Path to the threat intel data')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                       help='Hugging Face model name')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, 
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='models/threat-model',
                       help='Output directory for the trained model')
    
    args = parser.parse_args()
    
    # Train the model
    model_path, results = train_threat_model(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.max_length
    )
    
    # Print final results
    logger.info(f"Training completed. Final eval loss: {results['eval_loss']:.4f}")
    
if __name__ == "__main__":
    main() 