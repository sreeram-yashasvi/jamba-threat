#!/usr/bin/env python3
import argparse
from jamba.train import main as train_main

def main():
    parser = argparse.ArgumentParser(description='Jamba Threat Detection CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', required=True, help='Path to dataset')
    train_parser.add_argument('--target', default='is_threat', help='Target column name')
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: auto)')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    train_parser.add_argument('--save-dir', default='models', help='Directory to save models')
    train_parser.add_argument('--checkpoint', help='Path to checkpoint to resume from')
    train_parser.add_argument('--validate-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_main()