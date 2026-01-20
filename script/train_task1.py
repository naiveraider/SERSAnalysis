"""
Training script for Task 1
Trains models on data from datasets/1/
"""
import sys
import os
import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model
import argparse


class Tee:
    """Write to multiple streams (e.g., console + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(task_id: int, model_name: str):
    """
    Redirect stdout/stderr so that console output is also saved to results/.
    """
    # Project root is parent of this script's directory
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"task{task_id}_{model_name}_{timestamp}.log"

    log_file = open(log_path, "w", encoding="utf-8")

    # Use original stdout/stderr so we still see output in console
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[LOG] Writing training logs to: {log_path}")
    return log_file


def main():
    parser = argparse.ArgumentParser(description='Train models for Task 1')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'tcn', 'cnn_transformer'],
                        help='Model type (cnn, tcn, or cnn_transformer)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda, default: auto-select)')
    
    # TCN-specific parameters
    parser.add_argument('--tcn_num_channels', type=int, nargs='+', default=[64, 128, 256],
                        help='TCN model channel list (TCN model only)')
    parser.add_argument('--tcn_kernel_size', type=int, default=3,
                        help='TCN model kernel size (TCN model only)')
    parser.add_argument('--tcn_dropout', type=float, default=0.2,
                        help='TCN model dropout rate (TCN model only)')
    
    # CNN+Transformer-specific parameters
    parser.add_argument('--cnn_transformer_cnn_channels', type=int, nargs='+', default=[64, 128, 256],
                        help='CNN+Transformer CNN channel list (CNN+Transformer model only)')
    parser.add_argument('--cnn_transformer_d_model', type=int, default=256,
                        help='CNN+Transformer embedding dimension (CNN+Transformer model only)')
    parser.add_argument('--cnn_transformer_nhead', type=int, default=8,
                        help='CNN+Transformer number of attention heads (CNN+Transformer model only)')
    parser.add_argument('--cnn_transformer_num_layers', type=int, default=2,
                        help='CNN+Transformer number of transformer layers (CNN+Transformer model only)')
    parser.add_argument('--cnn_transformer_dim_feedforward', type=int, default=512,
                        help='CNN+Transformer feedforward dimension (CNN+Transformer model only)')
    parser.add_argument('--cnn_transformer_dropout', type=float, default=0.1,
                        help='CNN+Transformer dropout rate (CNN+Transformer model only)')
    
    args = parser.parse_args()

    # Set up logging so all console output is also written to results/
    setup_logging(task_id=1, model_name=args.model)
    
    # Prepare model-specific parameters
    model_kwargs = {}
    if args.model == 'tcn':
        model_kwargs = {
            'num_channels': args.tcn_num_channels,
            'kernel_size': args.tcn_kernel_size,
            'dropout': args.tcn_dropout,
        }
    elif args.model == 'cnn_transformer':
        model_kwargs = {
            'cnn_channels': args.cnn_transformer_cnn_channels,
            'd_model': args.cnn_transformer_d_model,
            'nhead': args.cnn_transformer_nhead,
            'num_layers': args.cnn_transformer_num_layers,
            'dim_feedforward': args.cnn_transformer_dim_feedforward,
            'dropout': args.cnn_transformer_dropout,
        }
    
    print("=" * 60)
    print("Training Task 1")
    print("=" * 60)
    
    # Train a single multi-class model on all folders under datasets/1/
    # Folder names (e.g., Glycine, Phenylalanine, etc.) will be used as class labels
    train_model(
        data_dir="datasets",
        task_id=1,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=f"models/1/{args.model}",
        validation_split=args.validation_split,
        device=args.device,
        **model_kwargs
    )


if __name__ == "__main__":
    main()

