"""
Training script for Task 4
Trains models on data from datasets/4/
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train models for Task 4')
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
    print("Training Task 4")
    print("=" * 60)
    
    # Train a single multi-class model on all folders under datasets/4/
    train_model(
        data_dir="datasets",
        task_id=4,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=f"models/4/{args.model}",
        validation_split=args.validation_split,
        device=args.device,
        **model_kwargs
    )


if __name__ == "__main__":
    main()

