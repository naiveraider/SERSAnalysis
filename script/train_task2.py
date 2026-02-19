"""
Training script for Task 2
Trains models on data from datasets/2/
"""
import sys
import os
import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model
from src.model.model_factory import get_available_models
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
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"task{task_id}_{model_name}_{timestamp}.log"

    log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[LOG] Writing training logs to: {log_path}")
    return log_file


def main():
    parser = argparse.ArgumentParser(description='Train models for Task 2')
    parser.add_argument('--model', type=str, default='cnn', 
                        choices=get_available_models(),
                        help=f'Model type: {", ".join(get_available_models())}')
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
    
    # Mamba/S4-specific parameters
    parser.add_argument('--mamba_d_model', type=int, default=256,
                        help='Mamba/S4 model dimension (Mamba/S4 model only)')
    parser.add_argument('--mamba_n_layers', type=int, default=4,
                        help='Mamba/S4 number of layers (Mamba/S4 model only)')
    parser.add_argument('--mamba_d_state', type=int, default=64,
                        help='Mamba/S4 state dimension (Mamba/S4 model only)')
    parser.add_argument('--mamba_dropout', type=float, default=0.1,
                        help='Mamba/S4 dropout rate (Mamba/S4 model only)')
    
    # ViT-specific parameters
    parser.add_argument('--vit_patch_size', type=int, default=16,
                        help='ViT patch size (ViT model only)')
    parser.add_argument('--vit_d_model', type=int, default=256,
                        help='ViT model dimension (ViT model only)')
    parser.add_argument('--vit_nhead', type=int, default=8,
                        help='ViT number of attention heads (ViT model only)')
    parser.add_argument('--vit_num_layers', type=int, default=6,
                        help='ViT number of transformer layers (ViT model only)')
    parser.add_argument('--vit_dim_feedforward', type=int, default=1024,
                        help='ViT feedforward dimension (ViT model only)')
    parser.add_argument('--vit_dropout', type=float, default=0.1,
                        help='ViT dropout rate (ViT model only)')
    
    # Static Hybrid-specific parameters
    parser.add_argument('--static_hybrid_cnn_channels', type=int, nargs='+', default=[64, 128, 256],
                        help='Static Hybrid CNN channel list (Static Hybrid model only)')
    parser.add_argument('--static_hybrid_rnn_hidden', type=int, default=256,
                        help='Static Hybrid RNN hidden dimension (Static Hybrid model only)')
    parser.add_argument('--static_hybrid_rnn_layers', type=int, default=2,
                        help='Static Hybrid RNN layers (Static Hybrid model only)')
    parser.add_argument('--static_hybrid_rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='Static Hybrid RNN type (LSTM or GRU)')
    parser.add_argument('--static_hybrid_dropout', type=float, default=0.2,
                        help='Static Hybrid dropout rate (Static Hybrid model only)')

    # LSTM/GRU/InceptionTime/MiniROCKET parameters
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--lstm_dropout', type=float, default=0.2)
    parser.add_argument('--inceptiontime_n_filters', type=int, default=32)
    parser.add_argument('--inceptiontime_depth', type=int, default=6)
    parser.add_argument('--inceptiontime_dropout', type=float, default=0.2)
    parser.add_argument('--minirocket_num_kernels', type=int, default=1000)
    parser.add_argument('--minirocket_seed', type=int, default=42)
    parser.add_argument('--minirocket_dropout', type=float, default=0.2)
    
    args = parser.parse_args()

    # Set up logging so all console output is also written to results/
    setup_logging(task_id=2, model_name=args.model)
    
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
    elif args.model in ['mamba', 's4']:
        model_kwargs = {
            'd_model': args.mamba_d_model,
            'n_layers': args.mamba_n_layers,
            'd_state': args.mamba_d_state,
            'dropout': args.mamba_dropout,
        }
    elif args.model == 'vit':
        model_kwargs = {
            'patch_size': args.vit_patch_size,
            'd_model': args.vit_d_model,
            'nhead': args.vit_nhead,
            'num_layers': args.vit_num_layers,
            'dim_feedforward': args.vit_dim_feedforward,
            'dropout': args.vit_dropout,
        }
    elif args.model == 'static_hybrid':
        model_kwargs = {
            'cnn_channels': args.static_hybrid_cnn_channels,
            'rnn_hidden': args.static_hybrid_rnn_hidden,
            'rnn_layers': args.static_hybrid_rnn_layers,
            'rnn_type': args.static_hybrid_rnn_type,
            'dropout': args.static_hybrid_dropout,
        }
    elif args.model == 'lstm':
        model_kwargs = {'hidden_size': args.lstm_hidden_size, 'num_layers': args.lstm_num_layers, 'dropout': args.lstm_dropout}
    elif args.model == 'gru':
        model_kwargs = {'hidden_size': args.lstm_hidden_size, 'num_layers': args.lstm_num_layers, 'dropout': args.lstm_dropout}
    elif args.model == 'inceptiontime':
        model_kwargs = {'n_filters': args.inceptiontime_n_filters, 'depth': args.inceptiontime_depth, 'dropout': args.inceptiontime_dropout}
    elif args.model == 'minirocket':
        model_kwargs = {'num_kernels': args.minirocket_num_kernels, 'seed': args.minirocket_seed, 'dropout': args.minirocket_dropout}

    print("=" * 60)
    print("Training Task 2")
    print("=" * 60)
    
    # Train a single multi-class model on all folders under datasets/2/
    train_model(
        data_dir="datasets",
        task_id=2,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=f"models/2/{args.model}",
        validation_split=args.validation_split,
        device=args.device,
        **model_kwargs
    )


if __name__ == "__main__":
    main()

