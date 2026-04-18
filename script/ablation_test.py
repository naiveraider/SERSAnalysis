"""
Train ablation MeanMax models for Task 1 and average results.
Follows the same pattern as train_task1_selected_models.sh
"""
import sys
import os
import subprocess
import re
import tempfile
import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_n_times_and_average(results_file, label, n_runs, cmd_args):
    """Run a command n times, extract FINAL_TEST_ACC, and compute average."""
    sum_acc = 0
    count = 0
    accs = []
    
    for i in range(1, n_runs + 1):
        print(f"---------- {label} Run {i}/{n_runs} ----------")
        
        # Create a temporary file to capture output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as tmp:
            tmp_path = tmp.name
        
        try:
            # Run the training command and capture output
            result = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            full_output = result.stdout
            
            # Write output to temporary file
            with open(tmp_path, 'w') as f:
                f.write(full_output)
            
            # Print output to console
            print(full_output)
            
            # Extract FINAL_TEST_ACC using regex
            acc = None
            for line in full_output.split('\n'):
                match = re.search(r'FINAL_TEST_ACC=([0-9.]+)', line)
                if match:
                    acc = match.group(1)
            
            if acc is not None:
                sum_acc += float(acc)
                count += 1
                accs.append(acc)
                print(f"Extracted accuracy: {acc}%")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    # Calculate and display average
    if count > 0:
        avg = sum_acc / count
        print("")
        print("=" * 50)
        print(f"{label} - Average Test Accuracy ({count}/{n_runs} runs): {avg:.2f}%")
        print("=" * 50)
        
        # Write to results file
        with open(results_file, 'a') as f:
            f.write(f"Model: {label}\n")
            for idx, acc in enumerate(accs, 1):
                f.write(f"  Run {idx}: {acc}%\n")
            f.write(f"  Average: {avg:.2f}%\n")
            f.write("\n")
    else:
        print(f"WARNING: No successful runs for {label}")


def build_model_args(model):
    """Build extra arguments specific to each ablation model."""
    args = {}
    
    model_mapping = {
        'cnn_meanmax': [],
        'tcn_meanmax': ['--tcn_num_channels', '64', '128', '256', '--tcn_kernel_size', '3', '--tcn_dropout', '0.2'],
        'vit_meanmax': ['--vit_patch_size', '16', '--vit_d_model', '256', '--vit_nhead', '8', 
                        '--vit_num_layers', '6', '--vit_dim_feedforward', '1024', '--vit_dropout', '0.1'],
        'inceptiontime_meanmax': ['--inceptiontime_n_filters', '32', '--inceptiontime_depth', '6', '--inceptiontime_dropout', '0.2'],
        'lstm_meanmax': ['--lstm_hidden_size', '128', '--lstm_num_layers', '2', '--lstm_dropout', '0.2'],
        'cnn_lstm_meanmax': ['--lstm_hidden_size', '128', '--lstm_num_layers', '2', '--lstm_dropout', '0.2'],
        'cnn_transformer_meanmax': ['--cnn_transformer_cnn_channels', '64', '128', '256', '--cnn_transformer_d_model', '256',
                                    '--cnn_transformer_nhead', '8', '--cnn_transformer_num_layers', '2',
                                    '--cnn_transformer_dim_feedforward', '512', '--cnn_transformer_dropout', '0.1'],
        'mamba_meanmax': ['--mamba_d_model', '256', '--mamba_n_layers', '4', '--mamba_d_state', '64', '--mamba_dropout', '0.1'],
    }
    
    if model not in model_mapping:
        print(f"ERROR: Unknown model '{model}'. Supported models: {list(model_mapping.keys())}")
        sys.exit(1)
    
    return model_mapping[model]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train ablation MeanMax models for Task 1 and average results',
        usage='%(prog)s [options] <model1> [model2 ...]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/ablation_test.py lstm_meanmax cnn_meanmax
  N_RUNS=10 EPOCHS=100 python script/ablation_test.py lstm_meanmax vit_meanmax tcn_meanmax

Supported models:
  cnn_meanmax, tcn_meanmax, vit_meanmax, inceptiontime_meanmax,
  lstm_meanmax, cnn_lstm_meanmax, cnn_transformer_meanmax, mamba_meanmax
        """
    )
    
    parser.add_argument('models', nargs='*', help='Models to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs per model')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Check if models provided
    if not args.models:
        parser.print_help()
        sys.exit(0)
    
    # Get N_RUNS from environment variable if set
    n_runs = int(os.environ.get('N_RUNS', args.n_runs))
    epochs = int(os.environ.get('EPOCHS', args.epochs))
    
    # Setup results directory and file
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"ablation_meanmax_{timestamp}.txt"
    
    # Write header to results file
    with open(results_file, 'w') as f:
        f.write("Task 1 - Ablation MeanMax Models Test Accuracy Summary\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write(f"N_RUNS={n_runs}\n")
        f.write(f"Models: {' '.join(args.models)}\n")
        f.write("\n")
    
    # Process each model
    for model in args.models:
        print("")
        print("=" * 50)
        print(f"Training {model} for Task 1 ({n_runs} runs, result averaged)")
        print("=" * 50)
        
        # Build model-specific arguments
        extra_args = build_model_args(model)
        
        # Build command
        cmd = [
            'python', 'script/train_task1.py',
            '--model', model,
            '--epochs', str(epochs),
            '--batch_size', str(args.batch_size),
            '--learning_rate', str(args.learning_rate),
            '--validation_split', str(args.validation_split),
        ]
        
        if args.device:
            cmd.extend(['--device', args.device])
        
        cmd.extend(extra_args)
        
        # Run n times and average
        run_n_times_and_average(str(results_file), model, n_runs, cmd)
    
    print("")
    print(f"Results written to {results_file}")


if __name__ == '__main__':
    main()
