#!/usr/bin/env python3
"""Train ablation pooling models and average results."""
import sys
import os
import re
import datetime
import io
import contextlib
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / '.venv' / 'bin' / 'python'

if os.environ.get('SERS_ANALYSIS_SKIP_REEXEC') != '1' and VENV_PYTHON.exists():
    current_python = Path(sys.executable).resolve() if sys.executable else None
    target_python = VENV_PYTHON.resolve()
    if current_python != target_python:
        env = os.environ.copy()
        env['SERS_ANALYSIS_SKIP_REEXEC'] = '1'
        os.execve(str(target_python), [str(target_python), __file__, *sys.argv[1:]], env)

# Add parent directory to path
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_model


BASE_MODELS = (
    'cnn',
    'tcn',
    'vit',
    'inceptiontime',
    'lstm',
    'gru',
    'cnn_lstm',
    'cnn_transformer',
    'mamba',
)

POOLING_METHODS = ('mean', 'max', 'meanmax')

BASE_MODEL_PARAMS = {
    'cnn': {},
    'tcn': {
        'num_channels': [64, 128, 256],
        'kernel_size': 3,
        'dropout': 0.2,
    },
    'vit': {
        'patch_size': 16,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
    },
    'inceptiontime': {
        'n_filters': 32,
        'depth': 6,
        'dropout': 0.2,
    },
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'gru': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'cnn_lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'cnn_transformer': {
        'cnn_channels': [64, 128, 256],
        'd_model': 256,
        'nhead': 8,
        'num_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1,
    },
    'mamba': {
        'd_model': 256,
        'n_layers': 4,
        'd_state': 64,
        'dropout': 0.1,
    },
}


SUPPORTED_MODELS = tuple(
    f"{base_model}_{pooling_method}"
    for base_model in BASE_MODELS
    for pooling_method in POOLING_METHODS
)

FINAL_METRIC_KEYS = {
    'accuracy_score': 'FINAL_TEST_ACC',
    'f1_score': 'FINAL_TEST_F1',
    'jaccard_score': 'FINAL_TEST_JACCARD',
    'average_precision_score': 'FINAL_TEST_AUPRC',
    'roc_auc_score': 'FINAL_TEST_AUROC',
}


def compute_average(values, precision):
    """Average numeric metric values, skipping NaN entries."""
    numeric_values = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric_value):
            continue
        numeric_values.append(numeric_value)

    if not numeric_values:
        return 'nan'

    return f"{sum(numeric_values) / len(numeric_values):.{precision}f}"


def extract_final_metrics(full_output):
    """Extract parseable final metrics from training output."""
    metrics = {metric_name: None for metric_name in FINAL_METRIC_KEYS}

    for line in full_output.splitlines():
        for metric_name, output_key in FINAL_METRIC_KEYS.items():
            prefix = f'{output_key}='
            if line.startswith(prefix):
                metrics[metric_name] = line[len(prefix):].strip()

    return metrics


def run_n_times_and_average(results_file, label, n_runs, model_name, task_id, train_args):
    """Train a model n times, extract final metrics, and compute averages."""
    count = 0
    per_run_metrics = []
    
    for i in range(1, n_runs + 1):
        print(f"---------- {label} Run {i}/{n_runs} ----------")
        
        try:
            # Capture stdout to extract FINAL_TEST_ACC
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                # Train the model
                train_model(
                    model_name=model_name,
                    task_id=task_id,
                    **train_args
                )
            
            full_output = f.getvalue()
            
            # Print output to console
            print(full_output)
            
            metrics = extract_final_metrics(full_output)

            if all(value is not None for value in metrics.values()):
                count += 1
                per_run_metrics.append(metrics)
                print(
                    "Extracted metrics: "
                    f"accuracy_score={metrics['accuracy_score']}% "
                    f"f1_score={metrics['f1_score']} "
                    f"jaccard_score={metrics['jaccard_score']} "
                    f"average_precision_score={metrics['average_precision_score']} "
                    f"roc_auc_score={metrics['roc_auc_score']}"
                )
            else:
                print(f"WARNING: Could not extract all final metrics from run {i}")
        
        except Exception as e:
            print(f"ERROR in run {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate and display average
    if count > 0:
        average_metrics = {
            'accuracy_score': compute_average(
                [metrics['accuracy_score'] for metrics in per_run_metrics],
                precision=2,
            ),
            'f1_score': compute_average(
                [metrics['f1_score'] for metrics in per_run_metrics],
                precision=4,
            ),
            'jaccard_score': compute_average(
                [metrics['jaccard_score'] for metrics in per_run_metrics],
                precision=4,
            ),
            'average_precision_score': compute_average(
                [metrics['average_precision_score'] for metrics in per_run_metrics],
                precision=4,
            ),
            'roc_auc_score': compute_average(
                [metrics['roc_auc_score'] for metrics in per_run_metrics],
                precision=4,
            ),
        }
        print("")
        print("=" * 50)
        print(f"{label} - Average Test Metrics ({count}/{n_runs} runs):")
        print(f"  accuracy_score: {average_metrics['accuracy_score']}%")
        print(f"  f1_score: {average_metrics['f1_score']}")
        print(f"  jaccard_score: {average_metrics['jaccard_score']}")
        print(f"  average_precision_score: {average_metrics['average_precision_score']}")
        print(f"  roc_auc_score: {average_metrics['roc_auc_score']}")
        print("=" * 50)
        
        # Write to results file
        with open(results_file, 'a') as f:
            f.write(f"Model: {label}\n")
            for idx, metrics in enumerate(per_run_metrics, 1):
                f.write(
                    f"  Run {idx}: accuracy_score={metrics['accuracy_score']}%, "
                    f"f1_score={metrics['f1_score']}, "
                    f"jaccard_score={metrics['jaccard_score']}, "
                    f"average_precision_score={metrics['average_precision_score']}, "
                    f"roc_auc_score={metrics['roc_auc_score']}\n"
                )
            f.write(
                f"  Average: accuracy_score={average_metrics['accuracy_score']}%, "
                f"f1_score={average_metrics['f1_score']}, "
                f"jaccard_score={average_metrics['jaccard_score']}, "
                f"average_precision_score={average_metrics['average_precision_score']}, "
                f"roc_auc_score={average_metrics['roc_auc_score']}\n"
            )
            f.write("\n")
    else:
        print(f"WARNING: No successful runs for {label}")


def build_train_args(model, epochs, batch_size, learning_rate, validation_split, device):
    """Build training arguments for a specific ablation model."""
    
    train_args = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'validation_split': validation_split,
    }
    
    if device:
        train_args['device'] = device
    
    matched_base_model = next(
        (base_model for base_model in BASE_MODELS if model == f"{base_model}_mean" or model == f"{base_model}_max" or model == f"{base_model}_meanmax"),
        None,
    )

    if matched_base_model is None:
        print(f"ERROR: Unknown model '{model}'. Supported models: {list(SUPPORTED_MODELS)}")
        sys.exit(1)

    train_args.update(BASE_MODEL_PARAMS[matched_base_model])
    return train_args


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
                description='Train ablation pooling models for Task 1 and average results',
        usage='%(prog)s [options] <model1> [model2 ...]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog=f"""
Examples:
    python script/ablation_test.py lstm_mean lstm_max lstm_meanmax
    N_RUNS=10 EPOCHS=100 python script/ablation_test.py cnn_mean vit_max tcn_meanmax

Supported models:
    {', '.join(SUPPORTED_MODELS)}
                """
    )
    
    parser.add_argument('models', nargs='*', help='Models to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs per model')
    parser.add_argument('--device', '--cuda', dest='device', type=str, default=None, help='Device (cpu/cuda/cuda:0)')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID to train (default: 1)')
    
    args = parser.parse_args()
    
    # Check if models provided
    if not args.models:
        parser.print_help()
        sys.exit(0)
    
    # Get N_RUNS from environment variable if set
    n_runs = int(os.environ.get('N_RUNS', args.n_runs))
    epochs = int(os.environ.get('EPOCHS', args.epochs))
    task_id = int(os.environ.get('TASK_ID', args.task_id))
    
    # Setup results directory and file
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"ablation_pooling_task{task_id}_{timestamp}.txt"
    
    # Write header to results file
    with open(results_file, 'w') as f:
        f.write(f"Task {task_id} - Ablation Pooling Models Test Metrics Summary\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write(f"N_RUNS={n_runs}\n")
        f.write(f"Models: {' '.join(args.models)}\n")
        f.write("\n")
    
    # Process each model
    for model in args.models:
        print("")
        print("=" * 50)
        print(f"Training {model} for Task {task_id} ({n_runs} runs, result averaged)")
        print("=" * 50)
        
        # Build training arguments
        train_args = build_train_args(model, epochs, args.batch_size, args.learning_rate, args.validation_split, args.device)
        
        # Run n times and average
        run_n_times_and_average(str(results_file), model, n_runs, model, task_id, train_args)
    
    print("")
    print(f"Results written to {results_file}")


if __name__ == '__main__':
    main()
