"""
Train models for spectrum data classification - PyTorch implementation
Supports multiple models: CNN, TCN, etc.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .data_loader import SpectrumDataLoader
from .model.model_factory import create_model, get_available_models, get_model_description
from .model.cnn_model import get_optimizer_and_criterion
import argparse
import pickle
import json


class SpectrumDataset(Dataset):
    """PyTorch dataset class"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_model(
    data_dir: str = "datasets",
    task_id: int = None,
    model_name: str = "cnn",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "model/saved_model",
    validation_split: float = 0.2,
    device: str = None,
    folder_path: Path = None,
    **model_kwargs
):
    """
    Train model
    
    Args:
        data_dir: Dataset directory
        task_id: Task ID (1-4), if specified, only load data from datasets/{task_id}/
        model_name: Model name ('cnn' or 'tcn')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_save_path: Model save path
        validation_split: Validation set ratio
        device: Device ('cpu' or 'cuda')
        folder_path: Path to a specific folder (if provided, train model only on this folder's data)
        **model_kwargs: Model-specific parameters (e.g., num_channels, kernel_size for TCN)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("=" * 50)
    print("Loading data...")
    print("=" * 50)
    
    # Load data
    loader = SpectrumDataLoader(data_dir=data_dir, task_id=task_id)
    
    # If folder_path is provided, load data from that specific folder
    if folder_path is not None:
        folder_path = Path(folder_path)
        X_train, X_test, y_train, y_test, class_names = loader.prepare_data_from_folder(
            folder_path=folder_path,
            test_size=validation_split
        )
    else:
        X_train, X_test, y_train, y_test, class_names = loader.prepare_data(
            test_size=validation_split
        )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Check if we have enough data
    if len(X_train) == 0:
        raise ValueError("No training data available")
    
    if len(X_test) == 0:
        print("Warning: No test data available. Using training data for validation.")
        X_test = X_train
        y_test = y_train
    
    print(f"Input length: {X_train.shape[2]}")
    
    # Handle single class case - for classification, we need at least 2 classes
    # If only one class, we'll use 2 classes (the single class + a dummy class)
    num_classes = len(class_names)
    if num_classes == 1:
        print("Warning: Only one class found. This is not suitable for classification.")
        print("Consider using an autoencoder or ensuring multiple classes exist.")
        # For now, we'll still proceed but the model will have 1 output class
        # The training might not work well, but we'll try
    
    # Create dataset and data loader
    train_dataset = SpectrumDataset(X_train, y_train)
    test_dataset = SpectrumDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("\n" + "=" * 50)
    print(f"Creating {model_name.upper()} model...")
    print(f"Model description: {get_model_description(model_name)}")
    print("=" * 50)
    
    input_length = X_train.shape[2]
    model = create_model(
        model_name=model_name,
        input_length=input_length,
        num_classes=num_classes,
        device=device,
        **model_kwargs
    )
    
    # Display model structure
    print(model)
    
    # Get optimizer and loss function
    optimizer, criterion = get_optimizer_and_criterion(model, learning_rate)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Train model
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    os.makedirs(model_save_path, exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(model_save_path, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'input_length': input_length,
                'num_classes': len(class_names),
                'epoch': epoch,
                'val_acc': val_acc,
                'model_kwargs': model_kwargs,
            }, best_model_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model (validation accuracy: {best_val_acc:.2f}%)')
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Evaluating model...")
    print("=" * 50)
    
    final_val_loss, final_val_acc = validate(model, test_loader, criterion, device)
    print(f"Test loss: {final_val_loss:.4f}")
    print(f"Test accuracy: {final_val_acc:.2f}%")
    # Parseable line for scripts (e.g. average over multiple runs)
    print(f"FINAL_TEST_ACC={final_val_acc}")
    
    # Save final model
    final_model_path = os.path.join(model_save_path, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'input_length': input_length,
        'num_classes': len(class_names),
        'epoch': epochs,
        'val_acc': final_val_acc,
        'model_kwargs': model_kwargs,
    }, final_model_path)
    
    # Save model configuration
    config_path = os.path.join(model_save_path, 'model_config.json')
    config = {
        'model_name': model_name,
        'input_length': input_length,
        'num_classes': len(class_names),
        'model_kwargs': model_kwargs,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model configuration saved to: {config_path}")
    print(f"\nModel saved to: {final_model_path}")
    
    # Save class names and scaler
    class_names_path = os.path.join(model_save_path, 'class_names.pkl')
    with open(class_names_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"Class names saved to: {class_names_path}")
    
    scaler_path = os.path.join(model_save_path, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(loader.scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    return model, {'train_loss': train_losses, 'train_acc': train_accs,
                   'val_loss': val_losses, 'val_acc': val_accs}, class_names


def main():
    parser = argparse.ArgumentParser(description='Train models for spectrum data classification')
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Dataset directory path')
    parser.add_argument('--task_id', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Task ID (1-4), if specified, only load data from datasets/{task_id}/')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=get_available_models(),
                        help=f'Model type (available: {", ".join(get_available_models())})')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='model/saved_model',
                        help='Model save path')
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
    
    train_model(
        data_dir=args.data_dir,
        task_id=args.task_id,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=args.model_save_path,
        validation_split=args.validation_split,
        device=args.device,
        **model_kwargs
    )


if __name__ == "__main__":
    main()
