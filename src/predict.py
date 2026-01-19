"""
Use trained models for prediction - PyTorch implementation
Supports multiple models: CNN, TCN, etc.
"""
import os
import numpy as np
import torch
from pathlib import Path
import pickle
import json
from .data_loader import SpectrumDataLoader
from .model.model_factory import get_model_class, create_model


def load_model_and_classes(model_path: str, device: str = None):
    """
    Load model and class names
    
    Args:
        model_path: Model save directory
        device: Device ('cpu' or 'cuda')
        
    Returns:
        model, class_names, scaler, input_length, model_name, device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Try to load model configuration
    config_file = os.path.join(model_path, 'model_config.json')
    model_name = None
    model_kwargs = {}
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            model_name = config.get('model_name', 'cnn')
            model_kwargs = config.get('model_kwargs', {})
    
    # Try to load best model
    model_file = os.path.join(model_path, 'best_model.pth')
    if not os.path.exists(model_file):
        model_file = os.path.join(model_path, 'final_model.pth')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    input_length = checkpoint['input_length']
    num_classes = checkpoint['num_classes']
    
    # Get model name from checkpoint (if exists)
    if model_name is None:
        model_name = checkpoint.get('model_name', 'cnn')
    
    # Get model parameters from checkpoint (if exists)
    if not model_kwargs:
        model_kwargs = checkpoint.get('model_kwargs', {})
    
    # Create model and load weights
    model = create_model(
        model_name=model_name,
        input_length=input_length,
        num_classes=num_classes,
        device=device,
        **model_kwargs
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load class names
    class_names_file = os.path.join(model_path, 'class_names.pkl')
    with open(class_names_file, 'rb') as f:
        class_names = pickle.load(f)
    
    # Load scaler
    scaler_file = os.path.join(model_path, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, class_names, scaler, input_length, model_name, device


def predict_single_file(model, class_names, loader: SpectrumDataLoader, 
                       file_path: str, input_length: int, scaler=None, device='cpu'):
    """
    Predict single file
    
    Args:
        model: Trained model
        class_names: Class name list
        loader: Data loader
        file_path: CSV file path
        input_length: Input sequence length
        scaler: Standardizer (if provided)
        device: Device
        
    Returns:
        Prediction results
    """
    # Load spectrum data
    spectrum = loader.load_spectrum_from_file(Path(file_path))
    
    if len(spectrum) == 0:
        raise ValueError("No valid spectrum data found in file")
    
    # Extract intensity values
    intensities = spectrum[:, 1]
    
    # Adjust data length
    if len(intensities) < input_length:
        intensities = np.pad(intensities, (0, input_length - len(intensities)), 
                           mode='constant')
    elif len(intensities) > input_length:
        intensities = intensities[:input_length]
    
    # Standardize
    if scaler is not None:
        intensities = scaler.transform([intensities])[0]
    else:
        # Simple standardization
        intensities = (intensities - np.mean(intensities)) / (np.std(intensities) + 1e-8)
    
    # Convert to PyTorch tensor format (1, 1, length)
    X = torch.FloatTensor(intensities).reshape(1, 1, len(intensities)).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)
        probabilities = probabilities.cpu().numpy()[0]
    
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    
    return predicted_class, confidence, probabilities


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Use trained models for prediction')
    parser.add_argument('--model_path', type=str, default='model/saved_model',
                        help='Model save path')
    parser.add_argument('--file_path', type=str, required=True,
                        help='CSV file path to predict')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda, default: auto-select)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model, class_names, scaler, input_length, model_name, device = load_model_and_classes(
        args.model_path, args.device
    )
    print(f"Model type: {model_name.upper()}")
    print(f"Classes: {class_names}")
    print(f"Using device: {device}")
    
    # Create data loader
    loader = SpectrumDataLoader()
    
    # Make prediction
    print(f"\nPredicting file: {args.file_path}")
    predicted_class, confidence, all_probs = predict_single_file(
        model, class_names, loader, args.file_path, input_length, scaler, device
    )
    
    print(f"\nPrediction results:")
    print(f"  Class: {predicted_class}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"\nProbabilities for all classes:")
    for i, (class_name, prob) in enumerate(zip(class_names, all_probs)):
        print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()
