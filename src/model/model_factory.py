"""
Model factory: unified management of all models
"""
import torch.nn as nn
from typing import Dict, Callable, Any
from .cnn_model import create_cnn_model, SpectrumCNN
from .tcn_model import create_tcn_model, SpectrumTCN
from .cnn_transformer_model import create_cnn_transformer_model, SpectrumCNNTransformer
from .mamba_s4_model import create_mamba_s4_model, SpectrumMambaS4
from .vit_model import create_vit_model, SpectrumViT
from .static_hybrid_model import create_static_hybrid_model, SpectrumStaticHybrid


# Model registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'cnn': {
        'create_fn': create_cnn_model,
        'model_class': SpectrumCNN,
        'description': 'Standard 1D CNN model'
    },
    'tcn': {
        'create_fn': create_tcn_model,
        'model_class': SpectrumTCN,
        'description': 'Temporal Convolutional Network (TCN) model'
    },
    'cnn_transformer': {
        'create_fn': create_cnn_transformer_model,
        'model_class': SpectrumCNNTransformer,
        'description': 'CNN + Transformer hybrid model'
    },
    'mamba': {
        'create_fn': create_mamba_s4_model,
        'model_class': SpectrumMambaS4,
        'description': 'Mamba / S4 (State Space Model) for efficient long-range sequence modeling'
    },
    's4': {
        'create_fn': create_mamba_s4_model,
        'model_class': SpectrumMambaS4,
        'description': 'Mamba / S4 (State Space Model) for efficient long-range sequence modeling'
    },
    'vit': {
        'create_fn': create_vit_model,
        'model_class': SpectrumViT,
        'description': 'Vision Transformer (ViT) adapted for 1D spectrum data'
    },
    'static_hybrid': {
        'create_fn': create_static_hybrid_model,
        'model_class': SpectrumStaticHybrid,
        'description': 'Static Hybrid model combining CNN, RNN, and statistical features'
    }
}


def get_available_models() -> list:
    """
    Get all available model names
    
    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())


def create_model(model_name: str, input_length: int, num_classes: int, 
                 device: str = 'cpu', **kwargs) -> nn.Module:
    """
    Create model
    
    Args:
        model_name: Model name ('cnn', 'tcn', or 'cnn_transformer')
        input_length: Input sequence length
        num_classes: Number of classes
        device: Device ('cpu' or 'cuda')
        **kwargs: Model-specific parameters
        
    Returns:
        PyTorch model
        
    Raises:
        ValueError: If model name does not exist
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(get_available_models())
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available}")
    
    create_fn = MODEL_REGISTRY[model_name]['create_fn']
    
    # Call creation function with common parameters and specific parameters
    return create_fn(input_length=input_length, num_classes=num_classes, 
                     device=device, **kwargs)


def get_model_class(model_name: str):
    """
    Get model class
    
    Args:
        model_name: Model name
        
    Returns:
        Model class
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(get_available_models())
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available}")
    
    return MODEL_REGISTRY[model_name]['model_class']


def get_model_description(model_name: str) -> str:
    """
    Get model description
    
    Args:
        model_name: Model name
        
    Returns:
        Model description
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        return "Unknown model"
    
    return MODEL_REGISTRY[model_name]['description']
