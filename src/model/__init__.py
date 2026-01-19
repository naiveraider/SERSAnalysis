"""
Model module
"""
from .cnn_model import create_cnn_model, get_optimizer_and_criterion, SpectrumCNN
from .tcn_model import create_tcn_model, SpectrumTCN
from .cnn_transformer_model import create_cnn_transformer_model, SpectrumCNNTransformer
from .model_factory import (
    create_model, 
    get_available_models, 
    get_model_class, 
    get_model_description,
    MODEL_REGISTRY
)

__all__ = [
    'create_cnn_model', 
    'get_optimizer_and_criterion', 
    'SpectrumCNN',
    'create_tcn_model',
    'SpectrumTCN',
    'create_cnn_transformer_model',
    'SpectrumCNNTransformer',
    'create_model',
    'get_available_models',
    'get_model_class',
    'get_model_description',
    'MODEL_REGISTRY'
]

