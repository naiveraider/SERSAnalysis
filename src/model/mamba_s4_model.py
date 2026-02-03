"""
Mamba / S4 (State Space Model) for spectrum data classification - PyTorch implementation
Simplified implementation of state space models for sequence modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class S4Layer(nn.Module):
    """
    Simplified S4 (State Space Sequence) layer
    Implements a learnable state space model for sequence processing
    """
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super(S4Layer, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        x = self.norm(x)
        
        # State space computation (simplified)
        # For efficiency, we use a simplified version
        # Full S4 would use structured state space matrices
        
        # Apply state space transformation
        # Using a simplified recurrent form
        states = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # State update: s_t = A * s_{t-1} + B * x_t
            states = torch.matmul(states, self.A.T) + torch.matmul(x[:, t], self.B.T)
            # Output: y_t = C * s_t + D * x_t
            out = torch.matmul(states, self.C.T) + torch.matmul(x[:, t], self.D.T)
            outputs.append(out)
        
        x = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
        
        # Output projection
        x = self.output_proj(x)
        x = self.dropout(x)
        
        return x


class SpectrumMambaS4(nn.Module):
    """
    Mamba / S4 model for spectrum data classification
    Uses state space models for efficient long-range sequence modeling
    """
    def __init__(self, input_length: int, num_classes: int,
                 d_model: int = 256,
                 n_layers: int = 4,
                 d_state: int = 64,
                 dropout: float = 0.1):
        """
        Initialize Mamba/S4 model
        
        Args:
            input_length: Input sequence length
            num_classes: Number of classes
            d_model: Model dimension
            n_layers: Number of S4 layers
            d_state: State dimension
            dropout: Dropout rate
        """
        super(SpectrumMambaS4, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)
        
        # S4 layers
        self.s4_layers = nn.ModuleList([
            S4Layer(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization between layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.d_model = d_model
        self.input_length = input_length
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, sequence_length)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)
        
        # Input embedding: (batch_size, seq_len, 1) -> (batch_size, seq_len, d_model)
        x = self.input_embedding(x)
        
        # Apply S4 layers
        for s4_layer, norm in zip(self.s4_layers, self.layer_norms):
            residual = x
            x = s4_layer(x)
            x = norm(x + residual)  # Residual connection
        
        # Global pooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_mamba_s4_model(input_length: int, num_classes: int, device: str = 'cpu',
                          d_model: int = 256,
                          n_layers: int = 4,
                          d_state: int = 64,
                          dropout: float = 0.1) -> SpectrumMambaS4:
    """
    Create Mamba/S4 model for spectrum data classification
    
    Args:
        input_length: Input sequence length
        num_classes: Number of classes
        device: Device ('cpu' or 'cuda')
        d_model: Model dimension
        n_layers: Number of S4 layers
        d_state: State dimension
        dropout: Dropout rate
        
    Returns:
        PyTorch model
    """
    model = SpectrumMambaS4(
        input_length=input_length,
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        dropout=dropout
    )
    model = model.to(device)
    return model
