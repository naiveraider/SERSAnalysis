"""
CNN + Transformer hybrid model for spectrum data classification - PyTorch implementation
Combines CNN for local feature extraction and Transformer for global dependency modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention
    Compatible with PyTorch 2.x by accepting is_causal parameter
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        """
        Forward pass
        
        Args:
            src: Input tensor (seq_len, batch_size, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Key padding mask
            is_causal: Causal mask flag (for PyTorch 2.x compatibility, ignored)
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class SpectrumCNNTransformer(nn.Module):
    """
    CNN + Transformer hybrid model for spectrum data classification
    """
    def __init__(self, input_length: int, num_classes: int,
                 cnn_channels: list = [64, 128, 256],
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        """
        Initialize CNN + Transformer model
        
        Args:
            input_length: Input sequence length
            num_classes: Number of classes
            cnn_channels: CNN channel list for each layer
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super(SpectrumCNNTransformer, self).__init__()
        
        # CNN backbone for local feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.25)
            ))
            in_channels = out_channels
        
        # Calculate sequence length after CNN
        seq_len_after_cnn = input_length // (2 ** len(cnn_channels))
        
        # Projection layer to match transformer input dimension
        self.cnn_projection = nn.Linear(cnn_channels[-1], d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_length, dropout=dropout)
        
        # Transformer encoder layers
        # Use PyTorch's built-in TransformerEncoderLayer for better compatibility
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # Our input is (seq_len, batch_size, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature fusion: combine CNN and Transformer features
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
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
        self.seq_len_after_cnn = seq_len_after_cnn
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, sequence_length)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        cnn_features = x
        for cnn_layer in self.cnn_layers:
            cnn_features = cnn_layer(cnn_features)
        
        # CNN global feature (global average pooling)
        cnn_global = F.adaptive_avg_pool1d(cnn_features, 1).squeeze(-1)  # (batch_size, channels)
        cnn_global_proj = self.cnn_projection(cnn_global)  # (batch_size, d_model)
        
        # Prepare sequence features for Transformer
        # Reshape CNN features: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        cnn_seq = cnn_features.transpose(1, 2)  # (batch_size, seq_len, channels)
        cnn_seq_proj = self.cnn_projection(cnn_seq)  # (batch_size, seq_len, d_model)
        
        # Transformer expects (seq_len, batch_size, d_model)
        cnn_seq_proj = cnn_seq_proj.transpose(0, 1)  # (seq_len, batch_size, d_model)
        
        # Add positional encoding
        cnn_seq_proj = self.pos_encoder(cnn_seq_proj)
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(cnn_seq_proj)  # (seq_len, batch_size, d_model)
        
        # Global pooling: average over sequence dimension
        transformer_global = transformer_output.mean(dim=0)  # (batch_size, d_model)
        
        # Feature fusion: concatenate CNN and Transformer global features
        fused_features = torch.cat([cnn_global_proj, transformer_global], dim=1)  # (batch_size, d_model * 2)
        
        # Reshape for batch normalization (BN expects 2D or 4D)
        fused_features = fused_features.unsqueeze(-1)  # (batch_size, d_model * 2, 1)
        fused_features = self.fusion(fused_features.squeeze(-1))  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output


def create_cnn_transformer_model(input_length: int, num_classes: int, device: str = 'cpu',
                                 cnn_channels: list = None,
                                 d_model: int = 256,
                                 nhead: int = 8,
                                 num_layers: int = 2,
                                 dim_feedforward: int = 512,
                                 dropout: float = 0.1) -> SpectrumCNNTransformer:
    """
    Create CNN + Transformer hybrid model for spectrum data classification
    
    Args:
        input_length: Input sequence length
        num_classes: Number of classes
        device: Device ('cpu' or 'cuda')
        cnn_channels: CNN channel list for each layer
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward network dimension
        dropout: Dropout rate
        
    Returns:
        PyTorch model
    """
    if cnn_channels is None:
        cnn_channels = [64, 128, 256]
    
    model = SpectrumCNNTransformer(
        input_length=input_length,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    model = model.to(device)
    return model

