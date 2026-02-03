"""
Static Hybrid model for spectrum data classification - PyTorch implementation
Combines CNN, RNN (LSTM/GRU), and static features for classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectrumStaticHybrid(nn.Module):
    """
    Static Hybrid model combining CNN, RNN, and static feature extraction
    """
    def __init__(self, input_length: int, num_classes: int,
                 cnn_channels: list = [64, 128, 256],
                 rnn_hidden: int = 256,
                 rnn_layers: int = 2,
                 rnn_type: str = 'LSTM',
                 dropout: float = 0.2):
        """
        Initialize Static Hybrid model
        
        Args:
            input_length: Input sequence length
            num_classes: Number of classes
            cnn_channels: CNN channel list for each layer
            rnn_hidden: RNN hidden dimension
            rnn_layers: Number of RNN layers
            rnn_type: RNN type ('LSTM' or 'GRU')
            dropout: Dropout rate
        """
        super(SpectrumStaticHybrid, self).__init__()
        
        # CNN branch for local feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # Calculate sequence length after CNN
        seq_len_after_cnn = input_length // (2 ** len(cnn_channels))
        
        # RNN branch for sequential modeling
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=cnn_channels[-1],
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=True
            )
            rnn_output_size = rnn_hidden * 2  # bidirectional
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=cnn_channels[-1],
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=True
            )
            rnn_output_size = rnn_hidden * 2  # bidirectional
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'LSTM' or 'GRU'")
        
        # Static feature extraction (statistical features)
        self.static_feature_size = 8  # mean, std, min, max, median, q25, q75, range
        self.static_projection = nn.Linear(self.static_feature_size, 128)
        
        # Feature fusion
        fusion_input_size = cnn_channels[-1] + rnn_output_size + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self.seq_len_after_cnn = seq_len_after_cnn
        self.rnn_type = rnn_type
        
    def extract_static_features(self, x):
        """
        Extract statistical features from input
        
        Args:
            x: Input tensor (batch_size, 1, sequence_length)
            
        Returns:
            Static features (batch_size, static_feature_size)
        """
        x = x.squeeze(1)  # (batch_size, sequence_length)
        
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        median = x.median(dim=1, keepdim=True)[0]
        q25 = x.quantile(0.25, dim=1, keepdim=True)
        q75 = x.quantile(0.75, dim=1, keepdim=True)
        range_val = max_val - min_val
        
        static_features = torch.cat([
            mean, std, min_val, max_val, median, q25, q75, range_val
        ], dim=1)
        
        return static_features
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, sequence_length)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # CNN branch
        cnn_features = x
        for cnn_layer in self.cnn_layers:
            cnn_features = cnn_layer(cnn_features)
        
        # Global average pooling for CNN
        cnn_global = F.adaptive_avg_pool1d(cnn_features, 1).squeeze(-1)  # (batch_size, channels)
        
        # RNN branch
        # Reshape CNN features for RNN: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        cnn_seq = cnn_features.transpose(1, 2)  # (batch_size, seq_len, channels)
        
        rnn_output, _ = self.rnn(cnn_seq)  # (batch_size, seq_len, rnn_hidden * 2)
        # Use last time step
        rnn_global = rnn_output[:, -1]  # (batch_size, rnn_hidden * 2)
        
        # Static features
        static_features = self.extract_static_features(x)  # (batch_size, static_feature_size)
        static_proj = self.static_projection(static_features)  # (batch_size, 128)
        
        # Feature fusion
        fused_features = torch.cat([cnn_global, rnn_global, static_proj], dim=1)  # (batch_size, fusion_size)
        fused_features = self.fusion(fused_features)  # (batch_size, 256)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output


def create_static_hybrid_model(input_length: int, num_classes: int, device: str = 'cpu',
                              cnn_channels: list = None,
                              rnn_hidden: int = 256,
                              rnn_layers: int = 2,
                              rnn_type: str = 'LSTM',
                              dropout: float = 0.2) -> SpectrumStaticHybrid:
    """
    Create Static Hybrid model for spectrum data classification
    
    Args:
        input_length: Input sequence length
        num_classes: Number of classes
        device: Device ('cpu' or 'cuda')
        cnn_channels: CNN channel list for each layer
        rnn_hidden: RNN hidden dimension
        rnn_layers: Number of RNN layers
        rnn_type: RNN type ('LSTM' or 'GRU')
        dropout: Dropout rate
        
    Returns:
        PyTorch model
    """
    if cnn_channels is None:
        cnn_channels = [64, 128, 256]
    
    model = SpectrumStaticHybrid(
        input_length=input_length,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers,
        rnn_type=rnn_type,
        dropout=dropout
    )
    model = model.to(device)
    return model
