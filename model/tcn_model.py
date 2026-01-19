"""
TCN (Temporal Convolutional Network) model for spectrum data classification - PyTorch implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalBlock(nn.Module):
    """
    TCN basic block: contains causal convolution, weight normalization, ReLU and Dropout
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Remove extra padding from causal convolution
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SpectrumTCN(nn.Module):
    """
    TCN model for spectrum data classification
    """
    def __init__(self, input_length: int, num_classes: int, 
                 num_channels: list = [64, 128, 256], kernel_size: int = 3, dropout: float = 0.2):
        """
        Initialize TCN model
        
        Args:
            input_length: Input sequence length
            num_classes: Number of classes
            num_channels: Channel list for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super(SpectrumTCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Calculate padding to maintain sequence length
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                    stride=1, dilation=dilation_size, 
                                    padding=padding, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_channels[-1], 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, sequence_length)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        # TCN layers
        x = self.network(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # (batch_size, num_channels[-1])
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x


def create_tcn_model(input_length: int, num_classes: int, device: str = 'cpu',
                     num_channels: list = None, kernel_size: int = 3, 
                     dropout: float = 0.2) -> SpectrumTCN:
    """
    Create TCN model for spectrum data classification
    
    Args:
        input_length: Input sequence length
        num_classes: Number of classes
        device: Device ('cpu' or 'cuda')
        num_channels: Channel list for each layer
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        
    Returns:
        PyTorch model
    """
    if num_channels is None:
        num_channels = [64, 128, 256]
    
    model = SpectrumTCN(
        input_length=input_length,
        num_classes=num_classes,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
    model = model.to(device)
    return model
