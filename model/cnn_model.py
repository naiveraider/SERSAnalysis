"""
CNN model for spectrum data classification - PyTorch implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectrumCNN(nn.Module):
    """
    CNN model for spectrum data classification
    """
    
    def __init__(self, input_length: int, num_classes: int):
        """
        Initialize model
        
        Args:
            input_length: Input sequence length
            num_classes: Number of classes
        """
        super(SpectrumCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Calculate feature length after convolution and pooling
        # After 3 pooling operations, each divides by 2
        feature_length = input_length // (2 ** 3)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)
        
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
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # (batch_size, 256)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x


def create_cnn_model(input_length: int, num_classes: int, device: str = 'cpu') -> SpectrumCNN:
    """
    Create CNN model for spectrum data classification
    
    Args:
        input_length: Input sequence length
        num_classes: Number of classes
        device: Device ('cpu' or 'cuda')
        
    Returns:
        PyTorch model
    """
    model = SpectrumCNN(input_length=input_length, num_classes=num_classes)
    model = model.to(device)
    return model


def get_optimizer_and_criterion(model: nn.Module, learning_rate: float = 0.001):
    """
    Get optimizer and loss function
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        
    Returns:
        optimizer, criterion
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion
