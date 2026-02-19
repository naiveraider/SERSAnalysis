"""
MiniROCKET model for spectrum data classification - PyTorch implementation
Based on: Dempster et al. "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification"
Uses random convolutional kernels (fixed) + trainable classifier head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def _generate_minirocket_kernels(seq_len: int, num_kernels: int = 1000, seed: int = 42):
    """
    Generate random kernels for MiniROCKET.
    Each kernel: length 9, weights in {-1, 2}, bias in U(-1,1), dilation from 2^0 to 2^max.
    """
    np.random.seed(seed)
    kernels = []
    kernel_length = 9
    max_dilation = max(0, int(math.log2((seq_len - 1) / (kernel_length - 1))))

    for _ in range(num_kernels):
        weights = np.random.choice([-1, 2], size=kernel_length)
        bias = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.randint(0, max_dilation + 1) if max_dilation >= 0 else 1
        kernels.append((weights.astype(np.float32), bias, dilation))
    return kernels


class MiniRocketFeatures(nn.Module):
    """Fixed (non-trainable) MiniROCKET feature extraction"""

    def __init__(self, seq_len: int, num_kernels: int = 1000, seed: int = 42):
        super().__init__()
        kernels = _generate_minirocket_kernels(seq_len, num_kernels, seed)
        self.num_kernels = num_kernels
        self._register_kernels(kernels)
        self.seq_len = seq_len

    def _register_kernels(self, kernels):
        """Register each kernel as a buffer (fixed) and store dilation"""
        self.dilations = []
        for i, (weights, bias, dilation) in enumerate(kernels):
            w = torch.from_numpy(weights).view(1, 1, -1)  # (1, 1, 9)
            self.register_buffer(f'kernel_{i}', w)
            self.register_buffer(f'bias_{i}', torch.tensor([bias], dtype=torch.float32))
            self.dilations.append(dilation)

    def forward(self, x):
        # x: (batch, 1, seq_len)
        batch = x.shape[0]
        features = []
        for i in range(self.num_kernels):
            w = getattr(self, f'kernel_{i}')
            b = getattr(self, f'bias_{i}')
            d = self.dilations[i]
            # F.conv1d with dilation
            out = F.conv1d(x, w, b, dilation=d, padding=(w.shape[2] - 1) * d // 2)
            # Two features per kernel: max, ppv (proportion of positive values)
            f_max = out.max(dim=2)[0]  # (batch, 1)
            f_ppv = (out > 0).float().mean(dim=2)  # (batch, 1)
            features.extend([f_max, f_ppv])
        return torch.cat(features, dim=1)  # (batch, num_kernels * 2)


class SpectrumMiniRocket(nn.Module):
    """MiniROCKET for 1D spectrum classification: fixed kernels + trainable head"""

    def __init__(self, input_length: int, num_classes: int,
                 num_kernels: int = 1000,
                 seed: int = 42,
                 dropout: float = 0.2):
        super().__init__()
        self.features = MiniRocketFeatures(input_length, num_kernels, seed)
        num_features = num_kernels * 2
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


def create_minirocket_model(input_length: int, num_classes: int, device: str = 'cpu',
                            num_kernels: int = 1000, seed: int = 42,
                            dropout: float = 0.2, **kwargs) -> SpectrumMiniRocket:
    model = SpectrumMiniRocket(
        input_length=input_length, num_classes=num_classes,
        num_kernels=num_kernels, seed=seed, dropout=dropout
    )
    return model.to(device)
