"""
InceptionTime model for spectrum data classification - PyTorch implementation
Based on: Fawaz et al. "InceptionTime: Finding AlexNet for Time Series Classification"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Single Inception block with bottleneck and multi-scale convolutions"""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]
        self.kernel_sizes = kernel_sizes

        self.bottleneck = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=k, padding=padding, bias=False)
            )
        self.out_channels = out_channels * len(kernel_sizes)

    def forward(self, x):
        # Bottleneck
        x = self.bottleneck(x)
        # Parallel convolutions
        out = [conv(x) for conv in self.convs]
        return torch.cat(out, dim=1)


class SpectrumInceptionTime(nn.Module):
    """InceptionTime for 1D spectrum classification"""

    def __init__(self, input_length: int, num_classes: int,
                 n_filters: int = 32,
                 depth: int = 6,
                 kernel_sizes: list = None,
                 use_residual: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]
        self.use_residual = use_residual
        self.depth = depth
        n_kernels = len(kernel_sizes)

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        in_ch = 1
        for _ in range(depth):
            block = InceptionBlock(in_ch, n_filters, kernel_sizes)
            out_ch = block.out_channels  # n_filters * n_kernels
            self.blocks.append(block)
            if use_residual:
                self.shortcuts.append(nn.Conv1d(in_ch, out_ch, 1))
            else:
                self.shortcuts.append(None)
            in_ch = out_ch

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            out = block(x)
            if self.use_residual and shortcut is not None:
                res = shortcut(x)
                if res.shape[2] == out.shape[2]:
                    out = out + res
            x = F.relu(out)
        x = self.gap(x)
        return self.fc(x)


def create_inceptiontime_model(input_length: int, num_classes: int, device: str = 'cpu',
                               n_filters: int = 32, depth: int = 6,
                               kernel_sizes: list = None, use_residual: bool = True,
                               dropout: float = 0.2, **kwargs) -> SpectrumInceptionTime:
    model = SpectrumInceptionTime(
        input_length=input_length, num_classes=num_classes,
        n_filters=n_filters, depth=depth, kernel_sizes=kernel_sizes,
        use_residual=use_residual, dropout=dropout
    )
    return model.to(device)
