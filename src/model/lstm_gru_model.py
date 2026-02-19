"""
LSTM and GRU models for spectrum data classification - PyTorch implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumLSTM(nn.Module):
    """LSTM model for 1D spectrum sequence classification"""

    def __init__(self, input_length: int, num_classes: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x = x.transpose(1, 2)
        out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state: (batch, num_directions * hidden_size)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.fc(h)


class SpectrumGRU(nn.Module):
    """GRU model for 1D spectrum sequence classification"""

    def __init__(self, input_length: int, num_classes: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        gru_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, 1)
        out, h_n = self.gru(x)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.fc(h)


def create_lstm_model(input_length: int, num_classes: int, device: str = 'cpu',
                      hidden_size: int = 128, num_layers: int = 2,
                      bidirectional: bool = True, dropout: float = 0.2, **kwargs) -> SpectrumLSTM:
    model = SpectrumLSTM(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout
    )
    return model.to(device)


def create_gru_model(input_length: int, num_classes: int, device: str = 'cpu',
                     hidden_size: int = 128, num_layers: int = 2,
                     bidirectional: bool = True, dropout: float = 0.2, **kwargs) -> SpectrumGRU:
    model = SpectrumGRU(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout
    )
    return model.to(device)
