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


class SpectrumLSTMAttention(nn.Module):
    """LSTM with temporal attention pooling for 1D spectrum classification."""

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
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.Tanh(),
            nn.Linear(lstm_output_size, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)

        # Learn a weight for each time step and pool the sequence.
        attn_scores = self.attention(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(out * attn_weights, dim=1)
        return self.fc(context)


class SpectrumLSTMCNN(nn.Module):
    """LSTM followed by temporal CNN blocks for 1D spectrum classification."""

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
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(lstm_output_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, seq_len) -> (batch, seq_len, 1)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out.transpose(1, 2)  # (batch, hidden*num_directions, seq_len)
        features = self.temporal_cnn(out)
        return self.fc(features)


class SpectrumCNNLSTM(nn.Module):
    """CNN preprocessing followed by LSTM for 1D spectrum classification."""

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

        self.cnn_preprocess = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=64,
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
        # x: (batch, 1, seq_len)
        x = self.cnn_preprocess(x)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_features)
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.fc(h)


class SpectrumCNNBiLSTMAttention(nn.Module):
    """CNN preprocessing, BiLSTM, and temporal attention pooling."""

    def __init__(self, input_length: int, num_classes: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2

        self.cnn_preprocess = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.Tanh(),
            nn.Linear(lstm_output_size, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = self.cnn_preprocess(x)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_features)
        out, _ = self.lstm(x)
        attn_scores = self.attention(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(out * attn_weights, dim=1)
        return self.fc(context)


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


def create_lstm_attention_model(input_length: int, num_classes: int, device: str = 'cpu',
                                hidden_size: int = 128, num_layers: int = 2,
                                bidirectional: bool = True, dropout: float = 0.2,
                                **kwargs) -> SpectrumLSTMAttention:
    model = SpectrumLSTMAttention(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout
    )
    return model.to(device)


def create_lstm_cnn_model(input_length: int, num_classes: int, device: str = 'cpu',
                          hidden_size: int = 128, num_layers: int = 2,
                          bidirectional: bool = True, dropout: float = 0.2,
                          **kwargs) -> SpectrumLSTMCNN:
    model = SpectrumLSTMCNN(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout
    )
    return model.to(device)


def create_cnn_lstm_model(input_length: int, num_classes: int, device: str = 'cpu',
                          hidden_size: int = 128, num_layers: int = 2,
                          bidirectional: bool = True, dropout: float = 0.2,
                          **kwargs) -> SpectrumCNNLSTM:
    model = SpectrumCNNLSTM(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout
    )
    return model.to(device)


def create_stacked_lstm_model(input_length: int, num_classes: int, device: str = 'cpu',
                              hidden_size: int = 128, num_layers: int = 2,
                              bidirectional: bool = False, dropout: float = 0.2,
                              **kwargs) -> SpectrumLSTM:
    model = SpectrumLSTM(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=max(2, num_layers),
        bidirectional=False, dropout=dropout
    )
    return model.to(device)


def create_bilstm_model(input_length: int, num_classes: int, device: str = 'cpu',
                        hidden_size: int = 128, num_layers: int = 2,
                        bidirectional: bool = True, dropout: float = 0.2,
                        **kwargs) -> SpectrumLSTM:
    model = SpectrumLSTM(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=True, dropout=dropout
    )
    return model.to(device)


def create_cnn_bilstm_attention_model(input_length: int, num_classes: int, device: str = 'cpu',
                                      hidden_size: int = 128, num_layers: int = 2,
                                      bidirectional: bool = True, dropout: float = 0.2,
                                      **kwargs) -> SpectrumCNNBiLSTMAttention:
    model = SpectrumCNNBiLSTMAttention(
        input_length=input_length, num_classes=num_classes,
        hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=True, dropout=dropout
    )
    return model.to(device)
