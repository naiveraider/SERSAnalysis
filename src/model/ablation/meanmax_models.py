"""Pooling ablation variants for existing models.

Each variant supports mean, max, or mean+max pooling over the sequence/patch
dimension before the classification head. SpectrumGRUMeanMax already exists in
`lstm_gru_model.py` so it is intentionally not duplicated.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cnn_model import SpectrumCNN
from ..tcn_model import SpectrumTCN
from ..vit_model import SpectrumViT
from ..inceptiontime_model import SpectrumInceptionTime
from ..lstm_gru_model import SpectrumLSTM, SpectrumCNNLSTM, SpectrumGRU
from ..cnn_transformer_model import SpectrumCNNTransformer
from ..mamba_s4_model import SpectrumMambaS4


VALID_POOLING_TYPES = {'mean', 'max', 'meanmax'}


def _validate_pooling_type(pooling_type: str) -> str:
    pooling_type = pooling_type.lower()
    if pooling_type not in VALID_POOLING_TYPES:
        raise ValueError(
            f"Unsupported pooling_type: {pooling_type}. "
            f"Expected one of {sorted(VALID_POOLING_TYPES)}"
        )
    return pooling_type


def _pool_features(x, dim: int, pooling_type: str) -> torch.Tensor:
    pooling_type = _validate_pooling_type(pooling_type)

    if pooling_type == 'mean':
        return x.mean(dim=dim)
    if pooling_type == 'max':
        return x.max(dim=dim).values

    mean_pool = x.mean(dim=dim)
    max_pool = x.max(dim=dim).values
    return torch.cat([mean_pool, max_pool], dim=1)


def _feature_multiplier(pooling_type: str) -> int:
    return 2 if _validate_pooling_type(pooling_type) == 'meanmax' else 1


class SpectrumCNNMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
        # reuse the conv blocks from SpectrumCNN but reimplement final head
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.25)

        pooled_features = 256 * _feature_multiplier(self.pooling_type)
        self.fc1 = nn.Linear(pooled_features, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        feat = _pool_features(x, dim=2, pooling_type=self.pooling_type)

        feat = self.fc1(feat)
        feat = self.bn4(feat)
        feat = F.relu(feat)
        feat = self.dropout4(feat)

        feat = self.fc2(feat)
        feat = self.bn5(feat)
        feat = F.relu(feat)
        feat = self.dropout5(feat)

        return self.fc3(feat)


class SpectrumTCNMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int, device: str = 'cpu',
                 num_channels: list = None, kernel_size: int = 3, dropout: float = 0.2,
                 pooling_type: str = 'meanmax'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
        # reuse TCN network from original implementation
        nc = num_channels if num_channels is not None else [64, 128, 256]
        base = SpectrumTCN(input_length, num_classes, num_channels=nc,
                   kernel_size=kernel_size, dropout=dropout)
        # copy network layers
        self.network = base.network

        last_ch = num_channels[-1] if num_channels is not None else 256
        pooled_features = last_ch * _feature_multiplier(self.pooling_type)
        self.fc1 = nn.Linear(pooled_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.network(x)
        feat = _pool_features(x, dim=2, pooling_type=self.pooling_type)

        feat = self.fc1(feat)
        feat = self.bn1(feat)
        feat = F.relu(feat)
        feat = self.dropout1(feat)

        feat = self.fc2(feat)
        feat = self.bn2(feat)
        feat = F.relu(feat)
        feat = self.dropout2(feat)

        return self.fc3(feat)


class SpectrumViTMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 patch_size: int = 16, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024,
                 dropout: float = 0.1, pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
        base = SpectrumViT(input_length, num_classes, patch_size=patch_size,
                           d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout)
        self.patch_embedding = base.patch_embedding
        self.transformer_blocks = base.transformer_blocks
        pooled_features = d_model * _feature_multiplier(self.pooling_type)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_features),
            nn.Linear(pooled_features, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)  # (batch, num_patches+1, d_model)
        for block in self.transformer_blocks:
            x = block(x)
        tokens = x[:, 1:, :]
        feat = _pool_features(tokens, dim=1, pooling_type=self.pooling_type)
        return self.classifier(feat)


class SpectrumInceptionTimeMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 n_filters: int = 32, depth: int = 6, kernel_sizes: list = None,
                 use_residual: bool = True, dropout: float = 0.2,
                 pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
        base = SpectrumInceptionTime(input_length, num_classes,
                                     n_filters=n_filters, depth=depth,
                                     kernel_sizes=kernel_sizes, use_residual=use_residual,
                                     dropout=dropout)
        # reuse blocks and shortcuts
        self.blocks = base.blocks
        self.shortcuts = base.shortcuts
        self.use_residual = base.use_residual

        # final channel count equals base fc input
        in_ch = base.fc[1].in_features if isinstance(base.fc, nn.Sequential) else None
        # fallback if attribute not present
        if in_ch is None:
            in_ch = n_filters * (len(kernel_sizes) if kernel_sizes is not None else 3)

        pooled_features = in_ch * _feature_multiplier(self.pooling_type)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_features, 128),
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
        feat = _pool_features(x, dim=2, pooling_type=self.pooling_type)
        return self.classifier(feat)


class SpectrumLSTMMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.2,
                 pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
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
            nn.Linear(lstm_output_size * _feature_multiplier(self.pooling_type), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        features = _pool_features(out, dim=1, pooling_type=self.pooling_type)
        return self.fc(features)


class SpectrumGRUMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.2,
                 pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
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
            nn.Linear(gru_output_size * _feature_multiplier(self.pooling_type), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        features = _pool_features(out, dim=1, pooling_type=self.pooling_type)
        return self.fc(features)


class SpectrumCNNLSTMMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.2,
                 pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
        # reuse cnn preprocess from SpectrumCNNLSTM
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

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size * _feature_multiplier(self.pooling_type), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn_preprocess(x)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_features)
        out, _ = self.lstm(x)
        features = _pool_features(out, dim=1, pooling_type=self.pooling_type)
        return self.fc(features)


class SpectrumCNNTransformerMeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 cnn_channels: list = None, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 2,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
        if cnn_channels is None:
            base = SpectrumCNNTransformer(input_length, num_classes,
                                          d_model=d_model,
                                          nhead=nhead, num_layers=num_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout)
        else:
            base = SpectrumCNNTransformer(input_length, num_classes,
                                          cnn_channels=cnn_channels, d_model=d_model,
                                          nhead=nhead, num_layers=num_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout)
        # reuse internal modules
        self.cnn_layers = base.cnn_layers
        self.cnn_projection = base.cnn_projection
        self.pos_encoder = base.pos_encoder
        self.transformer_encoder = base.transformer_encoder
        self.d_model = d_model

        fusion_input_dim = d_model * (4 if self.pooling_type == 'meanmax' else 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # reuse classifier from base
        self.classifier = base.classifier

    def forward(self, x):
        # CNN feature extraction
        cnn_features = x
        for layer in self.cnn_layers:
            cnn_features = layer(cnn_features)

        # Sequence features for transformer
        cnn_seq = cnn_features.transpose(1, 2)  # (batch, seq_len, channels)
        cnn_seq_proj = self.cnn_projection(cnn_seq)  # (batch, seq_len, d_model)

        # Transformer expects (seq_len, batch, d_model)
        t_in = cnn_seq_proj.transpose(0, 1)
        t_in = self.pos_encoder(t_in)
        transformer_output = self.transformer_encoder(t_in)  # (seq_len, batch, d_model)
        tokens = transformer_output.transpose(0, 1)  # (batch, seq_len, d_model)

        pooled_cnn = _pool_features(cnn_seq_proj, dim=1, pooling_type=self.pooling_type)
        pooled_trans = _pool_features(tokens, dim=1, pooling_type=self.pooling_type)
        feat = torch.cat([pooled_cnn, pooled_trans], dim=1)
        fused = self.fusion(feat)
        return self.classifier(fused)


class SpectrumMambaS4MeanMax(nn.Module):
    def __init__(self, input_length: int, num_classes: int,
                 d_model: int = 256, n_layers: int = 4, d_state: int = 64,
                 dropout: float = 0.1, pooling_type: str = 'meanmax', device: str = 'cpu'):
        super().__init__()
        self.pooling_type = _validate_pooling_type(pooling_type)
        base = SpectrumMambaS4(input_length, num_classes,
                               d_model=d_model, n_layers=n_layers,
                               d_state=d_state, dropout=dropout)
        # reuse S4 stack
        self.input_embedding = base.input_embedding
        self.s4_layers = base.s4_layers
        self.layer_norms = base.layer_norms
        self.d_model = base.d_model

        pooled_features = self.d_model * _feature_multiplier(self.pooling_type)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, 1)
        x = self.input_embedding(x)  # (batch, seq_len, d_model)

        for s4_layer, norm in zip(self.s4_layers, self.layer_norms):
            residual = x
            x = s4_layer(x)
            x = norm(x + residual)

        feat = _pool_features(x, dim=1, pooling_type=self.pooling_type)
        return self.classifier(feat)


# factory create functions
def create_cnn_mean_model(input_length: int, num_classes: int, device: str = 'cpu'):
    model = SpectrumCNNMeanMax(input_length, num_classes, pooling_type='mean')
    return model.to(device)


def create_cnn_max_model(input_length: int, num_classes: int, device: str = 'cpu'):
    model = SpectrumCNNMeanMax(input_length, num_classes, pooling_type='max')
    return model.to(device)


def create_cnn_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu'):
    model = SpectrumCNNMeanMax(input_length, num_classes, pooling_type='meanmax')
    return model.to(device)


def create_tcn_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumTCNMeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_tcn_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumTCNMeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_tcn_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumTCNMeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)


def create_vit_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumViTMeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_vit_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumViTMeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_vit_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumViTMeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)


def create_inceptiontime_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumInceptionTimeMeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_inceptiontime_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumInceptionTimeMeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_inceptiontime_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumInceptionTimeMeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)


def create_lstm_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumLSTMMeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_lstm_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumLSTMMeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_lstm_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumLSTMMeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)


def create_gru_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumGRUMeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_gru_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumGRUMeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_gru_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumGRUMeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)


def create_cnn_lstm_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumCNNLSTMMeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_cnn_lstm_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumCNNLSTMMeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_cnn_lstm_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumCNNLSTMMeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)


def create_cnn_transformer_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumCNNTransformerMeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_cnn_transformer_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumCNNTransformerMeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_cnn_transformer_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumCNNTransformerMeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)


def create_mamba_mean_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumMambaS4MeanMax(input_length, num_classes, pooling_type='mean', **kwargs)
    return model.to(device)


def create_mamba_max_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumMambaS4MeanMax(input_length, num_classes, pooling_type='max', **kwargs)
    return model.to(device)


def create_mamba_meanmax_model(input_length: int, num_classes: int, device: str = 'cpu', **kwargs):
    model = SpectrumMambaS4MeanMax(input_length, num_classes, pooling_type='meanmax', **kwargs)
    return model.to(device)
