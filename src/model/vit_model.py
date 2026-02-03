"""
Vision Transformer (ViT) adapted for 1D spectrum data classification - PyTorch implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Patch embedding for 1D sequences
    Divides sequence into patches and projects them
    """
    def __init__(self, input_length: int, patch_size: int, d_model: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        
        # Linear projection for patches
        self.projection = nn.Linear(patch_size, d_model)
        
        # Learnable position embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 1, sequence_length)
            
        Returns:
            Embedded patches (batch_size, num_patches + 1, d_model)
        """
        batch_size = x.size(0)
        
        # Reshape to patches: (batch_size, 1, seq_len) -> (batch_size, num_patches, patch_size)
        x = x.squeeze(1)  # (batch_size, seq_len)
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Project patches: (batch_size, num_patches, patch_size) -> (batch_size, num_patches, d_model)
        x = self.projection(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches + 1, d_model)
        
        # Add position embeddings
        x = x + self.position_embedding
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x


class SpectrumViT(nn.Module):
    """
    Vision Transformer (ViT) adapted for 1D spectrum data classification
    """
    def __init__(self, input_length: int, num_classes: int,
                 patch_size: int = 16,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        """
        Initialize ViT model
        
        Args:
            input_length: Input sequence length
            num_classes: Number of classes
            patch_size: Size of each patch
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super(SpectrumViT, self).__init__()
        
        # Ensure input_length is divisible by patch_size
        if input_length % patch_size != 0:
            raise ValueError(f"input_length ({input_length}) must be divisible by patch_size ({patch_size})")
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(input_length, patch_size, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.d_model = d_model
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, sequence_length)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, num_patches + 1, d_model)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Extract CLS token (first token)
        cls_token = x[:, 0]  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(cls_token)
        
        return x


def create_vit_model(input_length: int, num_classes: int, device: str = 'cpu',
                    patch_size: int = 16,
                    d_model: int = 256,
                    nhead: int = 8,
                    num_layers: int = 6,
                    dim_feedforward: int = 1024,
                    dropout: float = 0.1) -> SpectrumViT:
    """
    Create ViT model for spectrum data classification
    
    Args:
        input_length: Input sequence length
        num_classes: Number of classes
        device: Device ('cpu' or 'cuda')
        patch_size: Size of each patch
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        
    Returns:
        PyTorch model
    """
    model = SpectrumViT(
        input_length=input_length,
        num_classes=num_classes,
        patch_size=patch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    model = model.to(device)
    return model
