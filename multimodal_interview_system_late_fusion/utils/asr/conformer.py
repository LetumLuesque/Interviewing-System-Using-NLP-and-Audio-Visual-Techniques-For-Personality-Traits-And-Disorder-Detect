"""
Conformer encoder — from scratch.

Architecture (per block):
    x = x + 0.5 * FFN(x)
    x = x + MHSA(x)
    x = x + ConvModule(x)
    x = x + 0.5 * FFN(x)
    x = LayerNorm(x)

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", Interspeech 2020.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =========================================================================== #
#  Sub-modules                                                                 #
# =========================================================================== #

class FeedForwardModule(nn.Module):
    """Pre-norm feed-forward block with expansion factor."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.act     = nn.SiLU()               # Swish
        self.drop    = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    """Pre-norm MHSA with relative positional encoding (simplified sinusoidal)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.drop(x)


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module:
        LayerNorm → pointwise conv (expand 2×) → GLU → depthwise conv →
        BatchNorm → SiLU → pointwise conv (back to d_model) → dropout
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2

        self.norm       = nn.LayerNorm(d_model)
        self.pw1        = nn.Conv1d(d_model, d_model * 2, kernel_size=1)      # pointwise ×2 for GLU
        self.glu        = nn.GLU(dim=1)                                        # halves channel dim
        self.dw         = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                    padding=padding, groups=d_model)           # depthwise
        self.bn         = nn.BatchNorm1d(d_model)
        self.act        = nn.SiLU()
        self.pw2        = nn.Conv1d(d_model, d_model, kernel_size=1)           # pointwise back
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)          # [B, d_model, T]
        x = self.pw1(x)                # [B, d_model*2, T]
        x = self.glu(x)               # [B, d_model, T]
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x)
        x = x.transpose(1, 2)         # [B, T, d_model]
        return x


# =========================================================================== #
#  Conformer Block                                                              #
# =========================================================================== #

class ConformerBlock(nn.Module):
    """Single Conformer encoder block."""

    def __init__(
        self,
        d_model:    int = 256,
        num_heads:  int = 4,
        kernel_size: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1  = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2  = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.norm(x)
        return x


# =========================================================================== #
#  Conv Subsampler (4× time reduction)                                         #
# =========================================================================== #

class ConvSubsampler(nn.Module):
    """
    2 × Conv2D (stride=2) to reduce time by 4× and project to d_model.

    Input:  [B, T, n_mels]
    Output: [B, T//4, d_model]
    """

    def __init__(self, n_mels: int = 80, d_model: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # After 2× stride-2 convs the freq dimension is ceil(n_mels/4)
        freq_out = math.ceil(math.ceil(n_mels / 2) / 2)
        self.linear = nn.Linear(256 * freq_out, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_mels]
        x = x.unsqueeze(1)             # [B, 1, T, n_mels]
        x = self.conv(x)               # [B, 256, T//4, freq_out]
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)     # [B, T//4, 256, freq_out]
        x = x.reshape(B, T, C * F)    # [B, T//4, 256 * freq_out]
        x = self.linear(x)            # [B, T//4, d_model]
        return x


# =========================================================================== #
#  Full Conformer Encoder                                                       #
# =========================================================================== #

class ConformerEncoder(nn.Module):
    """
    Complete Conformer encoder:
        ConvSubsampler → Linear projection → N × ConformerBlock → LayerNorm
    """

    def __init__(
        self,
        n_mels:      int = 80,
        d_model:     int = 256,
        num_layers:  int = 12,
        num_heads:   int = 4,
        kernel_size: int = 31,
        ff_expansion: int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.subsampler = ConvSubsampler(n_mels=n_mels, d_model=d_model)
        self.dropout    = nn.Dropout(dropout)
        self.layers     = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, ff_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:       [B, T, n_mels]
            lengths: [B] original frame lengths before subsampling (optional)

        Returns:
            encoded:          [B, T//4, d_model]
            output_lengths:   [B] or None
        """
        x = self.subsampler(x)     # [B, T//4, d_model]
        x = self.dropout(x)

        # Compute subsampled lengths for masking
        output_lengths = None
        if lengths is not None:
            # For Conv2d(stride=2, padding=1, kernel=3): L_out = (L_in + 1) // 2
            output_lengths = (lengths + 1) // 2
            output_lengths = (output_lengths + 1) // 2
            output_lengths = output_lengths.clamp(min=1).to(x.device)

            # Build key_padding_mask: True where position is PADDING
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= output_lengths.unsqueeze(1)
        else:
            mask = None

        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)

        x = self.norm(x)
        return x, output_lengths
