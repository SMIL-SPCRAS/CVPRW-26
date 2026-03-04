import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)].detach()
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, dropout: float = 0.1, positional_encoding: bool = True):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        # Keep parity with the original implementation used in the best run:
        # positional encoding dropout is fixed at its default (0.1), independent of model dropout.
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.positional_encoding is not None:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        attn_output, _ = self.self_attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.add_norm_after_attention(attn_output, query)
        ff_output = self.feed_forward(x)
        return self.add_norm_after_ff(ff_output, x)


class VisualDynamicModel(nn.Module):
    """
    Minimal model needed for best run:
    - temporal transformer over cached frame features
    - MLP head: Linear -> LayerNorm -> GELU -> Dropout -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        tr_layers: int = 5,
        dropout: float = 0.1,
        out_dim: int = 2,
        head_type: str = "mlp",
    ):
        super().__init__()
        self.head_type = str(head_type).lower()
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    input_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    positional_encoding=True,
                )
                for _ in range(tr_layers)
            ]
        )
        if self.head_type == "linear":
            self.head = nn.Linear(hidden_dim, out_dim)
        elif self.head_type == "mlp":
            h = int(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, out_dim),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Use mlp | linear")

    def _forward_backbone(self, features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # True for padding
        x = self.image_proj(features)
        for layer in self.transformer:
            # Keep behavior identical to the original training code.
            attn = layer(x, x, x, key_padding_mask=key_padding_mask)
            x = x + attn
        return x

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._forward_backbone(features, mask)
        return self.head(x)

    def forward_with_embeddings(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        - embeddings right before final Linear(h, out_dim)
        - predictions
        """
        x = self._forward_backbone(features, mask)
        if self.head_type == "linear":
            emb = x
            pred = self.head(x)
            return emb, pred
        # MLP head: take representation before final Linear.
        emb = self.head[0](x)
        emb = self.head[1](emb)
        emb = self.head[2](emb)
        emb = self.head[3](emb)
        pred = self.head[4](emb)
        return emb, pred
