import math
from typing import Dict, List, Optional, Tuple

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


class FusionModel(nn.Module):
    """
    Transformer fusion model over multimodal frame-level streams.

    Q/K/V source modalities are configurable via:
    - q_modality
    - k_modality
    - v_modality

    Allowed values are modality names from `modalities` plus reserved value `fusion`
    (mean of all projected modality streams).
    """

    def __init__(
        self,
        modality_input_dims: Dict[str, int],
        modalities: List[str],
        q_modality: str = "fusion",
        k_modality: str = "fusion",
        v_modality: str = "fusion",
        hidden_dim: int = 256,
        num_heads: int = 8,
        tr_layers: int = 5,
        dropout: float = 0.1,
        out_dim: int = 2,
        head_type: str = "mlp",
    ):
        super().__init__()
        self.modalities = [str(m) for m in modalities]
        if len(self.modalities) == 0:
            raise ValueError("modalities list is empty")
        self.q_modality = str(q_modality)
        self.k_modality = str(k_modality)
        self.v_modality = str(v_modality)
        for name in (self.q_modality, self.k_modality, self.v_modality):
            if name != "fusion" and name not in self.modalities:
                raise ValueError(f"Q/K/V modality '{name}' is not in modalities={self.modalities} and is not 'fusion'")

        self.head_type = str(head_type).lower()
        self.projections = nn.ModuleDict()
        for m in self.modalities:
            in_dim = int(modality_input_dims[m])
            self.projections[m] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
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

    def _select_stream(self, name: str, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        if name == "fusion":
            stacked = torch.stack([states[m] for m in self.modalities], dim=0)  # [M, B, T, H]
            return stacked.mean(dim=0)
        return states[name]

    def _forward_backbone(self, modality_features: Dict[str, torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        states = {}
        for m in self.modalities:
            if m not in modality_features:
                raise KeyError(f"Missing modality '{m}' in batch. Available: {list(modality_features.keys())}")
            states[m] = self.projections[m](modality_features[m])

        q = self._select_stream(self.q_modality, states)
        for layer in self.transformer:
            k = self._select_stream(self.k_modality, states)
            v = self._select_stream(self.v_modality, states)
            attn = layer(q, k, v, key_padding_mask=key_padding_mask)
            q = q + attn
            if self.q_modality in states:
                states[self.q_modality] = q
        return q

    def forward(self, modality_features: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._forward_backbone(modality_features, mask)
        return self.head(x)

    def forward_with_embeddings(
        self,
        modality_features: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_backbone(modality_features, mask)
        if self.head_type == "linear":
            emb = x
            pred = self.head(x)
            return emb, pred
        emb = self.head[0](x)
        emb = self.head[1](emb)
        emb = self.head[2](emb)
        emb = self.head[3](emb)
        pred = self.head[4](emb)
        return emb, pred

