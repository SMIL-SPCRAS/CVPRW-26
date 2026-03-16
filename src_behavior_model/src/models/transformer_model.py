from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def _extract_head_dropout_features(head: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Sequential):
        if len(head) < 2:
            raise ValueError("Sequential head must have at least 2 layers to extract pre-final features.")
        return head[:-1](hidden)
    return hidden


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, ff_dim)
        self.layer_2 = nn.Linear(ff_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192) -> None:
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
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        positional_encoding: bool = False,
        max_len: int = 8192,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            input_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = PositionWiseFeedForward(input_dim, ff_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        self.positional_encoding = PositionalEncoding(input_dim, dropout=dropout, max_len=max_len) if positional_encoding else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
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
            attn_mask=attn_mask,
            need_weights=False,
        )

        x = self.add_norm_after_attention(attn_output, query)
        ff_output = self.feed_forward(x)
        return self.add_norm_after_ff(ff_output, x)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        head_hidden_dim: int = 0,
        head_dropout: float = 0.1,
        positional_encoding: bool = True,
        gate_mode: str | None = None,
        max_seq_len: int = 8192,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0 for transformer model.")
        if d_model <= 0:
            d_model = input_dim
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        if isinstance(gate_mode, str) and gate_mode.strip().lower() in {"", "none", "null"}:
            gate_mode = None
        if gate_mode is not None and gate_mode not in {"bt", "bd", "t", "d"}:
            raise ValueError("gate_mode must be one of: none, bt, bd, t, d.")

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.gate_mode = gate_mode
        self.num_layers = int(num_layers)

        self.image_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )

        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    input_dim=self.d_model,
                    num_heads=nhead,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    positional_encoding=positional_encoding,
                    max_len=self.max_seq_len,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.gate_mode is not None:
            self.bt_gates = nn.ParameterList(
                [nn.Parameter(torch.empty(self.d_model, 1)) for _ in range(self.num_layers)]
            )
            self.bd_gates = nn.ParameterList(
                [nn.Parameter(torch.empty(self.d_model, self.d_model)) for _ in range(self.num_layers)]
            )
            self.t_gates = nn.ParameterList(
                [nn.Parameter(torch.empty(self.max_seq_len, 1)) for _ in range(self.num_layers)]
            )
            self.d_gates = nn.ParameterList(
                [nn.Parameter(torch.empty(self.d_model)) for _ in range(self.num_layers)]
            )

            for plist in (self.bt_gates, self.bd_gates, self.t_gates, self.d_gates):
                for p in plist:
                    if p.dim() >= 2:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.zeros_(p)

        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden_dim),
                nn.LayerNorm(head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 2),
            )
        else:
            self.head = nn.Linear(self.d_model, 2)

        self._init_weights()

    def encode(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.image_proj(features)  # [B, T, D]

        key_padding_mask = (~valid_mask) if valid_mask is not None else None
        for i, layer in enumerate(self.transformer):
            att = layer(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
            )
            if self.gate_mode is None:
                x = x + att
            else:
                alpha = self._compute_alpha(layer_idx=i, sequences=x)
                x = (1.0 - alpha) * x + alpha * att

        return x

    def predict_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def features_from_hidden(self, hidden: torch.Tensor, embedding_kind: str = "contextual") -> torch.Tensor:
        if embedding_kind == "contextual":
            return hidden
        if embedding_kind == "head_dropout":
            return _extract_head_dropout_features(self.head, hidden)
        raise ValueError("embedding_kind must be 'contextual' or 'head_dropout'.")

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features=features, valid_mask=valid_mask)
        return self.predict_from_hidden(hidden)

    def _compute_alpha(self, layer_idx: int, sequences: torch.Tensor) -> torch.Tensor:
        b, t, d = sequences.shape

        if self.gate_mode == "bt":
            w_bt = self.bt_gates[layer_idx]  # [D, 1]
            seq_flat = sequences.reshape(b * t, d)
            alpha_flat = torch.matmul(seq_flat, w_bt)
            alpha = torch.sigmoid(alpha_flat).view(b, t, 1)
            return alpha

        if self.gate_mode == "bd":
            seq_mean = sequences.mean(dim=1)  # [B, D]
            w_bd = self.bd_gates[layer_idx]  # [D, D]
            alpha_feat = torch.matmul(seq_mean, w_bd)
            alpha_feat = torch.sigmoid(alpha_feat)
            return alpha_feat.unsqueeze(1)  # [B, 1, D]

        if self.gate_mode == "t":
            if t > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {t} exceeds max_seq_len {self.max_seq_len} for gate_mode='t'."
                )
            alpha_t = self.t_gates[layer_idx][:t]  # [T, 1]
            return torch.sigmoid(alpha_t).unsqueeze(0)  # [1, T, 1]

        if self.gate_mode == "d":
            alpha_d = torch.sigmoid(self.d_gates[layer_idx])  # [D]
            return alpha_d.view(1, 1, d)  # [1, 1, D]

        raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_discr: int | None = None,
        kernel_size: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_discr = d_discr if d_discr is not None else max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_D = nn.Sequential(
            nn.Linear(d_model, d_discr, bias=False),
            nn.Linear(d_discr, d_model, bias=False),
        )
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
            bias=True,
        )
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.add_norm = AddAndNorm(d_model, dropout=dropout)

    def forward(self, sequences: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = sequences.shape
        a, b = self.in_proj(sequences).chunk(2, dim=-1)

        x = a.transpose(1, 2)  # [B, D, T]
        x = self.conv(x)[..., :seq_len].transpose(1, 2)
        x = F.silu(x)

        x = self._ssm(x)
        b = F.silu(b)
        out = self.out_proj(x * b)
        out = self.add_norm(self.D.view(1, 1, -1) * sequences, out)

        if valid_mask is not None:
            out = out.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return out

    def _ssm(self, sequences: torch.Tensor) -> torch.Tensor:
        # sequences: [B, T, D]
        batch_size, seq_len, d_model = sequences.shape
        d_state = self.A.shape[1]

        A = -self.A
        B = self.s_B(sequences)  # [B, T, S]
        C = self.s_C(sequences)  # [B, T, S]
        s = F.softplus(self.D.view(1, 1, -1) + self.s_D(sequences))  # [B, T, D]

        A_bar = torch.exp(A).view(1, 1, d_model, d_state) * s.unsqueeze(-1)  # [B, T, D, S]
        B_bar = B.unsqueeze(2) * s.unsqueeze(-1)  # [B, T, D, S]
        X_bar = B_bar * sequences.unsqueeze(-1)  # [B, T, D, S]

        h = torch.zeros(batch_size, d_model, d_state, device=sequences.device, dtype=sequences.dtype)
        outs = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + X_bar[:, t]
            out_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [B, D]
            outs.append(out_t)
        return torch.stack(outs, dim=1)  # [B, T, D]


class TextMambaRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 0,
        head_dropout: float = 0.1,
        mamba_d_state: int = 8,
        mamba_kernel_size: int = 3,
        mamba_d_discr: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0 for mamba model.")
        if d_model <= 0:
            d_model = input_dim

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)

        d_discr = None if int(mamba_d_discr) <= 0 else int(mamba_d_discr)

        self.image_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )
        self.mamba = nn.ModuleList(
            [
                MambaBlock(
                    d_model=self.d_model,
                    d_state=int(mamba_d_state),
                    d_discr=d_discr,
                    kernel_size=int(mamba_kernel_size),
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(self.d_model, head_hidden_dim),
                nn.LayerNorm(head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 2),
            )
        else:
            self.head = nn.Linear(self.d_model, 2)

        self._init_weights()

    def encode(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.image_proj(features)
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        for block in self.mamba:
            x = block(x, valid_mask=valid_mask)

        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return x

    def predict_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def features_from_hidden(self, hidden: torch.Tensor, embedding_kind: str = "contextual") -> torch.Tensor:
        if embedding_kind == "contextual":
            return hidden
        if embedding_kind == "head_dropout":
            return _extract_head_dropout_features(self.head, hidden)
        raise ValueError("embedding_kind must be 'contextual' or 'head_dropout'.")

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(features=features, valid_mask=valid_mask)
        return self.predict_from_hidden(hidden)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
