import torch
import torch.nn as nn

from typing import Optional


class AttnPool1d(nn.Module):
    """Self-attentive pooling over time with an (optional) padding mask.

    h:    (B, T, C)
    mask: (B, T) 1/True for valid frames
    """

    def __init__(self, in_dim: int, attn_dim: int = 128, 
                 attn_dropout: float = 0.0, attn_use_std: bool = True):
        super().__init__()
        self.attn_use_std = attn_use_std
        self.proj = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(attn_dropout),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.proj(h).squeeze(-1)  # (B, T)
        if attn_mask is not None:
            m = attn_mask.to(dtype=torch.bool, device=h.device)
            scores = scores.masked_fill(~m, -1e4)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        mu = (h * w).sum(dim=1)
        if not self.attn_use_std:
            return mu

        var = ((h - mu.unsqueeze(1)) ** 2 * w).sum(dim=1)
        std = (var + 1e-6).sqrt().clamp_min(1e-6)
        return torch.cat([mu, std], dim=-1)