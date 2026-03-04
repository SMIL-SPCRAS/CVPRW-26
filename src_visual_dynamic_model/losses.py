import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        gold: torch.Tensor,
        pred: torch.Tensor,
        weights: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # gold/pred: [B, T, 2]
        gold = torch.nan_to_num(gold, nan=0.0)
        pred = torch.nan_to_num(pred, nan=0.0)
        if mask is None:
            mask = torch.ones_like(gold[..., 0], dtype=torch.float32, device=gold.device)
        else:
            mask = mask.to(dtype=torch.float32, device=gold.device)
        mask = mask.unsqueeze(-1)  # [B, T, 1]

        count = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        gold_mean = (gold * mask).sum(dim=1) / count
        pred_mean = (pred * mask).sum(dim=1) / count

        gold_centered = (gold - gold_mean.unsqueeze(1)) * mask
        pred_centered = (pred - pred_mean.unsqueeze(1)) * mask

        cov = (gold_centered * pred_centered).sum(dim=1) / count
        gold_var = (gold_centered ** 2).sum(dim=1) / count
        pred_var = (pred_centered ** 2).sum(dim=1) / count

        denom = gold_var + pred_var + (gold_mean - pred_mean) ** 2 + self.eps
        ccc_vals = 2.0 * cov / denom
        loss = 1.0 - ccc_vals
        if weights is not None:
            loss = loss * weights
        return torch.mean(loss)

