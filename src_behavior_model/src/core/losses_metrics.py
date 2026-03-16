from typing import Dict

import torch
from torch import nn


class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient loss."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps
        )
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + self.eps)
        return 1 - ccc


@torch.no_grad()
def ccc_score(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.float()
    y = y.float()

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + eps)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + eps)
    return float(ccc.item())


@torch.no_grad()
def compute_va_ccc(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    ccc_valence = ccc_score(preds[:, 0], targets[:, 0])
    ccc_arousal = ccc_score(preds[:, 1], targets[:, 1])
    return {
        "ccc_valence": ccc_valence,
        "ccc_arousal": ccc_arousal,
        "ccc_mean": 0.5 * (ccc_valence + ccc_arousal),
    }

