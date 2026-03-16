import torch
import torch.nn as nn

from chimera_ml.losses.base import BaseLoss
from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.core.registry import LOSSES

from utils import ccc_1d


class CCCMSELoss(BaseLoss):
    """
    CCC loss for (valence, arousal). Optionally mixes in MSE for stability.

    loss =
      ccc_weight * (w_v * ccc_v + w_a * ccc_a) / (w_v + w_a)
    + mse_weight * (w_v * mse_v + w_a * mse_a) / (w_v + w_a)

    Keeps your:
      - seq-to-seq support
      - invalid target mask (-5)
      - min_valid fallback to weighted MSE
    """

    def __init__(self, 
                 ccc_weight: float = 1.0,
                 mse_weight: float = 0.0,
                 v_weight: float = 0.5,
                 a_weight: float = 0.5,
                 smooth_weight: float = 0.0,
                 smooth_type: str = "l1",
                 min_valid: int = 4):
        self.ccc_weight = float(ccc_weight)
        self.mse_weight = float(mse_weight)
        self.v_weight = float(v_weight)
        self.a_weight = float(a_weight)
        self.smooth_weight = float(smooth_weight)
        self.smooth_type = str(smooth_type)
        self.min_valid = int(min_valid)
        self._mse = nn.MSELoss()

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        preds = output.preds  # [B, 2] or [B, T, 2]
        targets = batch.targets  # [B, 2]  or [B, T, 2]

        if targets is None:
            raise ValueError("ValenceArousalCCCLoss requires batch.targets (got None).")
        
        # NEW: temporal smoothness
        smooth_loss = preds.new_tensor(0.0)
        if self.smooth_weight > 0.0 and preds.dim() == 3:
            # targets: [B,T,2]
            valid_t = (
                torch.isfinite(targets).all(dim=-1)
                & (targets[..., 0] != -5.0)
                & (targets[..., 1] != -5.0)
            )  # [B,T]

            pair_valid = valid_t[:, 1:] & valid_t[:, :-1]  # [B,T-1]
            if pair_valid.any():
                dp = preds[:, 1:, :] - preds[:, :-1, :]    # [B,T-1,2]
                if self.smooth_type == "l2":
                    per_pair = (dp ** 2).sum(dim=-1)       # [B,T-1]
                else:  # "l1"
                    per_pair = dp.abs().sum(dim=-1)        # [B,T-1]

                smooth_loss = (per_pair * pair_valid.float()).sum() / (pair_valid.float().sum() + 1e-8)
        
        # seq-to-seq support: [B,4,2] -> [B*4,2]
        if preds.dim() == 3:
            preds = preds.reshape(-1, 2)
            targets = targets.reshape(-1, 2)

        # mask invalid targets (-5)
        valid = torch.isfinite(targets).all(dim=1) & (targets[:, 0] != -5.0) & (targets[:, 1] != -5.0)
        preds = preds[valid]
        targets = targets[valid]

        # no valid seconds
        if preds.numel() == 0:
            return preds.new_tensor(0.0)

        preds_f = preds.float()
        targets_f = targets.float()

        if preds_f.shape[0] < self.min_valid:
            base = self._mse(preds_f, targets_f)
            return base + self.smooth_weight * smooth_loss

        wsum = self.v_weight + self.a_weight
        ccc_v = 1.0 - ccc_1d(preds_f[:, 0], targets_f[:, 0])
        ccc_a = 1.0 - ccc_1d(preds_f[:, 1], targets_f[:, 1])
        ccc = (self.v_weight * ccc_v + self.a_weight * ccc_a) / wsum

        loss = self.ccc_weight * ccc
        if self.mse_weight > 0.0:
            loss = loss + self.mse_weight * self._mse(preds_f, targets_f)

        loss = loss + self.smooth_weight * smooth_loss
        return loss
    

@LOSSES.register("ccc_mse_loss")
def ccc_mse_loss(*, 
                 ccc_weight: float = 1.0, 
                 mse_weight: float = 0.0, 
                 v_weight: float = 0.5,
                 a_weight: float = 0.5,
                 **_) -> BaseLoss:
    return CCCMSELoss(ccc_weight=ccc_weight, 
                      mse_weight=mse_weight,
                      v_weight=v_weight,
                      a_weight=a_weight)