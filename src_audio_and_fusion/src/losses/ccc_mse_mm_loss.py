import torch
import torch.nn as nn

from chimera_ml.losses.base import BaseLoss
from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.core.registry import LOSSES

from utils import ccc_1d


class CCCMSEMMLoss(BaseLoss):
    """
    CCC loss for (valence, arousal). Optionally mixes in MSE for stability.

    loss =
      ccc_weight * (w_v * ccc_v + w_a * ccc_a) / (w_v + w_a)
    + mse

    Keeps your:
      - seq-to-seq support
      - invalid target mask (-5)
      - min_valid fallback to weighted MSE
    """
    def __init__(self, ccc_weight=1.0, mse_weight=0.0, v_weight=0.5, a_weight=0.5, min_valid=4):
        self.ccc_weight = float(ccc_weight)
        self.mse_weight = float(mse_weight)
        self.v_weight = float(v_weight)
        self.a_weight = float(a_weight)
        self.min_valid = int(min_valid)
        self._mse = nn.MSELoss()

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        preds = output.preds
        targets = batch.targets
        if targets is None:
            raise ValueError("CCCMSEMMLoss requires batch.targets (got None).")

        # masks
        masks = batch.meta.get("masks", {}) if batch.meta else {}
        targets_sequence_mask = masks.get("targets_sequence_mask", None)
        targets_valid_mask = masks.get("targets_valid_mask", None)

        if preds.dim() == 3:
            preds = preds.reshape(-1, 2)
            targets = targets.reshape(-1, 2)

            if torch.is_tensor(targets_sequence_mask):
                targets_sequence_mask = targets_sequence_mask.reshape(-1).bool()
            if torch.is_tensor(targets_valid_mask):
                targets_valid_mask = targets_valid_mask.reshape(-1).bool()

        # base valid from -5 and finite
        valid = torch.isfinite(targets).all(dim=1) & (targets[:, 0] != -5.0) & (targets[:, 1] != -5.0)

        if torch.is_tensor(targets_sequence_mask):
            valid = valid & targets_sequence_mask.to(device=valid.device)
        if torch.is_tensor(targets_valid_mask):
            valid = valid & targets_valid_mask.to(device=valid.device)

        preds = preds[valid]
        targets = targets[valid]

        if preds.numel() == 0:
            return preds.new_tensor(0.0)

        preds_f = preds.float()
        targets_f = targets.float()

        if preds_f.shape[0] < self.min_valid:
            return self._mse(preds_f, targets_f)

        wsum = self.v_weight + self.a_weight
        ccc_v = 1.0 - ccc_1d(preds_f[:, 0], targets_f[:, 0])
        ccc_a = 1.0 - ccc_1d(preds_f[:, 1], targets_f[:, 1])
        ccc = (self.v_weight * ccc_v + self.a_weight * ccc_a) / wsum

        loss = self.ccc_weight * ccc
        if self.mse_weight > 0.0:
            loss = loss + self.mse_weight * self._mse(preds_f, targets_f)

        return loss
    

class CCCMSEMMSmoothLoss(BaseLoss):
    def __init__(
        self,
        ccc_weight=1.0,
        mse_weight=0.0,
        v_weight=0.5,
        a_weight=0.5,
        min_valid=4,
        temporal_diff_weight=0.0,
        temporal_diff_v_weight=1.0,
        temporal_diff_a_weight=1.0,
    ):
        self.ccc_weight = float(ccc_weight)
        self.mse_weight = float(mse_weight)
        self.v_weight = float(v_weight)
        self.a_weight = float(a_weight)
        self.min_valid = int(min_valid)

        self.temporal_diff_weight = float(temporal_diff_weight)
        self.temporal_diff_v_weight = float(temporal_diff_v_weight)
        self.temporal_diff_a_weight = float(temporal_diff_a_weight)

        self._mse = nn.MSELoss()

    def _temporal_smoothness_loss(
        self,
        preds: torch.Tensor,               # [B, L, 2]
        seq_mask: torch.Tensor | None,     # [B, L]
        valid_mask: torch.Tensor | None,   # [B, L]
    ) -> torch.Tensor:
        if preds.ndim != 3 or preds.shape[1] < 2:
            return preds.new_tensor(0.0)

        pair_mask = torch.ones(
            preds.shape[:2], device=preds.device, dtype=torch.bool
        )

        if seq_mask is not None:
            pair_mask = pair_mask & seq_mask.bool()

        if valid_mask is not None:
            pair_mask = pair_mask & valid_mask.bool()

        pair_mask = pair_mask[:, 1:] & pair_mask[:, :-1]   # [B, L-1]

        if not pair_mask.any():
            return preds.new_tensor(0.0)

        diffs = preds[:, 1:, :] - preds[:, :-1, :]         # [B, L-1, 2]
        diffs = diffs[pair_mask]                           # [N, 2]

        loss_v = (diffs[:, 0] ** 2).mean()
        loss_a = (diffs[:, 1] ** 2).mean()

        wsum = self.temporal_diff_v_weight + self.temporal_diff_a_weight
        return (
            self.temporal_diff_v_weight * loss_v
            + self.temporal_diff_a_weight * loss_a
        ) / max(wsum, 1e-8)

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        preds = output.preds
        targets = batch.targets
        if targets is None:
            raise ValueError("CCCMSEMMLoss requires batch.targets (got None).")

        masks = batch.meta.get("masks", {}) if batch.meta else {}
        targets_sequence_mask = masks.get("targets_sequence_mask", None)
        targets_valid_mask = masks.get("targets_valid_mask", None)

        temporal_loss = preds.new_tensor(0.0)

        if preds.dim() == 3:
            temporal_loss = self._temporal_smoothness_loss(
                preds=preds,
                seq_mask=targets_sequence_mask if torch.is_tensor(targets_sequence_mask) else None,
                valid_mask=targets_valid_mask if torch.is_tensor(targets_valid_mask) else None,
            )

            preds = preds.reshape(-1, 2)
            targets = targets.reshape(-1, 2)

            if torch.is_tensor(targets_sequence_mask):
                targets_sequence_mask = targets_sequence_mask.reshape(-1).bool()
            if torch.is_tensor(targets_valid_mask):
                targets_valid_mask = targets_valid_mask.reshape(-1).bool()

        valid = torch.isfinite(targets).all(dim=1) & (targets[:, 0] != -5.0) & (targets[:, 1] != -5.0)

        if torch.is_tensor(targets_sequence_mask):
            valid = valid & targets_sequence_mask.to(device=valid.device)

        if torch.is_tensor(targets_valid_mask):
            valid = valid & targets_valid_mask.to(device=valid.device)

        preds = preds[valid]
        targets = targets[valid]

        if preds.numel() == 0:
            return self.temporal_diff_weight * temporal_loss

        preds_f = preds.float()
        targets_f = targets.float()

        if preds_f.shape[0] < self.min_valid:
            base_loss = self._mse(preds_f, targets_f)
            return base_loss + self.temporal_diff_weight * temporal_loss

        wsum = self.v_weight + self.a_weight
        ccc_v = 1.0 - ccc_1d(preds_f[:, 0], targets_f[:, 0])
        ccc_a = 1.0 - ccc_1d(preds_f[:, 1], targets_f[:, 1])
        ccc = (self.v_weight * ccc_v + self.a_weight * ccc_a) / max(wsum, 1e-8)

        loss = self.ccc_weight * ccc

        if self.mse_weight > 0.0:
            loss = loss + self.mse_weight * self._mse(preds_f, targets_f)

        if self.temporal_diff_weight > 0.0:
            loss = loss + self.temporal_diff_weight * temporal_loss

        return loss


@LOSSES.register("ccc_mse_mm_loss")
def ccc_mse_mm_loss(*, 
                    ccc_weight: float = 1.0, 
                    mse_weight: float = 0.0, 
                    v_weight: float = 0.5,
                    a_weight: float = 0.5,
                    **_) -> BaseLoss:
    return CCCMSEMMLoss(ccc_weight=ccc_weight, 
                        mse_weight=mse_weight,
                        v_weight=v_weight,
                        a_weight=a_weight)


@LOSSES.register("ccc_mse_mm_smooth_loss")
def ccc_mse_mm_smooth_loss(
    *,
    ccc_weight: float = 1.0,
    mse_weight: float = 0.0,
    v_weight: float = 0.5,
    a_weight: float = 0.5,
    min_valid: int = 4,
    temporal_diff_weight: float = 0.0,
    temporal_diff_v_weight: float = 1.0,
    temporal_diff_a_weight: float = 1.0,
    **_,
):
    return CCCMSEMMSmoothLoss(
        ccc_weight=ccc_weight,
        mse_weight=mse_weight,
        v_weight=v_weight,
        a_weight=a_weight,
        min_valid=min_valid,
        temporal_diff_weight=temporal_diff_weight,
        temporal_diff_v_weight=temporal_diff_v_weight,
        temporal_diff_a_weight=temporal_diff_a_weight,
    )