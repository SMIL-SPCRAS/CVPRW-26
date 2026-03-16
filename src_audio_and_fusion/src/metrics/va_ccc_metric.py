from typing import Dict

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric

from utils import ccc_1d

class VACCCMetric(BaseMetric):
    """
    Computes:
      ccc_valence, ccc_arousal, ccc_va = (ccc_valence + ccc_arousal)/2
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._preds = []
        self._targets = []

    def update(self, output: ModelOutput, batch: Batch) -> None:
        # unlabeled test => skip
        if batch.targets is None:
            return

        preds = output.preds
        targets = batch.targets

        # seq2seq support
        if preds.dim() == 3:
            preds = preds.reshape(-1, 2)
            targets = targets.reshape(-1, 2)

        valid = torch.isfinite(targets).all(dim=1) & (targets[:,0] != -5.0) & (targets[:,1] != -5.0)
        preds = preds[valid]
        targets = targets[valid]

        if preds.numel() == 0:
            return
        
        self._preds.append(preds.detach().float().cpu())
        self._targets.append(targets.detach().float().cpu())

    def compute(self) -> Dict[str, float]:
        if not self._preds:
            return {
                "v_ccc_metric": float("nan"),
                "a_ccc_metric": float("nan"),
                "va_ccc_metric": float("nan"),
            }

        preds = torch.cat(self._preds, dim=0)
        targets = torch.cat(self._targets, dim=0)

        ccc_v = ccc_1d(preds[:, 0], targets[:, 0], eps=self.eps)
        ccc_a = ccc_1d(preds[:, 1], targets[:, 1], eps=self.eps)
        ccc_va = 0.5 * (ccc_v + ccc_a)

        return {
            "v_ccc_metric": float(ccc_v.item()),
            "a_ccc_metric": float(ccc_a.item()),
            "va_ccc_metric": float(ccc_va.item()),
        }


@METRICS.register("va_ccc_metric")
def va_ccc_metric(*, eps: float = 1e-8, **_) -> BaseMetric:
    return VACCCMetric(eps=float(eps))

