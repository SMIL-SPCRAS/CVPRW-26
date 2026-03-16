from dataclasses import dataclass
from typing import Dict, Optional

import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class EMACallback(BaseCallback):
    """
    Exponential Moving Average (EMA) of model weights.

    How it works:
      - During training: maintain shadow EMA weights updated from current model params.
      - For evaluation: swap model weights -> EMA weights (so val_* callbacks use EMA),
        then restore original weights before next training epoch/batch.

    IMPORTANT:
      Put this callback BEFORE framewise_eval_callback in config callbacks list
      so that framewise evaluation runs on EMA weights.
    """

    decay: float = 0.999
    update_every: int = 1  # update EMA every N optimizer steps (or train steps)
    use_ema_for_eval: bool = True
    use_ema_for_save: bool = True  # if your checkpoint uses current model params, EMA will be saved
    start_epoch: int = 0
    # internal state
    _ema: Optional[Dict[str, torch.Tensor]] = None
    _backup: Optional[Dict[str, torch.Tensor]] = None
    _ema_on: bool = False
    _cur_epoch: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.decay < 1.0):
            raise ValueError("decay must be in (0,1)")
        
        if self.update_every < 1:
            raise ValueError("update_every must be >= 1")
        
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be >= 0")

    # ---------- helpers ----------
    def _named_params(self, model):
        # EMA for ALL floating params (trainable + frozen), skip buffers
        for name, p in model.named_parameters():
            if p is None:
                continue
            if not torch.is_floating_point(p.data):
                continue
            yield name, p

    def _ema_enabled(self) -> bool:
        return self._cur_epoch >= self.start_epoch

    def _init_ema(self, model) -> None:
        self._ema = {}
        for name, p in self._named_params(model):
            self._ema[name] = p.detach().clone()

    @torch.no_grad()
    def _update_ema(self, model) -> None:
        if self._ema is None:
            self._init_ema(model)
            return

        d = float(self.decay)
        for name, p in self._named_params(model):
            if name not in self._ema:
                self._ema[name] = p.detach().clone()
            else:
                self._ema[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def _swap_to_ema(self, model) -> None:
        """Save current weights -> backup, then load EMA weights into model."""
        if self._ema is None or self._ema_on:
            return

        self._backup = {}
        for name, p in self._named_params(model):
            self._backup[name] = p.detach().clone()
            if name in self._ema:
                p.copy_(self._ema[name])
        self._ema_on = True

    @torch.no_grad()
    def _restore_from_backup(self, model) -> None:
        """Restore weights saved in backup (original model weights)."""
        if not self._ema_on:
            return
        if self._backup is None:
            self._ema_on = False
            return

        for name, p in self._named_params(model):
            if name in self._backup:
                p.copy_(self._backup[name])
        self._backup = None
        self._ema_on = False

    # -------- callback hooks --------
    def on_train_start(self, trainer) -> None:
        # initialize EMA from initial weights
        self._init_ema(trainer.model)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        self._cur_epoch = int(epoch)
        self._restore_from_backup(trainer.model)

    # Fallbacks: some engines call these names instead
    def on_batch_end(self, trainer, global_step: int, logs: Dict[str, float]) -> None:
        if not self._ema_enabled():
            return
        
        # try to update only during training
        if (int(global_step) % self.update_every) == 0:
            self._update_ema(trainer.model)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]):
        """
        Key trick:
          - After training epoch is done, before other eval callbacks run,
            swap model to EMA so val_framewise uses EMA weights.
          - Restore at next epoch start / next train batch start.
        """
        self._cur_epoch = int(epoch)
        if not self._ema_enabled():
            return
        
        if self.use_ema_for_eval or self.use_ema_for_save:
            self._swap_to_ema(trainer.model)

    def on_fit_end(self, trainer) -> None:
        # leave model in EMA state for final save if desired
        if not self._ema_enabled():
            return
        
        if self.use_ema_for_save:
            self._swap_to_ema(trainer.model)
        else:
            self._restore_from_backup(trainer.model)


@CALLBACKS.register("ema_callback")
def ema_callback(**params):
    return EMACallback(**params)