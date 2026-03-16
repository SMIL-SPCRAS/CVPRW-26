import torch

from chimera_ml.core.registry import SCHEDULERS


class NamedReduceLROnPlateauScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, target_group_names=("head",), **kwargs):
        self.target_group_names = set(target_group_names)
        super().__init__(optimizer, **kwargs)

    def _reduce_lr(self, epoch):
        for i, group in enumerate(self.optimizer.param_groups):
            gname = group.get("name", None)
            if gname not in self.target_group_names:
                continue

            old_lr = group["lr"]

            min_lr = self.min_lrs[i] if isinstance(self.min_lrs, (list, tuple)) else self.min_lrs

            new_lr = max(old_lr * self.factor, min_lr)

            if new_lr < old_lr - self.eps:
                group["lr"] = new_lr


@SCHEDULERS.register("named_reduceonplateau_scheduler")
def named_reduceonplateau_scheduler(*, optimizer, target_group_names=("head",), **params):
    return NamedReduceLROnPlateauScheduler(
        optimizer,
        target_group_names=target_group_names,
        **params
    )