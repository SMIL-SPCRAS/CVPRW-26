from typing import Any, Optional

import torch
from chimera_ml.core.registry import OPTIMIZERS


@OPTIMIZERS.register("adamw_two_group_optimizer")
def adamw_two_group_optimizer(
    *,
    model: torch.nn.Module,
    lr_backbone: float = 1e-5,
    lr_head: float = 5e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    context: Optional[dict[str, Any]] = None,
    **_,
) -> torch.optim.Optimizer:
    """
    AdamW with two parameter groups:
      - backbone params (usually frozen or partially unfrozen) with lr_backbone
      - head params with lr_head
    Assumes model has attributes: .backbone and .head (as in our HF model).
    """
    backbone = getattr(model, "backbone", None)
    head = getattr(model, "head", None)

    if backbone is None or head is None:
        # fallback: single group
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr_head, weight_decay=weight_decay, betas=betas, eps=eps)

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"})

    if not param_groups:
        raise ValueError("No trainable parameters found (all requires_grad=False).")

    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)
