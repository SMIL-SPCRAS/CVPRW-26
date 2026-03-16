from typing import Dict

import torch
from torch import nn
from transformers import AutoModel


class TextVARegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        head_hidden_dim: int = 0,
        head_dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.dropout = nn.Dropout(dropout)
        if self.head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, self.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.head_dropout),
                nn.Linear(self.head_hidden_dim, 2),
            )
        else:
            self.head = nn.Linear(hidden_size, 2)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-8)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            raise RuntimeError("Backbone output does not contain last_hidden_state or pooler_output.")

        pooled = self.dropout(pooled)
        return self.head(pooled)


def build_model_from_checkpoint(
    checkpoint: Dict,
    device: torch.device,
    freeze_backbone: bool = False,
) -> TextVARegressor:
    head_hidden_dim = int(checkpoint.get("head_hidden_dim", 0))
    head_dropout = float(checkpoint.get("head_dropout", checkpoint.get("dropout", 0.1)))
    model = TextVARegressor(
        model_name=checkpoint["model_name"],
        dropout=float(checkpoint.get("dropout", 0.1)),
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        freeze_backbone=freeze_backbone,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
    }
