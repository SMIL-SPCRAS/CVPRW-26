from typing import Any, Optional

import torch
import torch.nn as nn

from chimera_ml.models.base import BaseModel
from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.core.registry import MODELS

from audio.models.common_models import AttnPool1d


class WavLMS2SModelV1(BaseModel):
    def __init__(
        self,
        model_name: str = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
        freeze_backbone: bool = False,
        unfreeze_last_n_layers: int = 0,
        pooling: str = "mean",
        head_dropout: float = 0.1,
        input_normalize: bool = True,

        attn_hidden_dim: Optional[int] = 128,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        try:
            from transformers import AutoModel
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                "transformers is required. Install with: pip install -U transformers"
            ) from e

        self.model_name = model_name
        self.pooling = pooling
        self.input_normalize = input_normalize

        self.backbone = AutoModel.from_pretrained("microsoft/wavlm-large")
        hidden_size = self.backbone.config.hidden_size

        ckpt_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        sd = load_file(ckpt_path)

        ssl_sd = {k[len("ssl_model."):]: v for k, v in sd.items() if k.startswith("ssl_model.")}
        self.backbone.load_state_dict(ssl_sd, strict=False)

        cfg = self.backbone.config

        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None:
            # Some configs use `hidden_dim`
            hidden_size = getattr(cfg, "hidden_dim")

        self.pooling = pooling
        
        if "attn" in self.pooling:
            attn_use_std = True if self.pooling == "attn_stats" else False
            self.pool = AttnPool1d(
                in_dim=hidden_size,
                attn_dim=attn_hidden_dim,
                attn_dropout=attn_dropout,
                attn_use_std=attn_use_std
            )

            head_in = hidden_size * 2 if attn_use_std else hidden_size    
        elif self.pooling == "stats":
            self.pool = None
            head_in = hidden_size * 2
        elif self.pooling == "mean":
            self.pool = None
            head_in = hidden_size
        else:
            raise ValueError(f"Unsupported pooling='{self.pooling}'. Use one of: mean|stats|attn")

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(head_dropout),
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 2),
        )

        if freeze_backbone or (unfreeze_last_n_layers and unfreeze_last_n_layers > 0):
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_layers and unfreeze_last_n_layers > 0 and hasattr(self.backbone, "encoder"):
            layers = getattr(self.backbone.encoder, "layers", None)
            if layers is not None:
                for layer in layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def _pool_chunks(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor], n_chunks: int = 4) -> torch.Tensor:
        B, T, C = h.shape
        
        cuts = [int(round(i * T / n_chunks)) for i in range(n_chunks + 1)]
        outs = []
        for i in range(n_chunks):
            start, end = cuts[i], cuts[i + 1]
            hh = h[:, start:end, :]
            mm = attn_mask[:, start:end] if attn_mask is not None else None
            outs.append(self._pool(hh, attn_mask=mm))  # [B, head_in]

        return torch.stack(outs, dim=1)  # [B,4,head_in]

    @staticmethod
    def _make_attention_mask(audio_len: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=audio_len.device).unsqueeze(0)  # [1, T]
        return (idx < audio_len.unsqueeze(1)).long()  # [B, T]

    def _normalize_audio(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std
    
    def _normalize_audio_masked(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is None:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (x - mean) / std

        m = attn_mask.to(x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=1, keepdim=True) / denom
        var = ((x - mean) ** 2 * m).sum(dim=1, keepdim=True) / denom
        std = (var + 1e-6).sqrt().clamp_min(1e-6)

        x = (x - mean) / std
        return x * m

    def _get_feature_mask(self, att_mask: Optional[torch.Tensor], feat_len: int) -> Optional[torch.Tensor]:
        """Map raw waveform attention_mask -> feature-level mask for pooling."""
        if att_mask is None:
            return None

        # Wav2Vec2/WavLM backbones usually provide this helper.
        fn = getattr(self.backbone, "_get_feature_vector_attention_mask", None)
        if callable(fn):
            return fn(feat_len, att_mask, add_adapter=False)

        # Fallback: approximate (works if feature length ~= input length)
        if att_mask.shape[1] == feat_len:
            return att_mask

        # Last resort: downsample by ratio
        ratio = att_mask.shape[1] / float(feat_len)
        idx = (torch.arange(feat_len, device=att_mask.device) * ratio).long().clamp_max(att_mask.shape[1] - 1)
        return att_mask.index_select(1, idx)

    def _pool(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if "attn" in self.pooling:
            if attn_mask is not None:
                bad = attn_mask.sum(dim=1) == 0
                if bad.any():
                    attn_mask = attn_mask.clone()
                    attn_mask[bad, 0] = 1
            
            return self.pool(h, attn_mask=attn_mask)

        if attn_mask is None:
            if self.pooling == "mean":
                return h.mean(dim=1)
            mean = h.mean(dim=1)
            std = h.std(dim=1, unbiased=False).clamp_min(1e-6)
            return torch.cat([mean, std], dim=-1)

        m = attn_mask.to(h.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        if self.pooling == "mean":
            return (h * m).sum(dim=1) / denom

        mean = (h * m).sum(dim=1) / denom
        var = ((h - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        std = (var + 1e-6).sqrt().clamp_min(1e-6)
        return torch.cat([mean, std], dim=-1)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch.inputs["audio"]
        if x.dtype != torch.float32:
            x = x.float()

        attn_mask = batch.meta.get("attention_mask", None)
        if attn_mask is None:
            audio_len = batch.meta.get("audio_len", None)
            if audio_len is not None:
                attn_mask = self._make_attention_mask(audio_len.to(x.device), x.size(1))
        elif attn_mask is not None:
            attn_mask = attn_mask.long()

        if self.input_normalize:
            x = self._normalize_audio_masked(x, attn_mask)

        kwargs = {"input_values": x, "return_dict": True}
        if attn_mask is not None:
            kwargs["attention_mask"] = attn_mask

        out = self.backbone(**kwargs)
        h = out.last_hidden_state

        feat_mask = self._get_feature_mask(attn_mask, h.size(1))
        emb = self._pool_chunks(h, attn_mask=feat_mask, n_chunks=4)

        # --- NEW ---
        z = emb
        feat256 = None
        for idx, layer in enumerate(self.head):
            z = layer(z)
            if idx == 3:
                feat256 = z
                
        preds = z
        return ModelOutput(preds=preds, aux={"features": feat256})
    

@MODELS.register("wavlm_s2s_model_v1")
def wavlm_s2s_model_v1(
    *,
    model_name: str = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
    freeze_backbone: bool = False,
    unfreeze_last_n_layers: int = 0,
    pooling: str = "mean",
    head_dropout: float = 0.1,
    input_normalize: bool = True,
    attn_hidden_dim: int = 128,
    attn_dropout: float = 0.2,
    context: Optional[dict[str, Any]] = None,
    **_,
) -> BaseModel:
    return WavLMS2SModelV1(
        model_name=str(model_name),
        freeze_backbone=bool(freeze_backbone),
        unfreeze_last_n_layers=int(unfreeze_last_n_layers),
        pooling=str(pooling),
        head_dropout=float(head_dropout),
        input_normalize=bool(input_normalize),
        attn_hidden_dim=int(attn_hidden_dim),
        attn_dropout=float(attn_dropout),
    )
