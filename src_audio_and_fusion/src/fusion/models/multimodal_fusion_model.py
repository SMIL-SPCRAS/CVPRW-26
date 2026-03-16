from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera_ml.core.batch import Batch
from chimera_ml.models.base import BaseModel
from chimera_ml.core.types import ModelOutput
from chimera_ml.core.registry import MODELS
    

class MultimodalFusionModelV4(BaseModel):
    """
    Reliability-gated multimodal fusion:
      1) framewise modalities -> projected to hidden
      2) gated fusion across frame modalities per frame
      3) audio tokens -> projected to hidden
      4) learnable bottleneck latents attend to audio
      5) fused frame tokens attend to audio latents
      6) temporal encoder over frame sequence
      7) regression head -> [B, L, 2]

    Expected batch:
      batch.inputs:
        - audio_embedding:   [B, T_a, D_a]
        - audio_prediction:  [B, T_a, 2]       optional
        - {mod}_embedding:   [B, L, D_m]
        - {mod}_prediction:  [B, L, 2]         optional

      batch.meta["masks"]:
        - targets_sequence_mask: [B, L]
        - input_valid_mask["audio"]: [B, T_a]
        - input_valid_mask[mod]:     [B, L]
    """

    def __init__(
        self,
        frame_modalities: List[str],
        *,
        use_predictions: bool = False,
        hidden: int = 256,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_temporal_layers: int = 2,
        num_audio_latents: int = 4,
        modality_priors: Optional[dict[str, float]] = None,
    ):
        super().__init__()
        self.frame_modalities = list(frame_modalities)
        self.use_predictions = bool(use_predictions)
        self.hidden = int(hidden)
        self.num_audio_latents = int(num_audio_latents)

        # lazy projectors: no need to know input dims upfront
        self.frame_projectors = nn.ModuleDict({
            mod: nn.LazyLinear(hidden) for mod in self.frame_modalities
        })
        self.frame_gate_mlps = nn.ModuleDict({
            mod: nn.Sequential(
                nn.LazyLinear(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for mod in self.frame_modalities
        })

        self.audio_projector = nn.LazyLinear(hidden)

        # learnable bottleneck latents for audio compression
        self.audio_latents = nn.Parameter(torch.randn(1, self.num_audio_latents, hidden) * 0.02)

        self.audio_latent_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.frame_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_frame_pre = nn.LayerNorm(hidden)
        self.norm_audio_latents = nn.LayerNorm(hidden)
        self.norm_after_audio = nn.LayerNorm(hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_temporal_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        # priors for modality gating
        modality_priors = modality_priors or {}
        prior_vals = []
        for mod in self.frame_modalities:
            p = float(modality_priors.get(mod, 1.0))
            prior_vals.append(max(p, 1e-6))
        prior_tensor = torch.tensor(prior_vals, dtype=torch.float32)
        prior_log = torch.log(prior_tensor / prior_tensor.sum())
        self.register_buffer("modality_log_priors", prior_log, persistent=True)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T]
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        return (x * w).sum(dim=1) / denom

    def _build_frame_tokens(
        self,
        batch: Batch,
        input_valid_mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          fused_frame: [B, L, H]
          any_valid:   [B, L] bool
        """
        reps = []
        gate_logits = []
        valid_masks = []

        for i, mod in enumerate(self.frame_modalities):
            emb_key = f"{mod}_embedding"
            pred_key = f"{mod}_prediction"

            if emb_key not in batch.inputs:
                continue

            x = batch.inputs[emb_key]  # [B, L, D]
            if self.use_predictions and pred_key in batch.inputs:
                x = torch.cat([x, batch.inputs[pred_key]], dim=-1)

            x = self.frame_projectors[mod](x)  # [B, L, H]

            vmask = input_valid_mask.get(mod, None)
            if vmask is None:
                raise ValueError(f"Missing input_valid_mask['{mod}'].")

            vmask_f = vmask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * vmask_f

            logit = self.frame_gate_mlps[mod](x).squeeze(-1)  # [B, L]
            logit = logit + self.modality_log_priors[i]

            # invalid positions for this modality should not receive weight
            logit = logit.masked_fill(~vmask, -1e4)

            reps.append(x)
            gate_logits.append(logit)
            valid_masks.append(vmask)

        if not reps:
            raise ValueError("No framewise modalities found in batch.inputs.")

        reps = torch.stack(reps, dim=2)               # [B, L, M, H]
        gate_logits = torch.stack(gate_logits, dim=2) # [B, L, M]
        valid_masks = torch.stack(valid_masks, dim=2) # [B, L, M]

        # if all modalities invalid at position -> weights become 0
        any_valid = valid_masks.any(dim=2)            # [B, L]
        weights = torch.softmax(gate_logits, dim=2)   # [B, L, M]
        weights = weights * valid_masks.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)

        fused_frame = (reps * weights.unsqueeze(-1)).sum(dim=2)  # [B, L, H]
        fused_frame = fused_frame * any_valid.unsqueeze(-1).to(dtype=fused_frame.dtype)

        return fused_frame, any_valid

    def _inject_audio_context(
        self,
        frame_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        frame_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        frame_tokens: [B, L, H]
        audio_tokens: [B, T_a, H]
        frame_mask:   [B, L] bool
        audio_mask:   [B, T_a] bool
        """
        B = frame_tokens.shape[0]

        # 1) compress audio into bottleneck latents
        latents = self.audio_latents.expand(B, -1, -1)  # [B, M, H]

        # MultiheadAttention with all keys masked can produce unstable outputs,
        # so handle rows with no valid audio explicitly.
        has_audio = audio_mask.any(dim=1)  # [B]

        if has_audio.any():
            latents_valid = latents[has_audio]
            audio_valid = audio_tokens[has_audio]
            audio_kpm = ~audio_mask[has_audio]  # True => ignore

            latents_upd, _ = self.audio_latent_attn(
                query=latents_valid,
                key=audio_valid,
                value=audio_valid,
                key_padding_mask=audio_kpm,
                need_weights=False,
            )
            latents = latents.clone()
            latents[has_audio] = self.norm_audio_latents(latents_valid + latents_upd)

        # 2) let frame tokens attend to audio latents
        out = frame_tokens
        if has_audio.any():
            q = self.norm_frame_pre(frame_tokens[has_audio])
            lat = latents[has_audio]

            # no padding mask needed for latents
            ctx, _ = self.frame_to_audio_attn(
                query=q,
                key=lat,
                value=lat,
                need_weights=False,
            )

            out = out.clone()
            out[has_audio] = self.norm_after_audio(frame_tokens[has_audio] + ctx)

        out = out * frame_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out, latents

    def forward(self, batch: Batch) -> ModelOutput:
        meta = batch.meta or {}
        masks = meta.get("masks", {})

        input_valid_mask = masks.get("input_valid_mask", {})
        targets_sequence_mask = masks.get("targets_sequence_mask", None)
        if targets_sequence_mask is None:
            raise ValueError("Missing batch.meta['masks']['targets_sequence_mask'].")

        # [B, L]
        frame_seq_mask = targets_sequence_mask.bool()

        # 1) fuse frame modalities with reliability gates
        frame_tokens, frame_any_valid = self._build_frame_tokens(batch, input_valid_mask)

        # 2) audio tokens
        if "audio_embedding" not in batch.inputs:
            raise ValueError("Missing 'audio_embedding' in batch.inputs.")

        audio_x = batch.inputs["audio_embedding"]  # [B, T_a, D_a]
        if self.use_predictions:
            if "audio_prediction" not in batch.inputs:
                raise ValueError("use_predictions=True but 'audio_prediction' missing.")
            audio_x = torch.cat([audio_x, batch.inputs["audio_prediction"]], dim=-1)

        audio_tokens = self.audio_projector(audio_x)  # [B, T_a, H]
        audio_mask = input_valid_mask.get("audio", None)
        if audio_mask is None:
            raise ValueError("Missing input_valid_mask['audio'].")

        audio_tokens = audio_tokens * audio_mask.unsqueeze(-1).to(dtype=audio_tokens.dtype)

        # 3) inject audio context through bottleneck latents
        frame_mask = frame_seq_mask & frame_any_valid
        fused_tokens, audio_latents = self._inject_audio_context(
            frame_tokens=frame_tokens,
            audio_tokens=audio_tokens,
            frame_mask=frame_mask,
            audio_mask=audio_mask.bool(),
        )

        # 4) temporal modeling on frame sequence
        # TransformerEncoder expects True in src_key_padding_mask for padded positions
        temporal_out = self.temporal_encoder(
            fused_tokens,
            src_key_padding_mask=~frame_seq_mask,
        )
        temporal_out = temporal_out * frame_seq_mask.unsqueeze(-1).to(dtype=temporal_out.dtype)

        # 5) regression head
        preds = self.head(temporal_out)  # [B, L, 2]

        aux = {
            "embs": temporal_out,
            "frame_tokens": frame_tokens,
            "audio_latents": audio_latents,
            "frame_valid_mask": frame_mask,
            "audio_valid_mask": audio_mask,
        }
        return ModelOutput(preds=preds, aux=aux)


@MODELS.register("multimodal_fusion_model_v4")
def multimodal_fusion_model_v4(
    *,
    frame_modalities: List[str],
    use_predictions: bool = False,
    hidden: int = 256,
    dropout: float = 0.1,
    num_heads: int = 4,
    num_temporal_layers: int = 2,
    num_audio_latents: int = 4,
    modality_priors: Optional[dict[str, float]] = None,
    context: Optional[dict[str, Any]] = None,
    **_,
) -> BaseModel:
    return MultimodalFusionModelV4(
        frame_modalities=list(frame_modalities),
        use_predictions=bool(use_predictions),
        hidden=int(hidden),
        dropout=float(dropout),
        num_heads=int(num_heads),
        num_temporal_layers=int(num_temporal_layers),
        num_audio_latents=int(num_audio_latents),
        modality_priors=modality_priors,
    )
