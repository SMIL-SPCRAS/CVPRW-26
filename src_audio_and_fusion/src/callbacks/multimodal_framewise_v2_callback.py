from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple, Any
import bisect
import pickle
import math

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS, METRICS

from utils import TensorMetricAdapter


@dataclass
class MultiModalFrameWiseCallback(BaseCallback):
    """
    Reruns inference on a chosen loader at epoch end and computes framewise metrics.

    Works with variable-length windows without using trainer.predictions_cache.

    Steps:
      1) Iterate loader -> model(batch) -> get window preds [B,T,2] and optional feats [B,T,D] / [B,D]
      2) For each sample, map it to frames [start_frame, end_frame) and resample preds/features to L
      3) Aggregate overlaps per frame: mean | center_weighted | last
      4) Load per-frame GT from files and match; optionally fill missing pred frames by nearest pred frame
      5) Compute metric and log; optionally dump big framewise pickle
    """

    loader_name: str = "val"        # which loader key to evaluate
    log_prefix: str = "val_frame"

    metric_name: str = "va_ccc_metric"
    metric_params: Optional[dict] = None
    eps: float = 1e-8

    # overlap aggregation
    overlap_strategy: str = "center_weighted"   # mean | center_weighted | gaussian_weighted | last
    gaussian_sigma_ratio: float = 0.22

    # smoothing after overlap aggregation
    smoothing: str = "none"                     # none | ema | moving_average | gaussian
    smoothing_valence_param: float = 0.0        # ema: alpha in (0,1], ma/gauss: kernel size
    smoothing_arousal_param: float = 0.0

    # GT files
    ann_root: str = "."
    ann_split_dir: str = "Validation_Set"
    ann_ext: str = ".txt"
    missing_value: float = -5.0

    # how to find video id in per-sample meta
    video_id_keys: List[str] = field(
        default_factory=lambda: ["vid", "segment_name", "video_name", "full_video_name", "filename"]
    )

    # dumping
    dump_pickle: bool = False
    pickle_path: str = "./frame_dump.pkl"
    pickle_numpy: bool = True
    frame_index_offset: int = 1  # frame 0 -> 00001.jpg if offset=1

    # fill missing GT-valid frames from nearest predicted frame
    fill_missing_gt_frames: bool = False

    submission_template_path: str = None
    submission_out_path: str = None

    def __post_init__(self) -> None:
        allowed_overlap = ("mean", "center_weighted", "gaussian_weighted", "last")
        if self.overlap_strategy not in allowed_overlap:
            raise ValueError(f"overlap_strategy must be one of: {allowed_overlap}")

        allowed_smoothing = ("none", "ema", "moving_average", "gaussian")
        if self.smoothing not in allowed_smoothing:
            raise ValueError(f"smoothing must be one of: {allowed_smoothing}")

        params = dict(self.metric_params or {})
        params.setdefault("eps", self.eps)
        metric = METRICS.create(self.metric_name, **params)
        self._metric = TensorMetricAdapter(metric)

        self._gt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ---------- helpers ----------
    def _normalize_video_id(self, raw: str) -> str:
        s = Path(str(raw)).stem
        s = s.split("___", 1)[0]
        return s

    def _resample_va_to_num_frames(self, p_t2: torch.Tensor, L: int) -> torch.Tensor:
        """
        p_t2: [Tpred, 2]
        returns [L, 2]
        """
        Tpred = int(p_t2.shape[0])
        L = int(L)

        if L <= 0:
            return p_t2[:0]

        if Tpred <= 0:
            return torch.zeros((L, 2), dtype=p_t2.dtype, device=p_t2.device)

        if Tpred == L:
            return p_t2.contiguous()

        if Tpred == 1:
            return p_t2.expand(L, -1).contiguous()

        x = p_t2.transpose(0, 1).unsqueeze(0)  # [1,2,T]
        x = F.interpolate(x, size=L, mode="linear", align_corners=True)
        x = x.squeeze(0).transpose(0, 1).contiguous()  # [L,2]
        return x

    def _get_video_id(self, meta: dict) -> str:
        for k in self.video_id_keys:
            v = meta.get(k, None)
            if v:
                return self._normalize_video_id(str(v))
        return ""

    def _get_window_frames(self, meta: dict) -> Tuple[int, int]:
        s = int(meta.get("start_frame", 0) or 0)
        e = int(meta.get("end_frame", 0) or 0)
        return s, e  # end exclusive

    def _tri_weight(self, j: int, L: int) -> float:
        L = max(int(L), 1)
        center = (L - 1) / 2.0
        half = max(L / 2.0, 1.0)
        w = 1.0 - abs(j - center) / half
        return float(max(w, 0.0))

    def _gauss_weight(self, j: int, L: int) -> float:
        L = max(int(L), 1)
        if L <= 1:
            return 1.0
        center = (L - 1) / 2.0
        sigma = max(float(L) * float(self.gaussian_sigma_ratio), 1e-6)
        z = (float(j) - center) / sigma
        return float(math.exp(-0.5 * z * z))

    def _weight_at(self, j: int, L: int) -> float:
        if self.overlap_strategy == "mean":
            return 1.0
        if self.overlap_strategy == "center_weighted":
            return self._tri_weight(j, L)
        if self.overlap_strategy == "gaussian_weighted":
            return self._gauss_weight(j, L)
        if self.overlap_strategy == "last":
            return 1.0
        raise ValueError(f"Unsupported overlap_strategy: {self.overlap_strategy}")

    def _ensure_odd_kernel(self, k: int) -> int:
        k = int(round(k))
        if k <= 1:
            return 1
        if k % 2 == 0:
            k += 1
        return k

    def _ema_smooth_1d(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        x: [F]
        """
        alpha = float(alpha)
        if x.numel() <= 1 or alpha <= 0.0:
            return x
        alpha = min(alpha, 1.0)

        y = x.clone()
        for i in range(1, x.shape[0]):
            y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
        return y

    def _moving_average_smooth_1d(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        x: [F]
        """
        k = self._ensure_odd_kernel(k)
        if x.numel() <= 1 or k <= 1:
            return x

        pad = k // 2
        xx = x.view(1, 1, -1)
        xx = F.pad(xx, (pad, pad), mode="replicate")
        kernel = torch.ones((1, 1, k), dtype=x.dtype, device=x.device) / float(k)
        y = F.conv1d(xx, kernel)
        return y.view(-1)

    def _gaussian_smooth_1d(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        x: [F]
        """
        k = self._ensure_odd_kernel(k)
        if x.numel() <= 1 or k <= 1:
            return x

        sigma = max(k / 6.0, 1e-6)  # roughly covers +-3 sigma
        center = (k - 1) / 2.0
        xs = torch.arange(k, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * ((xs - center) / sigma) ** 2)
        kernel = kernel / kernel.sum().clamp_min(1e-8)

        pad = k // 2
        xx = x.view(1, 1, -1)
        xx = F.pad(xx, (pad, pad), mode="replicate")
        y = F.conv1d(xx, kernel.view(1, 1, -1))
        return y.view(-1)

    def _smooth_channel(self, x: torch.Tensor, param: float) -> torch.Tensor:
        """
        x: [F]
        """
        if self.smoothing == "none" or param <= 0:
            return x

        if self.smoothing == "ema":
            return self._ema_smooth_1d(x, alpha=float(param))

        if self.smoothing == "moving_average":
            return self._moving_average_smooth_1d(x, k=int(round(param)))

        if self.smoothing == "gaussian":
            return self._gaussian_smooth_1d(x, k=int(round(param)))

        raise ValueError(f"Unsupported smoothing: {self.smoothing}")

    def _smooth_va_sequence(self, x_f2: torch.Tensor) -> torch.Tensor:
        """
        x_f2: [F,2]
        """
        if x_f2.numel() == 0 or self.smoothing == "none":
            return x_f2

        v = self._smooth_channel(x_f2[:, 0], self.smoothing_valence_param)
        a = self._smooth_channel(x_f2[:, 1], self.smoothing_arousal_param)
        return torch.stack([v, a], dim=-1)

    def _load_gt(self, video_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_id in self._gt_cache:
            return self._gt_cache[video_id]

        path = Path(self.ann_root) / self.ann_split_dir / f"{video_id}{self.ann_ext}"
        if not path.exists():
            raise FileNotFoundError(f"Per-frame GT file not found: {path}")

        df = pd.read_csv(path)
        gt = torch.tensor(df[["valence", "arousal"]].to_numpy(), dtype=torch.float32)  # [F,2]
        valid = (gt[:, 0] != self.missing_value) & (gt[:, 1] != self.missing_value)
        valid = valid & torch.isfinite(gt).all(dim=1)

        self._gt_cache[video_id] = (gt, valid)
        return gt, valid

    def _find_loader(self, trainer):
        for attr in ("_val_loaders", "_train_loaders", "_test_loaders", "_loaders"):
            d = getattr(trainer, attr, None)
            if isinstance(d, dict) and self.loader_name in d:
                return d[self.loader_name]
        return None

    def _to_time_major_va(self, p: torch.Tensor) -> torch.Tensor:
        # returns [T,2]
        if not torch.is_tensor(p):
            p = torch.as_tensor(p)
        while p.ndim > 2 and p.shape[0] == 1:
            p = p.squeeze(0)
        if p.ndim == 1:
            if p.numel() != 2:
                raise ValueError(f"Expected flattened VA vector of size 2, got shape {tuple(p.shape)}")
            return p.view(1, 2)
        if p.ndim != 2:
            raise ValueError(f"Expected 1D/2D preds per sample, got {tuple(p.shape)}")
        if p.shape[-1] == 2:
            return p.contiguous()

        if p.shape[0] == 2:
            return p.transpose(0, 1).contiguous()
        raise ValueError(f"Cannot interpret preds shape: {tuple(p.shape)}")

    # ---------- main ----------
    @torch.no_grad()
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]) -> None:
        loader = self._find_loader(trainer)
        if loader is None:
            return

        device = next(trainer.model.parameters()).device
        trainer.model.eval()
        use_amp = bool(getattr(trainer.config, "mixed_precision", False)) and (device.type == "cuda")

        # per_video[vid][fr] = (sum_pred[2], sum_w)
        per_video: DefaultDict[str, Dict[int, Tuple[torch.Tensor, float]]] = defaultdict(dict)

        pbar = tqdm(loader, desc=f"[{self.log_prefix} epoch {epoch}]", leave=False)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            for batch_raw in pbar:
                batch = trainer._to_device(batch_raw, device)
                out = trainer.model(batch)

                preds = out.preds.detach().cpu()  # [B,T,2] or [B,2]
                sample_meta = batch.meta.get("sample_meta", None) if batch.meta else None
                if not isinstance(sample_meta, list):
                    continue

                # lengths from padding mask (varlen collate)
                masks = batch.meta.get("masks", {}) if batch.meta else {}
                ttm = masks.get("targets_sequence_mask", None)
                lengths = None
                if torch.is_tensor(ttm):
                    lengths = ttm.detach().cpu().bool().sum(dim=1).tolist()

                B = int(preds.shape[0]) if preds.ndim >= 2 else 0

                for i in range(B):
                    meta_i = sample_meta[i] if isinstance(sample_meta[i], dict) else {}
                    vid = self._get_video_id(meta_i)
                    if not vid:
                        continue

                    s_f, e_f = self._get_window_frames(meta_i)
                    if e_f <= s_f:
                        continue
                    L = int(e_f - s_f)

                    # slice per-sample preds sequence
                    p_i = preds[i]
                    p_t2 = self._to_time_major_va(p_i)  # [Tpred,2]

                    # if collate provided true length, crop sequence
                    if lengths is not None:
                        Li = int(lengths[i])
                        if Li > 0:
                            p_t2 = p_t2[:Li]

                    if p_t2.numel() == 0:
                        continue

                    p_l2 = self._resample_va_to_num_frames(p_t2, L)  # [L,2]

                    m = per_video[vid]

                    for j, fr in enumerate(range(s_f, e_f)):
                        p = p_l2[j].view(-1)

                        if self.overlap_strategy == "last":
                            m[fr] = (p.clone(), 1.0)
                            continue

                        w = self._weight_at(j, L)
                        if w <= 0.0:
                            continue

                        if fr in m:
                            sp, sw = m[fr]
                            m[fr] = (sp + p * w, sw + float(w))
                        else:
                            m[fr] = (p.clone() * w, float(w))

        vids = sorted(per_video.keys())
        if not vids:
            return

        fp_list: List[torch.Tensor] = []
        ft_list: List[torch.Tensor] = []
        frame_dump: Optional[Dict[str, Dict[str, Any]]] = {}

        matched_frames_total = 0
        annotated_frames_total = 0
        oob_pred_frames_total = 0
        filled_frames_total = 0

        for vid in vids:
            try:
                gt, valid_gt = self._load_gt(vid)
            except Exception as e:
                try:
                    trainer.logger.warning(f"[{self.log_prefix}] skip {vid}: {e}")
                except Exception:
                    pass
                continue

            F_gt = int(gt.shape[0])
            annotated_frames_total += int(valid_gt.sum().item())

            frames = per_video[vid]
            if not frames:
                continue

            # optional: ensure last annotated valid frame exists
            if valid_gt.any():
                last_gt = int(torch.where(valid_gt)[0][-1].item())
                if last_gt not in frames:
                    prev = max((f for f in frames.keys() if f < last_gt), default=None)
                    if prev is not None:
                        frames[last_gt] = frames[prev]
                        filled_frames_total += 1

            # optional: fill all missing GT-valid frames by nearest prediction frame
            if self.fill_missing_gt_frames:
                valid_idx = torch.where(valid_gt)[0].tolist()
                pred_keys = sorted(frames.keys())
                if pred_keys:
                    pred_set = set(pred_keys)

                    for fr in valid_idx:
                        if fr in pred_set:
                            continue

                        pos = bisect.bisect_left(pred_keys, fr)
                        if pos == 0:
                            ref = pred_keys[0]
                        elif pos == len(pred_keys):
                            ref = pred_keys[-1]
                        else:
                            left = pred_keys[pos - 1]
                            right = pred_keys[pos]
                            ref = left if (fr - left) <= (right - fr) else right

                        frames[fr] = frames[ref]
                        filled_frames_total += 1
                        pred_set.add(fr)
                        pred_keys.insert(pos, fr)

            # build dense aggregated per-frame prediction for this video
            dense_pred = torch.full((F_gt, 2), float("nan"), dtype=torch.float32)
            dense_has_pred = torch.zeros(F_gt, dtype=torch.bool)

            for fr in sorted(frames.keys()):
                if fr < 0 or fr >= F_gt:
                    oob_pred_frames_total += 1
                    continue

                sp, sw = frames[fr]
                if sw <= 0.0:
                    continue

                dense_pred[fr] = (sp / float(sw)).view(2)
                dense_has_pred[fr] = True

            # smooth only on frames where we have dense predictions;
            # simplest stable choice: smooth the whole dense sequence after nearest fill over missing predicted frames
            pred_idx = torch.where(dense_has_pred)[0].tolist()
            if pred_idx:
                dense_for_smooth = dense_pred.clone()

                # fill holes by nearest predicted frame so smoothing is stable
                pred_keys = pred_idx
                for fr in range(F_gt):
                    if dense_has_pred[fr]:
                        continue

                    pos = bisect.bisect_left(pred_keys, fr)
                    if pos == 0:
                        ref = pred_keys[0]
                    elif pos == len(pred_keys):
                        ref = pred_keys[-1]
                    else:
                        left = pred_keys[pos - 1]
                        right = pred_keys[pos]
                        ref = left if (fr - left) <= (right - fr) else right

                    dense_for_smooth[fr] = dense_for_smooth[ref]

                dense_for_smooth = self._smooth_va_sequence(dense_for_smooth)

                # restore only frames that were originally predicted
                dense_pred[dense_has_pred] = dense_for_smooth[dense_has_pred]

            # collect matched GT-valid frames
            for fr in torch.where(valid_gt & dense_has_pred)[0].tolist():
                p = dense_pred[fr].view(1, 2)
                t = gt[fr].view(1, 2)

                fp_list.append(p)
                ft_list.append(t)
                matched_frames_total += 1

                if frame_dump is not None:
                    fr_no = int(fr) + int(self.frame_index_offset)
                    key = f"{vid}/{fr_no:05d}.jpg"
                    if self.pickle_numpy:
                        frame_dump[key] = {
                            "embedding": None,
                            "prediction": p.view(-1).cpu().numpy(),
                            "label": t.view(-1).cpu().numpy(),
                        }
                    else:
                        frame_dump[key] = {
                            "embedding": None,
                            "prediction": p.view(-1),
                            "label": t.view(-1),
                        }

        if not fp_list:
            return

        fp = torch.cat(fp_list, dim=0)
        ft = torch.cat(ft_list, dim=0)

        self._metric.reset()
        self._metric.update(fp, ft)
        out = self._metric.compute()

        for k, v in out.items():
            logs[f"{self.log_prefix}/{k}"] = float(v)

        coverage = (matched_frames_total / annotated_frames_total) if annotated_frames_total > 0 else 0.0

        msg_metrics = " ".join([f"{self.log_prefix}/{k}={float(v):.4f}" for k, v in out.items()])
        msg_stats = (
            f"{self.log_prefix}/matched_frames={matched_frames_total} "
            f"{self.log_prefix}/annotated_frames={annotated_frames_total} "
            f"{self.log_prefix}/coverage={coverage:.3f} "
            f"{self.log_prefix}/filled_frames={filled_frames_total} "
            f"{self.log_prefix}/oob_pred_frames={oob_pred_frames_total}"
        )

        try:
            trainer.logger.info(f"[{self.log_prefix} epoch {epoch}] {msg_metrics}")
            trainer.logger.info(f"[{self.log_prefix} epoch {epoch}] {msg_stats}")
        except Exception:
            print(f"[{self.log_prefix} epoch {epoch}] {msg_metrics}")
            print(f"[{self.log_prefix} epoch {epoch}] {msg_stats}")

        if frame_dump is not None and len(frame_dump) > 0:
            p = Path(self.pickle_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("wb") as f:
                pickle.dump(frame_dump, f, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                trainer.logger.info(f"[{self.log_prefix}] dumped {len(frame_dump)} frames to {p}")
            except Exception:
                pass

        if frame_dump is not None and self.submission_template_path and self.submission_out_path:
            template_df = pd.read_csv(self.submission_template_path)

            valence_list = []
            arousal_list = []

            for image_location in template_df["image_location"].tolist():
                item = frame_dump.get(image_location, None)
                if item is None:
                    raise KeyError(f"Prediction not found for {image_location}")

                pred = item["prediction"]
                if not torch.is_tensor(pred):
                    pred = torch.as_tensor(pred, dtype=torch.float32)
                else:
                    pred = pred.detach().cpu().float()

                pred = pred.clamp(-1.0, 1.0)

                valence_list.append(float(pred[0].item()))
                arousal_list.append(float(pred[1].item()))

            out_df = pd.DataFrame({
                "image_location": template_df["image_location"],
                "valence": valence_list,
                "arousal": arousal_list,
            })
            out_df.to_csv(self.submission_out_path, index=False)

        if getattr(trainer, "mlflow_logger", None) is not None:
            trainer.mlflow_logger.log_metrics(
                {f"{self.log_prefix}/{k}": float(v) for k, v in out.items()},
                step=epoch,
            )


@CALLBACKS.register("multimodal_framewise_v2_callback")
def multimodal_framewise_v2_callback(**params):
    return MultiModalFrameWiseCallback(**params)