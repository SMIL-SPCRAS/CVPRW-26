from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple, Any
import bisect
import pickle

import pandas as pd
import torch
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

    overlap_strategy: str = "center_weighted"  # mean | center_weighted | last

    # GT file settings
    ann_root: str = "."
    ann_split_dir: str = "Validation_Set"
    ann_ext: str = ".txt"
    missing_value: float = -5.0

    # how to find video id in per-sample meta
    video_id_keys: List[str] = field(default_factory=lambda: ["vid", "segment_name", "video_name", "full_video_name", "filename"])

    # dumping
    dump_pickle: bool = False
    pickle_path: str = "./frame_dump.pkl"
    pickle_numpy: bool = True
    frame_index_offset: int = 1  # frame 0 -> 00001.jpg if offset=1

    fill_missing_gt_frames: bool = False

    def __post_init__(self) -> None:
        if self.overlap_strategy not in ("mean", "center_weighted", "last"):
            raise ValueError("overlap_strategy must be one of: mean|center_weighted|last")

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

    @staticmethod
    def resample_va_to_num_frames(p_t2: torch.Tensor, L: int) -> torch.Tensor:
        """
        p_t2: [T, 2]
        returns: [L, 2]
        """
        T = int(p_t2.shape[0])
        L = int(L)

        if T == L:
            return p_t2
        if T == 1:
            return p_t2.expand(L, -1)

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
            return p.view(1, 2)
        if p.ndim != 2:
            raise ValueError(f"Expected 1D/2D preds per sample, got {tuple(p.shape)}")
        if p.shape[-1] == 2:
            return p
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

                    Tpred = int(p_t2.shape[0])

                    m = per_video[vid]

                    # case 1: constant pred -> apply to all frames
                    if Tpred == 1:
                        p0 = p_t2[0].view(-1)
                        for j, fr in enumerate(range(s_f, e_f)):
                            if self.overlap_strategy == "last":
                                m[fr] = (p0.clone(), 1.0)
                                continue

                            w = 1.0
                            if self.overlap_strategy == "center_weighted":
                                w = self._tri_weight(j, L)
                                if w <= 0.0:
                                    continue

                            if fr in m:
                                sp, sw = m[fr]
                                m[fr] = (sp + p0 * w, sw + float(w))
                            else:
                                m[fr] = (p0.clone() * w, float(w))
                        continue

                    # case 2: Tpred >= L -> take first L (no resample)
                    if Tpred >= L:
                        for j, fr in enumerate(range(s_f, e_f)):
                            p = p_t2[j].view(-1)
                            if self.overlap_strategy == "last":
                                m[fr] = (p.clone(), 1.0)
                                continue

                            w = 1.0
                            if self.overlap_strategy == "center_weighted":
                                w = self._tri_weight(j, L)
                                if w <= 0.0:
                                    continue

                            if fr in m:
                                sp, sw = m[fr]
                                m[fr] = (sp + p * w, sw + float(w))
                            else:
                                m[fr] = (p.clone() * w, float(w))
                        continue

                    # case 3: Tpred < L -> write only first Tpred frames; rest will be filled by nearest later
                    for j in range(Tpred):
                        fr = s_f + j
                        if fr >= e_f:
                            break
                        p = p_t2[j].view(-1)

                        if self.overlap_strategy == "last":
                            m[fr] = (p.clone(), 1.0)
                            continue

                        w = 1.0
                        if self.overlap_strategy == "center_weighted":
                            w = self._tri_weight(j, L)
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
        frame_dump: Optional[Dict[str, Dict[str, Any]]] = {} if self.dump_pickle else None

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

            F = int(gt.shape[0])
            annotated_frames_total += int(valid_gt.sum().item())
            
            frames = per_video[vid]
            if not frames:
                continue

            if valid_gt.any():
                last_gt = int(torch.where(valid_gt)[0][-1].item())
                if last_gt not in frames:
                    prev = max((f for f in frames.keys() if f < last_gt), default=None)
                    if prev is not None:
                        frames[last_gt] = frames[prev]
                        filled_frames_total += 1

            # fill missing GT-valid frames by nearest prediction frame
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

            for fr in sorted(frames.keys()):
                if fr < 0 or fr >= F:
                    oob_pred_frames_total += 1
                    continue
                if not bool(valid_gt[fr]):
                    continue

                sp, sw = frames[fr]
                if sw <= 0.0:
                    continue

                p = (sp / float(sw)).view(1, 2)
                t = gt[fr].view(1, 2)

                fp_list.append(p)
                ft_list.append(t)
                matched_frames_total += 1

                if frame_dump is not None:
                    fr_no = int(fr) + int(self.frame_index_offset)  # 0->1 => 00001
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

        if getattr(trainer, "mlflow_logger", None) is not None:
            trainer.mlflow_logger.log_metrics(
                {f"{self.log_prefix}/{k}": float(v) for k, v in out.items()},
                step=epoch,
            )


@CALLBACKS.register("multimodal_framewise_callback")
def multimodal_framewise_callback(**params):
    return MultiModalFrameWiseCallback(**params)