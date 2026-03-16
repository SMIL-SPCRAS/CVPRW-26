from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple, Any
import pickle
import bisect

import pandas as pd
import torch
import torch.nn.functional as F

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS, METRICS

from utils import TensorMetricAdapter


@dataclass
class FrameWiseEvalCallback(BaseCallback):
    """
    Frame-wise validation for an AUDIO window model using per-frame GT files.

    Pipeline:
      1) Inference on validation loader -> collect window preds + meta.
      2) Expand each window pred to frames [start_frame, end_frame).
      3) Resolve overlaps (mean | center_weighted | last) for PREDICTIONS only.
      4) Load per-frame GT from files (Validation_set/<video_id>.txt).
      5) Match by frame index; skip unannotated frames (-5,-5 by default).
      6) Compute metrics via registry and log.

    Meta requirements (per window):
      - video name in one of: full_video_name / video_name / filename
      - either start_frame/end_frame OR start_time/end_time + fps
    """

    loader_name: str = "val_full"
    log_prefix: str = "val_frame"

    metric_name: str = "va_ccc_metric"
    metric_params: Optional[dict] = None
    eps: float = 1e-8

    overlap_strategy: str = "center_weighted"  # mean | center_weighted | last
    resample_mode: str = "linear"   # "nearest"

    # GT file settings
    ann_root: str = "."
    ann_split_dir: str = "Validation_Set"
    ann_ext: str = ".txt"

    # missing marker in GT
    missing_value: float = -5.0

    # meta keys to find video id
    video_id_keys: List[str] = field(default_factory=lambda: ["segment_name"])
    save_pred_csv: bool = False
    pred_csv_dir: str = "./val_frame_dumps"

    # features
    dump_pickle: bool = False
    pickle_path: str = "./val_frame_dump.pkl"
    pickle_numpy: bool = True
    frame_index_offset: int = 0
    
    def __post_init__(self) -> None:
        if self.overlap_strategy not in ("keep_all", "keep_first", "keep_last"):
            raise ValueError("overlap_strategy must be one of: keep_all|keep_first|keep_last")
        
        if self.resample_mode not in ("linear", "nearest"):
            raise ValueError("resample_mode must be one of: linear|nearest")

        params = dict(self.metric_params or {})
        params.setdefault("eps", self.eps)
        metric = METRICS.create(self.metric_name, **params)
        self._metric = TensorMetricAdapter(metric)

        # cache: vid -> (gt[F,2], valid_mask[F])
        self._gt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ---------------- helpers ----------------
    def _tri_weight(self, fr: int, start: int, end: int) -> float:
        length = max(end - start, 1)
        center = start + (length - 1) / 2.0
        half = max(length / 2.0, 1.0)
        w = 1.0 - abs(fr - center) / half
        return float(max(w, 0.0))

    def _normalize_video_id(self, raw: str) -> str:
        """
        Examples:
          "479.mov" -> "479"
          "/path/to/479.avi" -> "479"
          "479___123_223_0007.mov" -> "479"
        """
        stem = Path(str(raw)).stem
        stem = stem.split("___", 1)[0]
        return stem

    def _get_video_id(self, meta: dict) -> str:
        for k in self.video_id_keys:
            v = meta.get(k, None)
            if v:
                return self._normalize_video_id(str(v))
        return ""

    def _load_gt(self, video_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          gt: [F,2] float32 CPU
          valid: [F] bool
        """
        if video_id in self._gt_cache:
            return self._gt_cache[video_id]

        path = Path(self.ann_root) / self.ann_split_dir / f"{video_id}{self.ann_ext}"
        if not path.exists():
            raise FileNotFoundError(f"Per-frame GT file not found: {path}")

        df = pd.read_csv(path)  # columns: valence, arousal
        gt = torch.tensor(df[["valence", "arousal"]].to_numpy(), dtype=torch.float32)  # [F,2] CPU

        # mark unannotated frames as invalid (so frame indices still match)
        valid = (gt[:, 0] != self.missing_value) & (gt[:, 1] != self.missing_value)
        valid = valid & torch.isfinite(gt).all(dim=1)

        self._gt_cache[video_id] = (gt, valid)
        return gt, valid

    def _get_window_frames(self, meta: dict) -> Tuple[int, int]:
        """
        Prefer start_frame/end_frame; else compute from start_time/end_time and fps using round()
        (matches your chunking script behavior).
        """
        start_frame = int(meta.get("start_frame", 0) or 0)
        end_frame = int(meta.get("end_frame", 0) or 0)
        if end_frame > start_frame:
            return start_frame, end_frame

        st = meta.get("start_time", None)
        et = meta.get("end_time", None)
        fps = meta.get("fps", None)
        if st is None or et is None or fps is None:
            return start_frame, end_frame

        try:
            st_f = float(st)
            et_f = float(et)
            fps_f = float(fps)
            if et_f > st_f and fps_f > 0:
                start_frame = int(round(st_f * fps_f))
                end_frame = int(round(et_f * fps_f))
        except Exception:
            pass

        return start_frame, end_frame
    
    def _to_time_major_emb(self, e: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(e):
            e = torch.as_tensor(e)

        # squeeze [1,T,D] -> [T,D]
        while e.ndim > 2 and e.shape[0] == 1:
            e = e.squeeze(0)

        if e.ndim == 1:
            return e.view(1, -1)  # [1,D]

        if e.ndim != 2:
            raise ValueError(f"Expected 1D/2D embedding per sample, got {tuple(e.shape)}")

        if e.shape[0] > e.shape[1]:
            return e.transpose(0, 1).contiguous()  # [T,D]
        return e  # [T,D]

    def _resample_seq_to_len_nearest(self, seq_td: torch.Tensor, L: int) -> torch.Tensor:
        """Resample [T,D] -> [L,D] (nearest)."""
        T = int(seq_td.shape[0])
        D = int(seq_td.shape[1])
        if L <= 0:
            return seq_td[:0]
        if T == 0:
            return seq_td.new_zeros((L, D))
        if T == L:
            return seq_td
        if L == 1:
            return seq_td[0:1]

        idx = torch.linspace(0, T - 1, steps=L)
        idx = torch.round(idx).to(dtype=torch.long).clamp(0, T - 1)
        return seq_td[idx]
    
    def _resample_seq_to_len_linear(self, seq_td: torch.Tensor, L: int) -> torch.Tensor:
        """Linear resample [T,D] -> [L,D] using interpolate()."""
        T = int(seq_td.shape[0])
        D = int(seq_td.shape[1])

        if L <= 0:
            return seq_td[:0]
        if T == 0:
            return seq_td.new_zeros((L, D))
        if T == L:
            return seq_td
        if T == 1:
            return seq_td.expand(L, D).clone()

        x = seq_td.transpose(0, 1).unsqueeze(0)          # [1, D, T]
        y = F.interpolate(x, size=L, mode="linear", align_corners=False)  # [1, D, L]
        return y.squeeze(0).transpose(0, 1).contiguous() # [L, D]

    # ---- MINIMAL ADD: helpers for seq2seq support ----
    def _to_time_major_va(self, p: torch.Tensor) -> torch.Tensor:
        """Normalize a single-sample prediction to shape [T, 2] (time-major).

        Supported inputs:
          - [2]   -> [1,2]
          - [T,2] -> as-is
          - [2,T] -> transpose to [T,2]
          - [1,T,2] -> squeeze to [T,2]
        """
        if not torch.is_tensor(p):
            p = torch.as_tensor(p)

        # remove trivial leading dims (e.g. [1,T,2])
        while p.ndim > 2 and p.shape[0] == 1:
            p = p.squeeze(0)

        if p.ndim == 1:
            if p.numel() != 2:
                raise ValueError(f"Expected VA vector with 2 elems, got shape {tuple(p.shape)}")
            return p.view(1, 2)

        if p.ndim != 2:
            raise ValueError(f"Expected 1D or 2D preds per sample, got shape {tuple(p.shape)}")

        # [T,2]
        if p.shape[-1] == 2:
            return p
        # [2,T]
        if p.shape[0] == 2:
            return p.transpose(0, 1)

        raise ValueError(f"Cannot interpret preds shape as VA sequence: {tuple(p.shape)}")

    # -----------------------------------------------

    def _save_video_csv(
        self,
        vid: str,
        epoch: int,
        vid_fp: List[torch.Tensor],
        vid_ft: List[torch.Tensor],
    ) -> None:
        """
        Save per-video CSV with 4 columns:
        target_valence, target_arousal, pred_valence, pred_arousal

        video id is encoded in filename.
        """
        if not getattr(self, "save_pred_csv", False):
            return
        if not vid_fp:
            return

        out_dir = Path(f"{self.pred_csv_dir}_epoch{epoch}")
        out_dir.mkdir(parents=True, exist_ok=True)

        vfp = torch.cat(vid_fp, dim=0)  # [K,2]
        vft = torch.cat(vid_ft, dim=0)  # [K,2]

        df = pd.DataFrame({
            "target_valence": vft[:, 0].cpu().numpy(),
            "target_arousal": vft[:, 1].cpu().numpy(),
            "pred_valence":   vfp[:, 0].cpu().numpy(),
            "pred_arousal":   vfp[:, 1].cpu().numpy(),
        })

        csv_path = out_dir / f"{vid}.csv"
        df.to_csv(csv_path, index=False)

    # ---------------- main hook ----------------
    @torch.no_grad()
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]) -> None:
        cache = getattr(trainer, "predictions_cache", None)
        cached = cache.get(self.loader_name)
        if cached is None:
            return
        
        preds = getattr(cached, "preds", None)
        meta_all = getattr(cached, "sample_meta", None)
        feats = getattr(cached, "features", None)

        if preds is None or meta_all is None:
            return
        
        if len(meta_all) == 0:
            return
        
        # preds: [N,2] or [N,T,2] / [N,2,T]
        preds = preds.detach().cpu()
        if preds.ndim == 3 and preds.shape[-1] != 2 and preds.shape[1] == 2:
            preds = preds.transpose(1, 2).contiguous()  # [N,T,2]

        # feats: [N,D] or [N,T,D] or [N,D,T]
        if feats is not None:
            feats = feats.detach().cpu()
            if feats.ndim == 3 and feats.shape[1] > feats.shape[2]:
                # [N,D,T] -> [N,T,D]
                feats = feats.transpose(1, 2).contiguous()

        if len(meta_all) != preds.shape[0]:
            n = min(len(meta_all), preds.shape[0])
            meta_all = meta_all[:n]
            preds = preds[:n]
            if feats is not None:
                feats = feats[:n]

        # ---- overlap-aware per-frame PRED aggregation (single container) ----
        per_video: DefaultDict[str, Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor], float]]] = defaultdict(dict)

        for i, meta in enumerate(meta_all):
            if not isinstance(meta, dict) or not meta:
                continue

            vid = self._get_video_id(meta)
            if not vid:
                continue

            start_frame, end_frame = self._get_window_frames(meta)
            if end_frame <= start_frame:
                continue

            # preds -> [T,2]
            try:
                p_t2 = self._to_time_major_va(preds[i])
            except Exception as e:
                try:
                    trainer.logger.warning(f"[{self.log_prefix}] bad preds shape for {vid}: {e}")
                except Exception:
                    print(f"[{self.log_prefix}] bad preds shape for {vid}: {e}")
                continue

            L = int(end_frame - start_frame)

            if self.resample_mode == "linear": # [L,2]
                p_t2 = self._resample_seq_to_len_linear(p_t2, L)
            else:
                p_t2 = self._resample_seq_to_len_nearest(p_t2, L)

            # feats -> [L,D] or None
            e_tD = None
            if feats is not None:
                e_i = feats[i]
                e_td = self._to_time_major_emb(e_i)  # [T,D] or [1,D]
                if e_td.shape[0] == 1 and L > 1:
                    e_tD = e_td.expand(L, e_td.shape[1]).clone()
                else:
                    # [L,D]
                    if self.resample_mode == "linear":
                        e_tD = self._resample_seq_to_len_linear(e_td, L)
                    else:
                        e_tD = self._resample_seq_to_len_nearest(e_td, L)

            m = per_video[vid]

            for j, fr in enumerate(range(start_frame, end_frame)):
                p = p_t2[j].view(-1)  # [2]
                e = e_tD[j].view(-1) if e_tD is not None else None  # [D] or None

                if self.overlap_strategy == "last":
                    m[fr] = (p.clone(), (e.clone() if e is not None else None), 1.0)
                    continue

                # mean / center_weighted
                w = 1.0
                if self.overlap_strategy == "center_weighted":
                    w = self._tri_weight(fr, start_frame, end_frame)
                    if w <= 0.0:
                        continue

                if fr in m:
                    sp, se, sw = m[fr]
                    sp = sp + p * w
                    if e is not None:
                        se = (se + e * w) if se is not None else (e.clone() * w)

                    m[fr] = (sp, se, sw + float(w))
                else:
                    sp = p.clone() * w
                    se = (e.clone() * w) if e is not None else None
                    m[fr] = (sp, se, float(w))

        vids = sorted(per_video.keys())
        if not vids:
            return

        # ---- match with per-frame GT (intersection only) ----
        fp_list: List[torch.Tensor] = []
        ft_list: List[torch.Tensor] = []

        frame_dump: Optional[Dict[str, Dict[str, Any]]] = {} if self.dump_pickle else None

        matched_frames_total = 0
        annotated_frames_total = 0
        oob_frames_total = 0

        for vid in vids:
            try:
                gt, valid = self._load_gt(vid)  # gt [F,2], valid [F]
            except Exception as e:
                # if GT missing/broken, just skip this video
                try:
                    trainer.logger.warning(f"[{self.log_prefix}] skip {vid}: {e}")
                except Exception:
                    print(f"[{self.log_prefix}] skip {vid}: {e}")
                continue

            F = int(gt.shape[0])
            annotated_frames_total += int(valid.sum().item())

            frames = per_video[vid]

            # --- FIX: pad missing last annotated frame (off-by-one tail) ---
            if valid.any():
                last_gt = int(torch.where(valid)[0][-1].item())
                if last_gt not in frames:
                    prev = max((f for f in frames.keys() if f < last_gt), default=None)
                    if prev is not None:
                        frames[last_gt] = frames[prev]

            valid_idx = torch.where(valid)[0].tolist()
            if not valid_idx:
                continue

            pred_keys = sorted(frames.keys())
            if not pred_keys:
                continue

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

                pred_set.add(fr)
                pred_keys.insert(pos, fr)

            # -------------------------------------------------------------

            vid_fp: List[torch.Tensor] = []
            vid_ft: List[torch.Tensor] = []

            for fr in sorted(frames.keys()):
                if fr < 0 or fr >= F:
                    oob_frames_total += 1
                    continue
                if not bool(valid[fr]):
                    continue

                sp, se, sw = frames[fr]
                if sw <= 0:
                    continue

                p = (sp / float(sw)).view(1, 2)
                t = gt[fr].view(1, 2)

                fp_list.append(p)
                ft_list.append(t)

                vid_fp.append(p)
                vid_ft.append(t)

                matched_frames_total += 1

                # dump entry
                if frame_dump is not None:
                    fr_idx = int(fr) + int(self.frame_index_offset)
                    key = f"{vid}/{fr_idx + 1:05d}.jpg"

                    emb = None
                    if se is not None:
                        emb = (se / float(sw)).view(-1)

                    if self.pickle_numpy:
                        frame_dump[key] = {
                            "embedding": None if emb is None else emb.cpu().numpy(),
                            "prediction": p.view(-1).cpu().numpy(),
                            "label": t.view(-1).cpu().numpy(),
                        }
                    else:
                        frame_dump[key] = {
                            "embedding": emb,
                            "prediction": p.view(-1),
                            "label": t.view(-1),
                        }

            self._save_video_csv(vid=vid, epoch=epoch, vid_fp=vid_fp, vid_ft=vid_ft)

        if not fp_list:
            return

        fp = torch.cat(fp_list, dim=0)  # [M,2]
        ft = torch.cat(ft_list, dim=0)  # [M,2]

        # ---- metrics ----
        self._metric.reset()
        self._metric.update(fp, ft)
        out = self._metric.compute()

        for k, v in out.items():
            logs[f"{self.log_prefix}/{k}"] = float(v)

        # diagnostics (handy to detect mismatch/off-by-one)
        coverage = (matched_frames_total / annotated_frames_total) if annotated_frames_total > 0 else 0.0

        # --- metrics message ---
        msg_metrics = " ".join([f"{self.log_prefix}/{k}={float(v):.4f}" for k, v in out.items()])

        # --- stats message (separate log line) ---
        msg_stats = (
            f"{self.log_prefix}/matched_frames={matched_frames_total} "
            f"{self.log_prefix}/annotated_frames={annotated_frames_total} "
            f"{self.log_prefix}/coverage={coverage:.3f} "
            f"{self.log_prefix}/oob_pred_frames={oob_frames_total}"
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


@CALLBACKS.register("framewise_eval_callback")
def framewise_eval_callback(**params):
    return FrameWiseEvalCallback(**params)
