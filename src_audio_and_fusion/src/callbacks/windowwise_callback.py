from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class WindowWiseCallback(BaseCallback):
    loader_name: str = "val_full"

    # overlap handling: keep_all | keep_first | keep_last
    # keep_first: overlap "belongs" to earlier window
    # keep_last : overlap "belongs" to later window
    overlap_strategy: str = "keep_first"

    # GT settings for per-frame targets_frames
    ann_root: str = "."
    ann_split_dir: str = "Validation_Set"
    ann_ext: str = ".txt"
    missing_value: float = -5.0

    # meta -> video id
    video_id_keys: List[str] = field(default_factory=lambda: ["segment_name"])

    dump_pickle: bool = False
    pickle_path: str = "./val_frame_dump.pkl"
    pickle_numpy: bool = True

    def __post_init__(self) -> None:
        if self.overlap_strategy not in ("keep_all", "keep_first", "keep_last"):
            raise ValueError("overlap_strategy must be one of: keep_all|keep_first|keep_last")
        
        self._gt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ---------------- meta helpers ----------------
    def _normalize_video_id(self, raw: str) -> str:
        stem = Path(str(raw)).stem
        stem = stem.split("___", 1)[0]
        return stem

    def _get_video_id(self, meta: dict) -> str:
        for k in self.video_id_keys:
            v = meta.get(k, None)
            if v:
                return self._normalize_video_id(str(v))
        return ""

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

    # ---------------- GT helpers ----------------
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

    # ---------------- overlap filter ----------------
    def _assign_use_ranges(self, wins: List[dict]) -> List[dict]:
        """
        wins sorted by (start_frame, end_frame).
        Adds use_start_frame/use_end_frame (non-overlapping assignment) without dropping windows.
        Windows whose usable interval becomes empty are kept (but will get empty use_mask and can be skipped later).
        """
        if not wins:
            return wins

        if self.overlap_strategy == "keep_all":
            for w in wins:
                w["use_start_frame"] = w["start_frame"]
                w["use_end_frame"] = w["end_frame"]
            return wins

        if self.overlap_strategy == "keep_first":
            last_end = -10**18
            for w in wins:
                us = max(int(w["start_frame"]), int(last_end))
                ue = int(w["end_frame"])
                w["use_start_frame"] = us
                w["use_end_frame"] = ue
                last_end = max(int(last_end), int(w["end_frame"]))
            return wins

        # keep_last
        for i, w in enumerate(wins):
            s = int(w["start_frame"])
            e = int(w["end_frame"])
            if i == len(wins) - 1:
                w["use_start_frame"] = s
                w["use_end_frame"] = e
            else:
                nxt_s = int(wins[i + 1]["start_frame"])
                # overlap => earlier window loses overlap to the next window
                w["use_start_frame"] = s
                w["use_end_frame"] = min(e, nxt_s) if nxt_s < e else e
        return wins

    # ---------------- main hook ----------------
    @torch.no_grad()
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]) -> None:
        cache = getattr(trainer, "predictions_cache", None)
        cached = cache.get(self.loader_name)
        if cached is None:
            return

        preds = getattr(cached, "preds", None)
        targets = getattr(cached, "targets", None)
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

        targets = targets.detach().cpu()
        if targets.ndim == 3 and targets.shape[-1] != 2 and targets.shape[1] == 2:
            targets = targets.transpose(1, 2).contiguous()  # [N,T,2]

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
            targets = targets[:n]
            if feats is not None:
                feats = feats[:n]

        wins_by_vid = defaultdict(list)
        for idx, meta in enumerate(meta_all):
            if not isinstance(meta, dict) or not meta:
                continue

            vid = self._get_video_id(meta)
            if not vid:
                continue

            start_frame, end_frame = self._get_window_frames(meta)
            if end_frame <= start_frame:
                continue

            wins_by_vid[vid].append({
                "idx": idx,
                "meta": meta,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            })

        if not wins_by_vid:
            return

        # group windows by video
        dump: Dict[str, Dict[str, Any]] = {}
        for vid, wins in wins_by_vid.items():
            wins.sort(key=lambda x: (x["start_frame"], x["end_frame"]))
            wins = self._assign_use_ranges(wins)

            gt, vmask = self._load_gt(vid)
            F = int(gt.shape[0])
            for w in wins:
                idx = w["idx"]
                s_f = w["start_frame"]
                e_f = w["end_frame"]
                us_f = int(w["use_start_frame"])
                ue_f = int(w["use_end_frame"])

                # per-frame GT slice for this window (clamped)
                if gt is not None and vmask is not None and F is not None:
                    ss = max(0, min(int(s_f), F))
                    ee = max(0, min(int(e_f), F))
                    if ee <= ss:
                        continue

                    target_frames = gt[ss:ee]      # [L,2]
                    valid_frames = vmask[ss:ee] # [L]

                    # clamp use-range into [ss, ee)
                    us = max(ss, min(us_f, ee))
                    ue = max(ss, min(ue_f, ee))
                    base_start = ss
                else:
                    L = max(int(e_f - s_f), 0)
                    if L <= 0:
                        continue

                    target_frames = torch.full((L, 2), self.missing_value, dtype=torch.float32)
                    valid_frames = torch.zeros((L,), dtype=torch.bool)
                    us = max(s_f, min(us_f, e_f))
                    ue = max(s_f, min(ue_f, e_f))
                    base_start = s_f

                L = int(target_frames.shape[0])
                # ---- build use_mask [L] ----
                use_mask = torch.zeros((L,), dtype=torch.bool)
                a = int(us - base_start)
                b = int(ue - base_start)
                a = max(0, min(a, L))
                b = max(0, min(b, L))
                if b > a:
                    use_mask[a:b] = True

                # skip windows that contribute nothing after masking (optional)
                if not use_mask.any():
                    continue

                pred_i = preds[idx]
                feat_i = feats[idx]
                tgt_i = targets[idx]
                
                key = f"{vid}___{int(s_f):06d}_{int(e_f):06d}__{idx:06d}"

                dump[key] = {
                    "vid": vid,
                    "start_frame": int(s_f),
                    "end_frame": int(e_f),
                    "meta": w["meta"],
                    "use_start_frame": us_f,
                    "use_end_frame": ue_f,
                    "embedding": feat_i.cpu().numpy().astype(np.float32),
                    "prediction": pred_i.cpu().numpy().astype(np.float32),
                    "label_w": tgt_i.cpu().numpy().astype(np.float32),
                    "label_f": target_frames.cpu().numpy().astype(np.float32),
                    "valid_f": valid_frames.cpu().numpy().astype(np.bool_),
                }
                
        if not dump:
            return

        out_path = Path(self.pickle_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            trainer.logger.info(f"[WindowWiseCallback:{self.loader_name}] dumped {len(dump)} windows -> {out_path}")
        except Exception:
            print(f"[WindowWiseCallback:{self.loader_name}] dumped {len(dump)} windows -> {out_path}")


@CALLBACKS.register("windowwise_callback")
def windowwise_callback(**params):
    return WindowWiseCallback(**params)