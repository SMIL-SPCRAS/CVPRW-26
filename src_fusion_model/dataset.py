import bisect
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class IndexConfig:
    annotations_root: Path
    split: str  # train | val
    cache_path: Optional[Path] = None
    filter_invalid: bool = False


@dataclass
class WindowConfig:
    window_length: int
    hop_length: int
    time_delay: int = 0


def _safe_float(value) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def _parse_annotation_txt(path: Path, filter_invalid: bool) -> List[Tuple[int, float, float]]:
    rows: List[Tuple[int, float, float]] = []
    frame_idx = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if frame_idx == 0 and line.lower().startswith("valence"):
                continue
            frame_idx += 1
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                valence = float("nan")
                arousal = float("nan")
            else:
                valence = _safe_float(parts[0])
                arousal = _safe_float(parts[1])
                if valence is None:
                    valence = float("nan")
                if arousal is None:
                    arousal = float("nan")
            if filter_invalid and (valence == -5 or arousal == -5):
                continue
            rows.append((frame_idx, valence, arousal))
    return rows


def build_index(cfg: IndexConfig) -> pd.DataFrame:
    if cfg.cache_path is not None and cfg.cache_path.is_file():
        df = pd.read_csv(cfg.cache_path, low_memory=False)
        df["video"] = df["video"].astype(str)
        df["image_location"] = df["image_location"].astype(str)
        return df

    split = cfg.split.lower()
    if split == "train":
        split_dir = "Train_Set"
    elif split in ("val", "valid", "validation"):
        split_dir = "Validation_Set"
    else:
        raise ValueError(f"Unsupported split for training: {cfg.split}")

    ann_dir = cfg.annotations_root / split_dir
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotations folder not found: {ann_dir}")

    rows = []
    for txt_path in sorted(ann_dir.glob("*.txt")):
        video = txt_path.stem
        parsed = _parse_annotation_txt(txt_path, filter_invalid=cfg.filter_invalid)
        for frame_idx, valence, arousal in parsed:
            rows.append(
                {
                    "video": str(video),
                    "frame": int(frame_idx),
                    "valence": float(valence),
                    "arousal": float(arousal),
                    "image_location": f"{video}/{frame_idx:05d}.jpg",
                }
            )
    df = pd.DataFrame(rows)
    if cfg.cache_path is not None:
        cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.cache_path, index=False)
    return df


def _load_pickle_compat(path: Path):
    # Compatibility for pickles created with numpy 2.x (`numpy._core`) loaded in numpy 1.x env.
    if "numpy._core" not in sys.modules:
        import numpy as _np

        sys.modules["numpy._core"] = _np.core
        sys.modules["numpy._core.multiarray"] = _np.core.multiarray
    with path.open("rb") as f:
        return pickle.load(f)


class PKLFeatureResolver:
    _CACHE: Dict[str, tuple] = {}

    def __init__(
        self,
        features_root: Path,
        modality: str,
        split: str,
        input_mode: str = "both",  # embedding | prediction | both
    ):
        self.features_root = Path(features_root)
        self.modality = str(modality)
        self.split = str(split)
        self.input_mode = str(input_mode).lower()
        if self.input_mode not in {"embedding", "prediction", "both"}:
            raise ValueError(f"Unknown input_mode={input_mode}. Use embedding | prediction | both")

        self.pkl_path = self.features_root / self.modality / f"{self.split}.pkl"
        if not self.pkl_path.is_file():
            raise FileNotFoundError(f"Modality pkl not found: {self.pkl_path}")

        cache_key = str(self.pkl_path.resolve())
        if cache_key not in self._CACHE:
            records = _load_pickle_compat(self.pkl_path)
            if not isinstance(records, dict):
                raise ValueError(f"Expected dict in {self.pkl_path}, got {type(records)}")
            frame_to_key: Dict[str, Dict[int, str]] = {}
            frame_lists: Dict[str, np.ndarray] = {}
            for key in records.keys():
                parts = str(key).replace("\\", "/").split("/")
                if len(parts) < 2:
                    continue
                video = str(parts[0])
                try:
                    frame_idx = int(Path(parts[-1]).stem)
                except ValueError:
                    continue
                frame_to_key.setdefault(video, {})[frame_idx] = str(key)
            for video, m in frame_to_key.items():
                frame_lists[video] = np.array(sorted(m.keys()), dtype=np.int32)

            sample_val = next(iter(records.values()))
            emb_dim = int(np.asarray(sample_val["embedding"]).shape[0]) if "embedding" in sample_val else 0
            pred_dim = int(np.asarray(sample_val["prediction"]).shape[0]) if "prediction" in sample_val else 0
            if self.input_mode == "embedding":
                feat_dim = emb_dim
            elif self.input_mode == "prediction":
                feat_dim = pred_dim
            else:
                feat_dim = emb_dim + pred_dim
            if feat_dim <= 0:
                raise ValueError(
                    f"Invalid feature dim for modality={self.modality}, mode={self.input_mode}: emb={emb_dim}, pred={pred_dim}"
                )

            self._CACHE[cache_key] = (records, frame_to_key, frame_lists, feat_dim)

        self.records, self.frame_to_key, self.frame_lists, self.feature_dim = self._CACHE[cache_key]

    def _build_vec(self, rec: dict) -> np.ndarray:
        emb = np.asarray(rec.get("embedding", np.array([], dtype=np.float32)), dtype=np.float32).reshape(-1)
        pred = np.asarray(rec.get("prediction", np.array([], dtype=np.float32)), dtype=np.float32).reshape(-1)
        if self.input_mode == "embedding":
            return emb
        if self.input_mode == "prediction":
            return pred
        return np.concatenate([emb, pred], axis=0)

    @staticmethod
    def _nearest(frames: np.ndarray, target: int) -> Optional[int]:
        if frames.size == 0:
            return None
        pos = int(np.searchsorted(frames, int(target), side="left"))
        if pos == 0:
            return int(frames[0])
        if pos >= len(frames):
            return int(frames[-1])
        prev_n = int(frames[pos - 1])
        next_n = int(frames[pos])
        if abs(target - prev_n) <= abs(next_n - target):
            return prev_n
        return next_n

    def resolve_feature(self, video: str, frame_idx: int, allow_nearest: bool) -> Tuple[Optional[np.ndarray], bool]:
        video = str(video)
        fmap = self.frame_to_key.get(video)
        if fmap is None:
            return None, True
        req = int(frame_idx)
        key = fmap.get(req)
        if key is None and allow_nearest:
            near = self._nearest(self.frame_lists.get(video, np.array([], dtype=np.int32)), req)
            if near is not None:
                key = fmap.get(int(near))
        if key is None:
            return None, True
        rec = self.records.get(key)
        if rec is None:
            return None, True
        vec = self._build_vec(rec).astype(np.float32)
        if vec.size == 0:
            return None, True
        return vec, False


class VideoWindowArranger:
    def __init__(self, df: pd.DataFrame, window_cfg: WindowConfig):
        self.df = df
        self.window_cfg = window_cfg
        self.video_data = self._group_by_video(df)
        self.windows = self._build_windows()

    def _group_by_video(self, df: pd.DataFrame) -> Dict[str, dict]:
        data = {}
        for video, group in df.groupby("video"):
            g = group.sort_values("frame")
            frames = g["frame"].astype(int).to_numpy()
            valence = g["valence"].to_numpy(dtype=np.float32)
            arousal = g["arousal"].to_numpy(dtype=np.float32)
            labels = np.stack([valence, arousal], axis=1)
            valid = (~np.isnan(labels)).all(axis=1) & (labels != -5).all(axis=1)

            if self.window_cfg.time_delay and self.window_cfg.time_delay > 0:
                td = int(self.window_cfg.time_delay)
                if len(labels) > 0:
                    labels = np.concatenate([labels[td:], np.repeat(labels[-1][None, :], repeats=td, axis=0)], axis=0)
                    valid = np.concatenate([valid[td:], np.repeat(valid[-1:], repeats=td, axis=0)], axis=0)

            data[str(video)] = {"frames": frames, "labels": labels, "valid": valid}
        return data

    def _build_windows(self) -> List[dict]:
        windows: List[dict] = []
        wlen = int(self.window_cfg.window_length)
        hop = int(self.window_cfg.hop_length)
        for video, info in self.video_data.items():
            length = len(info["frames"])
            if length == 0:
                continue
            if length <= wlen:
                windows.append({"video": video, "start": 0, "length": length})
                continue
            start = 0
            while start + wlen <= length:
                windows.append({"video": video, "start": start, "length": wlen})
                start += hop
            last_start = length - wlen
            if windows[-1]["video"] != video or windows[-1]["start"] != last_start:
                windows.append({"video": video, "start": last_start, "length": wlen})
        return windows


class FusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_root: Path,
        modalities: List[str],
        split: str,
        window_cfg: WindowConfig,
        allow_nearest: bool,
        modality_input_mode: str = "both",
    ):
        self.window_cfg = window_cfg
        self.allow_nearest = bool(allow_nearest)
        self.split = str(split)
        self.modalities = [str(m) for m in modalities]
        if len(self.modalities) == 0:
            raise ValueError("At least one modality must be provided")
        self.resolvers = {
            m: PKLFeatureResolver(features_root=features_root, modality=m, split=split, input_mode=modality_input_mode)
            for m in self.modalities
        }
        self.feature_dims = {m: int(self.resolvers[m].feature_dim) for m in self.modalities}

        arranger = VideoWindowArranger(df, window_cfg)
        self.video_data = arranger.video_data
        self.windows = arranger.windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        item = self.windows[idx]
        video = item["video"]
        start = int(item["start"])
        length = int(item["length"])
        frames = self.video_data[video]["frames"]
        labels = self.video_data[video]["labels"]
        valid = self.video_data[video]["valid"]

        end = min(start + length, len(frames))
        frame_numbers = frames[start:end].tolist()
        label_seq = labels[start:end]
        valid_seq = valid[start:end]

        wlen = int(self.window_cfg.window_length)
        pad_len = wlen - len(frame_numbers)
        if pad_len > 0:
            if len(frame_numbers) == 0:
                frame_numbers = [1] * wlen
                label_seq = np.zeros((wlen, 2), dtype=np.float32)
                valid_seq = np.zeros((wlen,), dtype=bool)
            else:
                frame_numbers = frame_numbers + [frame_numbers[-1]] * pad_len
                pad_labels = np.repeat(label_seq[-1][None, :], repeats=pad_len, axis=0)
                label_seq = np.concatenate([label_seq, pad_labels], axis=0)
                valid_seq = np.concatenate([valid_seq, np.zeros((pad_len,), dtype=bool)], axis=0)

        modality_features = {}
        per_mod_missing = {}
        for modality in self.modalities:
            resolver = self.resolvers[modality]
            dim = int(self.feature_dims[modality])
            feats = []
            miss = []
            for fidx in frame_numbers:
                vec, missing = resolver.resolve_feature(video, int(fidx), self.allow_nearest)
                if missing or vec is None:
                    vec = np.zeros((dim,), dtype=np.float32)
                    missing = True
                feats.append(vec.astype(np.float32))
                miss.append(bool(missing))
            modality_features[modality] = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)
            per_mod_missing[modality] = torch.tensor(miss, dtype=torch.bool)

        missing_any = torch.zeros((wlen,), dtype=torch.bool)
        for modality in self.modalities:
            missing_any = missing_any | per_mod_missing[modality]

        labels_tensor = torch.tensor(label_seq, dtype=torch.float32)
        valid_mask = torch.tensor(valid_seq, dtype=torch.bool)

        actual_len = min(length, wlen)
        if actual_len <= 0:
            pos_indices = torch.zeros((wlen,), dtype=torch.long)
        else:
            pos_indices = torch.arange(start, start + actual_len, dtype=torch.long)
            if len(pos_indices) < wlen:
                pad = torch.full((wlen - len(pos_indices),), int(pos_indices[-1]), dtype=torch.long)
                pos_indices = torch.cat([pos_indices, pad], dim=0)

        return {
            "modality_features": modality_features,
            "labels": labels_tensor,
            "valid_mask": valid_mask,
            "missing_any_mask": missing_any,
            "indices": pos_indices,
            "length": length,
            "video": video,
        }


def collate_fusion(batch: List[dict]) -> dict:
    modalities = list(batch[0]["modality_features"].keys())
    modality_features = {
        m: torch.stack([b["modality_features"][m] for b in batch], dim=0) for m in modalities
    }
    return {
        "modality_features": modality_features,
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "missing_any_mask": torch.stack([b["missing_any_mask"] for b in batch], dim=0),
        "indices": torch.stack([b["indices"] for b in batch], dim=0),
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "videos": [b["video"] for b in batch],
        "modalities": modalities,
    }

