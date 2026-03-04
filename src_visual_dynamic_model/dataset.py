import math
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
                    "video": video,
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


class FeatureResolver:
    def __init__(self, features_root: Path, split: str):
        self.features_root = Path(features_root)
        self.split = str(split)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[int, int]]] = {}

    def _load_video(self, video: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        video = str(video)
        if video in self._cache:
            return self._cache[video]
        path = self.features_root / self.split / f"{video}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Feature file not found: {path}")
        data = np.load(path)
        frames = data["frames"].astype(np.int32)
        feats = data["features"].astype(np.float32)
        frame_to_idx = {int(f): i for i, f in enumerate(frames.tolist())}
        self._cache[video] = (frames, feats, frame_to_idx)
        return self._cache[video]

    @staticmethod
    def _nearest_frame(cached_frames: np.ndarray, frame_idx: int) -> Optional[int]:
        if cached_frames.size == 0:
            return None
        pos = int(np.searchsorted(cached_frames, int(frame_idx), side="left"))
        if pos == 0:
            return int(cached_frames[0])
        if pos >= len(cached_frames):
            return int(cached_frames[-1])
        prev_n = int(cached_frames[pos - 1])
        next_n = int(cached_frames[pos])
        if abs(frame_idx - prev_n) <= abs(next_n - frame_idx):
            return prev_n
        return next_n

    def resolve_feature(self, video: str, frame_idx: int, allow_nearest: bool) -> Tuple[np.ndarray, bool]:
        frames, feats, frame_to_idx = self._load_video(video)
        req = int(frame_idx)
        if req in frame_to_idx:
            return feats[frame_to_idx[req]], False
        if allow_nearest:
            near = self._nearest_frame(frames, req)
            if near is not None:
                return feats[frame_to_idx[int(near)]], False
        return None, True

    def infer_feature_dim(self, split: str) -> int:
        split_dir = self.features_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Feature split directory not found: {split_dir}")
        npz_files = sorted(split_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No feature files in {split_dir}")
        sample = np.load(npz_files[0])
        return int(sample["features"].shape[1])


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


class VisualDynamicDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_root: Path,
        split: str,
        window_cfg: WindowConfig,
        allow_nearest: bool,
    ):
        self.window_cfg = window_cfg
        self.allow_nearest = bool(allow_nearest)
        self.resolver = FeatureResolver(features_root=features_root, split=split)
        arranger = VideoWindowArranger(df, window_cfg)
        self.video_data = arranger.video_data
        self.windows = arranger.windows
        self.feature_dim = self.resolver.infer_feature_dim(split)

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

        feat_list = []
        missing_flags = []
        for fidx in frame_numbers:
            feat, missing = self.resolver.resolve_feature(video, int(fidx), self.allow_nearest)
            if missing or feat is None:
                feat = np.zeros((self.feature_dim,), dtype=np.float32)
                missing = True
            feat_list.append(feat.astype(np.float32))
            missing_flags.append(bool(missing))

        features = torch.tensor(np.stack(feat_list, axis=0), dtype=torch.float32)
        labels_tensor = torch.tensor(label_seq, dtype=torch.float32)
        valid_mask = torch.tensor(valid_seq, dtype=torch.bool)
        missing_mask = torch.tensor(missing_flags, dtype=torch.bool)

        actual_len = min(length, wlen)
        if actual_len <= 0:
            pos_indices = torch.zeros((wlen,), dtype=torch.long)
        else:
            pos_indices = torch.arange(start, start + actual_len, dtype=torch.long)
            if len(pos_indices) < wlen:
                pad = torch.full((wlen - len(pos_indices),), int(pos_indices[-1]), dtype=torch.long)
                pos_indices = torch.cat([pos_indices, pad], dim=0)

        return {
            "features": features,
            "labels": labels_tensor,
            "valid_mask": valid_mask,
            "missing_mask": missing_mask,
            "video": video,
            "indices": pos_indices,
            "length": length,
        }


def collate_visual_dynamic(batch: List[dict]) -> dict:
    return {
        "features": torch.stack([b["features"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "missing_mask": torch.stack([b["missing_mask"] for b in batch], dim=0),
        "indices": torch.stack([b["indices"] for b in batch], dim=0),
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "videos": [b["video"] for b in batch],
    }

