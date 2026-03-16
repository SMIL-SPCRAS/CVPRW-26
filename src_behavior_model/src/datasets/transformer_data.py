from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .text_data import add_stream_name_column, ensure_columns, read_csv_with_fallback


def maybe_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit and limit > 0:
        return df.head(limit).copy()
    return df


def load_embeddings_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError(f"Invalid embeddings cache format: {path}")
    embeddings = payload["embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"Embeddings tensor must be 2D in cache: {path}")
    return embeddings.float().cpu()


def _resolve_group_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "stream_name" in df.columns:
        return "stream_name"
    if "full_video_name" in df.columns:
        return "full_video_name"
    if "video_name" in df.columns:
        return "video_name"
    raise ValueError("Could not resolve group column: none of preferred/full_video_name/video_name exists.")


def _resolve_window_params(window_size: int, window_stride: int) -> Tuple[int, int]:
    if window_size <= 0:
        return 0, 0
    stride = window_stride if window_stride > 0 else window_size
    return int(window_size), int(max(1, stride))


def _build_window_ranges(seq_len: int, window_size: int, window_stride: int) -> List[Tuple[int, int]]:
    if seq_len <= 0:
        return []
    if window_size <= 0 or window_size >= seq_len:
        return [(0, seq_len)]

    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        ranges.append((start, end))
        if end >= seq_len:
            break
        start += window_stride
    return ranges


def _drop_overlaps_in_group(
    grp: pd.DataFrame,
    start_col: str,
    end_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    if start_col not in grp.columns or end_col not in grp.columns or grp.empty:
        return grp, np.ones(len(grp), dtype=bool), 0

    keep_mask = np.zeros(len(grp), dtype=bool)
    starts = grp[start_col].to_numpy()
    ends = grp[end_col].to_numpy()
    last_kept_end = None
    dropped = 0

    for i, (start_value, end_value) in enumerate(zip(starts, ends)):
        if pd.isna(start_value) or pd.isna(end_value):
            keep_mask[i] = True
            last_kept_end = end_value if last_kept_end is None else last_kept_end
            continue
        if last_kept_end is None or float(start_value) >= float(last_kept_end):
            keep_mask[i] = True
            last_kept_end = float(end_value)
        else:
            dropped += 1

    return grp.loc[keep_mask].copy(), keep_mask, dropped


def build_transformer_sequences(
    df: pd.DataFrame,
    embeddings: torch.Tensor,
    group_col: str,
    val_col: str = "valence",
    aro_col: str = "arousal",
    start_col: str = "start_frame",
    end_col: str = "end_frame",
    window_size: int = 0,
    window_stride: int = 0,
    drop_overlapping_segments: bool = False,
) -> Tuple[List[Dict], str, pd.DataFrame | None]:
    df = df.reset_index(drop=True).copy()
    if len(df) != int(embeddings.shape[0]):
        raise ValueError(
            f"CSV rows ({len(df)}) do not match embeddings rows ({int(embeddings.shape[0])})."
        )
    ensure_columns(df, ["video_name", val_col, aro_col], "transformer_df")
    resolved_group_col = _resolve_group_col(df, group_col)
    resolved_window_size, resolved_window_stride = _resolve_window_params(window_size, window_stride)

    df["_row_idx"] = np.arange(len(df), dtype=np.int64)
    sort_cols = [c for c in [start_col, end_col] if c in df.columns]
    sequences: List[Dict] = []
    total_dropped_overlap = 0
    debug_frames: List[pd.DataFrame] = []

    for group_value, grp in df.groupby(resolved_group_col, sort=False):
        if sort_cols:
            grp = grp.sort_values(sort_cols, kind="mergesort")
        if drop_overlapping_segments:
            grp_kept, keep_mask, dropped_overlap = _drop_overlaps_in_group(
                grp=grp,
                start_col=start_col,
                end_col=end_col,
            )
            total_dropped_overlap += int(dropped_overlap)
            debug_grp = grp.copy()
            debug_grp["is_kept_after_overlap_filter"] = keep_mask.astype(bool)
            debug_frames.append(debug_grp)
            grp = grp_kept

        full_row_idx = grp["_row_idx"].to_numpy(dtype=np.int64)
        full_targets = grp[[val_col, aro_col]].to_numpy(dtype=np.float32)
        window_ranges = _build_window_ranges(
            seq_len=int(len(full_row_idx)),
            window_size=resolved_window_size,
            window_stride=resolved_window_stride,
        )
        for start, end in window_ranges:
            row_idx = full_row_idx[start:end].copy()
            targets = full_targets[start:end].copy()
            sequences.append(
                {
                    "group": str(group_value),
                    "row_idx": torch.from_numpy(row_idx),
                    "features": embeddings[row_idx].clone(),
                    "labels": torch.from_numpy(targets),
                }
            )

    if drop_overlapping_segments:
        logging.info(
            "Sequence overlap filtering: dropped %d overlapping segments before windowing.",
            total_dropped_overlap,
        )
        debug_df = pd.concat(debug_frames, axis=0, ignore_index=True) if debug_frames else df.copy()
    else:
        debug_df = None

    return sequences, resolved_group_col, debug_df


class TransformerSequenceDataset(Dataset):
    def __init__(self, sequences: List[Dict]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        return self.sequences[idx]


def make_transformer_collate_fn():
    def collate_fn(batch: List[Dict]) -> Dict:
        batch_size = len(batch)
        max_len = max(int(item["features"].shape[0]) for item in batch)
        feat_dim = int(batch[0]["features"].shape[1])

        features = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
        labels = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
        valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        row_idx = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.long)
        groups: List[str] = []

        for i, item in enumerate(batch):
            length = int(item["features"].shape[0])
            features[i, :length] = item["features"]
            labels[i, :length] = item["labels"]
            valid_mask[i, :length] = True
            row_idx[i, :length] = item["row_idx"]
            groups.append(item["group"])

        return {
            "features": features,
            "labels": labels,
            "valid_mask": valid_mask,
            "row_idx": row_idx,
            "groups": groups,
        }

    return collate_fn


def load_transformer_split(
    csv_path: Path,
    embeddings_path: Path,
    group_col: str,
    limit: int,
    val_col: str = "valence",
    aro_col: str = "arousal",
    window_size: int = 0,
    window_stride: int = 0,
    drop_overlapping_segments: bool = False,
) -> Tuple[pd.DataFrame, List[Dict], str, int, pd.DataFrame | None]:
    df = read_csv_with_fallback(csv_path)
    df = add_stream_name_column(df)
    ensure_columns(df, ["video_name", "text", val_col, aro_col], str(csv_path))
    if df[[val_col, aro_col]].isna().any().any():
        raise ValueError(f"NaN detected in targets for {csv_path}.")
    df = maybe_limit(df, limit).reset_index(drop=True).copy()
    embeddings = load_embeddings_tensor(embeddings_path)
    if int(embeddings.shape[0]) < len(df):
        raise ValueError(
            f"Embeddings rows ({int(embeddings.shape[0])}) are fewer than CSV rows ({len(df)}) in {embeddings_path}."
        )
    embeddings = embeddings[: len(df)]

    sequences, resolved_group_col, debug_df = build_transformer_sequences(
        df=df,
        embeddings=embeddings,
        group_col=group_col,
        val_col=val_col,
        aro_col=aro_col,
        window_size=window_size,
        window_stride=window_stride,
        drop_overlapping_segments=drop_overlapping_segments,
    )
    feature_dim = int(embeddings.shape[1])
    return df, sequences, resolved_group_col, feature_dim, debug_df
