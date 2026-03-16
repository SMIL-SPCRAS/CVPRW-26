from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from ..core.losses_metrics import ccc_score


def _normalize_video_name(value: Any) -> str:
    text = str(value).strip()
    if "/" in text or "\\" in text:
        text = text.replace("\\", "/").split("/")[-1]
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text


def _prepare_and_merge_frame_tables(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    pred_video_col: str = "video_name",
    gt_video_col: str = "video_name",
    frame_col: str = "frame_idx",
    val_col: str = "valence",
    aro_col: str = "arousal",
    normalize_video_name: bool = True,
    filter_invalid_gt: bool = True,
    invalid_threshold: float = -4.9,
) -> pd.DataFrame:
    required_pred = [pred_video_col, frame_col, val_col, aro_col]
    required_gt = [gt_video_col, frame_col, val_col, aro_col]

    missing_pred = [c for c in required_pred if c not in pred_df.columns]
    missing_gt = [c for c in required_gt if c not in gt_df.columns]
    if missing_pred:
        raise ValueError(f"Prediction CSV missing columns: {missing_pred}")
    if missing_gt:
        raise ValueError(f"GT frame table missing columns: {missing_gt}")

    pred = pred_df[[pred_video_col, frame_col, val_col, aro_col]].copy()
    gt = gt_df[[gt_video_col, frame_col, val_col, aro_col]].copy()

    pred = pred.rename(
        columns={
            pred_video_col: "video_name",
            frame_col: "frame_idx",
            val_col: "valence_pred",
            aro_col: "arousal_pred",
        }
    )
    gt = gt.rename(
        columns={
            gt_video_col: "video_name",
            frame_col: "frame_idx",
            val_col: "valence_gt",
            aro_col: "arousal_gt",
        }
    )

    if filter_invalid_gt:
        gt_mask = (gt["valence_gt"] > invalid_threshold) & (gt["arousal_gt"] > invalid_threshold)
        gt_dropped = int((~gt_mask).sum())
        if gt_dropped > 0:
            logging.info(
                "Frame GT filtering: dropped %d invalid rows (threshold=%s).",
                gt_dropped,
                invalid_threshold,
            )
        gt = gt.loc[gt_mask].reset_index(drop=True)

    if normalize_video_name:
        pred["video_name"] = pred["video_name"].map(_normalize_video_name)
        gt["video_name"] = gt["video_name"].map(_normalize_video_name)

    pred["frame_idx"] = pred["frame_idx"].astype(int)
    gt["frame_idx"] = gt["frame_idx"].astype(int)

    merged = gt.merge(pred, on=["video_name", "frame_idx"], how="inner")
    merged = merged.dropna(
        subset=["valence_gt", "arousal_gt", "valence_pred", "arousal_pred"]
    ).reset_index(drop=True)

    if merged.empty:
        raise ValueError("No matched frame rows between prediction and GT after merge.")
    return merged


def evaluate_frame_ccc_from_dataframes(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    pred_video_col: str = "video_name",
    gt_video_col: str = "video_name",
    frame_col: str = "frame_idx",
    val_col: str = "valence",
    aro_col: str = "arousal",
    normalize_video_name: bool = True,
    filter_invalid_gt: bool = True,
    invalid_threshold: float = -4.9,
    merged_out_csv: Path | None = None,
) -> Dict[str, float]:
    merged = _prepare_and_merge_frame_tables(
        pred_df=pred_df,
        gt_df=gt_df,
        pred_video_col=pred_video_col,
        gt_video_col=gt_video_col,
        frame_col=frame_col,
        val_col=val_col,
        aro_col=aro_col,
        normalize_video_name=normalize_video_name,
        filter_invalid_gt=filter_invalid_gt,
        invalid_threshold=invalid_threshold,
    )
    if merged_out_csv is not None:
        merged_out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_out_csv, index=False)

    ccc_val = ccc_score(
        torch.tensor(merged["valence_pred"].values, dtype=torch.float32),
        torch.tensor(merged["valence_gt"].values, dtype=torch.float32),
    )
    ccc_aro = ccc_score(
        torch.tensor(merged["arousal_pred"].values, dtype=torch.float32),
        torch.tensor(merged["arousal_gt"].values, dtype=torch.float32),
    )
    ccc_mean = 0.5 * (ccc_val + ccc_aro)

    metrics = {
        "frame_rows_matched": float(len(merged)),
        "frame_ccc_valence": float(ccc_val),
        "frame_ccc_arousal": float(ccc_aro),
        "frame_ccc_mean": float(ccc_mean),
    }

    logging.info(
        "Frame-level metrics | "
        f"rows={int(metrics['frame_rows_matched'])} | "
        f"ccc_v={metrics['frame_ccc_valence']:.5f} | "
        f"ccc_a={metrics['frame_ccc_arousal']:.5f} | "
        f"ccc_mean={metrics['frame_ccc_mean']:.5f}"
    )
    return metrics


def evaluate_frame_ccc(
    pred_csv: Path,
    gt_csv: Path,
    pred_video_col: str = "video_name",
    gt_video_col: str = "video_name",
    frame_col: str = "frame_idx",
    val_col: str = "valence",
    aro_col: str = "arousal",
    normalize_video_name: bool = True,
    filter_invalid_gt: bool = True,
    invalid_threshold: float = -4.9,
    merged_out_csv: Path | None = None,
) -> Dict[str, float]:
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)
    return evaluate_frame_ccc_from_dataframes(
        pred_df=pred_df,
        gt_df=gt_df,
        pred_video_col=pred_video_col,
        gt_video_col=gt_video_col,
        frame_col=frame_col,
        val_col=val_col,
        aro_col=aro_col,
        normalize_video_name=normalize_video_name,
        filter_invalid_gt=filter_invalid_gt,
        invalid_threshold=invalid_threshold,
        merged_out_csv=merged_out_csv,
    )
