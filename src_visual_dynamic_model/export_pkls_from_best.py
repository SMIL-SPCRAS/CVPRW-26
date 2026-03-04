import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import (
    IndexConfig,
    WindowConfig,
    VisualDynamicDataset,
    build_index,
    collate_visual_dynamic,
)
from metrics import ccc
from model import VisualDynamicModel
from train import TrainConfig, _infer_feature_dim, load_config


def _frame_mask(lengths: torch.Tensor, t: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(t, device=device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def _build_test_index(test_list_path: Path) -> pd.DataFrame:
    if not test_list_path.is_file():
        raise FileNotFoundError(f"test_list_path not found: {test_list_path}")
    df = pd.read_csv(test_list_path)
    if "image_location" not in df.columns:
        raise ValueError("Test list must contain image_location column")
    rows = []
    for _, row in df.iterrows():
        img_loc = str(row["image_location"])
        parts = img_loc.replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        video = parts[0]
        try:
            frame_idx = int(Path(parts[-1]).stem)
        except ValueError:
            continue
        valence = row["valence"] if "valence" in df.columns else np.nan
        arousal = row["arousal"] if "arousal" in df.columns else np.nan
        rows.append(
            {
                "video": str(video),
                "frame": int(frame_idx),
                "image_location": img_loc,
                "valence": float(valence) if pd.notna(valence) else np.nan,
                "arousal": float(arousal) if pd.notna(arousal) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_loader(index_df: pd.DataFrame, cfg: TrainConfig, split: str) -> DataLoader:
    ds = VisualDynamicDataset(
        df=index_df,
        features_root=cfg.features_root,
        split=split,
        window_cfg=WindowConfig(
            window_length=cfg.window_length,
            hop_length=cfg.hop_length,
            time_delay=cfg.time_delay,
        ),
        allow_nearest=cfg.allow_nearest_train if split == "train" else cfg.allow_nearest_val,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_visual_dynamic,
    )


def _valid_label_mask(df: pd.DataFrame) -> np.ndarray:
    if ("valence" not in df.columns) or ("arousal" not in df.columns):
        return np.zeros((len(df),), dtype=bool)
    v = pd.to_numeric(df["valence"], errors="coerce").to_numpy(dtype=np.float32)
    a = pd.to_numeric(df["arousal"], errors="coerce").to_numpy(dtype=np.float32)
    finite = np.isfinite(v) & np.isfinite(a)
    not_invalid = (v != -5.0) & (a != -5.0) & (v != 5.0) & (a != 5.0)
    return finite & not_invalid


def _load_checkpoint_compat(model: VisualDynamicModel, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    # Compatibility with old wrapper model where params were under "temporal.*".
    if any(str(k).startswith("temporal.") for k in state.keys()):
        stripped = {}
        for k, v in state.items():
            if str(k).startswith("temporal."):
                stripped[str(k)[len("temporal."):]] = v
        model.load_state_dict(stripped, strict=True)
        return

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}; Unexpected: {unexpected}")


class _ValueHandler:
    def __init__(self, video_lengths: Dict[str, int], dim: int):
        self.dim = int(dim)
        self.records = {video: [[] for _ in range(length)] for video, length in video_lengths.items()}

    def update(self, values: np.ndarray, videos: List[str], indices: np.ndarray, lengths: np.ndarray, mask: np.ndarray):
        b, t, d = values.shape
        if d != self.dim:
            raise ValueError(f"Value dim mismatch: got {d}, expected {self.dim}")
        for i in range(b):
            video = videos[i]
            length = int(lengths[i])
            idxs = indices[i].tolist()
            for k in range(min(length, t)):
                if not bool(mask[i, k]):
                    continue
                pos = int(idxs[k])
                self.records[video][pos].append(values[i, k].astype(np.float32))

    def average(self) -> Dict[str, np.ndarray]:
        out = {}
        for video, per_frame in self.records.items():
            rows = []
            for vals in per_frame:
                if len(vals) > 0:
                    rows.append(np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32))
                else:
                    rows.append(np.full((self.dim,), np.nan, dtype=np.float32))
            out[video] = np.stack(rows, axis=0)
        return out


class _PredHandler:
    def __init__(self, video_lengths: Dict[str, int]):
        self.records = {
            video: {"Valence": [[] for _ in range(length)], "Arousal": [[] for _ in range(length)]}
            for video, length in video_lengths.items()
        }

    def update(self, outputs: np.ndarray, videos: List[str], indices: np.ndarray, lengths: np.ndarray, valid_mask: np.ndarray):
        b, t, _ = outputs.shape
        for i in range(b):
            video = videos[i]
            length = int(lengths[i])
            idxs = indices[i].tolist()
            for k in range(min(length, t)):
                if not bool(valid_mask[i, k]):
                    continue
                pos = int(idxs[k])
                self.records[video]["Valence"][pos].append(float(outputs[i, k, 0]))
                self.records[video]["Arousal"][pos].append(float(outputs[i, k, 1]))

    def average(self) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for video, emo_dict in self.records.items():
            out[video] = {}
            for emo, vals in emo_dict.items():
                out[video][emo] = np.array(
                    [float(np.mean(v)) if len(v) > 0 else np.nan for v in vals],
                    dtype=np.float32,
                )
        return out


def _save_flat_test_predictions(
    test_index: pd.DataFrame,
    sample_file: Path,
    trialwise_pred: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    missing_value: float = -5.0,
) -> Path:
    sample_df = pd.read_csv(sample_file)
    if "image_location" not in sample_df.columns:
        raise ValueError("Sample file must contain image_location column")

    frame_to_pos = {}
    for video, g in test_index.groupby("video"):
        frames = g.sort_values("frame")["frame"].astype(int).tolist()
        frame_to_pos[str(video)] = {fidx: pos for pos, fidx in enumerate(frames)}

    out_vals = np.full((len(sample_df),), float(missing_value), dtype=np.float32)
    out_aros = np.full((len(sample_df),), float(missing_value), dtype=np.float32)
    for i, img_loc in enumerate(sample_df["image_location"].tolist()):
        parts = str(img_loc).replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        video = parts[0]
        try:
            frame_idx = int(Path(parts[-1]).stem)
        except ValueError:
            continue
        pos_map = frame_to_pos.get(video)
        preds_v = trialwise_pred.get(video)
        if pos_map is None or preds_v is None:
            continue
        pos = pos_map.get(frame_idx)
        if pos is None:
            continue
        val = preds_v["Valence"][pos]
        aro = preds_v["Arousal"][pos]
        if np.isfinite(val):
            out_vals[i] = float(val)
        if np.isfinite(aro):
            out_aros[i] = float(aro)

    result = pd.DataFrame(
        {
            "image_location": sample_df["image_location"].astype(str).tolist(),
            "valence": out_vals,
            "arousal": out_aros,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return output_path


def _run_test_trialwise(
    cfg: TrainConfig,
    model: VisualDynamicModel,
    test_index: pd.DataFrame,
    device: torch.device,
    max_samples: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    if max_samples > 0:
        test_index = test_index.iloc[: int(max_samples)].reset_index(drop=True)

    loader = _build_loader(test_index, cfg, split="test")
    video_lengths = {video: len(g) for video, g in test_index.groupby("video")}
    pred_handler = _PredHandler(video_lengths)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Export test txt", leave=False):
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            indices = batch["indices"].to(device)
            videos = batch["videos"]
            frame_mask = _frame_mask(lengths, features.shape[1], device)

            preds = model(features, mask=frame_mask).detach().cpu().numpy()
            pred_handler.update(
                preds,
                videos,
                indices.detach().cpu().numpy(),
                lengths.detach().cpu().numpy(),
                frame_mask.detach().cpu().numpy(),
            )

    return test_index, pred_handler.average()


def _run_split(
    split: str,
    cfg: TrainConfig,
    model: VisualDynamicModel,
    index_df: pd.DataFrame,
    device: torch.device,
    max_samples: int,
) -> Tuple[Dict[str, dict], int, int]:
    if max_samples > 0:
        index_df = index_df.iloc[: int(max_samples)].reset_index(drop=True)

    loader = _build_loader(index_df, cfg, split=split)
    video_lengths = {video: len(g) for video, g in index_df.groupby("video")}
    pred_handler = _ValueHandler(video_lengths, dim=2)
    emb_dim = int(model.head[-1].in_features) if isinstance(model.head, torch.nn.Sequential) else int(model.head.in_features)
    emb_handler = _ValueHandler(video_lengths, dim=emb_dim)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Export {split}", leave=False):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            missing_mask = batch["missing_mask"].to(device)
            lengths = batch["lengths"].to(device)
            indices = batch["indices"].to(device)
            videos = batch["videos"]

            frame_mask = _frame_mask(lengths, features.shape[1], device)
            label_not_pos5 = (labels[..., 0] != 5.0) & (labels[..., 1] != 5.0)
            mask = frame_mask & valid_mask & (~missing_mask) & label_not_pos5

            emb, pred = model.forward_with_embeddings(features, mask=frame_mask)
            pred_handler.update(
                pred.detach().cpu().numpy(),
                videos,
                indices.detach().cpu().numpy(),
                lengths.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
            )
            emb_handler.update(
                emb.detach().cpu().numpy(),
                videos,
                indices.detach().cpu().numpy(),
                lengths.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
            )

    pred_avg = pred_handler.average()
    emb_avg = emb_handler.average()

    records: Dict[str, dict] = {}
    for video, group in index_df.groupby("video"):
        g = group.sort_values("frame").reset_index(drop=True)
        p = pred_avg.get(video)
        e = emb_avg.get(video)
        if p is None or e is None:
            continue
        n = min(len(g), p.shape[0], e.shape[0])
        for i in range(n):
            gt_v = float(g.iloc[i].get("valence", np.nan))
            gt_a = float(g.iloc[i].get("arousal", np.nan))
            if (not np.isfinite(gt_v)) or (not np.isfinite(gt_a)):
                continue
            if gt_v in (-5.0, 5.0) or gt_a in (-5.0, 5.0):
                continue
            pred_v = float(p[i, 0])
            pred_a = float(p[i, 1])
            emb_vec = e[i]
            if (not np.isfinite(pred_v)) or (not np.isfinite(pred_a)) or (not np.isfinite(emb_vec).all()):
                continue
            frame_num = int(g.iloc[i]["frame"])
            frame_name = f"{frame_num:05d}.jpg"
            key = f"{video}/{frame_name}"
            records[key] = {
                "embedding": emb_vec.astype(np.float32),
                "prediction": np.array([pred_v, pred_a], dtype=np.float32),
                "label": np.array([gt_v, gt_a], dtype=np.float32),
            }

    valid_count = int(_valid_label_mask(index_df).sum())
    saved_count = len(records)
    return records, valid_count, saved_count


def _save_split_pkl(path: Path, records: Dict[str, dict]) -> None:
    # Save only frame-keyed records:
    # {"video/00001.jpg": {"embedding": ..., "prediction": ..., "label": ...}, ...}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def _val_metrics(records: Dict[str, dict]) -> Dict[str, float]:
    if len(records) == 0:
        return {"ccc_v": 0.0, "ccc_a": 0.0, "va_score": 0.0}
    preds = np.stack([v["prediction"] for v in records.values()], axis=0)
    labels = np.stack([v["label"] for v in records.values()], axis=0)
    ccc_v = float(ccc(preds[:, 0], labels[:, 0]))
    ccc_a = float(ccc(preds[:, 1], labels[:, 1]))
    return {"ccc_v": ccc_v, "ccc_a": ccc_a, "va_score": 0.5 * (ccc_v + ccc_a)}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(Path(args.config))
    if args.num_workers >= 0:
        cfg.num_workers = int(args.num_workers)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.output_dir) if str(args.output_dir).strip() else (checkpoint_path.parent.parent / "export_pkls")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.cache_dir if cfg.cache_dir is not None else (out_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_index = build_index(
        IndexConfig(
            annotations_root=cfg.annotations_root,
            split="train",
            cache_path=cache_dir / "affwild2_train.csv",
            filter_invalid=False,
        )
    )
    val_index = build_index(
        IndexConfig(
            annotations_root=cfg.annotations_root,
            split="val",
            cache_path=cache_dir / "affwild2_val.csv",
            filter_invalid=False,
        )
    )
    test_index = _build_test_index(cfg.test_list_path)

    input_dim = _infer_feature_dim(cfg.features_root, split="train")
    model = VisualDynamicModel(
        input_dim=input_dim,
        hidden_dim=int(cfg.hidden_dim),
        num_heads=int(cfg.num_transformer_heads),
        tr_layers=int(cfg.tr_layers),
        dropout=float(cfg.dropout),
        out_dim=int(cfg.out_dim),
        head_type=str(cfg.head_type),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _load_checkpoint_compat(model, checkpoint_path=checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    train_records, train_valid, train_saved = _run_split(
        split="train",
        cfg=cfg,
        model=model,
        index_df=train_index,
        device=device,
        max_samples=int(args.max_train_samples),
    )
    val_records, val_valid, val_saved = _run_split(
        split="val",
        cfg=cfg,
        model=model,
        index_df=val_index,
        device=device,
        max_samples=int(args.max_val_samples),
    )
    test_records, test_valid, test_saved = _run_split(
        split="test",
        cfg=cfg,
        model=model,
        index_df=test_index,
        device=device,
        max_samples=int(args.max_test_samples),
    )

    _save_split_pkl(out_dir / "train.pkl", train_records)
    _save_split_pkl(out_dir / "val.pkl", val_records)
    _save_split_pkl(out_dir / "test.pkl", test_records)

    test_index_for_txt, test_trialwise_pred = _run_test_trialwise(
        cfg=cfg,
        model=model,
        test_index=test_index,
        device=device,
        max_samples=int(args.max_test_samples),
    )
    flat_test_path = _save_flat_test_predictions(
        test_index=test_index_for_txt,
        sample_file=cfg.test_list_path,
        trialwise_pred=test_trialwise_pred,
        output_path=out_dir / "test_predictions" / cfg.test_list_path.name,
    )

    val_m = _val_metrics(val_records)
    summary = {
        "checkpoint": str(checkpoint_path),
        "ccc_v_val": float(val_m["ccc_v"]),
        "ccc_a_val": float(val_m["ccc_a"]),
        "va_score_val": float(val_m["va_score"]),
        "counts": {
            "train": {"valid_annotation_frames": int(train_valid), "saved_frames": int(train_saved)},
            "val": {"valid_annotation_frames": int(val_valid), "saved_frames": int(val_saved)},
            "test": {"valid_annotation_frames": int(test_valid), "saved_frames": int(test_saved)},
        },
        "output_dir": str(out_dir),
        "test_prediction_file": str(flat_test_path),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] PKLs saved to: {out_dir}")
    print(f"[VAL] ccc_v={val_m['ccc_v']:.6f} ccc_a={val_m['ccc_a']:.6f} va_score={val_m['va_score']:.6f}")
    print(f"[COUNT] train: valid={train_valid} saved={train_saved}")
    print(f"[COUNT] val: valid={val_valid} saved={val_saved}")
    print(f"[COUNT] test: valid={test_valid} saved={test_saved}")
    print(f"[DONE] Test prediction file saved: {flat_test_path}")


if __name__ == "__main__":
    main()
