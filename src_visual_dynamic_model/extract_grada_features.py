import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm
from torchvision.transforms import functional as F

from config_utils import load_toml_compat


def _safe_float(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def _parse_annotation_txt(path: Path) -> List[Tuple[int, float, float]]:
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
                valence = float("nan") if valence is None else float(valence)
                arousal = float("nan") if arousal is None else float(arousal)
            rows.append((frame_idx, valence, arousal))
    return rows


def _build_index_from_txt(annotations_root: Path, split: str) -> pd.DataFrame:
    split_name = "Train_Set" if split == "train" else "Validation_Set"
    ann_dir = annotations_root / split_name
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")
    rows = []
    for txt_path in sorted(ann_dir.glob("*.txt")):
        video = txt_path.stem
        parsed = _parse_annotation_txt(txt_path)
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
    return pd.DataFrame(rows)


def _build_index_from_test_list(test_list_path: Path) -> pd.DataFrame:
    if not test_list_path.is_file():
        raise FileNotFoundError(f"test_list_path not found: {test_list_path}")
    df = pd.read_csv(test_list_path)
    if "image_location" not in df.columns:
        raise ValueError("Test list must contain image_location")
    rows = []
    for img_loc in df["image_location"].tolist():
        parts = str(img_loc).replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        video = parts[0]
        stem = Path(parts[-1]).stem
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        rows.append({"video": str(video), "frame": int(frame_idx), "image_location": str(img_loc)})
    return pd.DataFrame(rows)


def _group_frames(df: pd.DataFrame) -> Dict[str, List[int]]:
    grouped = {}
    for video, g in df.groupby("video"):
        grouped[str(video)] = sorted(set(g["frame"].astype(int).tolist()))
    return grouped


class FrameResolver:
    def __init__(self, frames_root: Path, split: str, faces_dir: str = "01"):
        self.frames_root = Path(frames_root)
        self.split = str(split)
        self.faces_dir = str(faces_dir)

    def resolve(self, video: str, frame_idx: int) -> Optional[Path]:
        path = self.frames_root / self.split / str(video) / self.faces_dir / f"{int(frame_idx):05d}.jpg"
        return path if path.is_file() else None


def _extract_split(
    split: str,
    index_df: pd.DataFrame,
    frames_root: Path,
    features_root: Path,
    grada_repo_path: Path,
    grada_model_variant: str,
    grada_weights_path: str,
    feature_batch_size: int,
    save_grada_predictions: bool,
    max_videos: int,
    max_frames_per_video: int,
) -> None:
    if str(grada_repo_path) not in sys.path:
        sys.path.insert(0, str(grada_repo_path))
    from grada_emotion.models import load_model

    resolver = FrameResolver(frames_root=frames_root, split=split, faces_dir="01")
    out_split_dir = features_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = str(grada_weights_path).strip() if str(grada_weights_path).strip() else None
    variant = str(grada_model_variant).lower()
    model = load_model(
        variant=variant,
        device=device,
        weights_path=weights,
    )
    model = model.to(device)
    model.eval()
    model_device = next(model.parameters()).device

    missing_keys = set(getattr(model, "_missing_keys", []))
    can_predict_emo = not {"classifier.weight", "classifier.bias"}.intersection(missing_keys)
    can_predict_va = not {"regression.weight", "regression.bias"}.intersection(missing_keys)

    grouped = _group_frames(index_df)
    videos = sorted(grouped.keys())
    if max_videos > 0:
        videos = videos[: int(max_videos)]

    for video in tqdm(videos, desc=f"Extract {split}"):
        out_path = out_split_dir / f"{video}.npz"
        if out_path.is_file():
            continue
        frames = grouped[video]
        if max_frames_per_video > 0:
            frames = frames[: int(max_frames_per_video)]

        frame_list: List[int] = []
        feat_list: List[np.ndarray] = []
        va_list: List[np.ndarray] = []
        emo_list: List[np.ndarray] = []

        for i in range(0, len(frames), int(feature_batch_size)):
            chunk = frames[i : i + int(feature_batch_size)]
            batch_faces = []
            valid_frames = []
            for fidx in chunk:
                frame_path = resolver.resolve(video, int(fidx))
                if frame_path is None:
                    continue
                face = np.array(Image.open(frame_path).convert("RGB"))
                # Local preprocessing equivalent to GRADA preprocessing.
                expected_size = 240 if variant == "b1" else 380
                pil = Image.fromarray(face)
                old_size = pil.size  # (w, h)
                ratio = float(expected_size) / max(old_size)
                new_size = tuple(int(x * ratio) for x in old_size)
                pil = pil.resize(new_size, Resampling.BILINEAR)
                padded = Image.new("RGB", (expected_size, expected_size))
                padded.paste(pil, ((expected_size - new_size[0]) // 2, (expected_size - new_size[1]) // 2))
                t = F.pil_to_tensor(padded)
                t = F.convert_image_dtype(t, torch.float)
                t = F.normalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                batch_faces.append(t)
                valid_frames.append(int(fidx))
            if not batch_faces:
                continue

            batch = torch.stack(batch_faces, dim=0).to(model_device)
            with torch.no_grad():
                x = model.model(batch)
                x = model.dropout_after_pretrained(x)
                x = model.embeddings_layer(x)
                x = model.batchnorm(x)
                x = model.activation_embeddings(x)
                emb_np = x.detach().cpu().numpy().astype(np.float32)

                probs_np = None
                va_np = None
                if save_grada_predictions and can_predict_emo:
                    probs_np = torch.softmax(model.classifier(x), dim=-1).detach().cpu().numpy().astype(np.float32)
                if save_grada_predictions and can_predict_va:
                    va_np = model.regression(x).detach().cpu().numpy().astype(np.float32)

            for j, fidx in enumerate(valid_frames):
                frame_list.append(int(fidx))
                feat_list.append(emb_np[j])
                if save_grada_predictions:
                    if can_predict_va and va_np is not None:
                        va_list.append(va_np[j])
                    else:
                        va_list.append(np.zeros((2,), dtype=np.float32))
                    if can_predict_emo and probs_np is not None:
                        emo_list.append(np.asarray(probs_np[j], dtype=np.float32))
                    else:
                        emo_list.append(np.zeros((8,), dtype=np.float32))

        if not frame_list:
            continue

        frames_arr = np.array(frame_list, dtype=np.int32)
        features = np.stack(feat_list, axis=0).astype(np.float32)
        order = np.argsort(frames_arr)
        payload = {
            "frames": frames_arr[order],
            "features": features[order],
        }
        if save_grada_predictions:
            payload["va_pred"] = np.stack(va_list, axis=0).astype(np.float32)[order]
            payload["emo_probs"] = np.stack(emo_list, axis=0).astype(np.float32)[order]
        np.savez(out_path, **payload)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = load_toml_compat(Path(args.config))

    annotations_root = Path(cfg["annotations_root"])
    frames_root = Path(cfg["frames_root"])
    features_root = Path(cfg["features_root"])
    grada_repo_path = Path(cfg["grada_repo_path"])
    grada_model_variant = str(cfg.get("grada_model_variant", "b1"))
    grada_weights_path = str(cfg.get("grada_weights_path", ""))
    feature_batch_size = int(cfg.get("feature_batch_size", 32))
    save_grada_predictions = bool(cfg.get("save_grada_predictions", True))
    test_list_path = Path(str(cfg.get("test_list_path", "")).strip()) if str(cfg.get("test_list_path", "")).strip() else None

    splits = [args.split] if args.split != "all" else ["train", "val", "test"]
    for split in splits:
        if split in ("train", "val"):
            index_df = _build_index_from_txt(annotations_root=annotations_root, split=split)
        else:
            if test_list_path is None:
                raise ValueError("test_list_path is required for split=test")
            index_df = _build_index_from_test_list(test_list_path)
        _extract_split(
            split=split,
            index_df=index_df,
            frames_root=frames_root,
            features_root=features_root,
            grada_repo_path=grada_repo_path,
            grada_model_variant=grada_model_variant,
            grada_weights_path=grada_weights_path,
            feature_batch_size=feature_batch_size,
            save_grada_predictions=save_grada_predictions,
            max_videos=int(args.max_videos),
            max_frames_per_video=int(args.max_frames_per_video),
        )


if __name__ == "__main__":
    main()
