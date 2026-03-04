import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import WindowConfig, VisualDynamicDataset, collate_visual_dynamic
from model import VisualDynamicModel
from train import TrainConfig, _infer_feature_dim, load_config
from export_pkls_from_best import _build_test_index, _frame_mask, _load_checkpoint_compat


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

    def average(self):
        out = {}
        for video, emo_dict in self.records.items():
            out[video] = {}
            for emo, vals in emo_dict.items():
                out[video][emo] = np.array(
                    [float(np.mean(v)) if len(v) > 0 else np.nan for v in vals],
                    dtype=np.float32,
                )
        return out


def _build_test_loader(test_index: pd.DataFrame, cfg: TrainConfig) -> DataLoader:
    ds = VisualDynamicDataset(
        df=test_index,
        features_root=cfg.features_root,
        split="test",
        window_cfg=WindowConfig(
            window_length=cfg.window_length,
            hop_length=cfg.hop_length,
            time_delay=cfg.time_delay,
        ),
        allow_nearest=cfg.allow_nearest_val,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_visual_dynamic,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-file", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
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

    sample_path = Path(args.sample_file) if str(args.sample_file).strip() else cfg.test_list_path
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    test_index = _build_test_index(sample_path)
    if len(test_index) == 0:
        raise RuntimeError("Empty test index parsed from sample file.")

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

    test_loader = _build_test_loader(test_index, cfg)
    video_lengths = {video: len(g) for video, g in test_index.groupby("video")}
    pred_handler = _PredHandler(video_lengths)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Infer test", leave=False):
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
                frame_mask.detach().cpu().numpy(),  # keep all frames, same as sample file structure
            )

    pred_avg = pred_handler.average()
    frame_to_pos = {}
    for video, g in test_index.groupby("video"):
        frames = g.sort_values("frame")["frame"].astype(int).tolist()
        frame_to_pos[video] = {fidx: pos for pos, fidx in enumerate(frames)}

    sample_df = pd.read_csv(sample_path)
    if "image_location" not in sample_df.columns:
        raise ValueError("Sample file must contain image_location.")

    out_vals = np.full((len(sample_df),), -5.0, dtype=np.float32)
    out_aros = np.full((len(sample_df),), -5.0, dtype=np.float32)
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
        preds_v = pred_avg.get(video)
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

    if str(args.output_path).strip():
        output_path = Path(args.output_path)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(cfg.output_dir) / "test_predictions" / f"CVPR_6th_ABAW_VA_test_set_example_{run_id}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"[DONE] Submission file saved: {output_path}")
    print(f"[INFO] Rows: {len(result)} (same order/length as source sample)")


if __name__ == "__main__":
    main()

