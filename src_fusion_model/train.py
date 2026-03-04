import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
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

from config_utils import load_toml_compat
from dataset import IndexConfig, WindowConfig, FusionDataset, build_index, collate_fusion
from losses import CCCLoss
from metrics import ccc
from model import FusionModel


@dataclass
class TrainConfig:
    annotations_root: Path
    test_list_path: Path
    features_root: Path
    output_dir: Path
    cache_dir: Optional[Path] = None

    modalities: List[str] = None
    modality_input_mode: str = "both"  # embedding | prediction | both
    q_modality: str = "fusion"  # modality name or fusion
    k_modality: str = "fusion"  # modality name or fusion
    v_modality: str = "fusion"  # modality name or fusion

    window_length: int = 400
    hop_length: int = 150
    time_delay: int = 0
    allow_nearest_train: bool = True
    allow_nearest_val: bool = True

    head_type: str = "mlp"
    hidden_dim: int = 256
    num_transformer_heads: int = 8
    tr_layers: int = 5
    dropout: float = 0.1
    out_dim: int = 2

    batch_size: int = 8
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    scheduler_patience: int = 5
    scheduler_factor: float = 0.1
    scheduler_min_lr: float = 0.0
    use_amp: bool = False
    seed: int = 42

    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.0
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    run_test_after_train: bool = True


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _frame_mask(lengths: torch.Tensor, t: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(t, device=device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


class TrialOutputHandler:
    def __init__(self, video_lengths: Dict[str, int]):
        self.records = {
            video: {"Valence": [[] for _ in range(length)], "Arousal": [[] for _ in range(length)]}
            for video, length in video_lengths.items()
        }

    def update(self, values: np.ndarray, videos, indices, lengths, valid_mask):
        b, t, _ = values.shape
        for i in range(b):
            video = videos[i]
            length = int(lengths[i])
            idxs = indices[i].tolist()
            for k in range(min(length, t)):
                if not bool(valid_mask[i, k]):
                    continue
                pos = int(idxs[k])
                self.records[video]["Valence"][pos].append(float(values[i, k, 0]))
                self.records[video]["Arousal"][pos].append(float(values[i, k, 1]))

    def average(self):
        out = {}
        for video, emo_dict in self.records.items():
            out[video] = {}
            for emo, per_frame_values in emo_dict.items():
                out[video][emo] = np.array(
                    [float(np.mean(v)) if len(v) > 0 else np.nan for v in per_frame_values],
                    dtype=np.float32,
                )
        return out

    @staticmethod
    def concat(avg_records):
        valence = []
        arousal = []
        for _, emo_dict in avg_records.items():
            valence.append(emo_dict["Valence"])
            arousal.append(emo_dict["Arousal"])
        return {"Valence": np.concatenate(valence, axis=0), "Arousal": np.concatenate(arousal, axis=0)}


def _video_lengths(df: pd.DataFrame) -> Dict[str, int]:
    return {video: len(group) for video, group in df.groupby("video")}


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
        rows.append(
            {
                "video": str(video),
                "frame": int(frame_idx),
                "image_location": img_loc,
                "valence": np.nan,
                "arousal": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _save_annotation_style_predictions(
    index_df: pd.DataFrame,
    trialwise_pred: Dict[str, Dict[str, np.ndarray]],
    out_root: Path,
    set_name: str,
    missing_value: float = -5.0,
) -> Path:
    target_dir = out_root / set_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for video, group in index_df.groupby("video"):
        g = group.sort_values("frame")
        n = len(g)
        vals = np.full((n,), float(missing_value), dtype=np.float32)
        aros = np.full((n,), float(missing_value), dtype=np.float32)

        preds_v = trialwise_pred.get(str(video))
        if preds_v is not None:
            pred_val = preds_v.get("Valence")
            pred_aro = preds_v.get("Arousal")
            if pred_val is not None and pred_aro is not None:
                m = min(n, len(pred_val), len(pred_aro))
                vals[:m] = pred_val[:m]
                aros[:m] = pred_aro[:m]

        vals = np.where(np.isnan(vals), float(missing_value), vals)
        aros = np.where(np.isnan(aros), float(missing_value), aros)

        txt_path = target_dir / f"{video}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            f.write("valence,arousal\n")
            for i in range(n):
                f.write(f"{vals[i]:.6f},{aros[i]:.6f}\n")
    return target_dir


def _save_flat_test_predictions(
    test_index: pd.DataFrame,
    sample_file: Path,
    trialwise_pred: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> Path:
    sample_df = pd.read_csv(sample_file)
    if "image_location" not in sample_df.columns:
        raise ValueError("Sample file must contain image_location column")

    frame_to_pos = {}
    for video, g in test_index.groupby("video"):
        frames = g.sort_values("frame")["frame"].astype(int).tolist()
        frame_to_pos[str(video)] = {fidx: pos for pos, fidx in enumerate(frames)}

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


def _load_checkpoint_compat(model: FusionModel, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}; Unexpected: {unexpected}")


def _to_device_modalities(modality_features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {m: x.to(device) for m, x in modality_features.items()}


def _build_loader(index_df: pd.DataFrame, cfg: TrainConfig, split: str, shuffle: bool) -> Tuple[DataLoader, Dict[str, int]]:
    dataset = FusionDataset(
        df=index_df,
        features_root=cfg.features_root,
        modalities=cfg.modalities,
        split=split,
        window_cfg=WindowConfig(
            window_length=cfg.window_length,
            hop_length=cfg.hop_length,
            time_delay=cfg.time_delay,
        ),
        allow_nearest=cfg.allow_nearest_train if split == "train" else cfg.allow_nearest_val,
        modality_input_mode=cfg.modality_input_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fusion,
    )
    return loader, dataset.feature_dims


def _validate_modalities(features_root: Path, modalities: List[str], splits: List[str]) -> None:
    missing = []
    for m in modalities:
        for s in splits:
            p = Path(features_root) / str(m) / f"{s}.pkl"
            if not p.is_file():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing modality PKL files:\n- " + "\n- ".join(missing)
        )


def train_one_epoch(
    model: FusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CCCLoss,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, torch.Tensor]:
    model.train()
    total_loss = 0.0
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    weight_sum = torch.zeros(2, dtype=torch.float32, device=device)
    weight_count = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        modality_features = _to_device_modalities(batch["modality_features"], device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        missing_any_mask = batch["missing_any_mask"].to(device)
        lengths = batch["lengths"].to(device)

        any_modality = next(iter(modality_features.values()))
        frame_mask = _frame_mask(lengths, any_modality.shape[1], device)
        mask = frame_mask & valid_mask & (~missing_any_mask)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            preds = model(modality_features, mask=frame_mask)
            loss = criterion(labels, preds, weights=weights, mask=mask)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        weight_sum = weight_sum + weights
        weight_count += 1

    avg_weights = weight_sum / max(weight_count, 1)
    return total_loss / max(len(loader), 1), avg_weights.detach().cpu()


def eval_one_epoch(
    model: FusionModel,
    loader: DataLoader,
    criterion: CCCLoss,
    device: torch.device,
    video_lengths: Dict[str, int],
    use_amp: bool,
) -> Tuple[dict, Dict[str, Dict[str, np.ndarray]]]:
    model.eval()
    total_loss = 0.0
    amp_enabled = bool(use_amp and device.type == "cuda")
    weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    output_handler = TrialOutputHandler(video_lengths)
    label_handler = TrialOutputHandler(video_lengths)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            modality_features = _to_device_modalities(batch["modality_features"], device)
            labels = batch["labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            lengths = batch["lengths"].to(device)
            indices = batch["indices"].to(device)
            videos = batch["videos"]

            any_modality = next(iter(modality_features.values()))
            frame_mask = _frame_mask(lengths, any_modality.shape[1], device)
            # Keep parity with src_visual_dynamic_model: validate on all annotated frames.
            mask = frame_mask & valid_mask

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                preds = model(modality_features, mask=frame_mask)
                loss = criterion(labels, preds, weights=weights, mask=mask)
            total_loss += float(loss.item())

            output_handler.update(
                preds.detach().cpu().numpy(),
                videos,
                indices.detach().cpu().numpy(),
                lengths.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
            )
            label_handler.update(
                labels.detach().cpu().numpy(),
                videos,
                indices.detach().cpu().numpy(),
                lengths.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
            )

    pred_avg = output_handler.average()
    label_avg = label_handler.average()
    pred = TrialOutputHandler.concat(pred_avg)
    lab = TrialOutputHandler.concat(label_avg)

    vmask = ~(np.isnan(pred["Valence"]) | np.isnan(lab["Valence"]))
    amask = ~(np.isnan(pred["Arousal"]) | np.isnan(lab["Arousal"]))
    ccc_v = float(ccc(pred["Valence"][vmask], lab["Valence"][vmask])) if vmask.sum() > 0 else 0.0
    ccc_a = float(ccc(pred["Arousal"][amask], lab["Arousal"][amask])) if amask.sum() > 0 else 0.0
    va_score = 0.5 * (ccc_v + ccc_a)

    metrics = {
        "val_loss": total_loss / max(len(loader), 1),
        "ccc_valence": ccc_v,
        "ccc_arousal": ccc_a,
        "va_score": va_score,
    }
    return metrics, pred_avg


def run_test_export(cfg: TrainConfig, checkpoint_path: Path, run_dir: Path) -> Path:
    _validate_modalities(cfg.features_root, cfg.modalities, splits=["test"])
    test_index = _build_test_index(cfg.test_list_path)
    if len(test_index) == 0:
        raise RuntimeError("Empty test index parsed from test_list_path")
    if cfg.max_test_samples and cfg.max_test_samples > 0:
        test_index = test_index.iloc[: int(cfg.max_test_samples)].reset_index(drop=True)

    test_loader, feature_dims = _build_loader(test_index, cfg, split="test", shuffle=False)
    test_video_lengths = _video_lengths(test_index)

    model = FusionModel(
        modality_input_dims=feature_dims,
        modalities=cfg.modalities,
        q_modality=cfg.q_modality,
        k_modality=cfg.k_modality,
        v_modality=cfg.v_modality,
        hidden_dim=int(cfg.hidden_dim),
        num_heads=int(cfg.num_transformer_heads),
        tr_layers=int(cfg.tr_layers),
        dropout=float(cfg.dropout),
        out_dim=int(cfg.out_dim),
        head_type=str(cfg.head_type),
    )
    device = _device()
    _load_checkpoint_compat(model, checkpoint_path=checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    output_handler = TrialOutputHandler(test_video_lengths)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", leave=False):
            modality_features = _to_device_modalities(batch["modality_features"], device)
            lengths = batch["lengths"].to(device)
            indices = batch["indices"].to(device)
            videos = batch["videos"]

            any_modality = next(iter(modality_features.values()))
            frame_mask = _frame_mask(lengths, any_modality.shape[1], device)
            preds = model(modality_features, mask=frame_mask).detach().cpu().numpy()
            output_handler.update(
                preds,
                videos,
                indices.detach().cpu().numpy(),
                lengths.detach().cpu().numpy(),
                frame_mask.detach().cpu().numpy(),
            )

    trialwise_pred = output_handler.average()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flat_path = _save_flat_test_predictions(
        test_index=test_index,
        sample_file=cfg.test_list_path,
        trialwise_pred=trialwise_pred,
        output_path=run_dir / "test_predictions" / f"va_test_predictions_{stamp}.txt",
    )
    return flat_path


def load_config(path: Path) -> TrainConfig:
    data = load_toml_compat(path)
    cfg = TrainConfig(
        annotations_root=Path(data["annotations_root"]),
        test_list_path=Path(data["test_list_path"]),
        features_root=Path(data["features_root"]),
        output_dir=Path(data["output_dir"]),
        cache_dir=Path(data["cache_dir"]) if str(data.get("cache_dir", "")).strip() else None,
    )
    for k, v in data.items():
        if not hasattr(cfg, k):
            continue
        if k in ("annotations_root", "test_list_path", "features_root", "output_dir", "cache_dir"):
            continue
        setattr(cfg, k, v)
    if cfg.modalities is None:
        cfg.modalities = ["face_features", "llm_multimodal_featuress"]
    cfg.modalities = [str(m) for m in cfg.modalities]
    return cfg


def run_train(cfg: TrainConfig) -> Path:
    _set_seed(int(cfg.seed))
    device = _device()

    if int(cfg.num_workers) > 0:
        # Large PKL caches are expensive to replicate across worker processes (especially on Windows spawn).
        print(f"[WARN] num_workers={cfg.num_workers} may duplicate large PKL memory. Forcing num_workers=0.")
        cfg.num_workers = 0

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _validate_modalities(cfg.features_root, cfg.modalities, splits=["train", "val"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / run_id
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    cache_dir = cfg.cache_dir if cfg.cache_dir is not None else (run_dir / "cache")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
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
    if cfg.max_train_samples and cfg.max_train_samples > 0:
        train_index = train_index.iloc[: int(cfg.max_train_samples)].reset_index(drop=True)
    if cfg.max_val_samples and cfg.max_val_samples > 0:
        val_index = val_index.iloc[: int(cfg.max_val_samples)].reset_index(drop=True)

    train_loader, feature_dims = _build_loader(train_index, cfg, split="train", shuffle=True)
    val_loader, _ = _build_loader(val_index, cfg, split="val", shuffle=False)
    val_video_lengths = _video_lengths(val_index)

    model = FusionModel(
        modality_input_dims=feature_dims,
        modalities=cfg.modalities,
        q_modality=cfg.q_modality,
        k_modality=cfg.k_modality,
        v_modality=cfg.v_modality,
        hidden_dim=int(cfg.hidden_dim),
        num_heads=int(cfg.num_transformer_heads),
        tr_layers=int(cfg.tr_layers),
        dropout=float(cfg.dropout),
        out_dim=int(cfg.out_dim),
        head_type=str(cfg.head_type),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(cfg.scheduler_patience),
        factor=float(cfg.scheduler_factor),
        min_lr=float(cfg.scheduler_min_lr),
    )
    criterion = CCCLoss()

    metrics_path = logs_dir / f"metrics_{run_id}.csv"
    metrics_path.write_text("epoch,lr,train_loss,val_loss,ccc_valence,ccc_arousal,va_score,w_valence,w_arousal\n", encoding="utf-8")
    (logs_dir / f"config_{run_id}.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_score = float("-inf")
    no_improve = 0
    best_ckpt_path = checkpoints_dir / f"best_{run_id}_va_init.pt"
    for epoch in range(int(cfg.epochs)):
        train_loss, avg_weights = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=bool(cfg.use_amp),
        )
        val_metrics, val_trialwise_pred = eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            video_lengths=val_video_lengths,
            use_amp=bool(cfg.use_amp),
        )
        scheduler.step(val_metrics["va_score"])

        current = float(val_metrics["va_score"])
        if current > best_score + float(cfg.early_stop_min_delta):
            best_score = current
            no_improve = 0
            best_ckpt_path = checkpoints_dir / f"best_{run_id}_va{current:.4f}.pt"
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, best_ckpt_path)
            best_val_txt_root = run_dir / "best_val_predictions_txt"
            if best_val_txt_root.exists():
                shutil.rmtree(best_val_txt_root, ignore_errors=True)
            saved_dir = _save_annotation_style_predictions(
                index_df=val_index,
                trialwise_pred=val_trialwise_pred,
                out_root=best_val_txt_root,
                set_name="Validation_Set",
            )
            print(f"[INFO] Best val txt saved: {saved_dir}")
        else:
            no_improve += 1

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{optimizer.param_groups[0]['lr']:.8f},{train_loss:.6f},{val_metrics['val_loss']:.6f},"
                f"{val_metrics['ccc_valence']:.6f},{val_metrics['ccc_arousal']:.6f},{val_metrics['va_score']:.6f},"
                f"{avg_weights[0]:.6f},{avg_weights[1]:.6f}\n"
            )
        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_metrics['val_loss']:.4f} "
            f"| CCC(v,a)=({val_metrics['ccc_valence']:.3f},{val_metrics['ccc_arousal']:.3f}) "
            f"| va_score={val_metrics['va_score']:.3f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if no_improve >= int(cfg.early_stop_patience):
            print(f"[INFO] Early stop at epoch {epoch}, best_va={best_score:.4f}")
            break

    print(f"[DONE] Run dir: {run_dir}")
    print(f"[DONE] Best checkpoint: {best_ckpt_path}")
    print(f"[DONE] Best va_score: {best_score:.6f}")
    if bool(cfg.run_test_after_train):
        if best_ckpt_path.is_file():
            try:
                test_flat_path = run_test_export(cfg=cfg, checkpoint_path=best_ckpt_path, run_dir=run_dir)
                print(f"[DONE] Test predictions saved: {test_flat_path}")
            except Exception as exc:
                print(f"[WARN] Failed to export test predictions from best checkpoint: {exc}")
        else:
            print("[WARN] Best checkpoint file not found. Skipping test export.")
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(Path(args.config))
    run_train(cfg)
