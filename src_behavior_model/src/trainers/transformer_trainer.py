from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..core.losses_metrics import CCCLoss, compute_va_ccc
from ..datasets.frame_eval import evaluate_frame_ccc_from_dataframes
from ..datasets.frame_expand import expand_segment_predictions_to_frames
from ..datasets.transformer_data import (
    TransformerSequenceDataset,
    load_transformer_split,
    make_transformer_collate_fn,
)
from ..datasets.text_data import ensure_columns, read_csv_with_fallback
from ..models.transformer_model import TextMambaRegressor, TransformerRegressor, count_parameters


def _cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float,
    momentum: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    nesterov: bool,
) -> torch.optim.Optimizer:
    key = name.strip().lower()
    if key == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
        )
    if key == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
        )
    if key == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if key == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{name}'. Use adamw, adam, sgd, or rmsprop.")


def _masked_ccc_loss(preds: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, criterion: CCCLoss) -> torch.Tensor:
    valid_preds = preds[valid_mask]
    valid_labels = labels[valid_mask]
    if valid_preds.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=preds.device, requires_grad=True)
    loss_val = criterion(valid_preds[:, 0], valid_labels[:, 0])
    loss_aro = criterion(valid_preds[:, 1], valid_labels[:, 1])
    return 0.5 * (loss_val + loss_aro)


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CCCLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    losses: List[float] = []

    pbar = tqdm(loader, total=len(loader), desc=f"TrainT {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        preds = model(features=features, valid_mask=valid_mask)
        loss = _masked_ccc_loss(preds, labels, valid_mask, criterion)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CCCLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    n_rows: int,
    targets_np: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    losses: List[float] = []
    row_sum = np.zeros((n_rows, 2), dtype=np.float32)
    row_count = np.zeros((n_rows,), dtype=np.float32)

    pbar = tqdm(loader, total=len(loader), desc=f"ValT   {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        row_idx = batch["row_idx"]

        preds = model(features=features, valid_mask=valid_mask)
        loss = _masked_ccc_loss(preds, labels, valid_mask, criterion)
        losses.append(float(loss.item()))
        pbar.set_postfix(loss=f"{losses[-1]:.4f}")

        preds_cpu = preds.detach().cpu()
        valid_cpu = valid_mask.detach().cpu()
        row_idx_cpu = row_idx.cpu()

        for i in range(preds_cpu.shape[0]):
            cur_mask = valid_cpu[i]
            cur_idx = row_idx_cpu[i][cur_mask].numpy()
            if cur_idx.size == 0:
                continue
            cur_preds = preds_cpu[i][cur_mask].numpy()
            row_sum[cur_idx] += cur_preds
            row_count[cur_idx] += 1.0

    row_preds = np.full((n_rows, 2), np.nan, dtype=np.float32)
    observed_rows = row_count > 0
    if np.any(observed_rows):
        row_preds[observed_rows] = row_sum[observed_rows] / row_count[observed_rows, None]

    valid_rows = np.isfinite(row_preds).all(axis=1) & np.isfinite(targets_np).all(axis=1)
    if not np.any(valid_rows):
        metrics = {
            "val_loss": float(np.mean(losses)) if losses else float("nan"),
            "ccc_valence": float("nan"),
            "ccc_arousal": float("nan"),
            "ccc_mean": float("nan"),
        }
        return metrics, row_preds

    pred_t = torch.tensor(row_preds[valid_rows], dtype=torch.float32)
    target_t = torch.tensor(targets_np[valid_rows], dtype=torch.float32)
    metrics = compute_va_ccc(pred_t, target_t)
    metrics["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, row_preds


def save_checkpoint(
    model: torch.nn.Module,
    output_path: Path,
    config_payload: Dict[str, Any],
) -> None:
    torch.save(
        {"model_type": "transformer", "config": config_payload, "state_dict": model.state_dict()},
        output_path,
    )


def run_transformer_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_csv = Path(cfg.get("train_csv", "dataset/train_segment_Qwen3-VL-4B-Instruct.csv"))
    val_csv = Path(cfg.get("val_csv", "dataset/val_segment_Qwen3-VL-4B-Instruct.csv"))
    train_embeddings_pt = Path(str(cfg.get("train_embeddings_pt", "")))
    val_embeddings_pt = Path(str(cfg.get("val_embeddings_pt", "")))
    if not str(train_embeddings_pt):
        raise ValueError("Transformer mode requires 'train_embeddings_pt'.")
    if not str(val_embeddings_pt):
        raise ValueError("Transformer mode requires 'val_embeddings_pt'.")

    output_dir = Path(cfg.get("output_dir", "results/text_va_hf_transformer"))
    model_name = str(cfg.get("model_name", "michellejieli/emotion_text_classifier"))

    batch_size = int(cfg.get("batch_size", 16))
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    optimizer_name = str(cfg.get("optimizer", "adamw"))
    momentum = float(cfg.get("momentum", 0.9))
    adam_beta1 = float(cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(cfg.get("adam_beta2", 0.999))
    adam_eps = float(cfg.get("adam_eps", 1e-8))
    nesterov = bool(cfg.get("nesterov", False))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    seed = int(cfg.get("seed", 42))
    num_workers = int(cfg.get("num_workers", 0))
    raw_device = str(cfg.get("device", "auto")).lower()
    if raw_device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = raw_device
    limit_train = int(cfg.get("limit_train", 0))
    limit_val = int(cfg.get("limit_val", 0))
    selection_metric = str(cfg.get("selection_metric", "segment_ccc_mean")).lower()

    group_col = str(_cfg_get(cfg, ["group_col", "transformer_group_col"], "stream_name"))
    model_arch = str(_cfg_get(cfg, ["model_arch", "transformer_arch"], "transformer")).strip().lower()
    model_d_model = int(_cfg_get(cfg, ["model_d_model", "transformer_d_model"], 256))
    model_dropout = float(_cfg_get(cfg, ["model_dropout", "transformer_dropout"], 0.1))
    model_head_hidden_dim = int(_cfg_get(cfg, ["model_head_hidden_dim", "transformer_head_hidden_dim"], 0))
    model_head_dropout = float(_cfg_get(cfg, ["model_head_dropout", "transformer_head_dropout"], 0.1))

    transformer_nhead = int(_cfg_get(cfg, ["transformer_nhead"], 8))
    transformer_layers = int(_cfg_get(cfg, ["transformer_layers"], 2))
    transformer_ff_dim = int(_cfg_get(cfg, ["transformer_ff_dim"], 512))
    transformer_positional_encoding = bool(_cfg_get(cfg, ["transformer_positional_encoding"], True))
    transformer_gate_mode = _cfg_get(cfg, ["transformer_gate_mode"], None)
    transformer_max_seq_len = int(_cfg_get(cfg, ["transformer_max_seq_len"], 8192))

    window_size = int(_cfg_get(cfg, ["window_size", "transformer_window_size"], 0))
    window_stride = int(_cfg_get(cfg, ["window_stride", "transformer_window_stride"], 0))
    drop_overlapping_segments = bool(_cfg_get(cfg, ["drop_overlapping_segments"], False))

    mamba_layers = int(_cfg_get(cfg, ["mamba_layers", "transformer_mamba_layers"], transformer_layers))
    mamba_d_state = int(_cfg_get(cfg, ["mamba_d_state", "transformer_mamba_d_state"], 8))
    mamba_kernel_size = int(_cfg_get(cfg, ["mamba_kernel_size", "transformer_mamba_kernel_size"], 3))
    mamba_d_discr = int(_cfg_get(cfg, ["mamba_d_discr", "transformer_mamba_d_discr"], 0))

    frame_val_enabled = bool(cfg.get("frame_val_enabled", False))
    frame_val_gt_csv = Path(str(cfg.get("frame_val_gt_csv", "dataset/val_by_frame.csv")))
    frame_val_group_col = str(cfg.get("frame_val_group_col", "stream_name"))
    frame_val_pred_video_col = str(cfg.get("frame_val_pred_video_col", "video_name"))
    frame_val_gt_video_col = str(cfg.get("frame_val_gt_video_col", "video_name"))
    frame_val_normalize_video_name = bool(cfg.get("frame_val_normalize_video_name", True))
    frame_val_frame_col = str(cfg.get("frame_val_frame_col", "frame_idx"))
    frame_val_val_col = str(cfg.get("frame_val_val_col", "valence"))
    frame_val_aro_col = str(cfg.get("frame_val_aro_col", "arousal"))
    frame_val_filter_invalid_gt = bool(cfg.get("frame_val_filter_invalid_gt", True))
    frame_val_invalid_threshold = float(cfg.get("frame_val_invalid_threshold", -4.9))
    if selection_metric == "frame_ccc_mean" and not frame_val_enabled:
        raise ValueError("selection_metric='frame_ccc_mean' requires frame_val_enabled=true.")

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=True, indent=2, default=str)

    train_df, train_sequences, train_group_col_resolved, feature_dim, train_overlap_debug_df = load_transformer_split(
        csv_path=train_csv,
        embeddings_path=train_embeddings_pt,
        group_col=group_col,
        limit=limit_train,
        val_col=frame_val_val_col,
        aro_col=frame_val_aro_col,
        window_size=window_size,
        window_stride=window_stride,
        drop_overlapping_segments=drop_overlapping_segments,
    )
    val_df, val_sequences, val_group_col_resolved, val_feature_dim, val_overlap_debug_df = load_transformer_split(
        csv_path=val_csv,
        embeddings_path=val_embeddings_pt,
        group_col=group_col,
        limit=limit_val,
        val_col=frame_val_val_col,
        aro_col=frame_val_aro_col,
        window_size=window_size,
        window_stride=window_stride,
        drop_overlapping_segments=drop_overlapping_segments,
    )
    if feature_dim != val_feature_dim:
        raise ValueError(f"Train/val feature dims mismatch: {feature_dim} vs {val_feature_dim}")

    if drop_overlapping_segments:
        train_overlap_debug_path = output_dir / "train_overlap_filter.csv"
        val_overlap_debug_path = output_dir / "val_overlap_filter.csv"
        if train_overlap_debug_df is not None:
            train_overlap_debug_df.to_csv(train_overlap_debug_path, index=False)
            logging.info(f"Overlap filter report saved: {train_overlap_debug_path}")
        if val_overlap_debug_df is not None:
            val_overlap_debug_df.to_csv(val_overlap_debug_path, index=False)
            logging.info(f"Overlap filter report saved: {val_overlap_debug_path}")

    collate_fn = make_transformer_collate_fn()
    train_loader = DataLoader(
        TransformerSequenceDataset(train_sequences),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        TransformerSequenceDataset(val_sequences),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    frame_gt_df = None
    if frame_val_enabled:
        ensure_columns(val_df, ["start_frame", "end_frame"], str(val_csv))
        frame_gt_df = read_csv_with_fallback(frame_val_gt_csv)
        ensure_columns(
            frame_gt_df,
            [frame_val_gt_video_col, frame_val_frame_col, frame_val_val_col, frame_val_aro_col],
            str(frame_val_gt_csv),
        )
        logging.info(
            "Frame validation enabled | "
            f"group_col={frame_val_group_col} | gt_csv={frame_val_gt_csv}"
        )

    device = torch.device(device_name)
    if model_arch in {"transformer", "former"}:
        model = TransformerRegressor(
            input_dim=feature_dim,
            d_model=model_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=model_dropout,
            head_hidden_dim=model_head_hidden_dim,
            head_dropout=model_head_dropout,
            positional_encoding=transformer_positional_encoding,
            gate_mode=transformer_gate_mode,
            max_seq_len=transformer_max_seq_len,
        ).to(device)
    elif model_arch == "mamba":
        model = TextMambaRegressor(
            input_dim=feature_dim,
            d_model=model_d_model,
            num_layers=mamba_layers,
            dropout=model_dropout,
            head_hidden_dim=model_head_hidden_dim,
            head_dropout=model_head_dropout,
            mamba_d_state=mamba_d_state,
            mamba_kernel_size=mamba_kernel_size,
            mamba_d_discr=mamba_d_discr,
        ).to(device)
    else:
        raise ValueError("model_arch must be one of: transformer, mamba.")
    criterion = CCCLoss()
    optimizer = build_optimizer(
        name=optimizer_name,
        params=[p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        nesterov=nesterov,
    )

    params = count_parameters(model)
    if model_arch in {"transformer", "former"}:
        logging.info(
            f"Sequence model params: total={params['total']}, trainable={params['trainable']} | "
            f"arch={model_arch} | input_dim={feature_dim} | d_model={model_d_model} | nhead={transformer_nhead} | "
            f"layers={transformer_layers} | ff_dim={transformer_ff_dim} | "
            f"pos_enc={transformer_positional_encoding} | gate_mode={transformer_gate_mode}"
        )
    else:
        logging.info(
            f"Sequence model params: total={params['total']}, trainable={params['trainable']} | "
            f"arch={model_arch} | input_dim={feature_dim} | d_model={model_d_model} | "
            f"mamba_layers={mamba_layers} | mamba_d_state={mamba_d_state} | "
            f"mamba_kernel={mamba_kernel_size} | mamba_d_discr={mamba_d_discr}"
        )
    logging.info(
        f"Optimizer: {optimizer_name.lower()} | lr={lr} | wd={weight_decay} | "
        f"momentum={momentum} | nesterov={nesterov}"
    )
    logging.info(f"Selection metric: {selection_metric}")
    logging.info(
        f"Sequence grouping resolved | train_group_col={train_group_col_resolved} | "
        f"val_group_col={val_group_col_resolved} | train_sequences={len(train_sequences)} | val_sequences={len(val_sequences)} | "
        f"window_size={window_size} | window_stride={window_stride} | "
        f"drop_overlapping_segments={drop_overlapping_segments}"
    )

    best_score = float("-inf")
    no_improve_epochs = 0
    best_row: Dict[str, float] | None = None
    history: List[Dict[str, float]] = []
    best_ckpt_path = output_dir / "best_model.pt"
    frame_eval_error_count = 0
    frame_eval_last_error = ""
    val_targets_np = val_df[[frame_val_val_col, frame_val_aro_col]].to_numpy(dtype=np.float32)

    for epoch in range(1, epochs + 1):
        train_loss = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        segment_metrics, val_segment_preds = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            n_rows=len(val_df),
            targets_np=val_targets_np,
        )

        frame_metrics = {
            "frame_ccc_valence": float("nan"),
            "frame_ccc_arousal": float("nan"),
            "frame_ccc_mean": float("nan"),
            "frame_rows_matched": float("nan"),
        }
        if frame_val_enabled and frame_gt_df is not None:
            segment_pred_df = val_df.copy()
            segment_pred_df[frame_val_val_col] = val_segment_preds[:, 0]
            segment_pred_df[frame_val_aro_col] = val_segment_preds[:, 1]

            frame_expand_group_col = frame_val_group_col if frame_val_group_col in segment_pred_df.columns else "video_name"
            frame_pred_df = expand_segment_predictions_to_frames(
                df=segment_pred_df,
                group_col=frame_expand_group_col,
                start_col="start_frame",
                end_col="end_frame",
                val_col=frame_val_val_col,
                aro_col=frame_val_aro_col,
                output_video_col=frame_val_pred_video_col,
            )
            try:
                frame_metrics = evaluate_frame_ccc_from_dataframes(
                    pred_df=frame_pred_df,
                    gt_df=frame_gt_df,
                    pred_video_col=frame_val_pred_video_col,
                    gt_video_col=frame_val_gt_video_col,
                    frame_col=frame_val_frame_col,
                    val_col=frame_val_val_col,
                    aro_col=frame_val_aro_col,
                    normalize_video_name=frame_val_normalize_video_name,
                    filter_invalid_gt=frame_val_filter_invalid_gt,
                    invalid_threshold=frame_val_invalid_threshold,
                    merged_out_csv=None,
                )
            except Exception as exc:  # noqa: BLE001
                frame_eval_error_count += 1
                frame_eval_last_error = str(exc)
                logging.warning(
                    "Frame evaluation failed at epoch %d: %s. Marking frame metrics as NaN for this epoch.",
                    epoch,
                    frame_eval_last_error,
                )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": segment_metrics["val_loss"],
            "ccc_valence": segment_metrics["ccc_valence"],
            "ccc_arousal": segment_metrics["ccc_arousal"],
            "ccc_mean": segment_metrics["ccc_mean"],
            "frame_ccc_valence": frame_metrics["frame_ccc_valence"],
            "frame_ccc_arousal": frame_metrics["frame_ccc_arousal"],
            "frame_ccc_mean": frame_metrics["frame_ccc_mean"],
            "frame_rows_matched": frame_metrics["frame_rows_matched"],
        }
        history.append(row)

        if frame_val_enabled:
            logging.info(
                f"Epoch {epoch:02d} | train_loss={row['train_loss']:.5f} | "
                f"val_loss={row['val_loss']:.5f} | seg_v={row['ccc_valence']:.5f} | "
                f"seg_a={row['ccc_arousal']:.5f} | seg_mean={row['ccc_mean']:.5f} | "
                f"frm_v={row['frame_ccc_valence']:.5f} | frm_a={row['frame_ccc_arousal']:.5f} | "
                f"frm_mean={row['frame_ccc_mean']:.5f}"
            )
        else:
            logging.info(
                f"Epoch {epoch:02d} | train_loss={row['train_loss']:.5f} | "
                f"val_loss={row['val_loss']:.5f} | ccc_v={row['ccc_valence']:.5f} | "
                f"ccc_a={row['ccc_arousal']:.5f} | ccc_mean={row['ccc_mean']:.5f}"
            )

        if selection_metric == "frame_ccc_mean":
            current_score = float(row["frame_ccc_mean"])
        elif selection_metric == "segment_ccc_mean":
            current_score = float(row["ccc_mean"])
        elif selection_metric == "neg_val_loss":
            current_score = -float(row["val_loss"])
        else:
            raise ValueError(
                "Unsupported selection_metric. Use 'segment_ccc_mean', 'frame_ccc_mean', or 'neg_val_loss'."
            )

        if not np.isfinite(current_score):
            current_score = float("-inf")

        improved = (best_row is None) or (current_score > (best_score + early_stopping_min_delta))
        if improved:
            best_score = current_score
            best_row = dict(row)
            no_improve_epochs = 0
            checkpoint_cfg = {
                "model_name": model_name,
                "input_dim": feature_dim,
                "model_arch": model_arch,
                "model_d_model": model_d_model,
                "model_dropout": model_dropout,
                "model_head_hidden_dim": model_head_hidden_dim,
                "model_head_dropout": model_head_dropout,
                "group_col": group_col,
                "transformer_nhead": transformer_nhead,
                "transformer_layers": transformer_layers,
                "transformer_ff_dim": transformer_ff_dim,
                "transformer_positional_encoding": transformer_positional_encoding,
                "transformer_gate_mode": transformer_gate_mode,
                "transformer_max_seq_len": transformer_max_seq_len,
                "mamba_layers": mamba_layers,
                "mamba_d_state": mamba_d_state,
                "mamba_kernel_size": mamba_kernel_size,
                "mamba_d_discr": mamba_d_discr,
                "window_size": window_size,
                "window_stride": window_stride,
                "drop_overlapping_segments": drop_overlapping_segments,
            }
            save_checkpoint(model=model, output_path=best_ckpt_path, config_payload=checkpoint_cfg)
            logging.info(
                f"Saved best checkpoint to {best_ckpt_path} ({selection_metric}={best_score:.5f})"
            )
        else:
            no_improve_epochs += 1

        if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
            logging.info(
                "Early stopping: "
                f"no improvement in {selection_metric} for {no_improve_epochs} epoch(s) "
                f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})."
            )
            break

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    logging.info(f"Training history saved to {history_path}")

    return {
        "model_type": "transformer",
        "best_checkpoint": str(best_ckpt_path),
        "history_csv": str(history_path),
        "best_ccc_mean": float(best_row["ccc_mean"]) if best_row is not None else float("nan"),
        "best_ccc_valence": float(best_row["ccc_valence"]) if best_row is not None else float("nan"),
        "best_ccc_arousal": float(best_row["ccc_arousal"]) if best_row is not None else float("nan"),
        "best_frame_ccc_mean": float(best_row["frame_ccc_mean"]) if best_row is not None else float("nan"),
        "best_frame_ccc_valence": float(best_row["frame_ccc_valence"]) if best_row is not None else float("nan"),
        "best_frame_ccc_arousal": float(best_row["frame_ccc_arousal"]) if best_row is not None else float("nan"),
        "best_frame_rows_matched": float(best_row["frame_rows_matched"]) if best_row is not None else float("nan"),
        "best_val_loss": float(best_row["val_loss"]) if best_row is not None else float("nan"),
        "best_epoch": int(best_row["epoch"]) if best_row is not None else -1,
        "selection_metric": selection_metric,
        "best_selection_score": float(best_score),
        "optimizer": optimizer_name.lower(),
        "model_arch": model_arch,
        "model_d_model": model_d_model,
        "model_dropout": model_dropout,
        "model_head_hidden_dim": model_head_hidden_dim,
        "model_head_dropout": model_head_dropout,
        "group_col": group_col,
        "transformer_nhead": transformer_nhead,
        "transformer_layers": transformer_layers,
        "transformer_ff_dim": transformer_ff_dim,
        "transformer_positional_encoding": transformer_positional_encoding,
        "transformer_gate_mode": transformer_gate_mode,
        "transformer_max_seq_len": transformer_max_seq_len,
        "mamba_layers": mamba_layers,
        "mamba_d_state": mamba_d_state,
        "mamba_kernel_size": mamba_kernel_size,
        "mamba_d_discr": mamba_d_discr,
        "window_size": window_size,
        "window_stride": window_stride,
        "drop_overlapping_segments": drop_overlapping_segments,
        # Backward-compatible aliases for older summaries/search configs.
        "transformer_arch": model_arch,
        "transformer_d_model": model_d_model,
        "transformer_dropout": model_dropout,
        "transformer_head_hidden_dim": model_head_hidden_dim,
        "transformer_head_dropout": model_head_dropout,
        "transformer_mamba_layers": mamba_layers,
        "transformer_mamba_d_state": mamba_d_state,
        "transformer_mamba_kernel_size": mamba_kernel_size,
        "transformer_mamba_d_discr": mamba_d_discr,
        "transformer_window_size": window_size,
        "transformer_window_stride": window_stride,
        "frame_eval_error_count": int(frame_eval_error_count),
        "frame_eval_last_error": frame_eval_last_error,
        "output_dir": str(output_dir),
        "stopped_early": bool(early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience),
    }

