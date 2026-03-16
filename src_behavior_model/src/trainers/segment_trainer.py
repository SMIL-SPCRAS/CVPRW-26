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
from transformers import AutoTokenizer

from ..datasets.text_data import (
    TextRegressionDataset,
    add_stream_name_column,
    ensure_columns,
    make_collate_fn,
    read_csv_with_fallback,
)
from ..datasets.frame_eval import evaluate_frame_ccc_from_dataframes
from ..datasets.frame_expand import expand_segment_predictions_to_frames
from ..core.losses_metrics import CCCLoss, compute_va_ccc
from ..models.segment_model import TextVARegressor, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit(df, limit: int):
    if limit and limit > 0:
        return df.head(limit).copy()
    return df


def build_loaders(
    tokenizer,
    train_csv: Path,
    val_csv: Path,
    max_length: int,
    batch_size: int,
    num_workers: int,
    limit_train: int,
    limit_val: int,
):
    train_df = read_csv_with_fallback(train_csv)
    val_df = read_csv_with_fallback(val_csv)

    ensure_columns(train_df, ["video_name", "text", "valence", "arousal"], str(train_csv))
    ensure_columns(val_df, ["video_name", "text", "valence", "arousal"], str(val_csv))

    if train_df[["valence", "arousal"]].isna().any().any():
        raise ValueError("NaN detected in train valence/arousal.")
    if val_df[["valence", "arousal"]].isna().any().any():
        raise ValueError("NaN detected in val valence/arousal.")

    train_df = maybe_limit(train_df, limit_train)
    val_df = maybe_limit(val_df, limit_val)
    train_df = add_stream_name_column(train_df)
    val_df = add_stream_name_column(val_df)

    collate_fn = make_collate_fn(tokenizer)
    train_ds = TextRegressionDataset(train_df, tokenizer=tokenizer, max_length=max_length, has_targets=True)
    val_ds = TextRegressionDataset(val_df, tokenizer=tokenizer, max_length=max_length, has_targets=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, val_df.reset_index(drop=True).copy()


def step_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    criterion: CCCLoss,
) -> torch.Tensor:
    loss_val = criterion(preds[:, 0], labels[:, 0])
    loss_aro = criterion(preds[:, 1], labels[:, 1])
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

    pbar = tqdm(loader, total=len(loader), desc=f"Train {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = step_loss(preds, labels, criterion)

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
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    losses: List[float] = []
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    pbar = tqdm(loader, total=len(loader), desc=f"Val   {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = step_loss(preds, labels, criterion)

        losses.append(float(loss.item()))
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    if not all_preds:
        metrics = {
            "val_loss": float("nan"),
            "ccc_valence": float("nan"),
            "ccc_arousal": float("nan"),
            "ccc_mean": float("nan"),
        }
        return metrics, np.empty((0, 2), dtype=np.float32)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_va_ccc(preds, labels)
    metrics["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, preds.numpy().astype(np.float32, copy=False)


def save_checkpoint(
    model: TextVARegressor,
    output_path: Path,
    model_name: str,
    dropout: float,
    head_hidden_dim: int,
    head_dropout: float,
    max_length: int,
) -> None:
    torch.save(
        {
            "model_name": model_name,
            "dropout": dropout,
            "head_hidden_dim": int(head_hidden_dim),
            "head_dropout": float(head_dropout),
            "max_length": max_length,
            "state_dict": model.state_dict(),
        },
        output_path,
    )


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


def run_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_csv = Path(cfg.get("train_csv", "dataset/train_segment_Qwen3-VL-4B-Instruct.csv"))
    val_csv = Path(cfg.get("val_csv", "dataset/val_segment_Qwen3-VL-4B-Instruct.csv"))
    output_dir = Path(cfg.get("output_dir", "results/text_va_hf"))
    model_name = str(cfg.get("model_name", "michellejieli/emotion_text_classifier"))

    max_length = int(cfg.get("max_length", 256))
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
    dropout = float(cfg.get("dropout", 0.1))
    head_hidden_dim = int(cfg.get("head_hidden_dim", 0))
    head_dropout = float(cfg.get("head_dropout", dropout))
    freeze_backbone = bool(cfg.get("freeze_backbone", True))
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
    if head_hidden_dim < 0:
        raise ValueError("head_hidden_dim must be >= 0.")

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=True, indent=2, default=str)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader, val_df = build_loaders(
        tokenizer=tokenizer,
        train_csv=train_csv,
        val_csv=val_csv,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        limit_train=limit_train,
        limit_val=limit_val,
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
    model = TextVARegressor(
        model_name=model_name,
        dropout=dropout,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)
    criterion = CCCLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        name=optimizer_name,
        params=trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        nesterov=nesterov,
    )

    params = count_parameters(model)
    logging.info(
        f"Model params: total={params['total']}, trainable={params['trainable']} "
        f"(freeze_backbone={freeze_backbone})."
    )
    logging.info(
        f"Optimizer: {optimizer_name.lower()} | lr={lr} | wd={weight_decay} | "
        f"momentum={momentum} | nesterov={nesterov}"
    )
    if head_hidden_dim > 0:
        logging.info(
            f"Head: mlp(hidden_dim={head_hidden_dim}, head_dropout={head_dropout}) | emb_dropout={dropout}"
        )
    else:
        logging.info(f"Head: linear | emb_dropout={dropout}")
    logging.info(f"Selection metric: {selection_metric}")

    best_score = float("-inf")
    no_improve_epochs = 0
    best_row: Dict[str, float] | None = None
    history: List[Dict[str, float]] = []
    best_ckpt_path = output_dir / "best_model.pt"
    frame_eval_error_count = 0
    frame_eval_last_error = ""

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
        )

        frame_metrics = {
            "frame_ccc_valence": float("nan"),
            "frame_ccc_arousal": float("nan"),
            "frame_ccc_mean": float("nan"),
            "frame_rows_matched": float("nan"),
        }
        if frame_val_enabled and frame_gt_df is not None and len(val_segment_preds) == len(val_df):
            segment_pred_df = val_df.copy()
            segment_pred_df[frame_val_val_col] = val_segment_preds[:, 0]
            segment_pred_df[frame_val_aro_col] = val_segment_preds[:, 1]

            group_col = frame_val_group_col if frame_val_group_col in segment_pred_df.columns else "video_name"
            frame_pred_df = expand_segment_predictions_to_frames(
                df=segment_pred_df,
                group_col=group_col,
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
                    "Frame evaluation failed at epoch %d: %s. "
                    "Marking frame metrics as NaN for this epoch.",
                    epoch,
                    frame_eval_last_error,
                )
        elif frame_val_enabled:
            frame_eval_error_count += 1
            frame_eval_last_error = (
                "Frame validation mismatch: number of validation predictions "
                f"({len(val_segment_preds)}) != number of validation rows ({len(val_df)})."
            )
            logging.warning(
                "%s Marking frame metrics as NaN for this epoch.",
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
            save_checkpoint(
                model=model,
                output_path=best_ckpt_path,
                model_name=model_name,
                dropout=dropout,
                head_hidden_dim=head_hidden_dim,
                head_dropout=head_dropout,
                max_length=max_length,
            )
            logging.info(
                f"Saved best checkpoint to {best_ckpt_path} "
                f"({selection_metric}={best_score:.5f})"
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
        "head_hidden_dim": int(head_hidden_dim),
        "head_dropout": float(head_dropout),
        "emb_dropout": float(dropout),
        "frame_eval_error_count": int(frame_eval_error_count),
        "frame_eval_last_error": frame_eval_last_error,
        "output_dir": str(output_dir),
        "stopped_early": bool(early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience),
    }
