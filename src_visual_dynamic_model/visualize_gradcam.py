import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import functional as F

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import IndexConfig, build_index
from model import VisualDynamicModel
from train import TrainConfig, _build_test_index, _load_checkpoint_compat, _infer_feature_dim, _frame_mask, load_config


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Grad-CAM style visualization for VA prediction using end-to-end chain: "
            "GRADA image encoder -> temporal VA model."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--target-frame", type=int, default=0, help="Original frame id (e.g., 123). 0 means middle frame.")
    parser.add_argument("--target-output", type=str, default="valence", choices=["valence", "arousal", "both"])
    parser.add_argument(
        "--uniform-targets",
        type=int,
        default=0,
        help="If >0, pick this many uniformly sampled target frames across the whole video.",
    )
    parser.add_argument("--window-length", type=int, default=0, help="0 -> min(config.window_length, 64)")
    parser.add_argument("--faces-dir", type=str, default="01")
    parser.add_argument("--max-output-frames", type=int, default=24)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--grada-variant", type=str, default="")
    parser.add_argument("--grada-weights", type=str, default="")
    parser.add_argument(
        "--no-summary-pdf",
        action="store_true",
        help="Disable creation of combined PDF report(s).",
    )
    parser.add_argument(
        "--pdf-max-cols",
        type=int,
        default=10,
        help="Maximum target frames per page in summary PDF.",
    )
    return parser.parse_args()


def _get_split_index(cfg: TrainConfig, split: str, cache_dir: Optional[Path]) -> pd.DataFrame:
    if split == "test":
        return _build_test_index(cfg.test_list_path)
    cache_path = None
    if cache_dir is not None:
        suffix = "train" if split == "train" else "val"
        cache_path = cache_dir / f"affwild2_{suffix}.csv"
    return build_index(
        IndexConfig(
            annotations_root=cfg.annotations_root,
            split=split,
            cache_path=cache_path,
            filter_invalid=False,
        )
    )


def _select_window(frames: List[int], target_frame: int, window_length: int) -> Tuple[List[int], int]:
    n = len(frames)
    if n == 0:
        raise ValueError("Empty frame list for selected video.")

    if target_frame > 0:
        tgt_pos = int(np.argmin(np.abs(np.asarray(frames, dtype=np.int64) - int(target_frame))))
    else:
        tgt_pos = n // 2

    w = int(window_length)
    if w <= 0:
        w = min(n, 64)
    if w > n:
        w = n

    start = max(0, tgt_pos - (w // 2))
    start = min(start, max(0, n - w))
    end = start + w
    win = frames[start:end]
    tgt_in_window = tgt_pos - start
    return win, int(tgt_in_window)


def _resolve_frame_path(frames_root: Path, split: str, video: str, faces_dir: str, frame_idx: int) -> Optional[Path]:
    p = frames_root / split / str(video) / str(faces_dir) / f"{int(frame_idx):05d}.jpg"
    return p if p.is_file() else None


def _preprocess_face(img: Image.Image, expected_size: int) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int, int, int]]:
    old_size = img.size
    ratio = float(expected_size) / max(old_size)
    new_size = tuple(max(1, int(x * ratio)) for x in old_size)
    img = img.resize(new_size, Resampling.BILINEAR)
    padded = Image.new("RGB", (expected_size, expected_size))
    x0 = (expected_size - new_size[0]) // 2
    y0 = (expected_size - new_size[1]) // 2
    padded.paste(img, (x0, y0))

    t = F.pil_to_tensor(padded)
    t = F.convert_image_dtype(t, torch.float)
    t = F.normalize(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    vis = np.asarray(padded, dtype=np.uint8)
    return t, vis, (int(x0), int(y0), int(new_size[0]), int(new_size[1]))


def _crop_to_bbox(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int], margin: int = 2) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x0, y0, bw, bh = bbox
    x1 = x0 + bw
    y1 = y0 + bh
    m = max(0, int(margin))
    x0 = max(0, int(x0) - m)
    y0 = max(0, int(y0) - m)
    x1 = min(w, int(x1) + m)
    y1 = min(h, int(y1) + m)
    if x1 <= x0 or y1 <= y0:
        return image_rgb
    return image_rgb[y0:y1, x0:x1]


def _find_last_conv(model: nn.Module) -> Tuple[str, nn.Conv2d]:
    convs = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layer found in GRADA backbone. Cannot compute spatial Grad-CAM.")
    return convs[-1]


def _colormap(cam_01: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm

        mapped = cm.get_cmap("jet")(cam_01)[..., :3]
        return (mapped * 255.0).astype(np.uint8)
    except Exception:
        # Fallback without matplotlib: black->red gradient.
        red = (cam_01 * 255.0).astype(np.uint8)
        zeros = np.zeros_like(red, dtype=np.uint8)
        return np.stack([red, zeros, zeros], axis=-1)


def _overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    cam_img = Image.fromarray(np.clip(cam * 255.0, 0, 255).astype(np.uint8), mode="L").resize((w, h), Resampling.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = _colormap(cam_resized).astype(np.float32)
    base = image_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _load_grada_model(cfg: TrainConfig, args) -> Tuple[nn.Module, int]:
    raw_repo = str(getattr(cfg, "grada_repo_path", "")).strip()
    if not raw_repo:
        raise ValueError("grada_repo_path is empty in config. Set it in config or pass a config that contains it.")
    grada_repo = Path(raw_repo)
    if not grada_repo.is_dir():
        raise FileNotFoundError(f"grada_repo_path not found: {grada_repo}")
    if str(grada_repo) not in sys.path:
        sys.path.insert(0, str(grada_repo))
    from grada_emotion.models import load_model

    variant = str(args.grada_variant).strip() if str(args.grada_variant).strip() else str(getattr(cfg, "grada_model_variant", "b1"))
    weights = str(args.grada_weights).strip() if str(args.grada_weights).strip() else str(getattr(cfg, "grada_weights_path", "")).strip()
    weights = weights if weights else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        variant=str(variant).lower(),
        device=device,
        weights_path=weights,
    )
    model = model.to(device)
    model.eval()
    expected_size = 240 if str(variant).lower() == "b1" else 380
    return model, expected_size


def _build_temporal_model(cfg: TrainConfig, checkpoint_path: Path, device: torch.device) -> VisualDynamicModel:
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
    _load_checkpoint_compat(model, checkpoint_path=checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    return model


def _sample_uniform_frames(frames: List[int], n_targets: int) -> List[int]:
    n = len(frames)
    if n == 0:
        return []
    k = int(n_targets)
    if k <= 0:
        return []
    if k >= n:
        return [int(x) for x in frames]

    lin = np.linspace(0, n - 1, num=k)
    idx = np.clip(np.round(lin).astype(np.int64), 0, n - 1)

    uniq = []
    seen = set()
    for i in idx.tolist():
        ii = int(i)
        if ii not in seen:
            seen.add(ii)
            uniq.append(ii)

    if len(uniq) < k:
        for ii in range(n):
            if ii in seen:
                continue
            uniq.append(ii)
            if len(uniq) >= k:
                break

    uniq = sorted(uniq[:k])
    return [int(frames[i]) for i in uniq]


def _run_single_target(
    cfg: TrainConfig,
    args,
    checkpoint_path: Path,
    grada_model: nn.Module,
    expected_size: int,
    temporal_model: VisualDynamicModel,
    conv_name: str,
    conv_layer: nn.Conv2d,
    frames: List[int],
    frame_to_target: Dict[int, float],
    target_frame: int,
    target_output: str,
    out_dir: Path,
) -> Dict[str, object]:
    win_len = int(args.window_length) if int(args.window_length) > 0 else min(int(cfg.window_length), 64)
    win_frames, target_pos = _select_window(frames=frames, target_frame=int(target_frame), window_length=win_len)

    captures: Dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _inp, out):
        captures["acts"] = out if isinstance(out, torch.Tensor) else out[0]

    hook_handle = conv_layer.register_forward_hook(_forward_hook)

    try:
        frame_tensors: List[torch.Tensor] = []
        vis_images: List[np.ndarray] = []
        vis_bboxes: List[Tuple[int, int, int, int]] = []
        resolved_paths: List[str] = []
        for frame_idx in win_frames:
            p = _resolve_frame_path(cfg.frames_root, args.split, args.video, args.faces_dir, int(frame_idx))
            if p is None:
                img = Image.new("RGB", (expected_size, expected_size))
                resolved_paths.append("")
            else:
                img = Image.open(p).convert("RGB")
                resolved_paths.append(str(p))
            t, vis, bbox = _preprocess_face(img, expected_size=expected_size)
            frame_tensors.append(t)
            vis_images.append(vis)
            vis_bboxes.append(bbox)

        device = next(grada_model.parameters()).device
        batch = torch.stack(frame_tensors, dim=0).to(device)
        with torch.enable_grad():
            x = grada_model.model(batch)
            x = grada_model.dropout_after_pretrained(x)
            x = grada_model.embeddings_layer(x)
            x = grada_model.batchnorm(x)
            emb = grada_model.activation_embeddings(x)  # [T, D]

            static_preds = None
            if hasattr(grada_model, "regression"):
                try:
                    static_preds = grada_model.regression(emb)  # [T, 2]
                except Exception:
                    static_preds = None

            emb_seq = emb.unsqueeze(0)  # [1, T, D]
            lengths = torch.tensor([emb_seq.shape[1]], dtype=torch.long, device=device)
            mask = _frame_mask(lengths, emb_seq.shape[1], device)
            preds = temporal_model(emb_seq, mask=mask)  # [1, T, 2]

            out_idx = 0 if str(target_output).lower() == "valence" else 1
            dynamic_score = preds[0, int(target_pos), out_idx]
            has_static = bool(static_preds is not None and static_preds.ndim == 2 and static_preds.shape[1] > out_idx)
            static_score = static_preds[int(target_pos), out_idx] if has_static else dynamic_score.detach()

            conv_acts = captures.get("acts")
            if conv_acts is None:
                raise RuntimeError("Failed to capture conv activations for Grad-CAM.")

            grad_emb, grad_conv_dynamic = torch.autograd.grad(
                outputs=dynamic_score,
                inputs=[emb, conv_acts],
                retain_graph=bool(has_static),
                create_graph=False,
                allow_unused=False,
            )

            if has_static:
                grad_conv_static = torch.autograd.grad(
                    outputs=static_score,
                    inputs=conv_acts,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad_conv_static is None:
                    grad_conv_static = grad_conv_dynamic.detach()
            else:
                grad_conv_static = grad_conv_dynamic.detach()
    finally:
        hook_handle.remove()

    temporal_importance = grad_emb.detach().abs().mean(dim=1)  # [T]
    temporal_importance = temporal_importance / (temporal_importance.max() + 1e-8)
    temporal_importance_np = temporal_importance.detach().cpu().numpy().astype(np.float32)

    acts = conv_acts.detach()
    grads_dynamic = grad_conv_dynamic.detach()
    grads_static = grad_conv_static.detach()

    alpha_dyn = grads_dynamic.mean(dim=(2, 3), keepdim=True)
    cams_dynamic = torch.relu((alpha_dyn * acts).sum(dim=1))
    cams_dynamic = cams_dynamic / (cams_dynamic.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cams_dynamic_np = cams_dynamic.cpu().numpy().astype(np.float32)

    alpha_static = grads_static.mean(dim=(2, 3), keepdim=True)
    cams_static = torch.relu((alpha_static * acts).sum(dim=1))
    cams_static = cams_static / (cams_static.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cams_static_np = cams_static.cpu().numpy().astype(np.float32)

    cams_dynamic_weighted_np = np.clip(cams_dynamic_np * temporal_importance_np[:, None, None], 0.0, 1.0)

    pred_np = preds.detach().cpu().numpy()[0]
    val_np = pred_np[:, 0].astype(np.float32)
    aro_np = pred_np[:, 1].astype(np.float32)

    if static_preds is not None:
        static_np = static_preds.detach().cpu().numpy()
        static_val_np = static_np[:, 0].astype(np.float32)
        static_aro_np = static_np[:, 1].astype(np.float32) if static_np.shape[1] > 1 else np.full((len(win_frames),), np.nan, dtype=np.float32)
    else:
        static_val_np = np.full((len(win_frames),), np.nan, dtype=np.float32)
        static_aro_np = np.full((len(win_frames),), np.nan, dtype=np.float32)

    frames_out = out_dir / "frames"
    frames_weighted_out = out_dir / "frames_temporal_weighted"
    frames_out.mkdir(parents=True, exist_ok=True)
    frames_weighted_out.mkdir(parents=True, exist_ok=True)

    num_frames = len(win_frames)
    max_out = max(1, int(args.max_output_frames))
    if num_frames <= max_out:
        selected = list(range(num_frames))
    else:
        top = np.argsort(-temporal_importance_np)[: max(0, max_out - 1)].tolist()
        selected = sorted(set([int(target_pos)] + [int(i) for i in top]))

    for rank, i in enumerate(selected):
        overlay_static = _overlay_heatmap(vis_images[i], cams_static_np[i], alpha=0.45)
        overlay_dynamic = _overlay_heatmap(vis_images[i], cams_dynamic_weighted_np[i], alpha=0.45)
        frame_id = int(win_frames[i])
        imp = float(temporal_importance_np[i])
        suffix = "target" if i == int(target_pos) else "ctx"
        out_path = frames_out / f"{rank:03d}_{suffix}_f{frame_id:05d}_p{i:03d}_imp{imp:.3f}.png"
        out_path_weighted = frames_weighted_out / f"{rank:03d}_{suffix}_f{frame_id:05d}_p{i:03d}_imp{imp:.3f}.png"
        Image.fromarray(overlay_static).save(out_path)
        Image.fromarray(overlay_dynamic).save(out_path_weighted)

    table = pd.DataFrame(
        {
            "pos_in_window": list(range(num_frames)),
            "frame": [int(x) for x in win_frames],
            "pred_valence": val_np.tolist(),
            "pred_arousal": aro_np.tolist(),
            "static_pred_valence": static_val_np.tolist(),
            "static_pred_arousal": static_aro_np.tolist(),
            "temporal_importance": temporal_importance_np.tolist(),
            "static_cam_mean": cams_static_np.mean(axis=(1, 2)).tolist(),
            "dynamic_cam_mean": cams_dynamic_np.mean(axis=(1, 2)).tolist(),
            "joint_cam_mean": cams_dynamic_weighted_np.mean(axis=(1, 2)).tolist(),
            "is_target": [1 if i == int(target_pos) else 0 for i in range(num_frames)],
            "frame_path": resolved_paths,
        }
    )
    table.to_csv(out_dir / "frame_scores.csv", index=False)

    summary = {
        "split": str(args.split),
        "video": str(args.video),
        "checkpoint": str(checkpoint_path),
        "target_output": str(target_output),
        "target_frame_arg": int(target_frame),
        "target_pos_in_window": int(target_pos),
        "target_frame_effective": int(win_frames[int(target_pos)]),
        "target_score": float(frame_to_target.get(int(win_frames[int(target_pos)]), np.nan)),
        "predicted_static_score": float(static_score.detach().cpu().item()) if has_static else float("nan"),
        "predicted_dynamic_score": float(dynamic_score.detach().cpu().item()),
        "window_length": int(num_frames),
        "conv_layer": str(conv_name),
        "selected_saved_frames": int(len(selected)),
        "joint_map_formula": "dynamic_weighted_cam_t = temporal_importance_t * dynamic_cam_t",
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _find_target_overlay(run_dir: Path, weighted: bool) -> Optional[Path]:
    sub = "frames_temporal_weighted" if weighted else "frames"
    p = Path(run_dir) / sub
    if not p.is_dir():
        return None
    files = sorted(p.glob("*_target_*.png"))
    if files:
        return files[0]
    files = sorted(p.glob("*.png"))
    return files[0] if files else None


def _build_summary_pdf(session_dir: Path, output_name: str, rows: List[Dict[str, object]], max_cols: int = 10) -> Optional[Path]:
    if len(rows) == 0:
        return None
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return None

    out_pdf = session_dir / f"summary_{output_name}.pdf"
    rows_sorted = sorted(rows, key=lambda r: int(r.get("target_frame_effective", 0)))
    cols_per_page = max(1, int(max_cols))

    with PdfPages(out_pdf) as pdf:
        for start in range(0, len(rows_sorted), cols_per_page):
            chunk = rows_sorted[start : start + cols_per_page]
            n = len(chunk)
            # Force zero horizontal gaps: first column starts at x=0, last ends at x=1.
            title_band = 0.12
            bottom_band = 0.00
            row_gap = 0.01
            row_h = (1.0 - title_band - bottom_band - row_gap) / 2.0

            # Keep frames un-stretched and still remove column gaps:
            # make each subplot cell square in physical size.
            fig_h = 3.9
            fig_w = max(1.0, float(n) * fig_h * row_h)
            fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h), squeeze=False)
            if n == 1:
                axes = np.asarray(axes).reshape(2, 1)

            for j in range(n):
                x0 = float(j) / float(n)
                w = (1.0 - x0) if j == (n - 1) else (1.0 / float(n))
                axes[0, j].set_position([x0, bottom_band + row_h + row_gap, w, row_h])
                axes[1, j].set_position([x0, bottom_band, w, row_h])
                axes[0, j].set_facecolor("black")
                axes[1, j].set_facecolor("black")

            for j, row in enumerate(chunk):
                run_dir = Path(str(row.get("output_dir", "")))
                target_frame = int(row.get("target_frame_effective", -1))
                target_score = float(row.get("target_score", np.nan))
                pred_static = float(row.get("predicted_static_score", np.nan))
                pred_dynamic = float(row.get("predicted_dynamic_score", np.nan))

                static_img_path = _find_target_overlay(run_dir, weighted=False)
                dyn_img_path = _find_target_overlay(run_dir, weighted=True)

                if static_img_path is not None and static_img_path.is_file():
                    axes[0, j].imshow(np.asarray(Image.open(static_img_path).convert("RGB")))
                else:
                    axes[0, j].imshow(np.zeros((64, 64, 3), dtype=np.uint8))
                axes[0, j].set_title(
                    f"F={target_frame}  T={target_score:.3f}\nS={pred_static:.3f}  D={pred_dynamic:.3f}",
                    fontsize=9,
                    pad=2,
                )
                axes[0, j].axis("off")

                if dyn_img_path is not None and dyn_img_path.is_file():
                    axes[1, j].imshow(np.asarray(Image.open(dyn_img_path).convert("RGB")))
                else:
                    axes[1, j].imshow(np.zeros((64, 64, 3), dtype=np.uint8))
                axes[1, j].axis("off")

            fig.suptitle(
                f"{output_name.upper()} | F=Target Frame, T=Target Score, S=Predicted Static Score, D=Predicted Dynamic Score | Top=Static Model Attention, Bottom=Dynamic Model Attention",
                fontsize=10,
                y=0.995,
            )
            pdf.savefig(fig)
            plt.close(fig)

    return out_pdf


def main():
    args = _parse_args()
    cfg = load_config(Path(args.config))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cache_dir = cfg.cache_dir if cfg.cache_dir is not None else (cfg.output_dir / "cache")
    index_df = _get_split_index(cfg=cfg, split=args.split, cache_dir=cache_dir)
    g = index_df[index_df["video"].astype(str) == str(args.video)].sort_values("frame")
    if len(g) == 0:
        raise ValueError(f"Video '{args.video}' not found in split '{args.split}'.")
    frames = g["frame"].astype(int).tolist()

    grada_model, expected_size = _load_grada_model(cfg, args)
    device = next(grada_model.parameters()).device
    temporal_model = _build_temporal_model(cfg=cfg, checkpoint_path=checkpoint_path, device=device)
    conv_name, conv_layer = _find_last_conv(grada_model.model)
    out_root = Path(args.output_dir) if str(args.output_dir).strip() else (cfg.output_dir / "gradcam")
    session_dir = out_root / f"{args.split}_{args.video}_{datetime_safe()}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if int(args.uniform_targets) > 0:
        target_frames = _sample_uniform_frames(frames=frames, n_targets=int(args.uniform_targets))
        if not target_frames:
            raise RuntimeError("No target frames selected for uniform mode.")
    else:
        if int(args.target_frame) > 0:
            target_frames = [int(args.target_frame)]
        else:
            target_frames = [int(frames[len(frames) // 2])]

    outputs = ["valence", "arousal"] if str(args.target_output).lower() == "both" else [str(args.target_output).lower()]
    run_summaries: List[Dict[str, object]] = []
    total_runs = len(target_frames) * len(outputs)
    run_idx = 0

    for target_frame in target_frames:
        for out_name in outputs:
            frame_to_target = {}
            col = "valence" if out_name == "valence" else "arousal"
            if col in g.columns:
                for _, row in g.iterrows():
                    try:
                        frame_to_target[int(row["frame"])] = float(row[col])
                    except Exception:
                        frame_to_target[int(row["frame"])] = float("nan")
            run_idx += 1
            subdir = session_dir / f"target_f{int(target_frame):05d}_{out_name}"
            summary = _run_single_target(
                cfg=cfg,
                args=args,
                checkpoint_path=checkpoint_path,
                grada_model=grada_model,
                expected_size=expected_size,
                temporal_model=temporal_model,
                conv_name=conv_name,
                conv_layer=conv_layer,
                frames=frames,
                frame_to_target=frame_to_target,
                target_frame=int(target_frame),
                target_output=out_name,
                out_dir=subdir,
            )
            run_summaries.append(summary)
            print(
                f"[INFO] Done {run_idx}/{total_runs}: frame={summary['target_frame_effective']} "
                f"output={summary['target_output']} target={summary['target_score']:.6f} "
                f"static={summary['predicted_static_score']:.6f} dynamic={summary['predicted_dynamic_score']:.6f}"
            )

    pd.DataFrame(run_summaries).to_csv(session_dir / "runs_summary.csv", index=False)
    pdf_paths: Dict[str, str] = {}
    if not bool(args.no_summary_pdf):
        for out_name in outputs:
            out_rows = [r for r in run_summaries if str(r.get("target_output", "")).lower() == out_name]
            pdf_path = _build_summary_pdf(
                session_dir=session_dir,
                output_name=out_name,
                rows=out_rows,
                max_cols=int(args.pdf_max_cols),
            )
            if pdf_path is not None:
                pdf_paths[out_name] = str(pdf_path)

    session_summary = {
        "split": str(args.split),
        "video": str(args.video),
        "checkpoint": str(checkpoint_path),
        "uniform_targets": int(args.uniform_targets),
        "target_frame_arg": int(args.target_frame),
        "target_output_arg": str(args.target_output),
        "num_runs": int(len(run_summaries)),
        "targets_used": [int(x) for x in target_frames],
        "outputs_used": outputs,
        "summary_pdfs": pdf_paths,
        "conv_layer": str(conv_name),
        "output_dir": str(session_dir),
    }
    (session_dir / "summary.json").write_text(json.dumps(session_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Grad-CAM session saved to: {session_dir}")
    print(f"[INFO] Conv layer: {conv_name}")
    print(f"[INFO] Runs: {len(run_summaries)} | targets={len(target_frames)} | outputs={outputs}")
    if pdf_paths:
        for k, v in pdf_paths.items():
            print(f"[DONE] PDF ({k}): {v}")


def datetime_safe() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
