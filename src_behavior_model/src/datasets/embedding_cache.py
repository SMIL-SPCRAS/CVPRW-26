from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from .text_data import TextRegressionDataset, ensure_columns, make_collate_fn, read_csv_with_fallback


def _normalize_path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def _sanitize_for_path(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    normalized = text.replace("\\", "/").replace("/", "__")
    return "".join(ch if ch in allowed else "_" for ch in normalized)


def _build_cache_subdir_name(model_name: str, max_length: int, pooling: str) -> str:
    model_tag = _sanitize_for_path(model_name)
    return f"{model_tag}__L{int(max_length)}__{pooling}"


def _build_qwen_cache_subdir_name(branch: str, pool: str) -> str:
    return f"qwen__{_sanitize_for_path(branch)}__{_sanitize_for_path(pool)}"


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-8)
    return summed / counts


@torch.no_grad()
def _extract_embeddings(
    csv_path: Path,
    model_name: str,
    max_length: int,
    pooling: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
    progress_desc: str,
) -> torch.Tensor:
    df = read_csv_with_fallback(csv_path)
    ensure_columns(df, ["video_name", "text"], str(csv_path))
    df = df.reset_index(drop=True).copy()
    df["text"] = df["text"].fillna("").astype(str)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    if device_name == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device_name
    device = torch.device(resolved_device)
    model.to(device)

    dataset = TextRegressionDataset(df=df, tokenizer=tokenizer, max_length=max_length, has_targets=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer),
        drop_last=False,
    )

    rows: list[torch.Tensor] = []
    pbar = tqdm(loader, total=len(loader), desc=progress_desc, leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if pooling == "mean":
            pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
        elif pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unsupported embeddings.pooling='{pooling}'. Use 'mean' or 'cls'.")

        rows.append(pooled.detach().cpu().float())

    if not rows:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(rows, dim=0)


def _cache_valid(cache_path: Path, expected_meta: Dict[str, Any]) -> Tuple[bool, str]:
    if not cache_path.exists():
        return False, "cache file does not exist"

    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to load cache: {exc}"

    if not isinstance(payload, dict):
        return False, "cache payload is not a dict"
    if "embeddings" not in payload or "meta" not in payload:
        return False, "cache payload missing keys: embeddings/meta"

    embeddings = payload["embeddings"]
    meta = payload["meta"]
    if not isinstance(embeddings, torch.Tensor):
        return False, "cache embeddings is not a tensor"
    if embeddings.ndim != 2:
        return False, "cache embeddings tensor must be 2D"
    if not isinstance(meta, dict):
        return False, "cache meta is not a dict"

    required_keys = list(expected_meta.keys()) + ["feature_dim"]
    missing = [k for k in required_keys if k not in meta]
    if missing:
        return False, f"cache meta missing keys: {missing}"

    for key, expected in expected_meta.items():
        if key not in meta:
            return False, f"cache meta missing expected key: {key}"
        if str(meta[key]) != str(expected):
            return False, f"cache meta mismatch for '{key}': got={meta[key]} expected={expected}"

    if int(meta["num_rows"]) != int(embeddings.shape[0]):
        return False, "cache num_rows does not match embeddings tensor rows"
    if int(meta["feature_dim"]) != int(embeddings.shape[1]):
        return False, "cache feature_dim does not match embeddings tensor width"

    return True, "ok"


def _save_cache(cache_path: Path, embeddings: torch.Tensor, meta: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": embeddings.cpu().float(),
        "meta": meta,
    }
    torch.save(payload, cache_path)


def _load_qwen_embeddings(
    csv_path: Path,
    pickle_path: Path,
    branch: str,
    pool: str,
) -> torch.Tensor:
    df = read_csv_with_fallback(csv_path)
    ensure_columns(df, ["video_name"], str(csv_path))
    video_names = df["video_name"].astype(str).tolist()

    with pickle_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Qwen embeddings format: expected dict in {pickle_path}")

    rows: list[torch.Tensor] = []
    missing_names: list[str] = []
    for video_name in video_names:
        item = payload.get(video_name)
        if item is None:
            missing_names.append(video_name)
            continue
        if not isinstance(item, dict) or "embeddings" not in item:
            raise ValueError(f"Invalid Qwen payload item for '{video_name}' in {pickle_path}")
        embeddings = item["embeddings"]
        if branch not in embeddings:
            raise ValueError(
                f"Qwen branch '{branch}' not found for '{video_name}' in {pickle_path}. "
                f"Available: {list(embeddings.keys())}"
            )
        branch_payload = embeddings[branch]
        if pool not in branch_payload:
            raise ValueError(
                f"Qwen pool '{pool}' not found under branch '{branch}' for '{video_name}' in {pickle_path}. "
                f"Available: {list(branch_payload.keys())}"
            )
        rows.append(torch.as_tensor(branch_payload[pool], dtype=torch.float32).view(1, -1))

    if missing_names:
        preview = ", ".join(missing_names[:5])
        raise ValueError(
            f"Missing {len(missing_names)} Qwen embeddings rows for {pickle_path}. "
            f"Examples: {preview}"
        )

    if not rows:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(rows, dim=0)


def prepare_embedding_cache_for_split(
    split: str,
    csv_path: Path,
    cache_path: Path,
    model_name: str,
    max_length: int,
    pooling: str,
    batch_size: int,
    num_workers: int,
    device: str,
    use_cache: bool,
    force_reextract: bool,
) -> Dict[str, Any]:
    df = read_csv_with_fallback(csv_path)
    ensure_columns(df, ["video_name", "text"], str(csv_path))
    num_rows = int(len(df))
    expected_meta = {
        "model_name": model_name,
        "max_length": int(max_length),
        "pooling": pooling,
        "source_csv": _normalize_path_text(csv_path),
        "num_rows": num_rows,
        "split": split,
        "cache_subdir": cache_path.parent.name,
    }

    if use_cache and not force_reextract:
        valid, reason = _cache_valid(cache_path, expected_meta)
        if valid:
            payload = torch.load(cache_path, map_location="cpu")
            feature_dim = int(payload["embeddings"].shape[1]) if payload["embeddings"].ndim == 2 else 0
            logging.info(f"Embeddings [{split}] cache hit: {cache_path}")
            return {
                "split": split,
                "status": "cache_hit",
                "cache_path": str(cache_path),
                "num_rows": num_rows,
                "feature_dim": feature_dim,
            }
        logging.warning(f"Embeddings [{split}] cache invalid: {reason}. Re-extracting.")

    logging.info(f"Embeddings [{split}] extracting from {csv_path}")
    embeddings = _extract_embeddings(
        csv_path=csv_path,
        model_name=model_name,
        max_length=max_length,
        pooling=pooling,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device,
        progress_desc=f"Emb {split}",
    )
    feature_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    meta = dict(expected_meta)
    meta["feature_dim"] = feature_dim
    _save_cache(cache_path=cache_path, embeddings=embeddings, meta=meta)
    logging.info(
        f"Embeddings [{split}] saved: {cache_path} | rows={num_rows} | dim={feature_dim}"
    )
    return {
        "split": split,
        "status": "extracted",
        "cache_path": str(cache_path),
        "num_rows": num_rows,
        "feature_dim": feature_dim,
    }


def prepare_embedding_caches(
    embeddings_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    predict_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    if not bool(embeddings_cfg.get("enabled", False)):
        return {}

    base_cache_dir = Path(str(embeddings_cfg.get("cache_dir", "features")))
    source = str(embeddings_cfg.get("source", "hf_text")).strip().lower()
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    device = str(train_cfg.get("device", "cuda"))
    use_cache = bool(embeddings_cfg.get("use_cache", True))
    force_reextract = bool(embeddings_cfg.get("force_reextract", False))
    meta_subdir = bool(embeddings_cfg.get("meta_subdir", True))

    prepare_train = bool(embeddings_cfg.get("prepare_train", True))
    prepare_val = bool(embeddings_cfg.get("prepare_val", True))
    prepare_test = bool(embeddings_cfg.get("prepare_test", False))
    split_to_csv: Dict[str, Path] = {}
    if prepare_train:
        split_to_csv["train"] = Path(str(train_cfg.get("train_csv", "dataset/train_segment_Qwen3-VL-4B-Instruct.csv")))
    if prepare_val:
        split_to_csv["val"] = Path(str(train_cfg.get("val_csv", "dataset/val_segment_Qwen3-VL-4B-Instruct.csv")))
    if prepare_test:
        split_to_csv["test"] = Path(str(predict_cfg.get("test_csv", "dataset/test_segment.csv")))

    if source == "hf_text":
        model_name = str(train_cfg.get("model_name", ""))
        if not model_name:
            raise ValueError("Cannot resolve model_name for embeddings extraction.")
        max_length = int(train_cfg.get("max_length", 512))
        pooling = str(embeddings_cfg.get("pooling", "mean")).lower()

        if meta_subdir:
            cache_dir = base_cache_dir / _build_cache_subdir_name(
                model_name=model_name,
                max_length=max_length,
                pooling=pooling,
            )
        else:
            cache_dir = base_cache_dir

        logging.info(
            "Embeddings extraction uses HF text encoder | "
            f"model={model_name} | max_length={max_length} | batch_size={batch_size} | "
            f"num_workers={num_workers} | device={device}"
        )
        logging.info(f"Embeddings cache dir: {cache_dir}")

        results: Dict[str, Dict[str, Any]] = {}
        for split, csv_path in split_to_csv.items():
            if not csv_path.exists():
                raise FileNotFoundError(f"Embeddings source CSV for split '{split}' not found: {csv_path}")
            cache_path = cache_dir / f"{split}_embeddings.pt"
            results[split] = prepare_embedding_cache_for_split(
                split=split,
                csv_path=csv_path,
                cache_path=cache_path,
                model_name=model_name,
                max_length=max_length,
                pooling=pooling,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                use_cache=use_cache,
                force_reextract=force_reextract,
            )
        return results

    if source == "qwen":
        branch = str(embeddings_cfg.get("qwen_branch", "multimodal")).strip().lower()
        pool = str(embeddings_cfg.get("qwen_pool", "mean")).strip().lower()
        if prepare_test:
            split_to_csv["test"] = Path(
                str(embeddings_cfg.get("qwen_test_csv", "dataset/test_segment_Qwen3-VL-4B-Instruct.csv"))
            )
        split_to_pickle = {
            "train": Path(str(embeddings_cfg.get("qwen_train_pickle", "dataset/train_segment_embeddings_Qwen3-VL-4B-Instruct.pkl"))),
            "val": Path(str(embeddings_cfg.get("qwen_val_pickle", "dataset/val_segment_embeddings_Qwen3-VL-4B-Instruct.pkl"))),
            "test": Path(str(embeddings_cfg.get("qwen_test_pickle", "dataset/test_segment_embeddings_Qwen3-VL-4B-Instruct.pkl"))),
        }

        cache_dir = base_cache_dir / _build_qwen_cache_subdir_name(branch=branch, pool=pool) if meta_subdir else base_cache_dir
        logging.info(
            "Embeddings extraction uses Qwen cached segment embeddings | "
            f"branch={branch} | pool={pool}"
        )
        logging.info(f"Embeddings cache dir: {cache_dir}")

        results: Dict[str, Dict[str, Any]] = {}
        for split, csv_path in split_to_csv.items():
            if not csv_path.exists():
                raise FileNotFoundError(f"Embeddings source CSV for split '{split}' not found: {csv_path}")
            pickle_path = split_to_pickle[split]
            if not pickle_path.exists():
                raise FileNotFoundError(f"Qwen pickle for split '{split}' not found: {pickle_path}")

            cache_path = cache_dir / f"{split}_embeddings.pt"
            num_rows = int(len(read_csv_with_fallback(csv_path)))
            expected_meta = {
                "source_type": "qwen",
                "branch": branch,
                "pool": pool,
                "source_csv": _normalize_path_text(csv_path),
                "source_pickle": _normalize_path_text(pickle_path),
                "num_rows": num_rows,
                "split": split,
                "cache_subdir": cache_path.parent.name,
            }

            if use_cache and not force_reextract:
                valid, reason = _cache_valid(cache_path, expected_meta)
                if valid:
                    payload = torch.load(cache_path, map_location="cpu")
                    feature_dim = int(payload["embeddings"].shape[1]) if payload["embeddings"].ndim == 2 else 0
                    logging.info(f"Embeddings [{split}] cache hit: {cache_path}")
                    results[split] = {
                        "split": split,
                        "status": "cache_hit",
                        "cache_path": str(cache_path),
                        "num_rows": num_rows,
                        "feature_dim": feature_dim,
                    }
                    continue
                logging.warning(f"Embeddings [{split}] cache invalid: {reason}. Rebuilding from Qwen pickle.")

            embeddings = _load_qwen_embeddings(
                csv_path=csv_path,
                pickle_path=pickle_path,
                branch=branch,
                pool=pool,
            )
            feature_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
            meta = dict(expected_meta)
            meta["feature_dim"] = feature_dim
            _save_cache(cache_path=cache_path, embeddings=embeddings, meta=meta)
            logging.info(f"Embeddings [{split}] saved: {cache_path} | rows={num_rows} | dim={feature_dim}")
            results[split] = {
                "split": split,
                "status": "extracted",
                "cache_path": str(cache_path),
                "num_rows": num_rows,
                "feature_dim": feature_dim,
            }
        return results

    raise ValueError("Unsupported embeddings.source. Use 'hf_text' or 'qwen'.")
