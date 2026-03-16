from pathlib import Path
import datetime
import logging
import shutil

from src.core.config import get_config_section, load_config
from src.core.logger_setup import setup_logger
from src.search.search_utils import run_search
from src.datasets.embedding_cache import prepare_embedding_caches
from src.trainers.segment_trainer import run_training as run_segment_training
from src.trainers.transformer_trainer import run_transformer_training

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "text_va.toml"
RESULTS_ROOT = BASE_DIR / "results"


def _resolve_local_path(path_value: str, default_relative: str) -> Path:
    path = Path(path_value) if path_value else Path(default_relative)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _make_run_dir() -> Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RESULTS_ROOT / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_summary(run_dir: Path, lines: list[str]) -> None:
    summary_path = run_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    config = load_config(CONFIG_PATH)
    run_dir = _make_run_dir()
    log_file = run_dir / "session_log.txt"
    setup_logger(log_file=log_file)
    shutil.copy2(CONFIG_PATH, run_dir / "config_copy.toml")

    pipeline = dict(config.get("pipeline", {}))
    pipeline_mode = str(pipeline.get("mode", "segment")).lower()
    use_sequence_model = pipeline_mode in {"transformer", "model"}
    run_train = bool(pipeline.get("run_train", True))
    search_cfg = dict(config.get("search", {}))
    search_mode = str(search_cfg.get("mode", "single")).lower()
    search_params_file = _resolve_local_path(
        str(search_cfg.get("params_file", "configs/search_params.toml")),
        "configs/search_params.toml",
    )
    objective = str(search_cfg.get("objective", "best_ccc_mean"))

    logging.info("Pipeline: load config -> prepare embeddings -> train")
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Pipeline mode: {pipeline_mode}")
    logging.info(f"Search mode: {search_mode}")

    embedding_results: dict[str, dict] = {}
    embeddings_cfg = dict(config.get("embeddings", {}))
    if bool(embeddings_cfg.get("enabled", False)):
        logging.info("Step: prepare_embeddings")
        train_cfg_for_embeddings = dict(config.get("train", {}))
        predict_cfg_for_embeddings = dict(config.get("predict", {}))
        embedding_results = prepare_embedding_caches(
            embeddings_cfg=embeddings_cfg,
            train_cfg=train_cfg_for_embeddings,
            predict_cfg=predict_cfg_for_embeddings,
        )

    best_checkpoint = None
    best_train_result = None
    if run_train:
        logging.info("Step: train")
        train_cfg = get_config_section(config, "train")
        train_fn = run_segment_training
        if use_sequence_model:
            model_cfg = dict(config.get("model", config.get("transformer", {})))
            train_cfg.update(model_cfg)
            train_fn = run_transformer_training

            if "train_embeddings_pt" not in train_cfg and "train" in embedding_results:
                train_cfg["train_embeddings_pt"] = embedding_results["train"].get("cache_path")
            if "val_embeddings_pt" not in train_cfg and "val" in embedding_results:
                train_cfg["val_embeddings_pt"] = embedding_results["val"].get("cache_path")
            if not train_cfg.get("train_embeddings_pt") or not train_cfg.get("val_embeddings_pt"):
                raise ValueError(
                    "Model mode requires train/val embedding caches. "
                    "Enable [embeddings] prepare_train/prepare_val or set model.train_embeddings_pt / model.val_embeddings_pt."
                )

        frame_eval_cfg = dict(config.get("frame_eval", {}))
        frame_eval_enabled = bool(frame_eval_cfg.get("enabled", False))
        if frame_eval_enabled:
            train_cfg["frame_val_enabled"] = True
            train_cfg["frame_val_gt_csv"] = str(frame_eval_cfg.get("val_frame_gt_csv", "dataset/val_by_frame.csv"))
            train_cfg["frame_val_group_col"] = str(frame_eval_cfg.get("pred_group_col", "stream_name"))
            train_cfg["frame_val_pred_video_col"] = str(frame_eval_cfg.get("pred_video_col", "video_name"))
            train_cfg["frame_val_gt_video_col"] = str(frame_eval_cfg.get("gt_video_col", "video_name"))
            train_cfg["frame_val_normalize_video_name"] = bool(frame_eval_cfg.get("normalize_video_name", True))
            train_cfg["frame_val_frame_col"] = str(frame_eval_cfg.get("frame_col", "frame_idx"))
            train_cfg["frame_val_val_col"] = str(frame_eval_cfg.get("val_col", "valence"))
            train_cfg["frame_val_aro_col"] = str(frame_eval_cfg.get("aro_col", "arousal"))
            train_cfg["frame_val_filter_invalid_gt"] = bool(frame_eval_cfg.get("filter_invalid_gt", True))
            train_cfg["frame_val_invalid_threshold"] = float(frame_eval_cfg.get("invalid_threshold", -4.9))

        best_train_result, _ = run_search(
            base_train_cfg=train_cfg,
            run_dir=run_dir,
            mode=search_mode,
            search_params_file=search_params_file,
            objective=objective,
            train_fn=train_fn,
        )
        best_checkpoint = best_train_result["best_checkpoint"]
        logging.info(f"Train done. Best checkpoint: {best_checkpoint}")
        logging.info(f"Best {objective}: {best_train_result.get(objective)}")

    summary_lines = [
        f"run_dir = {run_dir}",
        f"log_file = {log_file}",
        f"search_mode = {search_mode}",
        f"pipeline_mode = {pipeline_mode}",
        f"objective = {objective}",
        f"best_checkpoint = {best_checkpoint}",
    ]
    if embeddings_cfg:
        summary_lines.append(f"emb_source = {embeddings_cfg.get('source', 'hf_text')}")
        if str(embeddings_cfg.get("source", "hf_text")).strip().lower() == "qwen":
            summary_lines.append(f"emb_qwen_branch = {embeddings_cfg.get('qwen_branch', 'multimodal')}")
            summary_lines.append(f"emb_qwen_pool = {embeddings_cfg.get('qwen_pool', 'mean')}")
    if best_train_result is not None:
        def _res(*keys: str):
            for key in keys:
                if key in best_train_result:
                    return best_train_result.get(key)
            return None

        summary_lines.append(f"model_type = {best_train_result.get('model_type')}")
        summary_lines.append(f"optimizer = {best_train_result.get('optimizer')}")
        summary_lines.append(f"head_hidden_dim = {best_train_result.get('head_hidden_dim')}")
        summary_lines.append(f"head_dropout = {best_train_result.get('head_dropout')}")
        summary_lines.append(f"emb_dropout = {best_train_result.get('emb_dropout')}")
        summary_lines.append(f"model_arch = {_res('model_arch', 'transformer_arch')}")
        summary_lines.append(f"model_d_model = {_res('model_d_model', 'transformer_d_model')}")
        summary_lines.append(f"model_dropout = {_res('model_dropout', 'transformer_dropout')}")
        summary_lines.append(f"model_head_hidden_dim = {_res('model_head_hidden_dim', 'transformer_head_hidden_dim')}")
        summary_lines.append(f"model_head_dropout = {_res('model_head_dropout', 'transformer_head_dropout')}")
        summary_lines.append(f"group_col = {_res('group_col')}")
        summary_lines.append(f"window_size = {_res('window_size', 'transformer_window_size')}")
        summary_lines.append(f"window_stride = {_res('window_stride', 'transformer_window_stride')}")
        summary_lines.append(f"drop_overlapping_segments = {_res('drop_overlapping_segments')}")
        summary_lines.append(f"transformer_nhead = {best_train_result.get('transformer_nhead')}")
        summary_lines.append(f"transformer_layers = {best_train_result.get('transformer_layers')}")
        summary_lines.append(f"transformer_ff_dim = {best_train_result.get('transformer_ff_dim')}")
        summary_lines.append(f"transformer_positional_encoding = {best_train_result.get('transformer_positional_encoding')}")
        summary_lines.append(f"transformer_gate_mode = {best_train_result.get('transformer_gate_mode')}")
        summary_lines.append(f"transformer_max_seq_len = {best_train_result.get('transformer_max_seq_len')}")
        summary_lines.append(f"mamba_layers = {_res('mamba_layers', 'transformer_mamba_layers')}")
        summary_lines.append(f"mamba_d_state = {_res('mamba_d_state', 'transformer_mamba_d_state')}")
        summary_lines.append(f"mamba_kernel_size = {_res('mamba_kernel_size', 'transformer_mamba_kernel_size')}")
        summary_lines.append(f"mamba_d_discr = {_res('mamba_d_discr', 'transformer_mamba_d_discr')}")
        summary_lines.append(f"selection_metric = {best_train_result.get('selection_metric')}")
        summary_lines.append(f"best_selection_score = {best_train_result.get('best_selection_score')}")
        summary_lines.append(f"best_epoch = {best_train_result.get('best_epoch')}")
        summary_lines.append(f"segment_ccc_mean = {best_train_result.get('best_ccc_mean')}")
        summary_lines.append(f"segment_ccc_valence = {best_train_result.get('best_ccc_valence')}")
        summary_lines.append(f"segment_ccc_arousal = {best_train_result.get('best_ccc_arousal')}")
        summary_lines.append(f"segment_val_loss = {best_train_result.get('best_val_loss')}")
        summary_lines.append(f"frame_ccc_mean = {best_train_result.get('best_frame_ccc_mean')}")
        summary_lines.append(f"frame_ccc_valence = {best_train_result.get('best_frame_ccc_valence')}")
        summary_lines.append(f"frame_ccc_arousal = {best_train_result.get('best_frame_ccc_arousal')}")
        summary_lines.append(f"frame_rows_matched = {best_train_result.get('best_frame_rows_matched')}")
        summary_lines.append(f"frame_eval_error_count = {best_train_result.get('frame_eval_error_count')}")
        summary_lines.append(f"frame_eval_last_error = {best_train_result.get('frame_eval_last_error')}")
        summary_lines.append(f"history_csv = {best_train_result.get('history_csv')}")
    for split_name, info in embedding_results.items():
        summary_lines.append(f"emb_{split_name}_status = {info.get('status')}")
        summary_lines.append(f"emb_{split_name}_path = {info.get('cache_path')}")
        summary_lines.append(f"emb_{split_name}_num_rows = {info.get('num_rows')}")
        summary_lines.append(f"emb_{split_name}_feature_dim = {info.get('feature_dim')}")
    _write_summary(run_dir, summary_lines)
    logging.info(f"Summary saved to {run_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
