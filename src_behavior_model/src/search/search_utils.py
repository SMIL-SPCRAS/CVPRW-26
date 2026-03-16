import copy
import logging
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from ..core.config import load_config
from ..trainers.segment_trainer import run_training


def _write_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _result_metrics_line(result: Dict[str, Any]) -> str:
    keys = [
        "model_type",
        "optimizer",
        "head_hidden_dim",
        "head_dropout",
        "emb_dropout",
        "model_arch",
        "model_d_model",
        "model_dropout",
        "model_head_hidden_dim",
        "model_head_dropout",
        "group_col",
        "window_size",
        "window_stride",
        "drop_overlapping_segments",
        "transformer_nhead",
        "transformer_layers",
        "transformer_ff_dim",
        "transformer_positional_encoding",
        "transformer_gate_mode",
        "transformer_max_seq_len",
        "mamba_layers",
        "mamba_d_state",
        "mamba_kernel_size",
        "mamba_d_discr",
        "selection_metric",
        "best_selection_score",
        "best_epoch",
        "best_ccc_mean",
        "best_ccc_valence",
        "best_ccc_arousal",
        "best_val_loss",
        "best_frame_ccc_mean",
        "best_frame_ccc_valence",
        "best_frame_ccc_arousal",
        "best_frame_rows_matched",
        "frame_eval_error_count",
        "frame_eval_last_error",
        "stopped_early",
    ]
    parts = []
    for key in keys:
        if key in result:
            parts.append(f"{key}={_fmt_value(result[key])}")
    return " | ".join(parts)


def _score_from_result(result: Dict[str, Any], objective: str) -> float:
    value = result.get(objective, float("-inf"))
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("-inf")


def _trial_cfg(base_cfg: Dict[str, Any], output_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)
    cfg["output_dir"] = str(output_dir)
    return cfg


def run_single(
    base_train_cfg: Dict[str, Any],
    run_dir: Path,
    objective: str,
    report_path: Path,
    train_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trial_dir = run_dir / "single"
    cfg = _trial_cfg(base_train_cfg, trial_dir, {})
    logging.info("[Search:single] Starting single run.")
    try:
        result = train_fn(cfg)
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)
        _write_line(report_path, f"[single] FAILED error={error_text}")
        raise RuntimeError(f"Single run failed: {error_text}") from exc
    score = _score_from_result(result, objective)
    _write_line(report_path, f"[single] score={score:.6f} checkpoint={result['best_checkpoint']}")
    _write_line(report_path, f"[single] metrics | {_result_metrics_line(result)}")
    return result, cfg


def run_exhaustive(
    base_train_cfg: Dict[str, Any],
    run_dir: Path,
    grid: Dict[str, List[Any]],
    objective: str,
    report_path: Path,
    train_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    param_names = list(grid.keys())
    best_score = float("-inf")
    best_result: Dict[str, Any] = {}
    best_cfg: Dict[str, Any] = {}

    _write_line(report_path, "=== FULL SEARCH ===")
    combo_id = 0
    n_failed = 0
    for combo in product(*(grid[p] for p in param_names)):
        combo_id += 1
        overrides = dict(zip(param_names, combo))
        trial_dir = run_dir / "trials" / f"full_{combo_id:03d}"
        cfg = _trial_cfg(base_train_cfg, trial_dir, overrides)

        logging.info(f"[Search:full] Trial {combo_id}: {overrides}")
        try:
            result = train_fn(cfg)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            error_text = str(exc)
            _write_line(report_path, f"[full #{combo_id}] FAILED overrides={overrides} error={error_text}")
            logging.warning("[Search:full] Trial %d failed: %s", combo_id, error_text)
            continue
        score = _score_from_result(result, objective)
        _write_line(report_path, f"[full #{combo_id}] overrides={overrides} score={score:.6f}")
        _write_line(report_path, f"[full #{combo_id}] metrics | {_result_metrics_line(result)}")

        if score > best_score:
            best_score = score
            best_result = result
            best_cfg = cfg
            _write_line(report_path, f"  -> BEST updated: score={best_score:.6f}")

    _write_line(report_path, f"=== FULL STATS total={combo_id} failed={n_failed} succeeded={combo_id - n_failed} ===")
    _write_line(report_path, f"=== FULL BEST score={best_score:.6f} cfg={best_cfg} ===")
    if best_result:
        _write_line(report_path, f"=== FULL BEST METRICS | {_result_metrics_line(best_result)} ===")
        return best_result, best_cfg
    raise RuntimeError("All FULL-search trials failed. See search_report.txt for details.")


def run_greedy(
    base_train_cfg: Dict[str, Any],
    run_dir: Path,
    grid: Dict[str, List[Any]],
    defaults: Dict[str, Any],
    objective: str,
    report_path: Path,
    train_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    param_names = list(grid.keys())
    current = {k: defaults.get(k, base_train_cfg.get(k)) for k in param_names}
    best_result: Dict[str, Any] = {}
    best_cfg: Dict[str, Any] = {}

    _write_line(report_path, "=== GREEDY SEARCH ===")
    _write_line(report_path, f"start={current}")

    trial_id = 0
    n_failed = 0
    for step_id, param_name in enumerate(param_names, start=1):
        step_best_score = float("-inf")
        step_best_val = current[param_name]
        step_best_result: Dict[str, Any] = {}
        step_best_cfg: Dict[str, Any] = {}

        values = list(grid[param_name])
        for value in values:
            trial_id += 1
            overrides = dict(current)
            overrides[param_name] = value
            trial_dir = run_dir / "trials" / f"greedy_s{step_id:02d}_{param_name}_{trial_id:03d}"
            cfg = _trial_cfg(base_train_cfg, trial_dir, overrides)

            logging.info(f"[Search:greedy] Step {step_id}/{len(param_names)} try {param_name}={value}")
            try:
                result = train_fn(cfg)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                error_text = str(exc)
                _write_line(
                    report_path,
                    f"[greedy step={step_id}] FAILED {param_name}={value} error={error_text}",
                )
                logging.warning(
                    "[Search:greedy] Step %d parameter %s=%s failed: %s",
                    step_id,
                    param_name,
                    value,
                    error_text,
                )
                continue
            score = _score_from_result(result, objective)
            _write_line(report_path, f"[greedy step={step_id}] {param_name}={value} score={score:.6f}")
            _write_line(report_path, f"[greedy step={step_id}] metrics | {_result_metrics_line(result)}")

            if score > step_best_score:
                step_best_score = score
                step_best_val = value
                step_best_result = result
                step_best_cfg = cfg

        current[param_name] = step_best_val
        if step_best_result:
            best_result = step_best_result
            best_cfg = step_best_cfg
            _write_line(
                report_path,
                f"[greedy step={step_id}] best {param_name}={step_best_val} score={step_best_score:.6f}",
            )
        else:
            _write_line(
                report_path,
                f"[greedy step={step_id}] no successful trials for parameter '{param_name}', keeping value={step_best_val}",
            )

    _write_line(report_path, f"=== GREEDY STATS failed={n_failed} ===")
    _write_line(report_path, f"=== GREEDY BEST cfg={current} ===")
    if best_result:
        _write_line(report_path, f"=== GREEDY BEST METRICS | {_result_metrics_line(best_result)} ===")
        return best_result, best_cfg
    raise RuntimeError("All GREEDY-search trials failed. See search_report.txt for details.")


def run_search(
    base_train_cfg: Dict[str, Any],
    run_dir: Path,
    mode: str,
    search_params_file: Path,
    objective: str = "best_ccc_mean",
    train_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = run_training,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report_path = run_dir / "search_report.txt"
    if report_path.exists():
        report_path.unlink()

    normalized_mode = mode.strip().lower()
    if normalized_mode == "single":
        return run_single(base_train_cfg, run_dir, objective, report_path, train_fn)

    search_cfg = load_config(search_params_file)
    grid = dict(search_cfg.get("grid", {}))
    defaults = dict(search_cfg.get("defaults", {}))

    if not grid:
        raise ValueError(f"Search mode '{mode}' requires non-empty [grid] in {search_params_file}.")

    if normalized_mode in {"full", "exhaustive"}:
        return run_exhaustive(base_train_cfg, run_dir, grid, objective, report_path, train_fn)
    if normalized_mode == "greedy":
        return run_greedy(base_train_cfg, run_dir, grid, defaults, objective, report_path, train_fn)

    raise ValueError(
        f"Invalid search mode '{mode}'. Use 'single', 'greedy', or 'full'."
    )
