import argparse
import copy
import csv
import itertools
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_utils import load_toml_compat
from train import TrainConfig, load_config, run_train


def _clone_cfg(cfg: TrainConfig) -> TrainConfig:
    return copy.deepcopy(cfg)


def _set_cfg_value(cfg: TrainConfig, key: str, value: Any) -> None:
    if not hasattr(cfg, key):
        raise ValueError(f"Unknown TrainConfig field in search grid: {key}")

    cur = getattr(cfg, key)
    as_path = isinstance(cur, Path) or key.endswith("_dir") or key.endswith("_path")
    if as_path:
        if value is None or str(value).strip() == "":
            setattr(cfg, key, None)
        else:
            setattr(cfg, key, Path(value))
        return
    setattr(cfg, key, value)


def _build_trials(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    names = list(grid.keys())
    values = [grid[k] for k in names]
    for combo in itertools.product(*values):
        yield dict(zip(names, combo))


def _save_rows(rows: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "search_results.csv"
    fields = ["trial_id", "status", "best_score", "run_dir", "params_json", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r["best_score"]))
        (out_dir / "best_trial.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SEARCH] Best trial id={best['trial_id']} score={float(best['best_score']):.6f}")
    print(f"[SEARCH] Summary saved: {csv_path}")


def run_search(config_path: Path, search_config_path: Path) -> None:
    base_cfg = load_config(config_path)
    sc = load_toml_compat(search_config_path)
    grid = dict(sc.get("grid", {}))
    opts = dict(sc.get("search", {}))
    max_trials = int(opts.get("max_trials", 0))
    run_test_after_train = bool(opts.get("run_test_after_train", False))
    save_every_trials = max(int(opts.get("save_every_trials", 1)), 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = base_cfg.output_dir / f"search_{timestamp}"
    trials_output_dir = search_dir / "trials"
    search_dir.mkdir(parents=True, exist_ok=True)
    trials_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, search_dir / "base_config_used.toml")
    shutil.copy2(search_config_path, search_dir / "search_params_used.toml")

    meta = {
        "config_path": str(config_path),
        "search_config_path": str(search_config_path),
        "max_trials": max_trials,
        "run_test_after_train": run_test_after_train,
        "grid_keys": list(grid.keys()),
    }
    (search_dir / "search_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[dict] = []
    for idx, overrides in enumerate(_build_trials(grid), start=1):
        if max_trials > 0 and idx > max_trials:
            break
        print(f"[SEARCH] Trial {idx}: {overrides}")
        cfg = _clone_cfg(base_cfg)
        cfg.output_dir = trials_output_dir
        cfg.run_test_after_train = run_test_after_train
        for k, v in overrides.items():
            _set_cfg_value(cfg, k, v)
        try:
            run_dir = run_train(cfg)
            # parse best score from logs/metrics_*.csv
            metrics_files = sorted((run_dir / "logs").glob("metrics_*.csv"))
            if not metrics_files:
                raise RuntimeError(f"No metrics csv in {run_dir / 'logs'}")
            import pandas as pd

            m = pd.read_csv(metrics_files[-1])
            best_score = float(m["va_score"].max()) if "va_score" in m.columns and len(m) > 0 else float("-inf")
            rows.append(
                {
                    "trial_id": idx,
                    "status": "ok",
                    "best_score": best_score,
                    "run_dir": str(run_dir),
                    "params_json": json.dumps(overrides, ensure_ascii=False),
                    "error": "",
                }
            )
        except Exception as ex:
            rows.append(
                {
                    "trial_id": idx,
                    "status": "error",
                    "best_score": float("-inf"),
                    "run_dir": "",
                    "params_json": json.dumps(overrides, ensure_ascii=False),
                    "error": f"{type(ex).__name__}: {ex}",
                }
            )
            print(f"[SEARCH][ERROR] Trial {idx} failed: {type(ex).__name__}: {ex}")

        if idx % save_every_trials == 0:
            _save_rows(rows, search_dir)

    _save_rows(rows, search_dir)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to src_fusion_model config toml")
    parser.add_argument(
        "--search-config",
        type=str,
        default=str(Path(__file__).resolve().parent / "search_params_fusion.toml"),
        help="Path to search params toml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_search(config_path=Path(args.config), search_config_path=Path(args.search_config))

