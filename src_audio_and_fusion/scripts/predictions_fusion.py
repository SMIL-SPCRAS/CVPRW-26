from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

def ccc_1d(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    mu_p = pred.mean()
    mu_t = target.mean()

    var_p = pred.var()
    var_t = target.var()

    cov = ((pred - mu_p) * (target - mu_t)).mean()
    return float((2.0 * cov) / (var_p + var_t + (mu_p - mu_t) ** 2 + eps))


def compute_va_ccc(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """
    preds, targets: [N,2]
    """
    valid = np.isfinite(targets).all(axis=1)
    valid &= targets[:, 0] != -5.0
    valid &= targets[:, 1] != -5.0

    preds = preds[valid]
    targets = targets[valid]

    if len(preds) == 0:
        return {
            "v_ccc_metric": float("nan"),
            "a_ccc_metric": float("nan"),
            "va_ccc_metric": float("nan"),
        }

    ccc_v = ccc_1d(preds[:, 0], targets[:, 0], eps=eps)
    ccc_a = ccc_1d(preds[:, 1], targets[:, 1], eps=eps)
    ccc_va = 0.5 * (ccc_v + ccc_a)

    return {
        "v_ccc_metric": float(ccc_v),
        "a_ccc_metric": float(ccc_a),
        "va_ccc_metric": float(ccc_va),
    }

def load_pickle(path: Path) -> Dict[str, dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain dict")

    data = {
        k: v
        for k, v in data.items()
        if not np.array_equal(np.asarray(v["label"], dtype=np.float32), np.array([-5.0, -5.0], dtype=np.float32))
    }
    
    return data

def extract_arrays(
    data: Dict[str, dict],
    keys: List[str],
    require_labels: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = []
    labels = []

    for k in keys:
        item = data[k]
        p = np.asarray(item["prediction"], dtype=np.float64).reshape(-1)
        if p.shape[0] != 2:
            raise ValueError(f"{k}: prediction shape is not 2")

        preds.append(p)

        if require_labels:
            if "label" not in item or item["label"] is None:
                raise ValueError(f"{k}: missing label")
            t = np.asarray(item["label"], dtype=np.float64).reshape(-1)
            if t.shape[0] != 2:
                raise ValueError(f"{k}: label shape is not 2")
            labels.append(t)

    preds = np.stack(preds, axis=0)
    labels_arr = np.stack(labels, axis=0) if require_labels else None
    return preds, labels_arr

def get_common_keys(dicts: List[Dict[str, dict]]) -> List[str]:
    common = None
    for d in dicts:
        ks = set(d.keys())
        common = ks if common is None else (common & ks)
    if not common:
        raise ValueError("No common keys across all submissions")
    return sorted(common)


def normalize_weights(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    if s <= eps:
        raise ValueError("Weight sum is zero")
    return w / s


def fuse_predictions_channelwise(
    preds_list: List[np.ndarray],
    w_v: np.ndarray,
    w_a: np.ndarray,
) -> np.ndarray:
    """
    preds_list: list of [N,2]
    w_v, w_a: [M]
    returns [N,2]
    """
    M = len(preds_list)
    if len(w_v) != M or len(w_a) != M:
        raise ValueError("Weight length mismatch")

    w_v = normalize_weights(w_v)
    w_a = normalize_weights(w_a)

    preds_stack = np.stack(preds_list, axis=0)  # [M,N,2]

    out = np.empty_like(preds_stack[0], dtype=np.float64)
    out[:, 0] = np.tensordot(w_v, preds_stack[:, :, 0], axes=(0, 0))
    out[:, 1] = np.tensordot(w_a, preds_stack[:, :, 1], axes=(0, 0))
    return out


# ----------------------------
# search
# ----------------------------

@dataclass
class SearchResult:
    models: List[str]
    w_v: List[float]
    w_a: List[float]
    v_ccc_metric: float
    a_ccc_metric: float
    va_ccc_metric: float


def grid_weights_2models(step: float) -> List[np.ndarray]:
    vals = np.arange(0.0, 1.0 + 1e-9, step, dtype=np.float64)
    out = []
    for x in vals:
        out.append(np.array([x, 1.0 - x], dtype=np.float64))
    return out


def random_dirichlet_weights(num_models: int, num_samples: int, alpha: float, rng: np.random.Generator) -> List[np.ndarray]:
    weights = rng.dirichlet(alpha=np.full(num_models, alpha, dtype=np.float64), size=num_samples)
    return [w.astype(np.float64) for w in weights]


def evaluate_candidate(
    model_names: List[str],
    preds_by_model: Dict[str, np.ndarray],
    targets: np.ndarray,
    w_v: np.ndarray,
    w_a: np.ndarray,
) -> SearchResult:
    preds_list = [preds_by_model[m] for m in model_names]
    fused = fuse_predictions_channelwise(preds_list, w_v=w_v, w_a=w_a)
    metrics = compute_va_ccc(fused, targets)

    return SearchResult(
        models=list(model_names),
        w_v=[float(x) for x in normalize_weights(w_v)],
        w_a=[float(x) for x in normalize_weights(w_a)],
        v_ccc_metric=float(metrics["v_ccc_metric"]),
        a_ccc_metric=float(metrics["a_ccc_metric"]),
        va_ccc_metric=float(metrics["va_ccc_metric"]),
    )


def search_for_subset(
    model_names: List[str],
    fit_preds_by_model: Dict[str, np.ndarray],
    fit_targets: np.ndarray,
    eval_preds_by_model: Dict[str, np.ndarray],
    eval_targets: np.ndarray,
    pair_grid_step: float,
    dirichlet_samples: int,
    dirichlet_alpha: float,
    rng: np.random.Generator,
    channelwise: bool = True,
) -> SearchResult:
    M = len(model_names)

    if M == 1:
        return evaluate_candidate(
            model_names=model_names,
            preds_by_model=eval_preds_by_model,
            targets=eval_targets,
            w_v=np.array([1.0], dtype=np.float64),
            w_a=np.array([1.0], dtype=np.float64),
        )

    candidates: List[Tuple[np.ndarray, np.ndarray]] = []

    if M == 2:
        grid = grid_weights_2models(pair_grid_step)
        if channelwise:
            for wv in grid:
                for wa in grid:
                    candidates.append((wv, wa))
        else:
            for w in grid:
                candidates.append((w, w))
    else:
        base = random_dirichlet_weights(M, dirichlet_samples, dirichlet_alpha, rng)
        if channelwise:
            base2 = random_dirichlet_weights(M, dirichlet_samples, dirichlet_alpha, rng)
            candidates.extend(list(zip(base, base2)))
        else:
            for w in base:
                candidates.append((w, w))

        candidates.append((
            np.full(M, 1.0 / M, dtype=np.float64),
            np.full(M, 1.0 / M, dtype=np.float64),
        ))
        for i in range(M):
            w = np.zeros(M, dtype=np.float64)
            w[i] = 1.0
            candidates.append((w.copy(), w.copy()))

    best_fit = None
    best_fit_key = None

    for w_v, w_a in candidates:
        r_fit = evaluate_candidate(
            model_names=model_names,
            preds_by_model=fit_preds_by_model,
            targets=fit_targets,
            w_v=w_v,
            w_a=w_a,
        )
        key = (r_fit.va_ccc_metric, r_fit.a_ccc_metric, r_fit.v_ccc_metric)
        if best_fit is None or key > best_fit_key:
            best_fit = (w_v.copy(), w_a.copy())
            best_fit_key = key

    assert best_fit is not None
    best_w_v, best_w_a = best_fit

    return evaluate_candidate(
        model_names=model_names,
        preds_by_model=eval_preds_by_model,
        targets=eval_targets,
        w_v=best_w_v,
        w_a=best_w_a,
    )


# ----------------------------
# saving test fusion
# ----------------------------

def save_fused_test(
    best: SearchResult,
    test_dicts_by_model: Dict[str, Dict[str, dict]],
    test_template_path: str,
    out_path: Path,
) -> None:
    model_names = best.models
    dicts = [test_dicts_by_model[m] for m in model_names]
    keys = get_common_keys(dicts)

    w_v = np.asarray(best.w_v, dtype=np.float64)
    w_a = np.asarray(best.w_a, dtype=np.float64)

    out = {}
    for k in keys:
        preds = []
        label = None
        for d in dicts:
            item = d[k]
            preds.append(np.asarray(item["prediction"], dtype=np.float64).reshape(2))
            if label is None:
                label = item.get("label", None)

        preds = np.stack(preds, axis=0)  # [M,2]
        fused = np.empty((2,), dtype=np.float64)
        fused[0] = np.dot(w_v, preds[:, 0])
        fused[1] = np.dot(w_a, preds[:, 1])

        out[k] = {
            "embedding": None,
            "prediction": fused.astype(np.float32),
            "label": label,
        }

    out_path_p = out_path / "best_fused_test.pkl"
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    with out_path_p.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    template_df = pd.read_csv(test_template_path)
    valence_list = []
    arousal_list = []

    for image_location in template_df["image_location"].tolist():
        item = out.get(image_location, None)
        if item is None:
            raise KeyError(f"Prediction not found for {image_location}")

        pred = item["prediction"]
        pred = np.clip(pred, min=-1.0, max=1.0)

        valence_list.append(float(pred[0]))
        arousal_list.append(float(pred[1]))

    out_df = pd.DataFrame({
        "image_location": template_df["image_location"],
        "valence": valence_list,
        "arousal": arousal_list,
    })
    
    out_df.to_csv(out_path / "prediction.txt", index=False)

# ----------------------------
# main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="root folder with submit_* subfolders")
    parser.add_argument("--fit-split", type=str, default="val", choices=["train", "val"],
                        help="split used for fitting fusion weights")
    parser.add_argument("--eval-split", type=str, default="val", choices=["train", "val"],
                        help="split used for evaluating fitted weights")
    parser.add_argument("--test-split", type=str, default="test", help="split used for final fused submission")
    parser.add_argument("--test-template-path", 
                        type=str, 
                        default="ABAW_VA_test_set_example.txt", help="template file for test submission")

    parser.add_argument("--include", type=str, nargs="*", default=None, help="only these submit folder names")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, help="exclude these submit folder names")
    parser.add_argument("--max-comb-size", type=int, default=3, help="max subset size to search; use 0 to search all sizes")
    parser.add_argument("--pair-grid-step", type=float, default=0.05, help="grid step for 2-model fusion")
    parser.add_argument("--dirichlet-samples", type=int, default=5000, help="random dirichlet samples for 3+ models")
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0, help="dirichlet alpha")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channelwise", action="store_true", help="search separate weights for valence and arousal")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="./fusion_search_out")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    root = Path(args.root)
    submit_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if args.include:
        include_set = set(args.include)
        submit_dirs = [p for p in submit_dirs if p.name in include_set]
    if args.exclude:
        exclude_set = set(args.exclude)
        submit_dirs = [p for p in submit_dirs if p.name not in exclude_set]

    if not submit_dirs:
        raise ValueError("No submit dirs selected")

    print(f"\nWeight fitting split: {args.fit_split}")
    print(f"Evaluation split: {args.eval_split}")
    print(f"Final submission split: {args.test_split}")

    print("Using submission folders:")
    for p in submit_dirs:
        print(" -", p.name)

    fit_dicts_by_model: Dict[str, Dict[str, dict]] = {}
    eval_dicts_by_model: Dict[str, Dict[str, dict]] = {}
    test_dicts_by_model: Dict[str, Dict[str, dict]] = {}

    for sd in submit_dirs:
        fit_path = sd / f"{args.fit_split}.pkl"
        eval_path = sd / f"{args.eval_split}.pkl"
        test_path = sd / f"{args.test_split}.pkl"

        if not fit_path.exists():
            raise FileNotFoundError(fit_path)
        if not eval_path.exists():
            raise FileNotFoundError(eval_path)
        if not test_path.exists():
            raise FileNotFoundError(test_path)

        fit_dicts_by_model[sd.name] = load_pickle(fit_path)
        eval_dicts_by_model[sd.name] = load_pickle(eval_path)
        test_dicts_by_model[sd.name] = load_pickle(test_path)

    common_fit_keys = get_common_keys(list(fit_dicts_by_model.values()))
    print(f"\nCommon fit keys: {len(common_fit_keys)}")

    fit_preds_by_model: Dict[str, np.ndarray] = {}
    fit_targets: Optional[np.ndarray] = None

    for name, d in fit_dicts_by_model.items():
        preds, labels = extract_arrays(d, common_fit_keys, require_labels=True)
        fit_preds_by_model[name] = preds
        if fit_targets is None:
            fit_targets = labels
        else:
            if labels.shape != fit_targets.shape:
                raise ValueError(f"Fit label shape mismatch for {name}")
    assert fit_targets is not None

    common_eval_keys = get_common_keys(list(eval_dicts_by_model.values()))
    print(f"Common eval keys: {len(common_eval_keys)}")

    eval_preds_by_model: Dict[str, np.ndarray] = {}
    eval_targets: Optional[np.ndarray] = None

    for name, d in eval_dicts_by_model.items():
        preds, labels = extract_arrays(d, common_eval_keys, require_labels=True)
        eval_preds_by_model[name] = preds
        if eval_targets is None:
            eval_targets = labels
        else:
            if labels.shape != eval_targets.shape:
                raise ValueError(f"Eval label shape mismatch for {name}")
    assert eval_targets is not None

    # Print single-model scores
    print("\nSingle model scores:")
    single_results: List[SearchResult] = []
    model_names_all = list(eval_preds_by_model.keys())

    for name in model_names_all:
        r = search_for_subset(
            model_names=[name],
            fit_preds_by_model=fit_preds_by_model,
            fit_targets=fit_targets,
            eval_preds_by_model=eval_preds_by_model,
            eval_targets=eval_targets,
            pair_grid_step=args.pair_grid_step,
            dirichlet_samples=args.dirichlet_samples,
            dirichlet_alpha=args.dirichlet_alpha,
            rng=rng,
            channelwise=args.channelwise,
        )
        single_results.append(r)
        print(
            f"{name:20s} "
            f"v={r.v_ccc_metric:.6f} "
            f"a={r.a_ccc_metric:.6f} "
            f"va={r.va_ccc_metric:.6f}"
        )

    results: List[SearchResult] = []
    max_comb_size = args.max_comb_size if args.max_comb_size > 0 else len(model_names_all)

    for k in range(1, min(max_comb_size, len(model_names_all)) + 1):
        print(f"\nSearching subsets of size {k}...")
        for subset in itertools.combinations(model_names_all, k):
            r = search_for_subset(
                model_names=list(subset),
                fit_preds_by_model=fit_preds_by_model,
                fit_targets=fit_targets,
                eval_preds_by_model=eval_preds_by_model,
                eval_targets=eval_targets,
                pair_grid_step=args.pair_grid_step,
                dirichlet_samples=args.dirichlet_samples,
                dirichlet_alpha=args.dirichlet_alpha,
                rng=rng,
                channelwise=args.channelwise,
            )
            results.append(r)

    results.sort(
        key=lambda x: (x.va_ccc_metric, x.a_ccc_metric, x.v_ccc_metric),
        reverse=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save leaderboard
    leaderboard = [asdict(r) for r in results]
    with (out_dir / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    print("\nTop results:")
    for i, r in enumerate(results[: args.topk], start=1):
        print(
            f"{i:02d}. models={r.models} "
            f"w_v={[round(x, 4) for x in r.w_v]} "
            f"w_a={[round(x, 4) for x in r.w_a]} "
            f"v={r.v_ccc_metric:.6f} "
            f"a={r.a_ccc_metric:.6f} "
            f"va={r.va_ccc_metric:.6f}"
        )

    best = results[0]
    print("\nBest ensemble:")
    print(json.dumps(asdict(best), indent=2, ensure_ascii=False))

    save_fused_test(
        best=best,
        test_dicts_by_model=test_dicts_by_model,
        test_template_path=args.test_template_path,
        out_path=out_dir,
    )

    print(f"\nSaved fused test predictions to: {out_dir / 'best_fused_test.pkl'}, {out_dir / 'prediction.txt'}")


if __name__ == "__main__":
    main()