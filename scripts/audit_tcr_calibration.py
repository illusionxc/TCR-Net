


from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44, 45, 46)
COMPONENTS = ("R_cons", "R_seq", "R_rule")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("exp_data/revision_2026/core"),
        help="Directory containing seed_<n>/tcr-net/details.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Read test outputs only after the selected design is frozen.",
    )
    parser.add_argument(
        "--method-dir",
        default="tcr-net",
        help="Per-seed method directory below seed_<n>.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def soft_risk(frame: pd.DataFrame) -> np.ndarray:
    logits = frame[["logit_low", "logit_mid", "logit_high"]].to_numpy(float)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1.0e-12, None)
    return probs @ np.asarray([0.0, 0.5, 1.0])


def fit_stats(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        field: (float(frame[field].min()), float(frame[field].max()))
        for field in COMPONENTS
    }


def normalized_components(
    frame: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    columns = []
    for field in COMPONENTS:
        low, high = stats[field]
        columns.append(
            (frame[field].to_numpy(float) - low) / max(high - low, 1.0e-6)
        )
    return np.column_stack(columns)


def predict(scores: np.ndarray, eta1: float, eta2: float) -> np.ndarray:
    result = np.zeros(len(scores), dtype=np.int64)
    result[scores >= eta1] = 1
    result[scores >= eta2] = 2
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision = np.zeros(3, dtype=float)
    recall = np.zeros(3, dtype=float)
    f1 = np.zeros(3, dtype=float)
    for label in range(3):
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        precision[label] = true_positive / max(
            true_positive + false_positive, 1
        )
        recall[label] = true_positive / max(true_positive + false_negative, 1)
        denominator = precision[label] + recall[label]
        f1[label] = (
            2.0 * precision[label] * recall[label] / denominator
            if denominator > 0.0
            else 0.0
        )
    low_count = int((y_true == 0).sum())
    far = float(((y_true == 0) & (y_pred != 0)).sum() / max(low_count, 1))
    return {
        "macro_f1": float(f1.mean()),
        "high_recall": float(recall[2]),
        "far": far,
        "macro_precision": float(precision.mean()),
    }


def component_grid() -> list[tuple[float, float, float]]:
    values = np.arange(0.0, 1.0001, 0.1)
    grid = {
        tuple(round(float(value), 10) for value in weights)
        for weights in itertools.product(values, repeat=3)
        if abs(sum(weights) - 1.0) <= 1.0e-9
    }
    grid.update(
        {
            (0.34, 0.33, 0.33),
            (0.50, 0.25, 0.25),
            (0.25, 0.50, 0.25),
            (0.25, 0.25, 0.50),
        }
    )
    return sorted(grid)


def threshold_grid(scores: np.ndarray) -> tuple[list[float], list[float]]:
    q1 = np.arange(0.45, 0.651, 0.025)
    q2 = np.arange(0.70, 0.901, 0.025)
    return (
        sorted({float(np.quantile(scores, q)) for q in q1}),
        sorted({float(np.quantile(scores, q)) for q in q2}),
    )


def best_thresholds(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    eta1_values, eta2_values = threshold_grid(scores)
    best: dict[str, float] | None = None
    for eta1 in eta1_values:
        for eta2 in eta2_values:
            if eta2 <= eta1:
                continue
            candidate = metrics(y_true, predict(scores, eta1, eta2))
            candidate.update({"eta_1": eta1, "eta_2": eta2})
            key = (
                candidate["macro_f1"],
                candidate["high_recall"],
                -candidate["far"],
            )
            if best is None or key > (
                best["macro_f1"],
                best["high_recall"],
                -best["far"],
            ):
                best = candidate
    if best is None:
        raise RuntimeError("No valid threshold pair")
    return best


def load_seed(root: Path, seed: int, method_dir: str) -> dict[str, object]:
    details = root / f"seed_{seed}" / method_dir / "details"
    train = pd.read_csv(details / "train_details.csv")
    val = pd.read_csv(details / "val_details.csv")
    stats = fit_stats(train)
    return {
        "stats": stats,
        "val_y": val["risk_label_true"].to_numpy(np.int64),
        "val_components": normalized_components(val, stats),
        "val_logit": soft_risk(val),
        "details": details,
    }


def fused_scores(
    payload: dict[str, object],
    weights: tuple[float, float, float],
    mix_logit: float,
    split: str,
) -> np.ndarray:
    component_values = payload[f"{split}_components"]
    logit_values = payload[f"{split}_logit"]
    return (
        mix_logit * logit_values
        + (1.0 - mix_logit)
        * (component_values @ np.asarray(weights, dtype=float))
    )


def main() -> None:
    args = parse_args()
    runs = {
        seed: load_seed(args.root, seed, args.method_dir)
        for seed in SEEDS
    }
    candidates = []
    for mix_logit in np.arange(0.0, 1.0001, 0.1):
        for weights in component_grid():
            per_seed = {}
            valid_candidate = True
            for seed, payload in runs.items():
                score = fused_scores(payload, weights, float(mix_logit), "val")
                try:
                    per_seed[seed] = best_thresholds(score, payload["val_y"])
                except RuntimeError:
                    valid_candidate = False
                    break
            if not valid_candidate:
                continue
            macro_values = [entry["macro_f1"] for entry in per_seed.values()]
            hrr_values = [entry["high_recall"] for entry in per_seed.values()]
            far_values = [entry["far"] for entry in per_seed.values()]
            candidates.append(
                {
                    "mix_logit": float(mix_logit),
                    "component_weights": list(weights),
                    "val_macro_mean": float(np.mean(macro_values)),
                    "val_macro_std": float(np.std(macro_values, ddof=1)),
                    "val_hrr_mean": float(np.mean(hrr_values)),
                    "val_far_mean": float(np.mean(far_values)),
                    "per_seed": per_seed,
                }
            )
    candidates.sort(
        key=lambda item: (
            item["val_macro_mean"],
            item["val_hrr_mean"],
            -item["val_far_mean"],
        ),
        reverse=True,
    )
    result = {
        "selection_data": "train normalization plus validation calibration",
        "test_read": bool(args.evaluate_test),
        "best": candidates[0],
        "top_10": candidates[:10],
    }

    if args.evaluate_test:
        frozen = candidates[0]
        test_rows = {}
        for seed, payload in runs.items():
            test = pd.read_csv(payload["details"] / "test_details.csv")
            payload["test_y"] = test["risk_label_true"].to_numpy(np.int64)
            payload["test_components"] = normalized_components(test, payload["stats"])
            payload["test_logit"] = soft_risk(test)
            score = fused_scores(
                payload,
                tuple(frozen["component_weights"]),
                frozen["mix_logit"],
                "test",
            )
            calibration = frozen["per_seed"][seed]
            test_rows[seed] = metrics(
                payload["test_y"],
                predict(score, calibration["eta_1"], calibration["eta_2"]),
            )
        result["test"] = {
            "per_seed": test_rows,
            "macro_mean": float(
                np.mean([row["macro_f1"] for row in test_rows.values()])
            ),
            "macro_std": float(
                np.std(
                    [row["macro_f1"] for row in test_rows.values()],
                    ddof=1,
                )
            ),
            "hrr_mean": float(
                np.mean([row["high_recall"] for row in test_rows.values()])
            ),
            "far_mean": float(np.mean([row["far"] for row in test_rows.values()])),
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
