


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = (
    "macro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "high_recall",
    "far",
    "high_f1",
)
FINAL_METHODS = {"TCR-Net", "w/o Rule"}
FAIR_BASELINES = {
    "XGBoost",
    "XGBoost+Rule",
    "GRU",
    "GRU+Rule",
    "RandomForest",
    "SVM",
    "LogisticRegression",
}
EXPECTED_SEEDS = (42, 43, 44, 45, 46)


def load_results(root: Path, suite: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((root / suite).glob("seed_*/*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("suite") != suite:
            continue
        test = payload["metrics"]["test"]
        row = {
            "suite": suite,
            "method": payload["method"],
            "kind": payload["kind"],
            "seed": int(payload["seed"]),
            "data_variant": payload["data_variant"],
            "config_sha256": payload["config_sha256"],
            "source_result": str(path.resolve()),
        }
        row.update({metric: float(test[metric]) for metric in METRICS})
        rows.append(row)
    return rows


def validate_protocol(frame: pd.DataFrame, methods: list[str]) -> None:
    for method in methods:
        group = frame[frame["method"] == method]
        seeds = tuple(sorted(group["seed"].astype(int).tolist()))
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{method}: expected seeds {EXPECTED_SEEDS}, found {seeds}"
            )
        if set(group["data_variant"]) != {"fixed_data"}:
            raise ValueError(f"{method}: non-matching data variants")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for method, group in frame.groupby("method", sort=False):
        row = {"method": method, "n_runs": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["macro_f1_mean", "high_recall_mean"],
        ascending=[False, False],
    )


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(index="seed", columns="method", values="macro_f1")
    full = pivot["TCR-Net"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260730)
    rows: list[dict] = []
    for method in pivot.columns:
        if method == "TCR-Net":
            continue
        other = pivot[method].to_numpy(dtype=float)
        diff = full - other
        boot = np.array([
            rng.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(100_000)
        ])
        t_two_sided = stats.ttest_rel(full, other, alternative="two-sided")
        t_one_sided = stats.ttest_rel(full, other, alternative="greater")
        try:
            w_two_sided = stats.wilcoxon(
                full,
                other,
                alternative="two-sided",
                method="exact",
            )
            w_one_sided = stats.wilcoxon(
                full,
                other,
                alternative="greater",
                method="exact",
            )
            wilcoxon_two_sided_p = float(w_two_sided.pvalue)
            wilcoxon_one_sided_p = float(w_one_sided.pvalue)
        except ValueError:
            wilcoxon_two_sided_p = float("nan")
            wilcoxon_one_sided_p = float("nan")
        rows.append({
            "comparison": f"TCR-Net vs {method}",
            "mean_paired_difference": float(diff.mean()),
            "paired_difference_std": float(diff.std(ddof=1)),
            "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
            "paired_t_two_sided_p": float(t_two_sided.pvalue),
            "paired_t_one_sided_p": float(t_one_sided.pvalue),
            "wilcoxon_exact_two_sided_p": wilcoxon_two_sided_p,
            "wilcoxon_exact_one_sided_p": wilcoxon_one_sided_p,
            "full_wins": int(np.sum(diff > 0)),
            "n_pairs": int(len(diff)),
        })
    return pd.DataFrame(rows).sort_values(
        "mean_paired_difference", ascending=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-root",
        default="exp_data/revision_2026",
    )
    parser.add_argument(
        "--final-root",
        default="exp_data/revision_2026_final",
    )
    parser.add_argument(
        "--fair-root",
        default="exp_data/revision_2026_fair",
    )
    args = parser.parse_args()

    original_root = Path(args.original_root)
    final_root = Path(args.final_root)
    fair_root = Path(args.fair_root)
    output_dir = final_root / "final_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_core = load_results(original_root, "core")
    final_core = load_results(final_root, "core")
    fair_core = load_results(fair_root, "core")
    combined_rows = [
        row
        for row in original_core
        if row["method"] not in FINAL_METHODS | FAIR_BASELINES
    ] + [
        row for row in final_core if row["method"] in FINAL_METHODS
    ] + [
        row for row in fair_core if row["method"] in FAIR_BASELINES
    ]
    core = pd.DataFrame(combined_rows)
    methods = sorted(core["method"].unique().tolist())
    validate_protocol(core, methods)

    ablations = pd.DataFrame(load_results(final_root, "ablations"))
    ablation_methods = sorted(ablations["method"].unique().tolist())
    validate_protocol(ablations, ablation_methods)

    core.to_csv(output_dir / "raw_core_results.csv", index=False)
    summarize(core).to_csv(output_dir / "main_results.csv", index=False)
    paired_tests(core).to_csv(
        output_dir / "paired_significance.csv", index=False,
    )
    summarize(ablations).to_csv(
        output_dir / "ablation_results.csv", index=False,
    )

    provenance = {
        "selection_protocol": (
            "TCR-Net design and lambda_proto selected on validation metrics; "
            "GRU and XGBoost select native argmax or the common ordered-threshold "
            "grid on validation metrics; all decisions are frozen for test"
        ),
        "expected_seeds": list(EXPECTED_SEEDS),
        "fixed_data_only": True,
        "final_methods_source": str((final_root / "core").resolve()),
        "fair_baselines_source": str((fair_root / "core").resolve()),
        "retained_controls_source": str((original_root / "core").resolve()),
        "final_ablations_source": str(
            (final_root / "ablations").resolve()
        ),
        "fair_baselines": sorted(set(methods).intersection(FAIR_BASELINES)),
        "retained_controls": sorted(
            set(methods).difference(FINAL_METHODS | FAIR_BASELINES)
        ),
        "notes": [
            "All means and sample standard deviations use the same five seeds.",
            "Paired tests compare seed-matched Macro-F1 values.",
            "Rule-aware GRU and XGBoost controls receive the same typed rule "
            "features as the TCR-Net joint head.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summarize(core).to_string(index=False))
    print()
    print(paired_tests(core).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
