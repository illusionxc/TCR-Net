


from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "exp_data" / "revision_2026"
REFERENCE_REVISION: Path | None = None
AGGREGATE = REVISION / "aggregated" / "aggregate_results.csv"
FIGURE_DIR = ROOT / "figures_auto"
FIGURE_DATA_DIR = REVISION / "figure_data"

SEEDS = [42, 43, 44, 45, 46]
TARGETED_SEEDS = [42, 43, 44]

ORIGINAL_METHODS = [
    "TCR-Net",
    "Unified-Multimodal",
    "w/o Rule",
    "Sequence+Semantic",
    "Content+Semantic",
    "Content-Only",
    "Rule-Based",
]
FIG5_METHODS = [
    "RandomForest",
    "SVM",
    "LogisticRegression",
    *ORIGINAL_METHODS,
]
TRANSFER_METHODS = [
    "TCR-Net",
    "GRU",
    "Unified-Multimodal",
    "XGBoost",
]
FAIR_STRONG_METHODS = [
    "TCR-Net",
    "w/o Rule",
    "XGBoost+Rule",
    "GRU+Rule",
    "GRU",
    "XGBoost",
]

METHOD_SLUGS = {
    "RandomForest": "randomforest",
    "SVM": "svm",
    "LogisticRegression": "logisticregression",
    "XGBoost": "xgboost",
    "XGBoost+Rule": "xgboost_rule",
    "GRU": "gru",
    "GRU+Rule": "gru_rule",
    "TCR-Net": "tcr-net",
    "Unified-Multimodal": "unified-multimodal",
    "w/o Rule": "w_o_rule",
    "Sequence+Semantic": "sequence_semantic",
    "Content+Semantic": "content_semantic",
    "Content-Only": "content-only",
    "Rule-Based": "rule-based",
}
SHORT = {
    "RandomForest": "RF",
    "SVM": "SVM",
    "LogisticRegression": "LogReg",
    "XGBoost": "XGB",
    "XGBoost+Rule": "XGB+Rule",
    "GRU": "GRU",
    "GRU+Rule": "GRU+Rule",
    "TCR-Net": "TCR-Net",
    "Unified-Multimodal": "Concat+MLP",
    "w/o Rule": "No-Rule",
    "Sequence+Semantic": "Seq+Sem",
    "Content+Semantic": "Content+Sem",
    "Content-Only": "Content-Only",
    "Rule-Based": "Rule-Based",
    "Bidirectional": "Bi",
    "Current-to-History": "C-to-H",
    "History-to-Current": "H-to-C",
    "No Cross-Attention": "No-CA",
}


FIG1_COLORS = {
    "RandomForest": "#D35400",
    "SVM": "#2C3E50",
    "LogisticRegression": "#27AE60",
    "TCR-Net": "#C0392B",
    "Unified-Multimodal": "#2E86DE",
    "w/o Rule": "#8E44AD",
    "Sequence+Semantic": "#16A085",
    "Content+Semantic": "#F39C12",
    "Content-Only": "#7F8C8D",
    "Rule-Based": "#1ABC9C",
}
FIG2_COLORS = {
    "TCR-Net": "#E74C3C",
    "Unified-Multimodal": "#1F77B4",
    "w/o Rule": "#9467BD",
    "Sequence+Semantic": "#2CA02C",
    "Content+Semantic": "#FF7F0E",
    "Content-Only": "#8C564B",
    "Rule-Based": "#17BECF",
}
STRONG_COLORS = {
    "TCR-Net": "#C0392B",
    "w/o Rule": "#8E44AD",
    "GRU": "#F39C12",
    "GRU+Rule": "#5B3C88",
    "Unified-Multimodal": "#2E86DE",
    "XGBoost": "#16A085",
    "XGBoost+Rule": "#117864",
}
SUPP_METHOD_COLORS = {
    "TCR-Net": "#C43D32",
    "Unified-Multimodal": "#2F7FC1",
    "Rule-Based": "#4F616E",
}
RISK_LOW = "#2F7FC1"
RISK_MID = "#D08A2D"
RISK_HIGH = "#C43D32"
ORIGINAL_ORANGE = "#F39C12"
ORIGINAL_BLUE = "#2E86DE"
GRID_COLOR = "#A9A9A9"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 7.8,
            "axes.linewidth": 0.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.2,
            "figure.dpi": 150,
            "savefig.dpi": 500,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def remove_axes_frames(fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.set_frame_on(False)
        ax.patch.set_edgecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)


def save_figure_family(fig: plt.Figure, stem: str) -> None:
    remove_axes_frames(fig)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        dpi=500,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=500,
        facecolor="white",
        edgecolor="none",
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.04,
        f"({label})",
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_y_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.45, zorder=0)


def add_experiment_note(
    fig: plt.Figure, text: str, *, bottom: bool = False
) -> None:
    fig.text(
        0.995,
        0.005 if bottom else 0.995,
        text,
        ha="right",
        va="bottom" if bottom else "top",
        fontsize=6.2,
        color="#555555",
    )


def aggregate_results() -> pd.DataFrame:
    final = pd.read_csv(AGGREGATE)
    comparison_path = REVISION / "final_comparison" / "main_results.csv"
    if comparison_path.exists():
        fair_core = pd.read_csv(comparison_path)
        fair_core["suite"] = "core"
        fair_core["data_variant"] = "fixed_data"
        fair_methods = set(fair_core["method"])
        final = final[
            ~(
                (final["suite"] == "core")
                & (final["data_variant"] == "fixed_data")
                & final["method"].isin(fair_methods)
            )
        ]
        final = pd.concat([final, fair_core], ignore_index=True, sort=False)
    if REFERENCE_REVISION is None:
        return final
    reference_path = (
        REFERENCE_REVISION / "aggregated" / "aggregate_results.csv"
    )
    reference = pd.read_csv(reference_path)
    keys = ["suite", "method", "data_variant"]
    final_keys = set(map(tuple, final[keys].astype(str).to_numpy()))
    reference_keys = reference[keys].astype(str).apply(tuple, axis=1)
    unchanged = reference.loc[~reference_keys.isin(final_keys)]
    return pd.concat([final, unchanged], ignore_index=True)


def detail_path(seed: int, method: str, split: str = "test") -> Path:
    return (
        REVISION
        / "core"
        / f"seed_{seed}"
        / METHOD_SLUGS[method]
        / "details"
        / f"{split}_details.csv"
    )


def load_detail(seed: int, method: str, split: str = "test") -> pd.DataFrame:
    path = detail_path(seed, method, split)
    if not path.exists() and REFERENCE_REVISION is not None:
        path = (
            REFERENCE_REVISION
            / "core"
            / f"seed_{seed}"
            / METHOD_SLUGS[method]
            / "details"
            / f"{split}_details.csv"
        )
    if not path.exists():
        raise FileNotFoundError(f"Missing formal detail file: {path}")
    return pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)


def ordered_aggregate(
    agg: pd.DataFrame,
    suite: str,
    methods: list[str],
    data_variant: str | None = None,
) -> pd.DataFrame:
    frame = agg[(agg["suite"] == suite) & agg["method"].isin(methods)].copy()
    if data_variant is not None:
        frame = frame[frame["data_variant"] == data_variant]
    frame["method"] = pd.Categorical(frame["method"], methods, ordered=True)
    return frame.sort_values("method").reset_index(drop=True)


def errorbar_bars(
    ax: plt.Axes,
    x: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    colors: list[str],
    width: float = 0.80,
) -> None:
    ax.bar(
        x,
        means,
        width=width,
        color=colors,
        yerr=stds,
        capsize=1.8,
        error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#333333"},
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )


def per_class_runs(methods: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for method in methods:
        for seed in SEEDS:
            frame = load_detail(seed, method)
            y_true = frame["risk_label_true"].to_numpy()
            y_pred = frame["risk_label_pred"].to_numpy()
            precision = precision_score(
                y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
            )
            recall = recall_score(
                y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
            )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "low_precision": precision[0],
                    "mid_precision": precision[1],
                    "high_precision": precision[2],
                    "low_recall": recall[0],
                    "mid_recall": recall[1],
                    "high_recall": recall[2],
                    "high_f1": f1_score(
                        y_true == 2, y_pred == 2, zero_division=0
                    ),
                }
            )
    return pd.DataFrame(rows)


def mcnemar_runs(methods: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in SEEDS:
        reference = load_detail(seed, "TCR-Net")
        y_true = reference["risk_label_true"].to_numpy()
        ref_correct = reference["risk_label_pred"].to_numpy() == y_true
        for method in methods:
            if method == "TCR-Net":
                score = 0.0
            else:
                other = load_detail(seed, method)
                if not np.array_equal(
                    reference["sample_id"].to_numpy(),
                    other["sample_id"].to_numpy(),
                ):
                    raise ValueError(f"Unaligned sample IDs for {method}, seed={seed}")
                other_correct = other["risk_label_pred"].to_numpy() == y_true
                b = int(np.sum(ref_correct & ~other_correct))
                c = int(np.sum(~ref_correct & other_correct))
                p_value = (
                    1.0
                    if b + c == 0
                    else float(binomtest(b, b + c, p=0.5).pvalue)
                )
                score = min(-np.log10(max(p_value, 1e-50)), 50.0)
            rows.append(
                {"method": method, "seed": seed, "minus_log10_p": score}
            )
    return pd.DataFrame(rows)


def plot_method_score_distribution() -> None:

    bins = np.linspace(0, 1, 56)
    centers = (bins[:-1] + bins[1:]) / 2
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    styles = [("-", RISK_LOW), ("--", RISK_MID), ("-", RISK_HIGH)]
    labels = ["Low risk", "Medium risk", "High risk"]

    fig, ax = plt.subplots(figsize=(4.05, 2.35), constrained_layout=True)
    for risk, ((linestyle, color), label) in enumerate(zip(styles, labels)):
        densities = []
        for seed in SEEDS:
            frame = load_detail(seed, "TCR-Net")
            values = frame.loc[
                frame["risk_label_true"] == risk, "R_total"
            ].to_numpy()
            hist, _ = np.histogram(values, bins=bins, density=True)
            densities.append(np.convolve(hist, kernel, mode="same"))
        matrix = np.vstack(densities)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        ax.plot(
            centers,
            mean,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        ax.fill_between(
            centers,
            np.maximum(mean - std, 0),
            mean + std,
            color=color,
            alpha=0.13,
            linewidth=0,
        )

    thresholds = []
    for seed in SEEDS:
        frame = load_detail(seed, "TCR-Net")
        thresholds.append(
            [float(frame["eta_1"].iloc[0]), float(frame["eta_2"].iloc[0])]
        )
    thresholds = np.asarray(thresholds)
    eta_mean = thresholds.mean(axis=0)
    eta_std = thresholds.std(axis=0, ddof=1)
    ymax = ax.get_ylim()[1]
    ax.axvspan(0, eta_mean[0], color=RISK_LOW, alpha=0.055, linewidth=0)
    ax.axvspan(
        eta_mean[0], eta_mean[1], color=RISK_MID, alpha=0.065, linewidth=0
    )
    ax.axvspan(eta_mean[1], 1, color=RISK_HIGH, alpha=0.055, linewidth=0)
    for index, color in enumerate([RISK_MID, RISK_HIGH]):
        ax.axvspan(
            eta_mean[index] - eta_std[index],
            eta_mean[index] + eta_std[index],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.axvline(
            eta_mean[index], color=color, linestyle="--", linewidth=0.95
        )
        ax.text(
            eta_mean[index],
            ymax * 0.92,
            f"{eta_mean[index]:.2f}",
            ha="center",
            va="top",
            color=color,
            fontweight="bold",
            fontsize=7.0,
        )
    region_centers = [
        eta_mean[0] / 2,
        (eta_mean[0] + eta_mean[1]) / 2,
        (eta_mean[1] + 1) / 2,
    ]
    for x, label, color in zip(
        region_centers,
        ["Release", "Review", "Block"],
        [RISK_LOW, RISK_MID, RISK_HIGH],
    ):
        ax.text(
            x,
            ymax * 0.035,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold",
            fontsize=7.0,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Final risk score $R_{\mathrm{total}}$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    add_y_grid(ax)
    add_experiment_note(fig, "5 training seeds; bands = sample SD")
    save_figure_family(fig, "fig_method_fig4")
    plt.close(fig)


def plot_fig_5_1(agg: pd.DataFrame) -> None:

    core = ordered_aggregate(agg, "core", FIG5_METHODS, "fixed_data")
    if len(core) != len(FIG5_METHODS):
        missing = sorted(set(FIG5_METHODS) - set(core["method"].astype(str)))
        raise ValueError(f"Fig. 5 missing methods: {missing}")
    per_class = per_class_runs(FIG5_METHODS)
    mcnemar = mcnemar_runs(FIG5_METHODS)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.to_csv(FIGURE_DATA_DIR / "fig5_overall_repeated.csv", index=False)
    per_class.to_csv(FIGURE_DATA_DIR / "fig5_per_class_runs.csv", index=False)
    mcnemar.to_csv(FIGURE_DATA_DIR / "fig5_mcnemar_runs.csv", index=False)

    fig, axes = plt.subplots(
        2, 5, figsize=(12.8, 4.75), constrained_layout=True
    )
    axes = axes.ravel()
    x = np.arange(len(core))
    labels = [SHORT[str(item)] for item in core["method"]]
    colors = [FIG1_COLORS[str(item)] for item in core["method"]]

    for ax, metric, ylabel, panel, limits in [
        (axes[0], "macro_f1", "Macro-F1 (%)", "a", (48, 86)),
        (axes[1], "weighted_f1", "Weighted-F1 (%)", "b", (54, 90)),
    ]:
        means = core[f"{metric}_mean"].to_numpy() * 100
        stds = core[f"{metric}_std"].to_numpy() * 100
        errorbar_bars(ax, x, means, stds, colors, width=0.90)
        ax.set_xticks(x, labels, rotation=27, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*limits)
        add_y_grid(ax)
        add_panel_label(ax, panel)

    ax = axes[2]
    width = 0.43
    for offset, metric, alpha in [
        (-width / 2, "precision", 0.95),
        (width / 2, "recall", 0.52),
    ]:
        ax.bar(
            x + offset,
            core[f"{metric}_mean"].to_numpy() * 100,
            width,
            yerr=core[f"{metric}_std"].to_numpy() * 100,
            capsize=1.5,
            color=colors,
            alpha=alpha,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
    ax.set_xticks(x, labels, rotation=27, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(45, 96)
    ax.legend(
        handles=[
            Patch(facecolor="#666666", alpha=0.95, label="Precision"),
            Patch(facecolor="#666666", alpha=0.52, label="Recall"),
        ],
        loc="lower right",
    )
    add_y_grid(ax)
    add_panel_label(ax, "c")

    ax = axes[3]
    ax.errorbar(
        x,
        core["high_recall_mean"].to_numpy() * 100,
        yerr=core["high_recall_std"].to_numpy() * 100,
        color=RISK_HIGH,
        marker="o",
        capsize=1.8,
        label="HRR",
    )
    ax.errorbar(
        x,
        core["far_mean"].to_numpy() * 100,
        yerr=core["far_std"].to_numpy() * 100,
        color=RISK_LOW,
        marker="s",
        capsize=1.8,
        label="FAR",
    )
    ax.set_xticks(x, labels, rotation=27, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 82)
    ax.legend(loc="upper right")
    add_y_grid(ax)
    add_panel_label(ax, "d")

    ax = axes[4]
    sig = (
        mcnemar.groupby("method")["minus_log10_p"]
        .agg(["mean", "std"])
        .reindex(FIG5_METHODS)
        .fillna(0)
    )
    errorbar_bars(
        ax,
        x,
        sig["mean"].to_numpy(),
        sig["std"].to_numpy(),
        colors,
        width=0.90,
    )
    ax.axhline(
        -np.log10(0.05),
        color=RISK_MID,
        linestyle="--",
        linewidth=0.75,
    )
    ax.set_xticks(x, labels, rotation=27, ha="right")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_ylim(0, 42)
    add_y_grid(ax)
    add_panel_label(ax, "e")

    grouped = per_class.groupby("method")
    original_x = np.arange(len(ORIGINAL_METHODS))
    original_labels = [SHORT[item] for item in ORIGINAL_METHODS]
    original_colors = [FIG1_COLORS[item] for item in ORIGINAL_METHODS]
    for ax, metric, title, panel in [
        (axes[5], "low_recall", "Low-risk recall", "f"),
        (axes[6], "mid_recall", "Medium-risk recall", "g"),
        (axes[7], "high_recall", "High-risk recall", "h"),
        (axes[8], "high_f1", "High-risk F1", "i"),
    ]:
        means = (
            grouped[metric].mean().reindex(ORIGINAL_METHODS).to_numpy() * 100
        )
        stds = (
            grouped[metric]
            .std(ddof=1)
            .reindex(ORIGINAL_METHODS)
            .fillna(0)
            .to_numpy()
            * 100
        )
        errorbar_bars(
            ax,
            original_x,
            means,
            stds,
            original_colors,
            width=0.90,
        )
        ax.set_xticks(
            original_x, original_labels, rotation=27, ha="right"
        )
        ax.set_title(title)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(35 if metric == "high_f1" else 20, 105 if metric != "high_f1" else 90)
        add_y_grid(ax)
        add_panel_label(ax, panel)

    ax = axes[9]
    ablation_order = [
        "Full",
        "w/o Consistency",
        "w/o Sequence",
        "w/o Rule",
        "w/o Prototype",
        "w/o transformer",
    ]
    ablation = ordered_aggregate(
        agg, "ablations", ablation_order, "fixed_data"
    )
    xa = np.arange(len(ablation))
    bars = ax.bar(
        xa,
        ablation["macro_f1_mean"].to_numpy() * 100,
        width=0.88,
        yerr=ablation["macro_f1_std"].to_numpy() * 100,
        capsize=1.8,
        color=ORIGINAL_ORANGE,
        edgecolor="none",
        linewidth=0,
        label="Macro-F1",
        zorder=2,
    )
    ax.errorbar(
        xa,
        ablation["high_recall_mean"].to_numpy() * 100,
        yerr=ablation["high_recall_std"].to_numpy() * 100,
        color=ORIGINAL_BLUE,
        marker="o",
        capsize=1.8,
        label="HRR",
        zorder=3,
    )
    ax.set_xticks(
        xa,
        ["Full", "-Cons", "-Seq", "-Rule", "-Proto", "-TFM"],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Score (%)")
    ax.legend(handles=[bars, ax.lines[0]], labels=["Macro-F1", "HRR"])
    add_y_grid(ax)
    add_panel_label(ax, "j")

    add_experiment_note(fig, "5 training seeds; error bars = sample SD")
    save_figure_family(fig, "fig_5_1")
    plt.close(fig)


def rule_active_mask(frame: pd.DataFrame) -> np.ndarray:
    cols = ["V_type", "V_dst", "V_role", "V_time", "V_size"]
    if all(column in frame.columns for column in cols):
        return frame[cols].to_numpy().sum(axis=1) > 0

    raise ValueError("Formal detail export lacks rule-violation columns")


def reference_rule_mask(seed: int, split: str = "test") -> tuple[np.ndarray, np.ndarray]:

    reference = load_detail(seed, "TCR-Net", split)
    return reference["sample_id"].to_numpy(), rule_active_mask(reference)


def threshold_sweep_runs() -> pd.DataFrame:
    quantiles = np.linspace(0.70, 0.95, 7)
    rows: list[dict] = []
    for method in ORIGINAL_METHODS:
        for seed in SEEDS:
            val = load_detail(seed, method, "val")
            test = load_detail(seed, method, "test")
            reference_ids, active = reference_rule_mask(seed, "test")
            if not np.array_equal(test["sample_id"].to_numpy(), reference_ids):
                raise ValueError(
                    f"Unaligned rule-slice sample IDs for {method}, seed={seed}"
                )
            val_score = val["R_total"].to_numpy()
            test_score = test["R_total"].to_numpy()
            eta1 = float(test["eta_1"].iloc[0])
            masks = {
                "all": np.ones(len(test), dtype=bool),
                "rule_active": active,
                "clean": ~active,
            }
            y_true = test["risk_label_true"].to_numpy()
            for quantile in quantiles:
                eta2 = max(float(np.quantile(val_score, quantile)), eta1 + 1e-6)
                y_pred = np.where(
                    test_score < eta1, 0, np.where(test_score < eta2, 1, 2)
                )
                for subset, mask in masks.items():
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "subset": subset,
                            "eta2_quantile": quantile,
                            "macro_f1": f1_score(
                                y_true[mask],
                                y_pred[mask],
                                labels=[0, 1, 2],
                                average="macro",
                                zero_division=0,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def slice_runs() -> pd.DataFrame:
    rows: list[dict] = []
    dimensions = ["object_type", "source_type", "intent_id", "rule_active"]
    for method in ORIGINAL_METHODS:
        for seed in SEEDS:
            frame = load_detail(seed, method).copy()
            reference_ids, active = reference_rule_mask(seed, "test")
            if not np.array_equal(frame["sample_id"].to_numpy(), reference_ids):
                raise ValueError(
                    f"Unaligned rule-slice sample IDs for {method}, seed={seed}"
                )
            frame["rule_active"] = active.astype(int)
            for dimension in dimensions:
                for bucket, group in frame.groupby(dimension, sort=True):
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "dimension": dimension,
                            "bucket": int(bucket),
                            "macro_f1": f1_score(
                                group["risk_label_true"],
                                group["risk_label_pred"],
                                labels=[0, 1, 2],
                                average="macro",
                                zero_division=0,
                            ),
                            "samples": len(group),
                        }
                    )
    return pd.DataFrame(rows)


def plot_fig_5_2() -> None:

    sweep = threshold_sweep_runs()
    slices = slice_runs()
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(FIGURE_DATA_DIR / "fig6_threshold_runs.csv", index=False)
    slices.to_csv(FIGURE_DATA_DIR / "fig6_slice_runs.csv", index=False)

    fig, axes = plt.subplots(
        2, 4, figsize=(12.2, 4.55), constrained_layout=True
    )
    axes = axes.ravel()
    for ax, subset, title, panel in zip(
        axes[:3],
        ["all", "rule_active", "clean"],
        ["All events", "Rule-active events", "No-rule events"],
        ["a", "b", "c"],
    ):


        panel_methods = (
            ["TCR-Net", "Rule-Based"]
            if subset == "rule_active"
            else ORIGINAL_METHODS
        )
        for method in panel_methods:
            data = (
                sweep[
                    (sweep["subset"] == subset)
                    & (sweep["method"] == method)
                ]
                .groupby("eta2_quantile")["macro_f1"]
                .agg(["mean", "std"])
                .reset_index()
            )
            x = data["eta2_quantile"].to_numpy()
            mean = data["mean"].to_numpy() * 100
            std = data["std"].to_numpy() * 100
            color = FIG2_COLORS[method]
            ax.plot(
                x,
                mean,
                color=color,
                marker="o",
                linewidth=1.45 if method == "TCR-Net" else 1.0,
                label=SHORT[method],
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.075,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel(r"Validation quantile for $\eta_2$")
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_xlim(0.675, 0.965)
        ax.set_ylim(34, 88)
        add_y_grid(ax)
        add_panel_label(ax, panel)

    ax = axes[3]
    dimensions = ["object_type", "source_type", "intent_id", "rule_active"]
    dimension_labels = ["Obj", "Src", "Intent", "Rule"]
    scenario = (
        slices.groupby(["method", "seed", "dimension"])["macro_f1"]
        .mean()
        .reset_index()
    )
    xd = np.arange(len(dimensions))
    for method in ORIGINAL_METHODS:
        data = (
            scenario[scenario["method"] == method]
            .groupby("dimension")["macro_f1"]
            .agg(["mean", "std"])
            .reindex(dimensions)
        )
        ax.errorbar(
            xd,
            data["mean"].to_numpy() * 100,
            yerr=data["std"].to_numpy() * 100,
            color=FIG2_COLORS[method],
            marker="o",
            capsize=1.5,
            linewidth=1.3 if method == "TCR-Net" else 0.95,
            label=SHORT[method],
        )
    ax.set_xticks(xd, dimension_labels)
    ax.set_ylabel("Mean bucket Macro-F1 (%)")
    ax.set_ylim(50, 90)
    add_y_grid(ax)
    add_panel_label(ax, "d")

    for offset, dimension in enumerate(dimensions):
        ax = axes[4 + offset]
        data = (
            slices[slices["dimension"] == dimension]
            .groupby(["method", "bucket"])["macro_f1"]
            .agg(["mean", "std"])
            .reset_index()
        )
        buckets = sorted(data["bucket"].unique())
        xb = np.arange(len(buckets))
        width = 0.90 / len(ORIGINAL_METHODS)
        center = (len(ORIGINAL_METHODS) - 1) / 2
        for index, method in enumerate(ORIGINAL_METHODS):
            method_data = (
                data[data["method"] == method]
                .set_index("bucket")
                .reindex(buckets)
            )
            ax.bar(
                xb + (index - center) * width,
                method_data["mean"].to_numpy() * 100,
                width,
                yerr=method_data["std"].to_numpy() * 100,
                capsize=0.9,
                error_kw={
                    "elinewidth": 0.45,
                    "capthick": 0.45,
                    "ecolor": "#333333",
                },
                color=FIG2_COLORS[method],
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
        ax.set_xticks(xb, [f"B{item}" for item in buckets])
        ax.set_title(dimension.replace("_", " "))
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_ylim(38, 92)
        add_y_grid(ax)
        add_panel_label(ax, chr(ord("e") + offset))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=7,
        columnspacing=0.75,
        handlelength=1.5,
    )
    add_experiment_note(
        fig,
        "5 training seeds; bands/error bars = sample SD",
        bottom=True,
    )
    save_figure_family(fig, "fig_5_2")
    plt.close(fig)


def pooled_detail(method: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        frame = load_detail(seed, method).copy()
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def row_normalized_confusion(method: str) -> np.ndarray:
    counts = np.zeros((3, 3), dtype=float)
    for seed in SEEDS:
        frame = load_detail(seed, method)
        counts += confusion_matrix(
            frame["risk_label_true"],
            frame["risk_label_pred"],
            labels=[0, 1, 2],
        )
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)


def plot_confusion_triplet() -> None:

    matrices = [
        ("TCR-Net", row_normalized_confusion("TCR-Net")),
        ("Concat+MLP", row_normalized_confusion("Unified-Multimodal")),
        ("Rule-Based", row_normalized_confusion("Rule-Based")),
    ]
    difference = matrices[0][1] - matrices[2][1]
    fig, axes = plt.subplots(
        2, 2, figsize=(4.35, 3.95), constrained_layout=True
    )
    for index, (ax, (title, matrix)) in enumerate(zip(axes.ravel()[:3], matrices)):
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        for row in range(3):
            for col in range(3):
                color = "white" if matrix[row, col] > 0.55 else "#222222"
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                )
        ax.set_title(title)
        ax.set_xticks([0, 1, 2], ["Low", "Mid", "High"])
        ax.set_yticks([0, 1, 2], ["Low", "Mid", "High"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        add_panel_label(ax, chr(ord("a") + index))

    ax = axes.ravel()[3]
    ax.imshow(
        difference,
        vmin=-0.35,
        vmax=0.35,
        cmap="RdBu_r",
        interpolation="nearest",
    )
    for row in range(3):
        for col in range(3):
            ax.text(
                col,
                row,
                f"{difference[row, col]:+.2f}",
                ha="center",
                va="center",
            )
    ax.set_title("TCR-Net minus Rule-Based")
    ax.set_xticks([0, 1, 2], ["Low", "Mid", "High"])
    ax.set_yticks([0, 1, 2], ["Low", "Mid", "High"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    add_panel_label(ax, "d")
    fig.suptitle(
        "5 training seeds; shared fixed test set",
        fontsize=6.2,
        color="#555555",
    )
    save_figure_family(fig, "fig_nature_supp1_confusion_triplet")
    plt.close(fig)


def style_violin(violin: dict, colors: list[str]) -> None:
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.62)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in violin:
            violin[key].set_color("#303030")
            violin[key].set_linewidth(0.65)


def plot_violin_risk() -> None:

    tcr = pooled_detail("TCR-Net")
    unified = pooled_detail("Unified-Multimodal")
    rule = pooled_detail("Rule-Based")
    fig, axes = plt.subplots(
        1, 2, figsize=(4.75, 2.25), constrained_layout=True
    )

    groups = [
        tcr.loc[tcr["risk_label_true"] == risk, "R_total"].to_numpy()
        for risk in [0, 1, 2]
    ]
    violin = axes[0].violinplot(groups, showmeans=False, showmedians=True)
    style_violin(violin, [RISK_LOW, RISK_MID, RISK_HIGH])
    axes[0].set_xticks([1, 2, 3], ["Low", "Medium", "High"])
    axes[0].set_ylabel(r"Final score $R_{\mathrm{total}}$")
    axes[0].set_title("TCR-Net score separation")
    add_y_grid(axes[0])
    add_panel_label(axes[0], "a")

    high_groups = [
        data.loc[data["risk_label_true"] == 2, "R_total"].to_numpy()
        for data in [tcr, unified, rule]
    ]
    violin = axes[1].violinplot(
        high_groups, showmeans=False, showmedians=True
    )
    style_violin(
        violin,
        [
            SUPP_METHOD_COLORS["TCR-Net"],
            SUPP_METHOD_COLORS["Unified-Multimodal"],
            SUPP_METHOD_COLORS["Rule-Based"],
        ],
    )
    axes[1].set_xticks([1, 2, 3], ["TCR-Net", "Concat+MLP", "Rule-Based"])
    axes[1].tick_params(axis="x", labelrotation=18)
    axes[1].set_ylabel(r"$R_{\mathrm{total}}$ on high-risk events")
    axes[1].set_title("High-risk score comparison")
    add_y_grid(axes[1])
    add_panel_label(axes[1], "b")
    fig.suptitle(
        "5 training seeds; descriptive pooled predictions",
        fontsize=6.2,
        color="#555555",
    )
    save_figure_family(fig, "fig_nature_supp6_violin_risk")
    plt.close(fig)


def plot_revision_targeted_results(agg: pd.DataFrame) -> None:

    fig, axes = plt.subplots(
        2, 4, figsize=(11.0, 5.05), constrained_layout=True
    )
    axes = axes.ravel()

    attention_order = [
        "Bidirectional",
        "Current-to-History",
        "History-to-Current",
        "No Cross-Attention",
    ]
    attention = ordered_aggregate(agg, "attention", attention_order, "fixed_data")
    xa = np.arange(len(attention))
    axes[0].errorbar(
        xa,
        attention["macro_f1_mean"].to_numpy() * 100,
        yerr=attention["macro_f1_std"].to_numpy() * 100,
        color=ORIGINAL_BLUE,
        marker="o",
        capsize=2,
    )
    axes[0].set_xticks(
        xa, [SHORT[item] for item in attention_order], rotation=22
    )
    axes[0].set_title("Attention direction (3 seeds)")
    axes[0].set_ylabel("Macro-F1 (%)")
    add_y_grid(axes[0])
    add_panel_label(axes[0], "a")

    history_order = ["K=4", "K=8", "K=12", "K=16"]
    history = ordered_aggregate(agg, "history", history_order)
    xh = np.arange(len(history))
    axes[1].errorbar(
        xh,
        history["macro_f1_mean"].to_numpy() * 100,
        yerr=history["macro_f1_std"].to_numpy() * 100,
        color=ORIGINAL_BLUE,
        marker="o",
        capsize=2,
    )
    axes[1].set_xticks(xh, ["4", "8", "12", "16"])
    axes[1].set_xlabel("History length K")
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_title("History window (3 seeds)")
    add_y_grid(axes[1])
    add_panel_label(axes[1], "b")

    scenario_order = [
        "target=transmission_inspection",
        "target=renewable_station",
        "target=substation_maintenance",
    ]
    scenario_labels = ["Transmission", "Renewable", "Substation"]
    cross = agg[
        (agg["suite"] == "cross_scenario")
        & agg["method"].isin(TRANSFER_METHODS)
        & agg["data_variant"].isin(scenario_order)
    ].copy()
    xc = np.arange(len(scenario_order))
    width = 0.19
    for index, method in enumerate(TRANSFER_METHODS):
        data = (
            cross[cross["method"] == method]
            .set_index("data_variant")
            .reindex(scenario_order)
        )
        axes[2].bar(
            xc + (index - 1.5) * width,
            data["macro_f1_mean"].to_numpy() * 100,
            width,
            yerr=data["macro_f1_std"].to_numpy() * 100,
            capsize=1.2,
            color=STRONG_COLORS[method],
            edgecolor="none",
            linewidth=0,
            label=SHORT[method],
        )
    axes[2].set_xticks(xc, scenario_labels, rotation=18)
    axes[2].set_ylabel("Macro-F1 (%)")
    axes[2].set_title("Leave-one-scenario-out (3 seeds)")
    axes[2].legend(ncol=2, loc="lower left")
    add_y_grid(axes[2])
    add_panel_label(axes[2], "c")

    generator = ordered_aggregate(
        agg, "generator_robustness", TRANSFER_METHODS, "generator_seed_sweep"
    )
    xg = np.arange(len(generator))
    errorbar_bars(
        axes[3],
        xg,
        generator["macro_f1_mean"].to_numpy() * 100,
        generator["macro_f1_std"].to_numpy() * 100,
        [STRONG_COLORS[str(item)] for item in generator["method"]],
    )
    axes[3].set_xticks(
        xg, [SHORT[str(item)] for item in generator["method"]], rotation=23
    )
    axes[3].set_ylabel("Macro-F1 (%)")
    axes[3].set_title("Generator-seed robustness")
    add_y_grid(axes[3])
    add_panel_label(axes[3], "d")

    ablation_order = [
        "Full",
        "w/o Consistency",
        "w/o Sequence",
        "w/o Rule",
        "w/o Prototype",
        "w/o transformer",
    ]
    ablation = ordered_aggregate(
        agg, "ablations", ablation_order, "fixed_data"
    )
    xab = np.arange(len(ablation))
    width = 0.36
    axes[4].bar(
        xab - width / 2,
        ablation["macro_f1_mean"].to_numpy() * 100,
        width,
        yerr=ablation["macro_f1_std"].to_numpy() * 100,
        capsize=1.5,
        color=ORIGINAL_ORANGE,
        edgecolor="none",
        linewidth=0,
        label="Macro-F1",
    )
    axes[4].bar(
        xab + width / 2,
        ablation["high_recall_mean"].to_numpy() * 100,
        width,
        yerr=ablation["high_recall_std"].to_numpy() * 100,
        capsize=1.5,
        color=ORIGINAL_BLUE,
        edgecolor="none",
        linewidth=0,
        label="HRR",
    )
    axes[4].set_xticks(
        xab,
        ["Full", "-Cons", "-Seq", "-Rule", "-Proto", "-TFM"],
        rotation=24,
    )
    axes[4].set_ylabel("Score (%)")
    axes[4].set_title("Component ablations (5 seeds)")
    axes[4].legend(loc="lower left")
    add_y_grid(axes[4])
    add_panel_label(axes[4], "e")

    strong = ordered_aggregate(agg, "core", FAIR_STRONG_METHODS, "fixed_data")
    xs = np.arange(len(strong))
    for ax, metric, title, panel in [
        (axes[5], "macro_f1", "Strong-baseline Macro-F1", "f"),
        (axes[6], "high_recall", "Strong-baseline HRR", "g"),
        (axes[7], "far", "Strong-baseline FAR", "h"),
    ]:
        errorbar_bars(
            ax,
            xs,
            strong[f"{metric}_mean"].to_numpy() * 100,
            strong[f"{metric}_std"].to_numpy() * 100,
            [STRONG_COLORS[str(item)] for item in strong["method"]],
        )
        ax.set_xticks(
            xs, [SHORT[str(item)] for item in strong["method"]], rotation=23
        )
        ax.set_ylabel("Score (%)" if metric == "macro_f1" else "Rate (%)")
        ax.set_title(title)
        add_y_grid(ax)
        add_panel_label(ax, panel)

    targeted_rows = pd.concat(
        [
            attention.assign(panel="attention"),
            history.assign(panel="history"),
            cross.assign(panel="cross_scenario"),
            generator.assign(panel="generator_robustness"),
            ablation.assign(panel="ablation"),
            strong.assign(panel="strong_baselines"),
        ],
        ignore_index=True,
        sort=False,
    )
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    targeted_rows.to_csv(
        FIGURE_DATA_DIR / "fig9_targeted_aggregate.csv", index=False
    )
    save_figure_family(fig, "fig_revision_targeted_results")
    plt.close(fig)


def main() -> None:
    global REVISION, REFERENCE_REVISION, AGGREGATE, FIGURE_DIR, FIGURE_DATA_DIR

    parser = argparse.ArgumentParser(
        description="Regenerate original TCR-Net figures and the separate revision figure."
    )
    parser.add_argument(
        "--revision-root",
        type=Path,
        default=REVISION,
        help="Formal revision result root.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Destination for PDF/PNG figure files.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help=(
            "Fallback root for unchanged comparator aggregates/details. "
            "Rows present in --revision-root always take precedence."
        ),
    )
    args = parser.parse_args()

    REVISION = args.revision_root.resolve()
    REFERENCE_REVISION = (
        args.reference_root.resolve()
        if args.reference_root is not None
        else None
    )
    AGGREGATE = REVISION / "aggregated" / "aggregate_results.csv"
    FIGURE_DATA_DIR = REVISION / "figure_data"
    FIGURE_DIR = args.figure_dir.resolve()
    configure_style()

    agg = aggregate_results()
    plot_method_score_distribution()
    plot_fig_5_1(agg)
    plot_fig_5_2()
    plot_confusion_triplet()
    plot_violin_risk()
    plot_revision_targeted_results(agg)
    print(f"Updated original and revision figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
