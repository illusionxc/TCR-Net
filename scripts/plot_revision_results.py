


from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "exp_data/revision_2026/aggregated/aggregate_results.csv"
OUTPUT = ROOT / "figures_auto"

COLORS = {
    "TCR-Net": "#0072B2",
    "GRU": "#E69F00",
    "Unified-Multimodal": "#009E73",
    "Sequence+Semantic": "#56B4E9",
    "RandomForest": "#CC79A7",
    "XGBoost": "#D55E00",
    "LogisticRegression": "#999999",
    "SVM": "#666666",
}


def read_rows() -> list[dict[str, str]]:
    with AGGREGATE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict[str, str]], suite: str) -> list[dict[str, str]]:
    return [row for row in rows if row["suite"] == suite]


def metric(row: dict[str, str], name: str) -> tuple[float, float]:
    return float(row[f"{name}_mean"]), float(row[f"{name}_std"])


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def plot_main(rows: list[dict[str, str]]) -> None:
    order = [
        "GRU",
        "TCR-Net",
        "Unified-Multimodal",
        "Sequence+Semantic",
        "RandomForest",
        "XGBoost",
        "LogisticRegression",
        "SVM",
    ]
    lookup = {row["method"]: row for row in select(rows, "core")}
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.35), constrained_layout=True)
    specifications = [
        ("macro_f1", "Macro-F1", (0.58, 0.86)),
        ("high_recall", "High-risk recall", (0.20, 0.67)),
        ("far", "False alarm rate", (-0.005, 0.145)),
    ]
    y = np.arange(len(order))
    for ax, (name, label, limits) in zip(axes, specifications, strict=True):
        for index, method in enumerate(order):
            mean, std = metric(lookup[method], name)
            ax.errorbar(
                mean,
                index,
                xerr=std,
                fmt="o",
                color=COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.2,
                elinewidth=1.0,
                markersize=5.5,
                zorder=3,
            )
        ax.set_xlim(*limits)
        ax.set_xlabel(label)
        ax.set_yticks(y)
        ax.invert_yaxis()
        style_axis(ax)
    axes[0].set_yticklabels(order)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    for label, ax in zip(("a", "b", "c"), axes, strict=True):
        ax.text(
            -0.16,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
    for extension in ("pdf", "png"):
        fig.savefig(
            OUTPUT / f"fig_revision_main_results.{extension}",
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_targeted(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.0), constrained_layout=True)

    ablation_order = [
        "Full",
        "w/o Consistency",
        "w/o Rule",
        "w/o Sequence",
        "w/o transformer",
    ]
    ablation_lookup = {row["method"]: row for row in select(rows, "ablations")}
    ax = axes[0, 0]
    for index, method in enumerate(ablation_order):
        mean, std = metric(ablation_lookup[method], "macro_f1")
        color = "#0072B2" if method == "Full" else "#777777"
        ax.errorbar(mean, index, xerr=std, fmt="o", color=color, capsize=2.2)
    ax.set_yticks(range(len(ablation_order)), ablation_order)
    ax.invert_yaxis()
    ax.set_xlim(0.69, 0.86)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Component ablations", loc="left", fontsize=9)
    style_axis(ax)

    attention_order = [
        "Current-to-History",
        "No Cross-Attention",
        "History-to-Current",
        "Bidirectional",
    ]
    attention_lookup = {row["method"]: row for row in select(rows, "attention")}
    ax = axes[0, 1]
    for index, method in enumerate(attention_order):
        mean, std = metric(attention_lookup[method], "macro_f1")
        color = "#0072B2" if method == "Current-to-History" else "#777777"
        ax.errorbar(mean, index, xerr=std, fmt="o", color=color, capsize=2.2)
    ax.set_yticks(range(len(attention_order)), attention_order)
    ax.invert_yaxis()
    ax.set_xlim(0.81, 0.845)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Cross-attention direction", loc="left", fontsize=9)
    style_axis(ax)

    history_order = ["K=4", "K=8", "K=12", "K=16"]
    history_lookup = {row["method"]: row for row in select(rows, "history")}
    ax = axes[1, 0]
    x = np.arange(len(history_order))
    means = [metric(history_lookup[method], "macro_f1")[0] for method in history_order]
    stds = [metric(history_lookup[method], "macro_f1")[1] for method in history_order]
    ax.errorbar(
        x,
        means,
        yerr=stds,
        color="#0072B2",
        marker="o",
        capsize=2.2,
        linewidth=1.2,
    )
    ax.set_xticks(x, [item.replace("K=", "") for item in history_order])
    ax.set_xlabel("History length K")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.81, 0.865)
    ax.set_title("History-window sensitivity", loc="left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)

    scenario_order = [
        "target=transmission_inspection",
        "target=renewable_station",
        "target=substation_maintenance",
    ]
    scenario_labels = ["Transmission", "Renewable", "Substation"]
    scenario_methods = ["TCR-Net", "GRU", "Unified-Multimodal", "XGBoost"]
    scenario_legend = {
        "TCR-Net": "TCR-Net",
        "GRU": "GRU",
        "Unified-Multimodal": "Unified",
        "XGBoost": "XGB",
    }
    scenario_rows = {
        (row["method"], row["data_variant"]): row
        for row in select(rows, "cross_scenario")
    }
    ax = axes[1, 1]
    width = 0.19
    x = np.arange(len(scenario_order))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(scenario_methods))
    for offset, method in zip(offsets, scenario_methods, strict=True):
        values = [
            metric(scenario_rows[(method, scenario)], "macro_f1")[0]
            for scenario in scenario_order
        ]
        ax.bar(
            x + offset,
            values,
            width,
            label=scenario_legend[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x, scenario_labels, rotation=16, ha="right")
    ax.set_ylim(0.62, 0.91)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Leave-one-scenario-out", loc="left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left")

    for label, ax in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        ax.text(
            -0.18,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
    for extension in ("pdf", "png"):
        fig.savefig(
            OUTPUT / f"fig_revision_targeted_results.{extension}",
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    plot_main(rows)
    plot_targeted(rows)
    print(OUTPUT / "fig_revision_main_results.pdf")
    print(OUTPUT / "fig_revision_targeted_results.pdf")


if __name__ == "__main__":
    main()
