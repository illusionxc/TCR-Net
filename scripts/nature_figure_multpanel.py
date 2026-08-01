


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_style() -> None:

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#9A9A9A",
            "axes.facecolor": "#F4F4F4",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "grid.color": "#BDBDBD",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.55,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


COLORS = {
    "TCR-Net": "#C0392B",
    "Unified-Multimodal": "#2E86DE",
    "w/o Rule": "#8E44AD",
    "Sequence+Semantic": "#16A085",
    "Content+Semantic": "#F39C12",
    "Content-Only": "#95A5A6",
    "Rule-Based": "#7F8C8D",
}
ABLATION_COLORS = {
    "Full": "#C0392B",
    "w/o Consistency": "#2E86DE",
    "w/o Sequence": "#16A085",
    "w/o Rule": "#8E44AD",
    "w/o Prototype": "#F39C12",
    "w/o transformer": "#95A5A6",
}
SUBSET_COLORS = {"all": "#C0392B", "rule_active": "#2E86DE", "clean": "#16A085"}
SCENARIO_COLORS = {
    "object_type": "#C0392B",
    "source_type": "#2E86DE",
    "intent_id": "#8E44AD",
    "rule_active": "#16A085",
}


def panel_tag(ax, text: str) -> None:
    ax.text(
        -0.16,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def load_data(report_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "main": pd.read_csv(report_dir / "tab_5_6_main_overall_performance.csv"),
        "perclass": pd.read_csv(report_dir / "tab_5_7_per_risk_class_results.csv"),
        "ablation": pd.read_csv(report_dir / "tab_5_8_ablation_results.csv"),
        "cross": pd.read_csv(report_dir / "fig_5_10_cross_scenario_data.csv"),
        "perturb": pd.read_csv(report_dir / "fig_5_11_distribution_perturbation_data.csv"),
    }


def draw_panel1(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    main = data["main"].copy()
    perclass = data["perclass"].copy()
    ablation = data["ablation"].copy()

    fig = plt.figure(figsize=(8.2, 5.3))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])


    main = main.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    x = np.arange(len(main))
    bars = ax1.bar(
        x,
        main["Macro-F1"].values * 100.0,
        color=[COLORS.get(m, "#777777") for m in main["Method"]],
        edgecolor="white",
        linewidth=0.4,
    )
    for b, v in zip(bars, main["Macro-F1"].values * 100.0):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.1f}", ha="center", fontsize=6.4)
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["TCR", "Unified", "-Rule", "Seq+Sem", "C+Sem", "Content", "Rule"], rotation=20, ha="right"
    )
    ax1.set_ylim(45, 86)
    ax1.grid(axis="y")
    panel_tag(ax1, "(a)")


    top = ["TCR-Net", "Unified-Multimodal", "w/o Rule", "Sequence+Semantic"]
    d = perclass[perclass["Method"].isin(top)].copy()
    d["Method"] = pd.Categorical(d["Method"], categories=top, ordered=True)
    d = d.sort_values("Method")
    xx = np.arange(len(top))
    w = 0.24
    rec_cols = [("Low-R", "#2E86DE", "Low-R"), ("Mid-R", "#F39C12", "Mid-R"), ("High-R", "#C0392B", "High-R")]
    for i, (col, c, lb) in enumerate(rec_cols):
        vals = d[col].values * 100.0
        ax2.bar(xx + (i - 1) * w, vals, width=w, color=c, edgecolor="white", linewidth=0.35, label=lb)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(["TCR", "Unified", "-Rule", "Seq+Sem"])
    ax2.set_ylabel("Recall (%)")
    ax2.set_ylim(20, 104)
    ax2.grid(axis="y")
    ax2.legend(loc="upper left", ncol=3, columnspacing=0.9, handlelength=1.2)
    panel_tag(ax2, "(b)")


    x2 = np.arange(len(main))
    ax3.plot(x2, main["HRR"].values * 100.0, color="#C0392B", marker="o", linewidth=1.6, label="HRR")
    ax3.plot(x2, main["FAR"].values * 100.0, color="#2E86DE", marker="s", linewidth=1.4, label="FAR")
    ax3.set_xticks(x2)
    ax3.set_xticklabels(
        ["TCR", "Unified", "-Rule", "Seq+Sem", "C+Sem", "Content", "Rule"], rotation=20, ha="right"
    )
    ax3.set_ylabel("Rate (%)")
    ax3.set_ylim(0, 82)
    ax3.grid(axis="y")
    ax3.legend(loc="upper right")
    panel_tag(ax3, "(c)")


    ab = ablation.copy()
    order = ["Full", "w/o Consistency", "w/o Sequence", "w/o Rule", "w/o Prototype", "w/o transformer"]
    ab["Model"] = pd.Categorical(ab["Model"], categories=order, ordered=True)
    ab = ab.sort_values("Model")
    xx2 = np.arange(len(ab))
    ax4.bar(
        xx2,
        ab["Macro-F1"].values * 100.0,
        color=[ABLATION_COLORS.get(m, "#888888") for m in ab["Model"]],
        edgecolor="white",
        linewidth=0.35,
        label="Macro-F1",
    )
    ax4.plot(xx2, ab["HRR"].values * 100.0, color="#2E86DE", marker="o", linewidth=1.2, label="HRR")
    ax4.set_xticks(xx2)
    ax4.set_xticklabels(["Full", "-Cons", "-Seq", "-Rule", "-Proto", "-TFM"], rotation=20, ha="right")
    ax4.set_ylabel("Score (%)")
    ax4.set_ylim(60, 86)
    ax4.grid(axis="y")
    ax4.legend(loc="upper right")
    panel_tag(ax4, "(d)")

    fig_path_pdf = out_dir / "fig_nature_panel1_overall.pdf"
    fig.savefig(fig_path_pdf)
    plt.close(fig)
    print(f"=> {fig_path_pdf}")


def draw_panel2(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    perturb = data["perturb"].copy()
    models = ["TCR-Net", "Unified-Multimodal", "w/o Rule", "Sequence+Semantic"]
    subsets = ["all", "rule_active", "clean"]

    fig = plt.figure(figsize=(8.2, 2.9))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, subset in zip(axes, subsets):
        for m in models:
            d = perturb[(perturb["Model"] == m) & (perturb["Subset"] == subset)].sort_values("Eta2Quantile")
            if d.empty:
                continue
            ax.plot(
                d["Eta2Quantile"].values,
                d["Macro-F1"].values * 100.0,
                color=COLORS.get(m, "#777777"),
                linewidth=1.5 if m == "TCR-Net" else 1.1,
                marker="o",
                markersize=2.4,
                alpha=0.98,
                label=m,
            )
        ax.set_title(f"{subset}", fontsize=8.2)
        ax.set_xlabel(r"$\eta_2$ quantile")
        if ax is axes[0]:
            ax.set_ylabel("Macro-F1 (%)")
        ax.set_xlim(0.675, 0.965)
        ax.set_ylim(34, 86)
        ax.grid(True)

    panel_tag(axes[0], "(a)")
    panel_tag(axes[1], "(b)")
    panel_tag(axes[2], "(c)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05), handlelength=2.5)

    fig_path_pdf = out_dir / "fig_nature_panel2_thresholds.pdf"
    fig.savefig(fig_path_pdf)
    plt.close(fig)
    print(f"=> {fig_path_pdf}")


def draw_panel3(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    cross = data["cross"].copy()
    models = ["TCR-Net", "Unified-Multimodal", "w/o Rule", "Sequence+Semantic"]

    fig = plt.figure(figsize=(8.2, 5.1))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])


    d_obj = cross[cross["ScenarioKey"] == "object_type"]
    buckets = sorted(d_obj["Bucket"].unique())
    x = np.arange(len(buckets))
    w = 0.18
    for i, m in enumerate(models):
        d = d_obj[d_obj["Model"] == m].sort_values("Bucket")
        ax1.bar(
            x + (i - 1.5) * w,
            d["Macro-F1"].values * 100.0,
            width=w,
            color=COLORS[m],
            edgecolor="white",
            linewidth=0.35,
            label=m if i == 0 else None,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"B{k}" for k in buckets])
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_title("By object type")
    ax1.set_ylim(40, 92)
    ax1.grid(axis="y")
    panel_tag(ax1, "(a)")


    d_src = cross[cross["ScenarioKey"] == "source_type"]
    buckets = sorted(d_src["Bucket"].unique())
    x = np.arange(len(buckets))
    for i, m in enumerate(models):
        d = d_src[d_src["Model"] == m].sort_values("Bucket")
        ax2.bar(
            x + (i - 1.5) * w,
            d["Macro-F1"].values * 100.0,
            width=w,
            color=COLORS[m],
            edgecolor="white",
            linewidth=0.35,
            label=m,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{k}" for k in buckets])
    ax2.set_title("By source type")
    ax2.set_ylim(40, 92)
    ax2.grid(axis="y")
    panel_tag(ax2, "(b)")


    d_int = cross[cross["ScenarioKey"] == "intent_id"]
    for m in models:
        d = d_int[d_int["Model"] == m].sort_values("Bucket")
        ax3.plot(
            d["Bucket"].values,
            d["Macro-F1"].values * 100.0,
            color=COLORS[m],
            linewidth=1.5 if m == "TCR-Net" else 1.1,
            marker="o",
            markersize=2.6,
            label=m,
        )
    ax3.set_xlabel("Intent bucket")
    ax3.set_ylabel("Macro-F1 (%)")
    ax3.set_xticks(sorted(d_int["Bucket"].unique()))
    ax3.set_ylim(52, 90)
    ax3.grid(True)
    panel_tag(ax3, "(c)")


    rows = []
    for m in models:
        dm = cross[cross["Model"] == m]
        for sk in ["object_type", "source_type", "intent_id", "rule_active"]:
            d = dm[dm["ScenarioKey"] == sk]
            if d.empty:
                continue
            rows.append({"Model": m, "Scenario": sk, "Macro-F1": d["Macro-F1"].mean() * 100.0})
    mean_df = pd.DataFrame(rows)
    scen_order = ["object_type", "source_type", "intent_id", "rule_active"]
    mean_df["Scenario"] = pd.Categorical(mean_df["Scenario"], categories=scen_order, ordered=True)
    mean_df = mean_df.sort_values(["Scenario", "Model"])

    x = np.arange(len(scen_order))
    for i, m in enumerate(models):
        d = mean_df[mean_df["Model"] == m].set_index("Scenario").reindex(scen_order)
        ax4.plot(
            x,
            d["Macro-F1"].values,
            marker="o",
            linewidth=1.5 if m == "TCR-Net" else 1.1,
            color=COLORS[m],
            label=m,
        )
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Obj", "Src", "Intent", "Rule"])
    ax4.set_title("Scenario-wise mean")
    ax4.set_ylim(52, 90)
    ax4.grid(True)
    panel_tag(ax4, "(d)")

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), handlelength=2.3)

    fig_path_pdf = out_dir / "fig_nature_panel3_scenarios.pdf"
    fig.savefig(fig_path_pdf)
    plt.close(fig)
    print(f"=> {fig_path_pdf}")


def main() -> None:
    apply_style()
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "exp_data" / "TCRNet_Final_Figures" / "tcr_paper" / "report_data"
    out_dir = root / "exp_data" / "TCRNet_Final_Figures" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(report_dir)
    draw_panel1(data, out_dir)
    draw_panel2(data, out_dir)
    draw_panel3(data, out_dir)


if __name__ == "__main__":
    main()
