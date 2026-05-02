"""
Unified plotting module (refactored).

：
1)  3  panel figure
2)  4-5 、2 （）
3) //， KDD 
4)  report_data CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Patch


MODEL_ORDER = [
    "TCR-Net",
    "Unified-Multimodal",
    "w/o Rule",
    "Sequence+Semantic",
    "Content+Semantic",
    "Content-Only",
    "Rule-Based",
]
MODEL_SHORT = {
    "TCR-Net": "TCR-Net",
    "Unified-Multimodal": "Unified-Multi",
    "w/o Rule": "No-Rule",
    "Sequence+Semantic": "Seq+Sem",
    "Content+Semantic": "Content+Sem",
    "Content-Only": "Content-Only",
    "Rule-Based": "Rule-Based",
}
# <=4 colors for method comparisons (user requirement)
# Figure-wise method palettes:
# - Each method has unique color within a figure.
# - fig_5_1 and fig_5_2 use different but internally consistent mappings.
FIG1_METHOD_COLORS = {
    "TCR-Net": "#C0392B",            # red
    "Unified-Multimodal": "#2E86DE", # blue
    "w/o Rule": "#8E44AD",           # purple
    "Sequence+Semantic": "#16A085",  # green
    "Content+Semantic": "#F39C12",   # orange
    "Content-Only": "#7F8C8D",       # gray
    "Rule-Based": "#1ABC9C",         # cyan-teal
}
FIG2_METHOD_COLORS = {
    "TCR-Net": "#E74C3C",            # bright red
    "Unified-Multimodal": "#1F77B4", # deep blue
    "w/o Rule": "#9467BD",           # lavender purple
    "Sequence+Semantic": "#2CA02C",  # olive green
    "Content+Semantic": "#FF7F0E",   # warm orange
    "Content-Only": "#8C564B",       # brown
    "Rule-Based": "#17BECF",         # cyan
}
SUBSET_COLORS = {"all": "#E74C3C", "rule_active": "#2E86DE", "clean": "#16A085"}
SCENARIO_COLORS = {
    "object_type": "#E74C3C",
    "source_type": "#2E86DE",
    "intent_id": "#8E44AD",
    "rule_active": "#16A085",
}
RISK_COLORS = {"Low": "#2E86DE", "Medium": "#F39C12", "High": "#E74C3C"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#9A9A9A",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.frameon": False,
            "legend.fontsize": 7.0,
            "grid.color": "#BCBCBC",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _panel(ax, txt: str) -> None:
    ax.text(-0.17, 1.03, txt, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def _style_xticks(ax, labels, rotation: float = 28.0, fs: float = 6.6) -> None:
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fs)


def _annotate_bars(
    ax,
    bars,
    fmt: str = "{:.1f}",
    dy: float = 0.8,
    fs: float = 3.0,
    stagger: bool = False,
    dy_alt: float = 1.35,
    rot: float = 0.0,
) -> None:
    for i, b in enumerate(bars):
        h = b.get_height()
        if np.isnan(h):
            continue
        off = dy_alt if (stagger and (i % 2 == 1)) else dy
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + off,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fs,
            rotation=rot,
        )


def _load_csv(report_dir: Path, name: str) -> pd.DataFrame:
    p = report_dir / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _draw_fig1(main: pd.DataFrame, perclass: pd.DataFrame, ablation: pd.DataFrame, out_dir: Path) -> None:
    # 2x5 large panel figure
    fig = plt.figure(figsize=(12.8, 4.6))
    gs = fig.add_gridspec(2, 5, wspace=0.32, hspace=0.38)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(5)]

    d = main.copy()
    d["Method"] = pd.Categorical(d["Method"], MODEL_ORDER, ordered=True)
    d = d.sort_values("Method").reset_index(drop=True)
    x = np.arange(len(d))

    # (a) Macro-F1 bars
    ax = axes[0]
    vals = d["Macro-F1"].to_numpy() * 100
    bars = ax.bar(x, vals, width=0.92, color=[FIG1_METHOD_COLORS[m] for m in d["Method"]], edgecolor="white", linewidth=0.35)
    ax.set_xticks(x)
    _style_xticks(ax, [MODEL_SHORT[m] for m in d["Method"]], rotation=26.0, fs=6.5)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(48, 86)
    ax.grid(axis="y")
    _annotate_bars(ax, bars, "{:.1f}", dy=0.42, fs=5.3)
    _panel(ax, "(a)")

    # (b) Weighted-F1 bars
    ax = axes[1]
    vals = d["Weighted-F1"].to_numpy() * 100
    bars = ax.bar(x, vals, width=0.92, color=[FIG1_METHOD_COLORS[m] for m in d["Method"]], edgecolor="white", linewidth=0.35)
    ax.set_xticks(x)
    _style_xticks(ax, [MODEL_SHORT[m] for m in d["Method"]], rotation=26.0, fs=6.5)
    ax.set_ylabel("Weighted-F1 (%)")
    ax.set_ylim(54, 88)
    ax.grid(axis="y")
    _annotate_bars(ax, bars, "{:.1f}", dy=0.42, fs=5.3)
    _panel(ax, "(b)")

    # (c) Precision / Recall
    ax = axes[2]
    w = 0.46
    # Keep method-color consistency inside fig_5_1:
    # same method color in every panel; metric difference by alpha only.
    b1 = ax.bar(
        x - w / 2,
        d["Precision"].to_numpy() * 100,
        width=w,
        color=[FIG1_METHOD_COLORS[m] for m in d["Method"]],
        alpha=0.95,
        edgecolor="white",
        linewidth=0.3,
        label="Precision",
    )
    b2 = ax.bar(
        x + w / 2,
        d["Recall"].to_numpy() * 100,
        width=w,
        color=[FIG1_METHOD_COLORS[m] for m in d["Method"]],
        alpha=0.55,
        edgecolor="white",
        linewidth=0.3,
        label="Recall",
    )
    ax.set_xticks(x)
    _style_xticks(ax, [MODEL_SHORT[m] for m in d["Method"]], rotation=26.0, fs=6.5)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(45, 96)
    ax.grid(axis="y")
    style_handles = [
        Patch(facecolor="#666666", edgecolor="white", linewidth=0.3, alpha=0.95, label="Precision"),
        Patch(facecolor="#666666", edgecolor="white", linewidth=0.3, alpha=0.55, label="Recall"),
    ]
    ax.legend(handles=style_handles, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.1)
    _annotate_bars(ax, b1, "{:.1f}", dy=0.28, fs=5.0, stagger=True, dy_alt=0.95)
    _annotate_bars(ax, b2, "{:.1f}", dy=0.28, fs=5.0, stagger=True, dy_alt=0.95)
    _panel(ax, "(c)")

    # (d) HRR and FAR lines
    ax = axes[3]
    ax.plot(x, d["HRR"].to_numpy() * 100, color=RISK_COLORS["High"], marker="o", linewidth=1.5, label="HRR")
    ax.plot(x, d["FAR"].to_numpy() * 100, color=RISK_COLORS["Low"], marker="s", linewidth=1.3, label="FAR")
    ax.set_xticks(x)
    _style_xticks(ax, [MODEL_SHORT[m] for m in d["Method"]], rotation=26.0, fs=6.5)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 82)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    _panel(ax, "(d)")

    # (e) McNemar p-values as -log10(p), clip floor
    ax = axes[4]
    pvals = d["p-value"].copy()
    pvals = pd.to_numeric(pvals, errors="coerce").fillna(1.0)
    score = -np.log10(np.clip(pvals.to_numpy(), 1e-300, 1.0))
    score = np.minimum(score, 40)
    bars = ax.bar(x, score, width=0.92, color=[FIG1_METHOD_COLORS[m] for m in d["Method"]], edgecolor="white", linewidth=0.35)
    ax.set_xticks(x)
    _style_xticks(ax, [MODEL_SHORT[m] for m in d["Method"]], rotation=26.0, fs=6.5)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_ylim(0, 42)
    ax.grid(axis="y")
    _annotate_bars(ax, bars, "{:.1f}", dy=0.3, fs=5.1)
    _panel(ax, "(e)")

    # row2 per-class + ablation
    p = perclass.copy()
    p["Method"] = pd.Categorical(p["Method"], MODEL_ORDER, ordered=True)
    p = p.sort_values("Method").reset_index(drop=True)
    xx = np.arange(len(p))

    def _bar_col(ax_, col, ttl):
        bars = ax_.bar(
            xx,
            p[col].to_numpy() * 100,
            width=0.92,
            color=[FIG1_METHOD_COLORS[m] for m in p["Method"]],
            edgecolor="white",
            linewidth=0.35,
        )
        ax_.set_xticks(xx)
        _style_xticks(ax_, [MODEL_SHORT[m] for m in p["Method"]], rotation=26.0, fs=6.4)
        ax_.set_ylabel("Recall (%)")
        ax_.set_ylim(20, 104)
        ax_.set_title(ttl, fontsize=7.5)
        ax_.grid(axis="y")
        _annotate_bars(ax_, bars, "{:.1f}", dy=0.3, fs=5.1)

    _bar_col(axes[5], "Low-R", "Low-risk Recall")
    _panel(axes[5], "(f)")
    _bar_col(axes[6], "Mid-R", "Medium-risk Recall")
    _panel(axes[6], "(g)")
    _bar_col(axes[7], "High-R", "High-risk Recall")
    _panel(axes[7], "(h)")

    # (i) High-F1
    ax = axes[8]
    bars = ax.bar(
        xx,
        p["High-F1"].to_numpy() * 100,
        width=0.92,
        color=[FIG1_METHOD_COLORS[m] for m in p["Method"]],
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xticks(xx)
    _style_xticks(ax, [MODEL_SHORT[m] for m in p["Method"]], rotation=26.0, fs=6.4)
    ax.set_ylabel("High-F1 (%)")
    ax.set_ylim(35, 90)
    ax.grid(axis="y")
    _annotate_bars(ax, bars, "{:.1f}", dy=0.3, fs=5.1)
    _panel(ax, "(i)")

    # (j) Ablation Macro-F1 + HRR
    ax = axes[9]
    a = ablation.copy()
    order = ["Full", "w/o Consistency", "w/o Sequence", "w/o Rule", "w/o Prototype", "w/o transformer"]
    a["Model"] = pd.Categorical(a["Model"], categories=order, ordered=True)
    a = a.sort_values("Model")
    xa = np.arange(len(a))
    b = ax.bar(xa, a["Macro-F1"].to_numpy() * 100, width=0.92, color="#F39C12", edgecolor="white", linewidth=0.35, label="Macro-F1")
    ax.plot(xa, a["HRR"].to_numpy() * 100, color="#2E86DE", marker="o", linewidth=1.2, label="HRR")
    ax.set_xticks(xa)
    _style_xticks(ax, ["Full", "-Cons", "-Seq", "-Rule", "-Proto", "-TFM"], rotation=24.0, fs=6.4)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(64, 86)
    ax.grid(axis="y")
    # Avoid overlap with bar-top labels.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.4,
    )
    _annotate_bars(ax, b, "{:.1f}", dy=0.25, fs=4.9)
    _panel(ax, "(j)")

    _save(fig, out_dir, "fig_5_1")


def _draw_fig2(cross: pd.DataFrame, perturb: pd.DataFrame, out_dir: Path) -> None:
    # 2x4 panel: threshold curves + cross-scenario slices
    fig = plt.figure(figsize=(12.2, 4.4))
    gs = fig.add_gridspec(2, 4, wspace=0.28, hspace=0.36)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]
    #  7 baseline
    models = MODEL_ORDER.copy()

    # row1: threshold all/rule_active/clean + scenario mean line
    subsets = ["all", "rule_active", "clean"]
    for i, subset in enumerate(subsets):
        ax = axes[i]
        for m in models:
            d = perturb[(perturb["Model"] == m) & (perturb["Subset"] == subset)].sort_values("Eta2Quantile")
            if d.empty:
                continue
            ax.plot(
                d["Eta2Quantile"].to_numpy(),
                d["Macro-F1"].to_numpy() * 100,
                color=FIG2_METHOD_COLORS[m],
                linewidth=1.5 if m == "TCR-Net" else 1.05,
                marker="o",
                markersize=2.4,
                label=MODEL_SHORT[m],
            )
        ax.set_title(f"{subset}", fontsize=7.6)
        ax.set_xlabel(r"$\eta_2$ quantile")
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_xlim(0.675, 0.965)
        ax.set_ylim(34, 86)
        ax.grid(True)
        _panel(ax, f"({chr(ord('a') + i)})")

    # row1 col4: scenario means by model
    ax = axes[3]
    rows = []
    for m in models:
        dm = cross[cross["Model"] == m]
        for sk in ["object_type", "source_type", "intent_id", "rule_active"]:
            d = dm[dm["ScenarioKey"] == sk]
            if not d.empty:
                rows.append({"Model": m, "Scenario": sk, "Macro-F1": d["Macro-F1"].mean() * 100})
    md = pd.DataFrame(rows)
    x = np.arange(4)
    for m in models:
        d = md[md["Model"] == m]
        if d.empty:
            continue
        d = d.set_index("Scenario").reindex(["object_type", "source_type", "intent_id", "rule_active"])
        ax.plot(x, d["Macro-F1"].to_numpy(), marker="o", linewidth=1.3, color=FIG2_METHOD_COLORS[m], label=MODEL_SHORT[m])
    ax.set_xticks(x)
    _style_xticks(ax, ["Obj", "Src", "Intent", "Rule"], rotation=0, fs=7.2)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(52, 90)
    ax.grid(True)
    _panel(ax, "(d)")

    # row2: object/source/intent/rule bars by bucket
    for j, sk in enumerate(["object_type", "source_type", "intent_id", "rule_active"]):
        ax = axes[4 + j]
        dsk = cross[cross["ScenarioKey"] == sk]
        if dsk.empty:
            ax.axis("off")
            continue
        buckets = sorted(dsk["Bucket"].unique())
        xb = np.arange(len(buckets))
        w = min(0.24, 0.9 / max(1, len(models)))

        center_shift = (len(models) - 1) / 2.0
        for i, m in enumerate(models):
            dm = dsk[dsk["Model"] == m].sort_values("Bucket")
            if dm.empty:
                continue
            ax.bar(
                xb + (i - center_shift) * w,
                dm["Macro-F1"].to_numpy() * 100,
                width=w,
                color=FIG2_METHOD_COLORS[m],
                edgecolor="white",
                linewidth=0.3,
            )
        # fig_5_2: 
        ax.set_xticks(xb)
        _style_xticks(ax, [f"B{k}" for k in buckets], rotation=0, fs=6.6)
        ax.set_xlim(-0.5, len(buckets) - 0.5)
        ax.set_title(sk.replace("_", " "), fontsize=7.5)
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_ylim(38, 92)
        ax.grid(axis="y")
        _panel(ax, f"({chr(ord('e') + j)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), handlelength=2.2)
    _save(fig, out_dir, "fig_5_2")


def _draw_fig3(main: pd.DataFrame, perclass: pd.DataFrame, mech: pd.DataFrame, cases: pd.DataFrame, detail_df: Optional[pd.DataFrame], out_dir: Path) -> None:
    # 2x4 panel: mechanism + cases + confusion + score distribution
    fig = plt.figure(figsize=(12.0, 4.4))
    gs = fig.add_gridspec(2, 4, wspace=0.3, hspace=0.36)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]

    # (a) mechanism grouped bars
    ax = axes[0]
    keep_v = ["Full", "w/o Consistency", "w/o Sequence", "w/o Rule"]
    keep_s = ["S_cons", "S_seq", "S_rule"]
    md = mech[mech["Variant"].isin(keep_v) & mech["Subset"].isin(keep_s)].copy()
    pv = md.pivot_table(index="Variant", columns="Subset", values="Macro-F1", aggfunc="mean")
    pv = pv.reindex(keep_v)[keep_s]
    xv = np.arange(len(pv))
    w = 0.29
    cols = ["#2E86DE", "#16A085", "#E74C3C"]
    for i, s in enumerate(keep_s):
        bars = ax.bar(xv + (i - 1) * w, pv[s].to_numpy() * 100, width=w, color=cols[i], edgecolor="white", linewidth=0.35, label=s)
        _annotate_bars(ax, bars, "{:.1f}", dy=0.15, fs=4.6, stagger=True, dy_alt=0.9)
    ax.set_xticks(xv)
    _style_xticks(ax, ["Full", "-Cons", "-Seq", "-Rule"], rotation=20.0, fs=6.4)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(0, 104)
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.0)
    _panel(ax, "(a)")

    # (b) case component heatmap
    ax = axes[1]
    sd = cases[["R_cons", "R_seq", "R_rule"]].to_numpy()
    im = ax.imshow(sd, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["R_cons", "R_seq", "R_rule"], rotation=20, ha="right")
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([str(x).replace("_", " ") for x in cases["Case"]])
    for i in range(sd.shape[0]):
        for j in range(sd.shape[1]):
            ax.text(j, i, f"{sd[i, j]:.2f}", ha="center", va="center", fontsize=6.7, color="black")
    ax.set_title("Case components", fontsize=7.6)
    _panel(ax, "(b)")

    # (c) rule violation heatmap
    ax = axes[2]
    rd = cases[["V_type", "V_dst", "V_role", "V_time", "V_size"]].to_numpy()
    ax.imshow(rd, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Type", "Dst", "Role", "Time", "Size"], rotation=20, ha="right")
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([str(x).replace("_", " ") for x in cases["Case"]])
    for i in range(rd.shape[0]):
        for j in range(rd.shape[1]):
            ax.text(j, i, f"{rd[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="black")
    ax.set_title("Rule evidence", fontsize=7.6)
    _panel(ax, "(c)")

    # (d) confusion matrices, TCR only
    ax = axes[3]
    if detail_df is not None and not detail_df.empty:
        cm = confusion_matrix(detail_df["risk_label_true"], detail_df["risk_label_pred"], labels=[0, 1, 2]).astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1.0)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_yticklabels(["Low", "Mid", "High"])
    else:
        ax.axis("off")
    ax.set_title("TCR confusion", fontsize=7.6)
    _panel(ax, "(d)")

    # (e) per-class precision lines (all 7 methods)
    ax = axes[4]
    p = perclass.copy()
    p["Method"] = pd.Categorical(p["Method"], MODEL_ORDER, ordered=True)
    p = p.sort_values("Method")
    x = np.arange(len(p))
    ax.plot(x, p["Low-P"].to_numpy() * 100, color="#2E86DE", marker="o", linewidth=1.2, label="Low-P")
    ax.plot(x, p["Mid-P"].to_numpy() * 100, color="#F39C12", marker="o", linewidth=1.2, label="Mid-P")
    ax.plot(x, p["High-P"].to_numpy() * 100, color="#E74C3C", marker="o", linewidth=1.2, label="High-P")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in p["Method"]], rotation=20, ha="right")
    ax.set_ylabel("Precision (%)")
    ax.set_ylim(40, 104)
    ax.grid(True)
    ax.legend(loc="lower right")
    _panel(ax, "(e)")

    # (f) per-class high metrics bars (all 7 methods)
    ax = axes[5]
    d = main.copy()
    d["Method"] = pd.Categorical(d["Method"], MODEL_ORDER, ordered=True)
    d = d.sort_values("Method")
    xx = np.arange(len(d))
    ww = 0.46
    b1 = ax.bar(xx - ww / 2, d["HRR"].to_numpy() * 100, width=ww, color=RISK_COLORS["High"], edgecolor="white", linewidth=0.3, label="HRR")
    b2 = ax.bar(xx + ww / 2, d["FAR"].to_numpy() * 100, width=ww, color=RISK_COLORS["Low"], edgecolor="white", linewidth=0.3, label="FAR")
    ax.set_xticks(xx)
    _style_xticks(ax, [MODEL_SHORT[m] for m in d["Method"]], rotation=26.0, fs=6.4)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 82)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    _annotate_bars(ax, b1, "{:.1f}", dy=0.22, fs=4.9, stagger=True, dy_alt=0.9)
    _annotate_bars(ax, b2, "{:.1f}", dy=0.22, fs=4.9, stagger=True, dy_alt=0.9)
    _panel(ax, "(f)")

    # (g) R_total distribution by true class
    ax = axes[6]
    if detail_df is not None and not detail_df.empty:
        for cls, c, lb in [(0, "#2E86DE", "Low"), (1, "#F39C12", "Mid"), (2, "#E74C3C", "High")]:
            vals = detail_df.loc[detail_df["risk_label_true"] == cls, "R_total"].to_numpy()
            if len(vals) > 0:
                ax.hist(vals, bins=28, alpha=0.35, density=True, color=c, label=lb)
        if "eta_1" in detail_df.columns:
            ax.axvline(float(detail_df["eta_1"].iloc[0]), color="#F39C12", linestyle="--", linewidth=1.0)
        if "eta_2" in detail_df.columns:
            ax.axvline(float(detail_df["eta_2"].iloc[0]), color="#E74C3C", linestyle="--", linewidth=1.0)
    ax.set_xlabel("R_total")
    ax.set_ylabel("Density")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    _panel(ax, "(g)")

    # (h) subset sample counts
    ax = axes[7]
    cnt = mech.groupby("Subset", observed=False)["Samples"].max().reindex(["S_cons", "S_seq", "S_rule"]).fillna(0)
    bars = ax.bar(np.arange(len(cnt)), cnt.to_numpy(), color=["#2E86DE", "#16A085", "#E74C3C"], edgecolor="white", linewidth=0.35)
    ax.set_xticks(np.arange(len(cnt)))
    _style_xticks(ax, ["S_cons", "S_seq", "S_rule"], rotation=0, fs=6.8)
    ax.set_ylabel("Samples")
    ax.grid(axis="y")
    _annotate_bars(ax, bars, "{:.0f}", dy=8.0, fs=6.3)
    _panel(ax, "(h)")

    _save(fig, out_dir, "fig_5_3")


def _draw_confusion_triplet(main: pd.DataFrame, out_dir: Path) -> None:
    """Must-have #1: class-wise confusion support (TCR / Unified / Rule-Based)."""
    wanted = ["TCR-Net", "Unified-Multimodal", "Rule-Based"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), constrained_layout=False)
    for ax, m in zip(axes, wanted):
        row = main[main["Method"] == m]
        if row.empty or "DetailCSV" not in row.columns:
            ax.axis("off")
            continue
        p = Path(str(row.iloc[0]["DetailCSV"]))
        if not p.exists():
            ax.axis("off")
            continue
        d = pd.read_csv(p)
        cm = confusion_matrix(d["risk_label_true"], d["risk_label_pred"], labels=[0, 1, 2]).astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1.0)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_yticklabels(["Low", "Mid", "High"])
        ax.set_title(MODEL_SHORT.get(m, m), fontsize=7.4)
    _panel(axes[0], "(a)")
    _panel(axes[1], "(b)")
    _panel(axes[2], "(c)")
    _save(fig, out_dir, "fig_5_confusion_triplet")


def _draw_ablation_delta(ablation: pd.DataFrame, out_dir: Path) -> None:
    """Must-have #2: aligned ΔMacro-F1 / ΔHRR / ΔFAR in one figure."""
    a = ablation.copy()
    full = a[a["Model"] == "Full"]
    if full.empty:
        return
    fm, fh, ff = float(full["Macro-F1"].iloc[0]), float(full["HRR"].iloc[0]), float(full["FAR"].iloc[0])
    keep = ["w/o Consistency", "w/o Sequence", "w/o Rule", "w/o Prototype", "w/o transformer"]
    a = a[a["Model"].isin(keep)].copy()
    a["Model"] = pd.Categorical(a["Model"], categories=keep, ordered=True)
    a = a.sort_values("Model")
    x = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(6.4, 2.6), constrained_layout=False)
    w = 0.24
    dm = (a["Macro-F1"].to_numpy() - fm) * 100.0
    dh = (a["HRR"].to_numpy() - fh) * 100.0
    df = (a["FAR"].to_numpy() - ff) * 100.0
    b1 = ax.bar(x - w, dm, width=w, color="#C0392B", edgecolor="white", linewidth=0.3, label=r"$\Delta$Macro-F1")
    b2 = ax.bar(x, dh, width=w, color="#2E86DE", edgecolor="white", linewidth=0.3, label=r"$\Delta$HRR")
    b3 = ax.bar(x + w, df, width=w, color="#16A085", edgecolor="white", linewidth=0.3, label=r"$\Delta$FAR")
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xticks(x)
    _style_xticks(ax, ["-Cons", "-Seq", "-Rule", "-Proto", "-TFM"], rotation=16, fs=6.8)
    ax.set_ylabel("Delta (pp)")
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.0)
    _annotate_bars(ax, b1, "{:.1f}", dy=0.12, fs=5.0)
    _annotate_bars(ax, b2, "{:.1f}", dy=0.12, fs=5.0)
    _annotate_bars(ax, b3, "{:.1f}", dy=0.12, fs=5.0)
    _panel(ax, "(a)")
    _save(fig, out_dir, "fig_5_ablation_delta")


def _draw_robustness_summary(cross: pd.DataFrame, perturb: pd.DataFrame, out_dir: Path) -> None:
    """Must-have #3: mean + range error-bar summary."""
    if cross.empty or perturb.empty:
        return
    p_all = perturb[perturb["Subset"] == "all"].copy() if "Subset" in perturb.columns else perturb.copy()
    c_agg = cross.groupby("Model", observed=False)["Macro-F1"].agg(["mean", "min", "max"]).reset_index()
    p_agg = p_all.groupby("Model", observed=False)["Macro-F1"].agg(["mean", "min", "max"]).reset_index()
    c_agg["Model"] = pd.Categorical(c_agg["Model"], categories=MODEL_ORDER, ordered=True)
    p_agg["Model"] = pd.Categorical(p_agg["Model"], categories=MODEL_ORDER, ordered=True)
    c_agg = c_agg.sort_values("Model")
    p_agg = p_agg.sort_values("Model")
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), constrained_layout=False)
    for ax, agg, ttl in [(axes[0], c_agg, "Cross-scenario"), (axes[1], p_agg, "Threshold perturbation")]:
        for i, row in enumerate(agg.itertuples()):
            if pd.isna(row.mean):
                continue
            m = row.Model
            col = FIG2_METHOD_COLORS.get(m, "#555555")
            lo = (row.mean - row.min) * 100.0
            hi = (row.max - row.mean) * 100.0
            ax.errorbar(i, row.mean * 100.0, yerr=[[lo], [hi]], fmt="o", color=col,
                        capsize=2.5, capthick=0.7, markersize=5.5, markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xticks(x)
        _style_xticks(ax, [MODEL_SHORT[m] for m in MODEL_ORDER], rotation=24, fs=6.2)
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_ylim(45, 90)
        ax.set_title(ttl, fontsize=7.6)
        ax.grid(axis="y")
    _panel(axes[0], "(a)")
    _panel(axes[1], "(b)")
    _save(fig, out_dir, "fig_5_robustness_summary")


def _draw_case_evidence(cases: pd.DataFrame, out_dir: Path) -> None:
    """Optional #4: case evidence (3 components + 5 rule triggers)."""
    if cases.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), constrained_layout=False)
    # left: components
    ax = axes[0]
    c_cols = ["R_cons", "R_seq", "R_rule"]
    x = np.arange(len(cases))
    w = 0.24
    for i, col in enumerate(c_cols):
        vals = cases[col].to_numpy()
        bars = ax.bar(x + (i - 1) * w, vals, width=w,
                      color=["#2E86DE", "#16A085", "#E74C3C"][i],
                      edgecolor="white", linewidth=0.3, label=col)
        _annotate_bars(ax, bars, "{:.2f}", dy=0.02, fs=5.1)
    ax.set_xticks(x)
    _style_xticks(ax, [str(c).replace("_", " ") for c in cases["Case"]], rotation=14, fs=6.5)
    ax.set_ylabel("Component score")
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=3, fontsize=6.4)
    _panel(ax, "(a)")

    # right: rule triggers
    ax = axes[1]
    r_cols = ["V_type", "V_dst", "V_role", "V_time", "V_size"]
    x = np.arange(len(cases))
    w = 0.14
    for i, col in enumerate(r_cols):
        ax.bar(x + (i - 2) * w, cases[col].to_numpy(), width=w, label=col, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    _style_xticks(ax, [str(c).replace("_", " ") for c in cases["Case"]], rotation=14, fs=6.5)
    ax.set_ylabel("Rule evidence value")
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=3, fontsize=6.0)
    _panel(ax, "(b)")
    _save(fig, out_dir, "fig_5_case_evidence")


def _draw_significance(main: pd.DataFrame, out_dir: Path) -> None:
    """Optional #5: McNemar p-value chart."""
    d = main.copy()
    d["Method"] = pd.Categorical(d["Method"], categories=MODEL_ORDER, ordered=True)
    d = d.sort_values("Method")
    pvals = pd.to_numeric(d["p-value"], errors="coerce").fillna(1.0).to_numpy()
    score = -np.log10(np.clip(pvals, 1e-300, 1.0))
    score = np.minimum(score, 50)
    fig, ax = plt.subplots(figsize=(6.8, 2.4), constrained_layout=False)
    x = np.arange(len(d))
    bars = ax.bar(x, score, width=0.9, color=[FIG1_METHOD_COLORS[m] for m in d["Method"]], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    _style_xticks(ax, [MODEL_SHORT[m] for m in d["Method"]], rotation=24, fs=6.4)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.grid(axis="y")
    _annotate_bars(ax, bars, "{:.1f}", dy=0.3, fs=5.0)
    _panel(ax, "(a)")
    _save(fig, out_dir, "fig_5_significance")


def plot_all_figures(report_data_dir: str, output_dir: str, detail_csv: str | None = None) -> list[str]:
    """
    Generate 3 main multi-panel figures + supplemental evidence figures:
      - fig_5_1 (2x5)
      - fig_5_2 (2x4)
      - fig_5_3 (2x4)
      - fig_5_confusion_triplet
      - fig_5_ablation_delta
      - fig_5_robustness_summary
      - fig_5_case_evidence
      - fig_5_significance
    """
    _apply_style()
    report_dir = Path(report_data_dir)
    out_dir = Path(output_dir)

    main = _load_csv(report_dir, "tab_5_6_main_overall_performance.csv")
    perclass = _load_csv(report_dir, "tab_5_7_per_risk_class_results.csv")
    ablation = _load_csv(report_dir, "tab_5_8_ablation_results.csv")
    cross = _load_csv(report_dir, "fig_5_10_cross_scenario_data.csv")
    perturb = _load_csv(report_dir, "fig_5_11_distribution_perturbation_data.csv")
    mech = _load_csv(report_dir, "fig_5_9_mechanism_subset_data.csv")
    cases = _load_csv(report_dir, "fig_5_12_representative_cases.csv")

    detail_df = None
    if detail_csv:
        p = Path(detail_csv)
        if p.exists():
            detail_df = pd.read_csv(p)

    _draw_fig1(main, perclass, ablation, out_dir)
    _draw_fig2(cross, perturb, out_dir)
    _draw_fig3(main, perclass, mech, cases, detail_df, out_dir)
    _draw_confusion_triplet(main, out_dir)
    _draw_ablation_delta(ablation, out_dir)
    _draw_robustness_summary(cross, perturb, out_dir)
    _draw_case_evidence(cases, out_dir)
    _draw_significance(main, out_dir)

    generated = [
        "fig_5_1",
        "fig_5_2",
        "fig_5_3",
        "fig_5_confusion_triplet",
        "fig_5_ablation_delta",
        "fig_5_robustness_summary",
        "fig_5_case_evidence",
        "fig_5_significance",
    ]
    print(f"  => {len(generated)}  panel  {out_dir.resolve()}")
    return generated
