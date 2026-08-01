


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "exp_data" / "TCRNet_Final_Figures" / "tcr_paper" / "report_data"
OUT = ROOT / "exp_data" / "TCRNet_Final_Figures" / "figures"

METHODS = [
    "TCR-Net",
    "Unified-Multimodal",
    "w/o Rule",
    "Sequence+Semantic",
    "Content+Semantic",
    "Content-Only",
    "Rule-Based",
]
SHORT = {
    "TCR-Net": "TCR",
    "Unified-Multimodal": "Unified",
    "w/o Rule": "No-Rule",
    "Sequence+Semantic": "Seq+Sem",
    "Content+Semantic": "Cont+Sem",
    "Content-Only": "Cont",
    "Rule-Based": "Rule",
}
COL = {
    "TCR-Net": "#C43D32",
    "Unified-Multimodal": "#2F7FC1",
    "w/o Rule": "#8B5FBF",
    "Sequence+Semantic": "#239B82",
    "Content+Semantic": "#D08A2D",
    "Content-Only": "#7C8790",
    "Rule-Based": "#4F616E",
}


def sty() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#8C8C8C",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.8,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def load(name: str) -> pd.DataFrame:
    p = REPORT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def panel(ax, t: str) -> None:
    ax.text(-0.14, 1.03, t, transform=ax.transAxes, fontsize=9, fontweight="bold")


def fig_confusion_triplet() -> None:
    main = load("tab_5_6_main_overall_performance.csv")
    chosen = ["TCR-Net", "Unified-Multimodal", "Rule-Based"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.25))
    for i, m in enumerate(chosen):
        ax = axs[i]
        r = main[main["Method"] == m]
        if r.empty:
            ax.axis("off")
            continue
        dpath = Path(str(r.iloc[0]["DetailCSV"]))
        if not dpath.exists():
            ax.axis("off")
            continue
        d = pd.read_csv(dpath)
        cm = confusion_matrix(d["risk_label_true"], d["risk_label_pred"], labels=[0, 1, 2]).astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1.0)
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for rr in range(3):
            for cc in range(3):
                v = cm[rr, cc]
                ax.text(cc, rr, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if v > 0.52 else "#1F1F1F")
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_yticklabels(["Low", "Mid", "High"])
        ax.set_title(f"{SHORT[m]} confusion", pad=2.5)
        panel(ax, f"({chr(ord('a') + i)})")
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_supp_confusion_triplet")


def fig_ablation_delta() -> None:
    ab = load("tab_5_8_ablation_results.csv")
    full = ab[ab["Model"] == "Full"]
    if full.empty:
        return
    fm, fh, ff = [float(full[c].iloc[0]) for c in ["Macro-F1", "HRR", "FAR"]]
    keep = ["w/o Consistency", "w/o Sequence", "w/o Rule", "w/o Prototype", "w/o transformer"]
    d = ab[ab["Model"].isin(keep)].copy()
    d["Model"] = pd.Categorical(d["Model"], categories=keep, ordered=True)
    d = d.sort_values("Model")
    x = np.arange(len(d))
    dm = (d["Macro-F1"].to_numpy() - fm) * 100
    dh = (d["HRR"].to_numpy() - fh) * 100
    df = (d["FAR"].to_numpy() - ff) * 100

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    w = 0.22
    b1 = ax.bar(x - w, dm, width=w, color="#C43D32", label="ΔMacro-F1")
    b2 = ax.bar(x, dh, width=w, color="#2F7FC1", label="ΔHRR")
    b3 = ax.bar(x + w, df, width=w, color="#239B82", label="ΔFAR")
    ax.axhline(0, color="#666666", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["-Cons", "-Seq", "-Rule", "-Proto", "-TFM"], rotation=16, ha="right")
    ax.set_ylabel("Delta (pp)")
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=3, handlelength=0.9, columnspacing=0.6)
    for bs in [b1, b2, b3]:
        for b in bs:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (0.25 if v >= 0 else -0.65), f"{v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=5.8)
    panel(ax, "(a)")
    save(fig, "fig_supp_ablation_delta")


def fig_robustness_summary() -> None:
    cross = load("fig_5_10_cross_scenario_data.csv")
    pert = load("fig_5_11_distribution_perturbation_data.csv")
    if cross.empty or pert.empty:
        return
    pert = pert[pert["Subset"] == "all"] if "Subset" in pert.columns else pert

    def agg(df):
        out = df.groupby("Model", observed=False)["Macro-F1"].agg(["mean", "min", "max"]).reset_index()
        out["Model"] = pd.Categorical(out["Model"], categories=METHODS, ordered=True)
        return out.sort_values("Model")

    c = agg(cross); p = agg(pert)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.25))
    for ax, d, ttl, lab in [(axs[0], c, "Cross-scenario", "(a)"), (axs[1], p, "Threshold perturbation", "(b)")]:
        x = np.arange(len(METHODS))
        for i, r in enumerate(d.itertuples()):
            if pd.isna(r.mean):
                continue
            m = r.Model
            lo = (r.mean - r.min) * 100
            hi = (r.max - r.mean) * 100
            ax.errorbar(i, r.mean * 100, yerr=[[lo], [hi]], fmt="o", color=COL[m],
                        capsize=2.3, markersize=4.6, markeredgecolor="white", markeredgewidth=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=22, ha="right")
        ax.set_ylim(45, 90)
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_title(ttl, pad=2.4)
        ax.grid(axis="y")
        panel(ax, lab)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig_supp_robustness_summary")


def fig_case_evidence() -> None:
    c = load("fig_5_12_representative_cases.csv")
    if c.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    names = [str(x).replace("_", " ") for x in c["Case"]]
    x = np.arange(len(c))


    ax = axs[0]
    cols = ["R_cons", "R_seq", "R_rule"]
    colors = ["#2F7FC1", "#239B82", "#C43D32"]
    w = 0.22
    for i, (col, cc) in enumerate(zip(cols, colors)):
        b = ax.bar(x + (i - 1) * w, c[col].to_numpy(), width=w, color=cc, edgecolor="white", linewidth=0.3, label=col)
        for bb in b:
            v = bb.get_height()
            ax.text(bb.get_x() + bb.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=14, ha="right")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", ncol=3, fontsize=6.1, handlelength=0.9)
    ax.grid(axis="y"); panel(ax, "(a)")


    ax = axs[1]
    cols = ["V_type", "V_dst", "V_role", "V_time", "V_size"]
    w = 0.14
    pal = ["#5DA5DA", "#60BD68", "#B2912F", "#B276B2", "#F15854"]
    for i, (col, cc) in enumerate(zip(cols, pal)):
        ax.bar(x + (i - 2) * w, c[col].to_numpy(), width=w, color=cc, edgecolor="white", linewidth=0.25, label=col)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=14, ha="right")
    ax.set_ylabel("Rule value")
    ax.legend(loc="upper left", ncol=3, fontsize=5.8, handlelength=0.8)
    ax.grid(axis="y"); panel(ax, "(b)")
    fig.subplots_adjust(wspace=0.34)
    save(fig, "fig_supp_case_evidence")


def fig_significance() -> None:
    m = load("tab_5_6_main_overall_performance.csv")
    if m.empty:
        return
    m["Method"] = pd.Categorical(m["Method"], categories=METHODS, ordered=True)
    m = m.sort_values("Method")
    p = pd.to_numeric(m["p-value"], errors="coerce").fillna(1.0).to_numpy()
    y = np.minimum(-np.log10(np.clip(p, 1e-300, 1.0)), 40)
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    b = ax.bar(x, y, width=0.86, color=[COL[k] for k in m["Method"]], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in m["Method"]], rotation=22, ha="right")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.grid(axis="y")
    for bb in b:
        v = bb.get_height()
        ax.text(bb.get_x() + bb.get_width()/2, v + 0.25, f"{v:.1f}", ha="center", fontsize=5.8)
    panel(ax, "(a)")
    save(fig, "fig_supp_significance")


def main() -> None:
    sty()
    fig_confusion_triplet()
    fig_ablation_delta()
    fig_robustness_summary()
    fig_case_evidence()
    fig_significance()
    print(f"=> supplemental figures saved to: {OUT}")


if __name__ == "__main__":
    main()
