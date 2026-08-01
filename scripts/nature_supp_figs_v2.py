


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
    "TCR-Net": "TCR-Net",
    "Unified-Multimodal": "Concat+MLP",
    "w/o Rule": "w/o Rule",
    "Sequence+Semantic": "Sequence+Semantic",
    "Content+Semantic": "Content+Semantic",
    "Content-Only": "Content-Only",
    "Rule-Based": "Rule-Based",
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
RISK = {"Low": "#2F7FC1", "Mid": "#D08A2D", "High": "#C43D32"}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.6,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#8E8E8E",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#D2D2D2",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.8,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def load(name: str) -> pd.DataFrame:
    p = REPORT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def tag(ax, t: str) -> None:
    ax.text(-0.14, 1.02, t, transform=ax.transAxes, fontsize=9, fontweight="bold")


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def fig1_confusion_triplet() -> None:
    tab = load("tab_5_6_main_overall_performance.csv")
    selected = ["TCR-Net", "Unified-Multimodal", "Rule-Based"]
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.35), constrained_layout=True)
    axes = axes.flatten()
    cms = {}
    for i, m in enumerate(selected):
        ax = axes[i]
        r = tab[tab["Method"] == m]
        if r.empty:
            ax.axis("off")
            continue
        p = Path(str(r.iloc[0]["DetailCSV"]))
        if not p.exists():
            ax.axis("off")
            continue
        d = pd.read_csv(p)
        cm = confusion_matrix(d["risk_label_true"], d["risk_label_pred"], labels=[0, 1, 2]).astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1.0)
        cms[m] = cm
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for rr in range(3):
            for cc in range(3):
                v = cm[rr, cc]
                ax.text(cc, rr, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.8, color=("white" if v > 0.52 else "#1f1f1f"))
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_yticklabels(["Low", "Mid", "High"])
        ax.set_title(f"{SHORT[m]}", pad=2)
        tag(ax, f"({chr(ord('a') + i)})")

    ax = axes[3]
    if "TCR-Net" in cms and "Rule-Based" in cms:
        dm = cms["TCR-Net"] - cms["Rule-Based"]
        ax.imshow(dm, cmap="RdBu_r", vmin=-0.45, vmax=0.45)
        for rr in range(3):
            for cc in range(3):
                v = dm[rr, cc]
                ax.text(cc, rr, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6.4, color=("#1f1f1f" if abs(v) < 0.22 else "white"))
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_yticklabels(["Low", "Mid", "High"])
        ax.set_title("TCR-Net minus Rule-Based", pad=2)
        tag(ax, "(d)")
    else:
        ax.axis("off")
    save(fig, "fig_nature_supp1_confusion_triplet")


def fig2_ablation_lollipop() -> None:

    ab = load("tab_5_8_ablation_results.csv")
    full = ab[ab["Model"] == "Full"]
    if full.empty:
        return
    fm, fh, ff = [float(full[c].iloc[0]) for c in ["Macro-F1", "HRR", "FAR"]]
    keep = ["w/o Consistency", "w/o Sequence", "w/o Rule", "w/o Prototype", "w/o transformer"]
    d = ab[ab["Model"].isin(keep)].copy()
    d["Model"] = pd.Categorical(d["Model"], categories=keep, ordered=True)
    d = d.sort_values("Model")
    y = np.arange(len(d))
    labels = ["No-Cons", "No-Seq", "No-Rule", "No-Proto", "No-TFM"]
    dm = (d["Macro-F1"].to_numpy() - fm) * 100
    dh = (d["HRR"].to_numpy() - fh) * 100
    df = (d["FAR"].to_numpy() - ff) * 100

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), sharey=True, constrained_layout=True)
    specs = [
        (axes[0], dm, "#C43D32", r"$\Delta$Macro-F1", "(a)"),
        (axes[1], dh, "#2F7FC1", r"$\Delta$HRR", "(b)"),
        (axes[2], df, "#239B82", r"$\Delta$FAR", "(c)"),
    ]
    for ax, vals, c, ttl, lab in specs:
        ax.axvline(0, color="#777777", linewidth=0.7)
        for i, v in enumerate(vals):
            ax.hlines(i, 0, v, color=c, linewidth=2.0, alpha=0.9)
            ax.plot(v, i, "o", color=c, markersize=5.2)
            dx = 0.18 if v >= 0 else -0.18
            ax.text(v + dx, i, f"{v:.1f}", va="center", ha=("left" if v >= 0 else "right"), fontsize=6.5)
        ax.set_title(ttl, pad=2)
        ax.grid(axis="x")
        tag(ax, lab)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Ablation")
    save(fig, "fig_nature_supp2_ablation_lollipop")


def fig3_robustness_errorbar() -> None:
    cross = load("fig_5_10_cross_scenario_data.csv")
    pert = load("fig_5_11_distribution_perturbation_data.csv")
    if cross.empty or pert.empty:
        return
    pert = pert[pert["Subset"] == "all"] if "Subset" in pert.columns else pert

    def agg(df):
        t = df.groupby("Model", observed=False)["Macro-F1"].agg(["mean", "min", "max"]).reset_index()
        t["Model"] = pd.Categorical(t["Model"], categories=METHODS, ordered=True)
        return t.sort_values("Model")

    c = agg(cross); p = agg(pert)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.25), constrained_layout=True)
    for ax, tab, title, lab in [(axes[0], c, "Cross-scenario", "(a)"), (axes[1], p, "Threshold perturbation", "(b)")]:
        x = np.arange(len(METHODS))
        for i, r in enumerate(tab.itertuples()):
            if pd.isna(r.mean):
                continue
            m = r.Model
            lo = (r.mean - r.min) * 100
            hi = (r.max - r.mean) * 100
            ax.errorbar(i, r.mean * 100, yerr=[[lo], [hi]], fmt="o",
                        color=COL[m], markersize=4.8, capsize=2.5, capthick=0.7,
                        markeredgecolor="white", markeredgewidth=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=22, ha="right")
        ax.set_ylim(45, 90)
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_title(title, pad=2)
        ax.grid(axis="y")
        tag(ax, lab)
    save(fig, "fig_nature_supp3_robustness_errorbar")


def fig4_case_dualpanel() -> None:
    c = load("fig_5_12_representative_cases.csv")
    if c.empty:
        return
    names = [str(x).replace("_", " ") for x in c["Case"]]
    x = np.arange(len(c))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.55), constrained_layout=True)


    ax = axes[0]
    for i, row in c.iterrows():
        vals = [row["R_cons"], row["R_seq"], row["R_rule"]]
        ax.plot([0, 1, 2], vals, marker="o", linewidth=1.6, label=names[i])
        for j, v in enumerate(vals):
            ax.text(j, v + 0.02, f"{v:.2f}", fontsize=6.2, ha="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["R_cons", "R_seq", "R_rule"])
    ax.set_ylabel("Score")
    ax.set_title("Component evidence", pad=2)
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=5.2, ncol=1)
    tag(ax, "(a)")


    ax = axes[1]
    rcols = ["V_type", "V_dst", "V_role", "V_time", "V_size"]
    for i, row in c.iterrows():
        vals = [row[k] for k in rcols]
        xx = np.arange(len(rcols))
        s = np.clip(np.array(vals) * 6 + 22, 20, 110)
        ax.scatter(xx, np.full_like(xx, i), s=s, c="#C43D32", alpha=0.7, edgecolors="white", linewidths=0.3)
        for j, v in enumerate(vals):
            ax.text(j, i + 0.16, f"{v:.1f}", ha="center", fontsize=5.8)
    ax.set_xticks(np.arange(len(rcols)))
    ax.set_xticklabels(["Type", "Destination", "Role", "Time", "Size"], rotation=12, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Rule-trigger evidence", pad=2)
    ax.grid(axis="x")
    tag(ax, "(b)")
    save(fig, "fig_nature_supp4_case_dualpanel")


def fig5_significance_strip() -> None:
    m = load("tab_5_6_main_overall_performance.csv")
    if m.empty:
        return
    m["Method"] = pd.Categorical(m["Method"], categories=METHODS, ordered=True)
    m = m.sort_values("Method")
    pvals = pd.to_numeric(m["p-value"], errors="coerce").fillna(1.0).to_numpy()
    sig = np.minimum(-np.log10(np.clip(pvals, 1e-300, 1.0)), 40)
    x = np.arange(len(m))

    fig, ax = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    ax.barh(x, sig, color=[COL[k] for k in m["Method"]], edgecolor="white", linewidth=0.35)
    ax.set_yticks(x); ax.set_yticklabels([SHORT[k] for k in m["Method"]])
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title("Pairwise significance vs TCR", pad=2)
    ax.grid(axis="x")
    for i, v in enumerate(sig):
        ax.text(v + 0.35, i, f"{v:.1f}", va="center", fontsize=6.2)
    tag(ax, "(a)")
    save(fig, "fig_nature_supp5_significance_strip")


def fig6_violin_risk() -> None:


    tab = load("tab_5_6_main_overall_performance.csv")
    top3 = ["TCR-Net", "Unified-Multimodal", "Rule-Based"]
    rows = []
    for m in top3:
        r = tab[tab["Method"] == m]
        if r.empty or "DetailCSV" not in r.columns:
            continue
        p = Path(str(r.iloc[0]["DetailCSV"]))
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "R_total" not in d.columns:
            continue
        for cls, name in [(0, "Low"), (1, "Mid"), (2, "High")]:
            vals = d.loc[d["risk_label_true"] == cls, "R_total"].to_numpy()
            for v in vals:
                rows.append({"Method": m, "Class": name, "R_total": float(v)})
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.8), constrained_layout=True)


    ax = axes[0]
    sub = df[df["Method"] == "TCR-Net"]
    groups = [sub[sub["Class"] == c]["R_total"].to_numpy() for c in ["Low", "Mid", "High"]]
    vp = ax.violinplot(groups, positions=[1, 2, 3], widths=0.70, showmeans=True, showextrema=False)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor([RISK["Low"], RISK["Mid"], RISK["High"]][i])
        b.set_alpha(0.60)
        b.set_edgecolor("none")
    vp["cmeans"].set_color("#333333")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_ylabel("Total risk score $R_{\\mathrm{total}}$")
    ax.set_title("TCR-Net score separation", pad=2)
    ax.grid(axis="y")
    tag(ax, "(a)")


    ax = axes[1]
    sub = df[df["Class"] == "High"]
    groups = [sub[sub["Method"] == m]["R_total"].to_numpy() for m in top3]
    vp = ax.violinplot(groups, positions=[1, 2, 3], widths=0.70, showmeans=True, showextrema=False)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor([COL[m] for m in top3][i])
        b.set_alpha(0.60)
        b.set_edgecolor("none")
    vp["cmeans"].set_color("#333333")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["TCR-Net", "Concat+MLP", "Rule-Based"], rotation=12, ha="right")
    ax.set_ylabel("$R_{\\mathrm{total}}$ on high-risk")
    ax.set_title("High-risk score comparison", pad=2)
    ax.grid(axis="y")
    tag(ax, "(b)")

    save(fig, "fig_nature_supp6_violin_risk")


def fig7_scatter_tradeoff() -> None:


    tab = load("tab_5_6_main_overall_performance.csv")
    if tab.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.65), constrained_layout=True)


    ax = axes[0]
    for _, r in tab.iterrows():
        m = r["Method"]
        x = float(r["FAR"]) * 100
        y = float(r["HRR"]) * 100
        s = 28 + float(r["Macro-F1"]) * 120
        ax.scatter(x, y, s=s, color=COL.get(m, "#666666"), alpha=0.85, edgecolors="white", linewidths=0.35)
        ax.text(x + 0.6, y + 0.15, SHORT.get(m, m), fontsize=5.2)
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("HRR (%)")
    ax.set_title("Safety-efficiency tradeoff")
    ax.grid(True)
    tag(ax, "(a)")


    ax = axes[1]
    for _, r in tab.iterrows():
        m = r["Method"]
        x = float(r["Macro-F1"]) * 100
        y = float(r["Weighted-F1"]) * 100
        ax.scatter(x, y, s=60, color=COL.get(m, "#666666"), alpha=0.9, edgecolors="white", linewidths=0.35)
        ax.text(x + 0.15, y + 0.08, SHORT.get(m, m), fontsize=5.2)
    ax.set_xlabel("Macro-F1 (%)")
    ax.set_ylabel("Weighted-F1 (%)")
    ax.set_title("Balanced vs global quality")
    ax.grid(True)
    tag(ax, "(b)")

    save(fig, "fig_nature_supp7_scatter_tradeoff")


def fig8_singlecol_1x2_a() -> None:

    cross = load("fig_5_10_cross_scenario_data.csv")
    pert = load("fig_5_11_distribution_perturbation_data.csv")
    if cross.empty or pert.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.55), constrained_layout=True)


    ax = axes[0]
    t = pert[pert["Model"] == "TCR-Net"].copy()
    for subset, c in [("all", "#C43D32"), ("rule_active", "#2F7FC1"), ("clean", "#239B82")]:
        d = t[t["Subset"] == subset].sort_values("Eta2Quantile")
        if d.empty:
            continue
        ax.plot(d["Eta2Quantile"], d["Macro-F1"] * 100, marker="o", linewidth=1.2, markersize=2.7, color=c, label=subset)
    ax.set_xlabel(r"$\eta_2$ quantile")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("TCR threshold sensitivity")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=4.9, ncol=1, frameon=False)
    tag(ax, "(a)")


    ax = axes[1]
    rows = []
    for m in ["TCR-Net", "Unified-Multimodal", "Rule-Based"]:
        dm = cross[cross["Model"] == m]
        for sk in ["object_type", "source_type", "intent_id", "rule_active"]:
            d = dm[dm["ScenarioKey"] == sk]
            if not d.empty:
                rows.append({"Method": m, "Scenario": sk, "Macro": d["Macro-F1"].mean() * 100})
    if rows:
        dd = pd.DataFrame(rows)
        sc = ["object_type", "source_type", "intent_id", "rule_active"]
        x = np.arange(len(sc))
        w = 0.22
        for i, m in enumerate(["TCR-Net", "Unified-Multimodal", "Rule-Based"]):
            d = dd[dd["Method"] == m].set_index("Scenario").reindex(sc)
            ax.bar(x + (i - 1) * w, d["Macro"].to_numpy(), width=w, color=COL[m], edgecolor="white", linewidth=0.3, label=SHORT[m])
        ax.set_xticks(x); ax.set_xticklabels(["Obj", "Src", "Intent", "Rule"], rotation=0)
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_title("Scenario mean")
        ax.grid(axis="y")
        ax.legend(loc="lower left", fontsize=4.7, ncol=1, frameon=False)
    tag(ax, "(b)")

    save(fig, "fig_nature_supp8_singlecol_1x2_a")


def fig9_singlecol_1x2_b() -> None:

    per = load("tab_5_7_per_risk_class_results.csv")
    main = load("tab_5_6_main_overall_performance.csv")
    if per.empty or main.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.55), constrained_layout=True)


    ax = axes[0]
    top = ["TCR-Net", "Unified-Multimodal", "w/o Rule", "Sequence+Semantic"]
    d = per[per["Method"].isin(top)].copy()
    d["Method"] = pd.Categorical(d["Method"], categories=top, ordered=True)
    d = d.sort_values("Method")
    x = np.arange(len(top))
    ax.plot(x, d["Low-R"] * 100, marker="o", linewidth=1.2, color=RISK["Low"], label="Low-R")
    ax.plot(x, d["Mid-R"] * 100, marker="o", linewidth=1.2, color=RISK["Mid"], label="Mid-R")
    ax.plot(x, d["High-R"] * 100, marker="o", linewidth=1.2, color=RISK["High"], label="High-R")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in top], rotation=20, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Class-wise recall profile")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=4.9, ncol=1, frameon=False)
    tag(ax, "(a)")


    ax = axes[1]
    d = main.copy()
    d["Method"] = pd.Categorical(d["Method"], categories=METHODS, ordered=True)
    d = d.sort_values("Method")
    p = pd.to_numeric(d["p-value"], errors="coerce").fillna(1.0).to_numpy()
    y = np.minimum(-np.log10(np.clip(p, 1e-300, 1.0)), 30)
    x = np.arange(len(d))
    ax.bar(x, y, color=[COL[m] for m in d["Method"]], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in d["Method"]], rotation=24, ha="right")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title("Significance vs TCR")
    ax.grid(axis="y")
    tag(ax, "(b)")

    save(fig, "fig_nature_supp9_singlecol_1x2_b")


def main() -> None:
    style()
    fig1_confusion_triplet()
    fig2_ablation_lollipop()
    fig3_robustness_errorbar()
    fig4_case_dualpanel()
    fig5_significance_strip()
    fig6_violin_risk()
    fig7_scatter_tradeoff()
    fig8_singlecol_1x2_a()
    fig9_singlecol_1x2_b()
    print(f"=> new nature supplemental figures saved to {OUT}")


if __name__ == "__main__":
    main()
