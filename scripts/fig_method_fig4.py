


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / "exp_data" / "TCRNet_Final_Figures" / "tcr_paper" / "baselines" / "tcr_net" / "details" / "test_details.csv"
OUT = ROOT / "exp_data" / "TCRNet_Final_Figures" / "figures"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


C_LOW  = "#2F7FC1"
C_MID  = "#D08A2D"
C_HIGH = "#C43D32"

def main():
    df = pd.read_csv(DETAIL)
    eta1 = float(df["eta_1"].iloc[0])
    eta2 = float(df["eta_2"].iloc[0])


    fig, ax = plt.subplots(figsize=(3.5, 2.0))


    n_bins = 20
    for cls, color, label, ls in [
        (0, C_LOW,  "Low risk",  "-"),
        (1, C_MID,  "Medium risk", "--"),
        (2, C_HIGH, "High risk", "-"),
    ]:
        vals = df.loc[df["risk_label_true"] == cls, "R_total"].to_numpy()
        if len(vals) < 3:
            continue
        counts, edges = np.histogram(vals, bins=n_bins, range=(0, 1), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, color=color, linewidth=1.6, linestyle=ls, label=label)

        ax.fill_between(centers, counts, alpha=0.06, color=color)


    ylim = ax.get_ylim()
    y_top = ylim[1]
    for eta, color, label_text in [
        (eta1, C_MID,  r"$\eta_1$ (release/review)"),
        (eta2, C_HIGH, r"$\eta_2$ (review/block)"),
    ]:
        ax.axvline(eta, color=color, linestyle="--", linewidth=1.0, zorder=3)

        ax.text(eta, y_top * 0.88, f"{eta:.2f}",
                ha="center", fontsize=6.5, color=color, fontweight="bold")


    ax.axvspan(0, eta1, alpha=0.03, color=C_LOW)
    ax.axvspan(eta1, eta2, alpha=0.03, color=C_MID)
    ax.axvspan(eta2, 1, alpha=0.03, color=C_HIGH)


    y_label = y_top * 0.03
    ax.text(eta1 / 2, y_label, "Release",
            ha="center", fontsize=7, color=C_LOW, fontweight="bold")
    ax.text((eta1 + eta2) / 2, y_label, "Review",
            ha="center", fontsize=7, color=C_MID, fontweight="bold")
    ax.text((1 + eta2) / 2, y_label, "Block",
            ha="center", fontsize=7, color=C_HIGH, fontweight="bold")


    ax.set_xlabel("Total risk score $R_{\\mathrm{total}}$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", handlelength=1.2, fontsize=6.5)
    ax.grid(axis="y", alpha=0.12, linewidth=0.4)


    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_method_fig4.pdf", facecolor="white")
    plt.close(fig)
    print(f"  => {OUT / 'fig_method_fig4.pdf'}")


if __name__ == "__main__":
    main()
