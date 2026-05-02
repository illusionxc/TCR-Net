"""
KDD-style Figure 1 — TCR-Net Architecture + Main Quantitative Results

Layout (asymmetric):
  ┌─────────────────────┬─────────────────┐
  │                     │  (b) Macro-F1   │
  │  (a) Architecture   │  bar chart      │
  │  Schematic          ├─────────────────┤
  │                     │  (c) HRR vs FAR │
  │                     │  scatter        │
  └─────────────────────┴─────────────────┘

: python kdd_figure1_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# KDD  rcParams
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'font.size': 8,
    'axes.linewidth': 1.2,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ═══════════════════════════════════════════════════════════════════
# KDD 
# ═══════════════════════════════════════════════════════════════════
C_BLUE   = '#1F77B4'   # TCR-Net 
C_ORANGE = '#FF7F0E'   # 1
C_GREEN  = '#2CA02C'   # 2
C_RED    = '#D62728'   # /
C_PURPLE = '#9467BD'   # 3
C_BROWN  = '#8C564B'   # 4
C_GRAY   = '#7F7F7F'   # 
C_LGRAY  = '#C0C0C0'   # 
C_BGRAY  = '#333333'   # 

# 7 
METHOD_COLORS = {
    'TCR-Net':             C_BLUE,
    'Unified-Multimodal':  C_ORANGE,
    'w/o Rule':            C_GREEN,
    'Sequence+Semantic':   C_PURPLE,
    'Content+Semantic':    C_BROWN,
    'Rule-Based':          C_GRAY,
    'Content-Only':        C_LGRAY,
}

# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
DATA_DIR = Path('exp_data/TCRNet_Final_Figures/tcr_paper/report_data')
df_main = pd.read_csv(DATA_DIR / 'tab_5_6_main_overall_performance.csv')


# ═══════════════════════════════════════════════════════════════════
# (a) 
# ═══════════════════════════════════════════════════════════════════
def draw_architecture(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')


    def rbox(x, y, w, h, text, color, fs=7, fc='white', alpha=0.92):
        style = "round,pad=0.25"
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=style,
            facecolor=color, edgecolor='none', alpha=alpha, zorder=2))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='bold', zorder=3)

    def arrow(x1, y1, x2, y2, color=C_BGRAY, lw=2.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=lw, shrinkA=4, shrinkB=4), zorder=1)

    def tag(x, y, text, fs=7, color=C_BGRAY):
        ax.text(x, y, text, ha='center', va='bottom', fontsize=fs,
                color=color, fontweight='bold', zorder=3)


    in_x, in_w = 0.2, 2.0
    inputs = [
        (in_x, 7.0, 'File\nStructure'),
        (in_x, 6.1, 'Task\nSemantics'),
        (in_x, 5.2, 'Transmission\nContext'),
        (in_x, 4.3, 'History\nWindow'),
    ]
    for x, y, t in inputs:
        rbox(x, y, in_w, 0.65, t, '#555555', 6.5)
    tag(1.2, 7.75, 'Inputs')


    enc_x, enc_y, enc_w, enc_h = 2.7, 5.0, 2.0, 2.2
    rbox(enc_x, enc_y, enc_w, enc_h, 'Cross-modal\nTransformer\nEncoder', C_BLUE, 7.5)
    tag(enc_x + enc_w/2, enc_y - 0.25, 'Shared Encoding')

    for x, y, _ in inputs:
        arrow(x + in_w, y + 0.325, enc_x, enc_y + enc_h/2)


    comp_x, comp_w, comp_h = 5.2, 2.4, 1.0
    components = [
        (comp_x, 6.6, 'R_cons  Semantic Consistency', '#1F77B4'),
        (comp_x, 5.3, 'R_seq   Temporal Deviation',  '#2CA02C'),
        (comp_x, 4.0, 'R_rule  Rule Conflict',       '#D62728'),
    ]
    for cx, cy, text, color in components:
        rbox(cx, cy, comp_w, comp_h, text, color, 7, 'white', 0.85)
        arrow(enc_x + enc_w, enc_y + enc_h/2, cx, cy + comp_h/2)


    fuse_x, fuse_y, fuse_w, fuse_h = 7.8, 4.3, 1.3, 1.6
    rbox(fuse_x, fuse_y, fuse_w, fuse_h, 'Fusion\nβ-weights\n+ η thresholds', C_BLUE, 7)
    tag(fuse_x + fuse_w/2, fuse_y - 0.25, 'Decision')

    for cx, cy, _, _ in components:
        arrow(cx + comp_w, cy + comp_h/2, fuse_x, fuse_y + fuse_h/2)


    out_x, out_y, out_w, out_h = 7.8, 1.8, 1.3, 0.7
    rbox(out_x, out_y, out_w, out_h, 'Grade', C_BLUE, 9, 'white', 0.95)
    tag(out_x + out_w/2, out_y - 0.2, '{Low, Medium, High}', 7.5)
    arrow(fuse_x + fuse_w/2, fuse_y, out_x + out_w/2, out_y + out_h)


# ═══════════════════════════════════════════════════════════════════
# (b) Macro-F1 
# ═══════════════════════════════════════════════════════════════════
def draw_macro_f1(ax):
    data = df_main.sort_values('Macro-F1', ascending=True).reset_index(drop=True)
    labels = data['Method'].tolist()
    values = (data['Macro-F1'] * 100).tolist()
    y = np.arange(len(labels))

    colors = [METHOD_COLORS.get(m, C_GRAY) for m in labels]
    bars = ax.barh(y, values, height=0.55, color=colors, zorder=2,
                   edgecolor='white', linewidth=0.5)

    for i, (v, lab) in enumerate(zip(values, labels)):
        ax.text(v + 1.5, i, f'{v:.1f}', va='center', fontsize=7.5,
                color=C_BGRAY, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Macro-F1 (%)', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.2, linewidth=0.6)
    ax.set_title('(b) Overall Performance', fontsize=9, fontweight='bold', pad=8)


# ═══════════════════════════════════════════════════════════════════
# (c) HRR vs FAR 
# ═══════════════════════════════════════════════════════════════════
def draw_hrr_far(ax):
    short_names = {
        'TCR-Net': 'TCR-Net', 'Unified-Multimodal': 'Unified',
        'Sequence+Semantic': 'Seq+Sem', 'Content+Semantic': 'C+Sem',
        'Content-Only': 'Content', 'Rule-Based': 'Rule', 'w/o Rule': '-Rule',
    }
    for _, row in df_main.iterrows():
        method = row['Method']
        hrr = row['HRR'] * 100
        far = row['FAR'] * 100
        color = METHOD_COLORS.get(method, C_GRAY)
        is_tcr = method == 'TCR-Net'
        ax.scatter(far, hrr, s=100 if is_tcr else 55, color=color,
                   edgecolor='white', linewidth=0.5, zorder=3,
                   marker='o' if is_tcr else 'D')
        label = short_names.get(method, method)
        ox = 2.5 if far < 15 else -4.5
        ax.text(far + ox, hrr, label, fontsize=7, va='center',
                color=C_BGRAY, fontweight='bold' if is_tcr else 'normal')

    ax.set_xlabel('False Alarm Rate, FAR (%)', fontsize=9, fontweight='bold')
    ax.set_ylabel('High-Risk Recall, HRR (%)', fontsize=9, fontweight='bold')
    ax.set_xlim(-2, 55)
    ax.set_ylim(0, 85)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_title('(c) Safety-Efficiency Trade-off', fontsize=9, fontweight='bold', pad=8)


# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
def main():
    out_dir = Path('exp_data/TCRNet_Final_Figures/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.5, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.06, right=0.97, top=0.94, bottom=0.07)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_architecture(ax_a)
    ax_a.set_title('(a) TCR-Net Architecture', fontsize=9, fontweight='bold', pad=8)
    draw_macro_f1(ax_b)
    draw_hrr_far(ax_c)

    pdf = out_dir / 'fig_kdd_architecture.pdf'
    fig.savefig(pdf, facecolor='white')
    print(f'  => {pdf}')


if __name__ == '__main__':
    main()
