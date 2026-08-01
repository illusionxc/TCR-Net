


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


C_TEAL    = '#006D77'
C_BLUE    = '#2D6A9F'
C_AMBER   = '#D98A3C'
C_RED     = '#A23B3B'
C_GREEN   = '#3A7D54'
C_GRAY    = '#8C8C8C'
C_LGRAY   = '#BFBFBF'
C_BGRAY   = '#4D4D4D'
C_WHITE   = '#FFFFFF'
C_BG      = '#F5F5F0'


METHOD_ORDER = ['TCR-Net', 'Unified-Multimodal', 'w/o Rule',
                'Sequence+Semantic', 'Content+Semantic',
                'Rule-Based', 'Content-Only']
METHOD_COLORS = [C_TEAL, C_BLUE, '#7A7A7A', '#A0A0A0',
                 '#B8B8B8', '#C8C8C8', '#D8D8D8']


DATA_DIR = Path('exp_data/TCRNet_Final_Figures/tcr_paper/report_data')
df_main = pd.read_csv(DATA_DIR / 'tab_5_6_main_overall_performance.csv')


def draw_architecture(ax):


    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')


    def box(x, y, w, h, text, color=C_TEAL, text_color='white', fs=7, alpha=0.9):

        style = "round,pad=0.3"
        bbox = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=style,
            facecolor=color, edgecolor='none', alpha=alpha, zorder=2)
        ax.add_patch(bbox)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, color=text_color, fontweight='bold', zorder=3)

    def arrow(x1, y1, x2, y2, color=C_BGRAY, lw=1.5):

        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=lw, shrinkA=3, shrinkB=3),
                    zorder=1)

    def label(x, y, text, fs=6.5, color=C_BGRAY, style='italic'):

        ax.text(x, y, text, ha='center', va='bottom', fontsize=fs,
                color=color, fontstyle=style, zorder=3)


    input_top = 7.0
    inputs = [
        (0.3, input_top, 2.0, 0.6, 'File\nStructure', C_GRAY),
        (0.3, input_top - 0.8, 2.0, 0.6, 'Task\nSemantics', C_GRAY),
        (0.3, input_top - 1.6, 2.0, 0.6, 'Transmission\nContext', C_GRAY),
        (0.3, input_top - 2.4, 2.0, 0.6, 'History\nWindow', C_GRAY),
    ]
    for x, y, w, h, t, c in inputs:
        box(x, y, w, h, t, c, 'white', 6)
    label(1.3, input_top + 0.3, 'Input Features', 7, C_BGRAY, 'normal')


    enc_x, enc_y, enc_w, enc_h = 2.8, 5.3, 2.2, 1.6
    box(enc_x, enc_y, enc_w, enc_h, '', C_TEAL, alpha=0.15)
    ax.text(enc_x + enc_w/2, enc_y + enc_h/2,
            'Cross-modal\nTransformer\nEncoder',
            ha='center', va='center', fontsize=7, color=C_TEAL,
            fontweight='bold', zorder=3)
    ax.text(enc_x + enc_w/2, enc_y - 0.15, 'Shared representation h',
            ha='center', va='top', fontsize=6, color=C_BGRAY, fontstyle='italic')


    for inp in inputs:
        arrow(inp[0] + inp[2], inp[1] + inp[3]/2, enc_x, enc_y + enc_h/2)


    comp_y = 1.1
    comp_h = 1.2
    comp_w = 2.2
    components = [
        (5.5, comp_y + 2.0, 'R_cons', C_BLUE, 'Semantic\nConsistency\nα₁·‖z−p‖² + α₂·‖r−r̂‖₁'),
        (5.5, comp_y, 'R_seq', C_GREEN, 'Temporal\nDeviation\n√((h−μ)ᵀΣ⁻¹(h−μ))'),
        (5.5, comp_y - 2.0, 'R_rule', C_RED, 'Rule\nConflict\nγ₁V₁+…+γ₅V₅'),
    ]
    for cx, cy, name, color, desc in components:
        box(cx, cy, comp_w, comp_h, '', color, alpha=0.10)

        ax.text(cx + 0.3, cy + comp_h/2, name, ha='center', va='center',
                fontsize=8, color=color, fontweight='bold', zorder=3)

        ax.text(cx + 1.5, cy + comp_h/2, desc, ha='center', va='center',
                fontsize=5.5, color=C_BGRAY, zorder=3)


    arrow(enc_x + enc_w, enc_y + enc_h/2,
          5.5, comp_y + 2.5 + comp_h/2)
    arrow(enc_x + enc_w, enc_y + enc_h/2,
          5.5, comp_y + comp_h/2)
    arrow(enc_x + enc_w, enc_y + enc_h/2,
          5.5, comp_y - 2.0 + comp_h/2)


    fuse_x, fuse_y, fuse_w, fuse_h = 8.3, 2.0, 1.3, 1.8
    box(fuse_x, fuse_y, fuse_w, fuse_h, '', C_TEAL, alpha=0.85)
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h/2,
            'Fusion\nβ-weight\n+ η thresholds',
            ha='center', va='center', fontsize=6.5, color='white',
            fontweight='bold', zorder=3)


    arrow(5.5 + comp_w, comp_y + 2.0 + comp_h/2, fuse_x, fuse_y + fuse_h*0.7)
    arrow(5.5 + comp_w, comp_y + comp_h/2, fuse_x, fuse_y + fuse_h/2)
    arrow(5.5 + comp_w, comp_y - 2.0 + comp_h/2, fuse_x, fuse_y + fuse_h*0.3)


    out_x, out_y, out_w, out_h = 8.3, 0.1, 1.3, 0.6
    box(out_x, out_y, out_w, out_h, 'Grade', C_TEAL, 'white', 8, alpha=0.95)
    label(out_x + out_w/2, out_y - 0.1, '{Low, Medium, High}', 6.5)
    arrow(fuse_x + fuse_w/2, fuse_y, out_x + out_w/2, out_y + out_h)


    legend_y = 0.05
    ax.text(5.5, legend_y,
            'R_cons: Semantic-consistency score\n'
            'R_seq: Temporal-deviation score\n'
            'R_rule:  Rule-conflict score',
            fontsize=5.5, color=C_BGRAY, va='bottom', fontfamily='monospace')


    ax.text(-0.08, 1.02, 'a', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def draw_macro_f1(ax):
    data = df_main.sort_values('Macro-F1', ascending=True).reset_index(drop=True)
    labels = data['Method'].tolist()
    values = (data['Macro-F1'] * 100).tolist()
    y = np.arange(len(labels))

    colors = [C_TEAL if m == 'TCR-Net' else C_LGRAY for m in labels]
    ax.barh(y, values, height=0.55, color=colors, zorder=2, edgecolor='white', linewidth=0.3)

    for i, (v, lab) in enumerate(zip(values, labels)):
        color = C_TEAL if lab == 'TCR-Net' else C_BGRAY
        ax.text(v + 1.5, i, f'{v:.1f}', va='center', fontsize=7,
                color=color, fontweight='bold' if lab == 'TCR-Net' else 'normal')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Macro-F1 (%)', fontsize=8)
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.15, linewidth=0.4)
    ax.text(-0.12, 1.04, 'b', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


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
        idx = METHOD_ORDER.index(method) if method in METHOD_ORDER else 0
        color = METHOD_COLORS[idx]
        is_tcr = method == 'TCR-Net'
        ax.scatter(far, hrr, s=80 if is_tcr else 45, color=color,
                   edgecolor='white', linewidth=0.5, zorder=3)
        label = short_names.get(method, method)
        ox = 2.0 if far < 15 else -4.0
        ax.text(far + ox, hrr, label, fontsize=6.5, va='center',
                color=C_BGRAY, fontweight='bold' if is_tcr else 'normal')

    ax.set_xlabel('False Alarm Rate, FAR (%)', fontsize=8)
    ax.set_ylabel('High-Risk Recall, HRR (%)', fontsize=8)
    ax.set_xlim(-2, 55)
    ax.set_ylim(0, 85)
    ax.grid(alpha=0.15, linewidth=0.4)
    ax.text(-0.12, 1.04, 'c', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def main():
    out_dir = Path('exp_data/TCRNet_Final_Figures/figures')
    out_dir.mkdir(parents=True, exist_ok=True)


    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.06, right=0.97, top=0.95, bottom=0.07)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_architecture(ax_a)
    draw_macro_f1(ax_b)
    draw_hrr_far(ax_c)

    pdf_path = out_dir / 'fig_nature_architecture.pdf'
    fig.savefig(pdf_path, facecolor='white')
    print(f'  => {pdf_path}')


if __name__ == '__main__':
    main()
