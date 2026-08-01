


import matplotlib.pyplot as plt
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
    'axes.spines.left': True,
    'axes.spines.bottom': True,
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
    'savefig.pad_inches': 0.1,
})


C_TEAL   = '#006D77'
C_BLUE   = '#2D6A9F'
C_AMBER  = '#D98A3C'
C_RED    = '#A23B3B'
C_GREEN  = '#3A7D54'
C_GRAY   = '#8C8C8C'
C_LGRAY  = '#BFBFBF'
C_BGRAY  = '#4D4D4D'


METHOD_ORDER = ['TCR-Net', 'Unified-Multimodal', 'w/o Rule',
                'Sequence+Semantic', 'Content+Semantic',
                'Rule-Based', 'Content-Only']
METHOD_COLORS = [C_TEAL, C_BLUE, C_GRAY, C_LGRAY, C_BGRAY, C_GRAY, C_LGRAY]

METHOD_COLORS_MAP = dict(zip(METHOD_ORDER, [
    C_TEAL,
    '#4A8BB7',
    '#7A7A7A',
    '#A0A0A0',
    '#B8B8B8',
    '#C8C8C8',
    '#D8D8D8',
]))


ABLATION_COLORS = {
    'Full': C_TEAL,
    'w/o Consistency': '#A0A0A0',
    'w/o Sequence': '#B0B0B0',
    'w/o Rule': C_RED,
    'w/o Prototype': '#B8B8B8',
    'w/o transformer': '#C0C0C0',
}


DATA_DIR = Path('exp_data/TCRNet_Final_Figures/tcr_paper/report_data')

def load_csv(name):
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

df_main = load_csv('tab_5_6_main_overall_performance.csv')
df_perclass = load_csv('tab_5_7_per_risk_class_results.csv')
df_ablation = load_csv('tab_5_8_ablation_results.csv')


def panel_a(ax):

    data = df_main.sort_values('Macro-F1', ascending=True).reset_index(drop=True)
    labels = data['Method'].tolist()
    values = (data['Macro-F1'] * 100).tolist()

    y = np.arange(len(labels))
    colors = [C_TEAL if m == 'TCR-Net' else C_LGRAY for m in labels]

    ax.barh(y, values, height=0.55, color=colors, zorder=2, edgecolor='white',
            linewidth=0.3)


    for i, (v, lab) in enumerate(zip(values, labels)):
        color = C_TEAL if lab == 'TCR-Net' else C_BGRAY
        ax.text(v + 1.2, i, f'{v:.1f}', va='center', fontsize=7, color=color,
                fontweight='bold' if lab == 'TCR-Net' else 'normal')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Macro-F1 (%)', fontsize=8)
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.2, linewidth=0.4)


    ax.text(-0.25, 1.04, 'a', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


def panel_b(ax):


    top4 = ['TCR-Net', 'Unified-Multimodal', 'w/o Rule', 'Sequence+Semantic']
    df_plot = df_perclass[df_perclass['Method'].isin(top4)].copy()
    df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=top4, ordered=True)
    df_plot = df_plot.sort_values('Method')


    Low_R = df_plot['Low-R'].values
    Low_P = df_plot['Low-P'].values
    Mid_R = df_plot['Mid-R'].values
    Mid_P = df_plot['Mid-P'].values
    High_R = df_plot['High-R'].values
    High_P = df_plot['High-P'].values


    def f1(p, r):
        return 2 * p * r / (p + r + 1e-10)

    low_f1 = f1(Low_P, Low_R) * 100
    mid_f1 = f1(Mid_P, Mid_R) * 100
    high_f1 = f1(High_P, High_R) * 100

    x = np.arange(len(top4))
    w = 0.22
    risk_colors = ['#4A7B9D', C_AMBER, C_RED]
    risk_labels = ['Low-risk', 'Medium-risk', 'High-risk']

    for risk_i in range(3):
        vals = [low_f1, mid_f1, high_f1][risk_i]
        off = (risk_i - 1) * w
        bars = ax.bar(x + off, vals, w, color=risk_colors[risk_i],
                      label=risk_labels[risk_i],
                      edgecolor='white', linewidth=0.3, zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{v:.0f}', ha='center', fontsize=6, color=C_BGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(['TCR-Net', 'Unified', '-Rule', 'Seq+Sem'],
                       fontsize=7.5)
    ax.set_ylabel('F1 score (%)', fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', handlelength=0.8, handleheight=0.8, fontsize=6.5)
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax.text(-0.18, 1.04, 'b', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


def panel_c(ax):

    data = df_main.copy()

    short_names = {
        'TCR-Net': 'TCR-Net',
        'Unified-Multimodal': 'Unified',
        'Sequence+Semantic': 'Seq+Sem',
        'Content+Semantic': 'C+Sem',
        'Content-Only': 'Content',
        'Rule-Based': 'Rule',
        'w/o Rule': '-Rule',
    }

    for _, row in data.iterrows():
        method = row['Method']
        hrr = row['HRR'] * 100
        far = row['FAR'] * 100
        color = METHOD_COLORS_MAP.get(method, C_GRAY)
        is_tcr = method == 'TCR-Net'
        ax.scatter(far, hrr, s=70 if is_tcr else 40, color=color,
                   edgecolor='white', linewidth=0.5, zorder=3)
        label = short_names.get(method, method)
        offset_x = 1.5
        if far > 15:
            offset_x = -3.5
        ax.text(far + offset_x, hrr, label, fontsize=6.5, va='center',
                color=C_BGRAY, fontweight='bold' if is_tcr else 'normal')

    ax.set_xlabel('FAR (%)', fontsize=8)
    ax.set_ylabel('HRR (%)', fontsize=8)
    ax.set_xlim(-2, 55)
    ax.set_ylim(0, 85)
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.text(-0.18, 1.04, 'c', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


def panel_d(ax):

    df = df_ablation[df_ablation['Model'] != 'Full'].copy()
    full_f1 = df_ablation[df_ablation['Model'] == 'Full']['Macro-F1'].values[0]

    variants = ['w/o Rule', 'w/o transformer', 'w/o Sequence',
                'w/o Consistency', 'w/o Prototype']
    df = df[df['Model'].isin(variants)]

    df['drop'] = (df['Macro-F1'] - full_f1) * 100
    df = df.sort_values('drop')

    labels = df['Model'].tolist()
    drops = df['drop'].tolist()

    x = np.arange(len(labels))
    colors = [ABLATION_COLORS.get(m, C_GRAY) for m in labels]

    bars = ax.bar(x, drops, width=0.5, color=colors, edgecolor='white',
                  linewidth=0.3, zorder=2)
    ax.axhline(0, color=C_BGRAY, linewidth=0.6)

    for bar, v in zip(bars, drops):
        offset = -4.5 if v < 0 else 1.5
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f'{v:.1f}', ha='center', fontsize=7, color=C_BGRAY,
                fontweight='bold')


    short_abl = {'w/o Rule': '-Rule', 'w/o transformer': '-TFM',
                 'w/o Sequence': '-Seq', 'w/o Consistency': '-Cons',
                 'w/o Prototype': '-Proto'}
    ax.set_xticks(x)
    ax.set_xticklabels([short_abl.get(m, m) for m in labels], fontsize=7.5)
    ax.set_ylabel(r'$\Delta$ Macro-F1 (pp)', fontsize=8)
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax.text(-0.18, 1.04, 'd', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


def main():
    out_dir = Path('exp_data/TCRNet_Final_Figures/figures')
    out_dir.mkdir(parents=True, exist_ok=True)


    fig = plt.figure(figsize=(7.2, 5.8))


    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.40,
                          left=0.08, right=0.97, top=0.94, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)


    pdf_path = out_dir / 'fig_nature_main.pdf'
    fig.savefig(pdf_path, facecolor='white')
    plt.close(fig)
    print(f'  => {pdf_path}')


if __name__ == '__main__':
    main()
