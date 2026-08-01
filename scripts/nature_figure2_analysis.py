


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


C_TEAL   = '#006D77'
C_BLUE   = '#2D6A9F'
C_AMBER  = '#D98A3C'
C_RED    = '#A23B3B'
C_GREEN  = '#3A7D54'
C_GRAY   = '#8C8C8C'
C_LGRAY  = '#BFBFBF'
C_BGRAY  = '#4D4D4D'


DATA_DIR = Path('exp_data/TCRNet_Final_Figures/tcr_paper/report_data')
df_main = pd.read_csv(DATA_DIR / 'tab_5_6_main_overall_performance.csv')
df_perclass = pd.read_csv(DATA_DIR / 'tab_5_7_per_risk_class_results.csv')
df_ablation = pd.read_csv(DATA_DIR / 'tab_5_8_ablation_results.csv')


df_mech = pd.DataFrame()
mech_path = DATA_DIR / 'fig_5_9_mechanism_subset_data.csv'
if mech_path.exists():
    df_mech = pd.read_csv(mech_path)


def draw_ablation(ax):

    df = df_ablation[df_ablation['Model'] != 'Full'].copy()
    full_f1 = df_ablation[df_ablation['Model'] == 'Full']['Macro-F1'].values[0]
    full_hrr = df_ablation[df_ablation['Model'] == 'Full']['HRR'].values[0]


    df['f1_drop'] = (df['Macro-F1'] - full_f1) * 100
    df['hrr_drop'] = (df['HRR'] - full_hrr) * 100
    df = df.sort_values('f1_drop')

    variants = df['Model'].tolist()
    short_names = {'w/o Consistency': '-Cons', 'w/o Sequence': '-Seq',
                   'w/o Rule': '-Rule', 'w/o Prototype': '-Proto',
                   'w/o transformer': '-TFM'}

    x = np.arange(len(variants))
    w = 0.30

    bars1 = ax.bar(x - w/2, df['f1_drop'].values, w, color=C_TEAL,
                   label=r'$\Delta$Macro-F1', edgecolor='white', linewidth=0.3, zorder=2)
    bars2 = ax.bar(x + w/2, df['hrr_drop'].values, w, color=C_AMBER,
                   label=r'$\Delta$HRR', edgecolor='white', linewidth=0.3, zorder=2)

    ax.axhline(0, color=C_BGRAY, linewidth=0.6)

    def _label_bars(bars, values):
        for bar, v in zip(bars, values):
            off = -5.5 if v < 0 else 1.5
            ax.text(bar.get_x() + bar.get_width()/2, v + off,
                    f'{v:.1f}', ha='center', fontsize=6.5, color=C_BGRAY)

    _label_bars(bars1, df['f1_drop'].values)
    _label_bars(bars2, df['hrr_drop'].values)

    ax.set_xticks(x)
    ax.set_xticklabels([short_names.get(m, m) for m in variants], fontsize=7.5)
    ax.set_ylabel('Performance change (pp)', fontsize=8)
    ax.legend(loc='lower left', handlelength=0.8, handleheight=0.8, fontsize=6.5)
    ax.grid(axis='y', alpha=0.15, linewidth=0.4)
    ax.text(-0.12, 1.04, 'a', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def draw_perclass_f1(ax):
    top4 = ['TCR-Net', 'Unified-Multimodal', 'w/o Rule', 'Sequence+Semantic']
    df = df_perclass[df_perclass['Method'].isin(top4)].copy()
    df['Method'] = pd.Categorical(df['Method'], top4, ordered=True)
    df = df.sort_values('Method')

    def f1(p, r):
        return 2 * p * r / (p + r + 1e-10)

    low_f1  = f1(df['Low-P'].values, df['Low-R'].values) * 100
    mid_f1  = f1(df['Mid-P'].values, df['Mid-R'].values) * 100
    high_f1 = f1(df['High-P'].values, df['High-R'].values) * 100

    x = np.arange(len(top4))
    w = 0.22
    colors = ['#4A7B9D', C_AMBER, C_RED]
    labels = ['Low-risk', 'Medium-risk', 'High-risk']

    for i, (vals, c, lb) in enumerate(zip([low_f1, mid_f1, high_f1], colors, labels)):
        off = (i - 1) * w
        bars = ax.bar(x + off, vals, w, color=c, label=lb,
                      edgecolor='white', linewidth=0.3, zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{v:.0f}', ha='center', fontsize=6, color=C_BGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(['TCR-Net', 'Unified', '-Rule', 'Seq+Sem'], fontsize=7.5)
    ax.set_ylabel('F1 score (%)', fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', handlelength=0.8, handleheight=0.8, fontsize=6.5)
    ax.grid(axis='y', alpha=0.15, linewidth=0.4)
    ax.text(-0.12, 1.04, 'b', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def draw_mechanism(ax):

    if df_mech.empty:
        ax.text(0.5, 0.5, 'Mechanism subset data\nnot available',
                ha='center', va='center', fontsize=8, color=C_GRAY)
        ax.axis('off')
        ax.text(-0.12, 1.04, 'c', transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='bottom', ha='left')
        return

    keep = ['Full', 'w/o Consistency', 'w/o Sequence', 'w/o Rule']
    subsets = ['S_cons', 'S_seq', 'S_rule']
    df = df_mech[df_mech['Variant'].isin(keep) & df_mech['Subset'].isin(subsets)]
    if df.empty:
        ax.text(0.5, 0.5, 'No mechanism subset data', ha='center', va='center',
                fontsize=8, color=C_GRAY)
        ax.axis('off')
        return

    pivot = df.pivot_table(index='Variant', columns='Subset',
                            values='Macro-F1', aggfunc='mean')
    pivot = pivot.reindex(keep)[subsets]

    x = np.arange(len(keep))
    w = 0.22
    colors = [C_BLUE, C_GREEN, C_RED]
    labels = [r'S_cons', r'S_seq', r'S_rule']

    for i, (col, c, lb) in enumerate(zip(subsets, colors, labels)):
        off = (i - 1) * w
        vals = pivot[col].to_numpy() * 100
        bars = ax.bar(x + off, vals, w, color=c, label=lb,
                      edgecolor='white', linewidth=0.3, zorder=2)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                        f'{v:.1f}', ha='center', fontsize=6, color=C_BGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(['Full', '-Cons', '-Seq', '-Rule'], fontsize=7.5)
    ax.set_ylabel('Macro-F1 (%)', fontsize=8)
    ax.legend(loc='lower left', handlelength=0.8, handleheight=0.8, fontsize=6.5)
    ax.grid(axis='y', alpha=0.15, linewidth=0.4)
    ax.text(-0.12, 1.04, 'c', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def main():
    out_dir = Path('exp_data/TCRNet_Final_Figures/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.35,
                          left=0.08, right=0.97, top=0.95, bottom=0.07)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_ablation(ax_a)
    draw_perclass_f1(ax_b)
    draw_mechanism(ax_c)

    pdf_path = out_dir / 'fig_nature_analysis.pdf'
    fig.savefig(pdf_path, facecolor='white')
    print(f'  => {pdf_path}')


if __name__ == '__main__':
    main()
