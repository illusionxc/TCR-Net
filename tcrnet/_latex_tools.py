"""
LaTeX utilities -- internal module, not exposed via tcrnet/__init__.py.

Functions:
    1. Fill LaTeX tables and figures with experiment data (fill)
    2. Insert generated figures into LaTeX file (insert)

Called by tcrnet_paper_pipeline.py, not as a standalone entry point.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import pandas as pd


# LaTeX environment locator (shared by both subcommands)

def find_env_block(text: str, label: str, env: str) -> tuple[int, int] | None:
    label_token = rf"\label{{{label}}}"
    pos = text.find(label_token)
    if pos < 0:
        return None

    begin_token = rf"\begin{{{env}}}"
    end_token = rf"\end{{{env}}}"

    before = text[:pos]
    depth = 0
    begin_pos = -1
    for m in re.finditer(rf"({re.escape(begin_token)}|{re.escape(end_token)})", before):
        token = m.group()
        if token == end_token:
            depth += 1
        elif token == begin_token:
            if depth == 0:
                begin_pos = m.start()
                break
            depth -= 1
    if begin_pos < 0:
        return None

    after = text[pos:]
    depth = 1
    end_pos = -1
    for m in re.finditer(rf"({re.escape(begin_token)}|{re.escape(end_token)})", after):
        token = m.group()
        if token == begin_token:
            depth += 1
        elif token == end_token:
            depth -= 1
            if depth == 0:
                end_pos = pos + m.end()
                break
    if end_pos < 0:
        return None

    return (begin_pos, end_pos)


def replace_env_by_label(text: str, label: str, env: str, new_block: str) -> tuple[str, bool]:
    span = find_env_block(text, label, env)
    if span is None:
        return text, False
    begin, end = span
    return text[:begin] + new_block + text[end:], True


def remove_env_by_label(text: str, label: str, env: str) -> tuple[str, bool]:
    span = find_env_block(text, label, env)
    if span is None:
        return text, False
    begin, end = span
    return text[:begin] + text[end:], True


# fill -- fill LaTeX tables and figures with experiment data

TABLE_SPECS = {
    "tab:5_1": ("tab_5_1_data_source_label_distribution.csv", "table"),
    "tab:5_2": ("tab_5_2_split_statistics.csv", "table"),
    "tab:5_3": ("tab_5_3_input_fields_and_modules.csv", "table"),
    "tab:5_4": ("tab_5_4_training_settings.csv", "table"),
    "tab:5_5": ("tab_5_5_baseline_input_scope.csv", "table*"),
    "tab:5_6": ("tab_5_6_main_overall_performance.csv", "table*"),
    "tab:5_7": ("tab_5_7_per_risk_class_results.csv", "table*"),
    "tab:5_8": ("tab_5_8_ablation_results.csv", "table*"),
}

FIGURE_PLACEHOLDERS = {
    "Figure~5-1 Main metric comparison across methods (placeholder).": ("fig:5_1", "fig_5_1", "Main metric comparison across baselines."),
    "Figure~5-2 Confusion-matrix comparison across methods (placeholder).": ("fig:5_2", "fig_5_2", "Three-level confusion matrices."),
    "Figure~5-3 Performance drop of ablation settings relative to full model (placeholder).": ("fig:5_3", "fig_5_3", "Ablation deltas."),
    "Figure~5-4 Performance curves under cross-scenario and perturbation settings (placeholder).": ("fig:5_4", "fig_5_4", "Robustness summary."),
    "Figure~5-5 Risk-component and evidence visualization on representative cases (placeholder).": ("fig:5_5", "fig_5_5", "Case evidence."),
}

CAPTION_REPLACEMENTS = {
    "Pre-transmission file-sharing scenario in power-team operations (placeholder).": "Pre-transmission file-sharing workflow and the deployment position of TCR-Net.",
    "Overall architecture of TCR-Net (double-column placeholder for Fig. 2).": "Overall TCR-Net architecture.",
    "Shared semantic encoding and event-history alignment (single-column placeholder for Fig. 3).": "Shared semantic encoding and event-history alignment.",
    "Joint-risk fusion and three-level decision boundaries (single-column placeholder for Fig. 4).": "Joint risk-score distribution and decision boundaries.",
    "Mechanism-subset results (placeholder figure; grouped bar charts).": "Mechanism-subset validation.",
    "Cross-scenario generalization results (placeholder figure; line-chart layout).": "Cross-scenario generalization.",
    "Distribution-perturbation results (placeholder figure; line-chart layout).": "Distribution-perturbation robustness.",
    "Representative case explanation results (placeholder figure).": "Representative case explanations.",
}

COMPACT_REMOVE_FIGURES = {"fig:5_5": "figure*", "fig:5_10": "figure*", "fig:5_11": "figure*", "fig:5_12": "figure*"}
COMPACT_REMOVE_TABLES = {"tab:5_3": "table", "tab:5_4": "table"}
FIGURE_WIDTHS = {"fig:5_1": r"0.92\textwidth", "fig:5_2": r"0.90\textwidth", "fig:5_3": r"0.90\textwidth", "fig:5_4": r"0.90\textwidth", "fig:5_9": r"0.90\textwidth"}


def _esc(value: object) -> str:
    if pd.isna(value):
        return "--"
    text = str(value)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        text = text.replace(a, b)
    return text


def _fmt(value: object, decimals: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, str):
        return _esc(value)
    try:
        x = float(value)
    except Exception:
        return _esc(value)
    if math.isnan(x):
        return "--"
    return f"{x:.{decimals}f}"


def _bold(text: str) -> str:
    return rf"\textbf{{{text}}}"


def _best_mask(df: pd.DataFrame, col: str, higher: bool = True, tol: float = 5e-7) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    target = vals.max() if higher else vals.min()
    return (vals - target).abs() <= tol


def _fmt_best(value: object, is_best: bool, decimals: int = 3) -> str:
    text = _fmt(value, decimals)
    return _bold(text) if is_best and text != "--" else text


def _fmt_p(value: object) -> str:
    if pd.isna(value):
        return "--"
    try:
        x = float(value)
    except Exception:
        return _esc(value)
    if math.isnan(x):
        return "--"
    if x < 1.0e-3:
        return r"$<10^{-3}$"
    return f"{x:.3f}"


def _tex_symbol(value: object) -> str:
    mapping = {"d": r"$d$", "K": r"$K$", "xi": r"$\xi$", "omega": r"$\omega$", "B": r"$B$", "E_max": r"$E_{\max}$", "E_pat": r"$E_{\mathrm{pat}}$"}
    if pd.isna(value):
        return "--"
    return mapping.get(str(value), _esc(value))


def _yesno(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return "Yes" if str(value).lower() in {"true", "1", "yes"} else "No"


def _rows_to_latex(rows: list[list[str]]) -> str:
    return "\n".join(" & ".join(row) + r" \\" for row in rows)


def _render_table_5_1(df: pd.DataFrame) -> str:
    rows = [[_esc(r["Data Dimension"]), _esc(r["Statistic"]), _esc(r["Description"])] for _, r in df.iterrows()]
    return rf"""\begin{{table}}[!t]
\caption{{Data source, task-type coverage, and label distribution.}}
\label{{tab:5_1}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabularx}}{{\columnwidth}}{{|p{{0.30\columnwidth}}|p{{0.23\columnwidth}}|X|}}
\hline
Data Dimension & Statistic & Description \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{table}}"""


def _render_table_5_2(df: pd.DataFrame) -> str:
    names = {"train": r"$\mathcal{D}^{\mathrm{tr}}$", "val": r"$\mathcal{D}^{\mathrm{val}}$", "test": r"$\mathcal{D}^{\mathrm{te}}$", "ood": r"$\mathcal{D}^{\mathrm{ood}}$"}
    rows = []
    for _, r in df.iterrows():
        rows.append([names.get(str(r["Split"]), _esc(r["Split"])), _fmt(r["Total"], 0), _fmt(r["Low"], 0), _fmt(r["Medium"], 0), _fmt(r["High"], 0), _esc(r["Description"])])
    return rf"""\begin{{table}}[!t]
\caption{{Train/validation/test and cross-scenario split.}}
\label{{tab:5_2}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabularx}}{{\columnwidth}}{{|p{{0.17\columnwidth}}|p{{0.10\columnwidth}}|p{{0.09\columnwidth}}|p{{0.12\columnwidth}}|p{{0.10\columnwidth}}|X|}}
\hline
Split & Total & Low & Med. & High & Description \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{table}}"""


def _render_table_5_3(df: pd.DataFrame) -> str:
    rows = [[_esc(r["Category"]), _esc(r["Main Content"]), _esc(r["Target Module"])] for _, r in df.iterrows()]
    return rf"""\begin{{table}}[!t]
\caption{{Input fields, rule items, and corresponding modules.}}
\label{{tab:5_3}}
\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\columnwidth}}{{|p{{0.25\columnwidth}}|X|p{{0.25\columnwidth}}|}}
\hline
Category & Main Content & Target Module \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{table}}"""


def _render_table_5_4(df: pd.DataFrame) -> str:
    rows = [[_esc(r["Configuration"]), _tex_symbol(r["Symbol"]), _esc(r["Value"]), _esc(r["Description"])] for _, r in df.iterrows()]
    return rf"""\begin{{table}}[!t]
\caption{{Major training and implementation settings.}}
\label{{tab:5_4}}
\centering\scriptsize\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabularx}}{{\columnwidth}}{{|p{{0.24\columnwidth}}|p{{0.12\columnwidth}}|p{{0.18\columnwidth}}|X|}}
\hline
Configuration & Symbol & Value & Description \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{table}}"""


def _render_table_5_5(df: pd.DataFrame) -> str:
    rows = [[_esc(r["Method"]), _yesno(r["Struct"]), _yesno(r["Sem"]), _yesno(r["Context"]), _yesno(r["History"]), _yesno(r["Rule_feat"]), _yesno(r["Three_comp"])] for _, r in df.iterrows()]
    return rf"""\begin{{table*}}[!t]
\caption{{Baseline methods and input scopes.}}
\label{{tab:5_5}}
\centering\small
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabularx}}{{\textwidth}}{{|p{{0.22\textwidth}}|Y|Y|Y|Y|Y|Y|}}
\hline
Method & Struct. & Sem. & Context & History & Rule feat. & 3-comp. \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{adjustbox}}
\end{{table*}}"""


def _render_table_5_6(df: pd.DataFrame) -> str:
    best = {"Macro-F1": _best_mask(df, "Macro-F1"), "Weighted-F1": _best_mask(df, "Weighted-F1"), "Precision": _best_mask(df, "Precision"), "Recall": _best_mask(df, "Recall"), "HRR": _best_mask(df, "HRR"), "FAR": _best_mask(df, "FAR", False)}
    rows = []
    for idx, r in df.iterrows():
        rows.append([_esc(r["Method"]), _fmt_best(r["Macro-F1"], bool(best["Macro-F1"].loc[idx])), _fmt_best(r["Weighted-F1"], bool(best["Weighted-F1"].loc[idx])), _fmt_best(r["Precision"], bool(best["Precision"].loc[idx])), _fmt_best(r["Recall"], bool(best["Recall"].loc[idx])), _fmt_best(r["HRR"], bool(best["HRR"].loc[idx])), _fmt_best(r["FAR"], bool(best["FAR"].loc[idx])), _fmt_p(r["p-value"])])
    return rf"""\begin{{table*}}[!t]
\caption{{Main overall performance on the test split.}}
\label{{tab:5_6}}
\centering\small
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabularx}}{{\textwidth}}{{|p{{0.21\textwidth}}|Y|Y|Y|Y|Y|Y|Y|}}
\hline
Method & Macro-F1 & Weighted-F1 & Precision & Recall & HRR & FAR & $p$-value \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{adjustbox}}
\end{{table*}}"""


def _render_table_5_7(df: pd.DataFrame) -> str:
    cols = ["Low-P", "Low-R", "Mid-P", "Mid-R", "High-P", "High-R", "High-F1"]
    best = {c: _best_mask(df, c) for c in cols}
    rows = [[_esc(r["Method"])] + [_fmt_best(r[c], bool(best[c].loc[idx])) for c in cols] for idx, r in df.iterrows()]
    return rf"""\begin{{table*}}[!t]
\caption{{Per-risk-class classification results.}}
\label{{tab:5_7}}
\centering\small
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabularx}}{{\textwidth}}{{|p{{0.18\textwidth}}|Y|Y|Y|Y|Y|Y|Y|}}
\hline
Method & Low-P & Low-R & Mid-P & Mid-R & High-P & High-R & High-F1 \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{adjustbox}}
\end{{table*}}"""


def _render_table_5_8(df: pd.DataFrame) -> str:
    best = {"Macro-F1": _best_mask(df, "Macro-F1"), "HRR": _best_mask(df, "HRR"), "FAR": _best_mask(df, "FAR", False), "Relative drop": _best_mask(df, "Relative drop", False)}
    rows = []
    for idx, r in df.iterrows():
        rows.append([_esc(r["Model"]), _fmt_best(r["Macro-F1"], bool(best["Macro-F1"].loc[idx])), _fmt_best(r["HRR"], bool(best["HRR"].loc[idx])), _fmt_best(r["FAR"], bool(best["FAR"].loc[idx])), _fmt_best(r["Relative drop"], bool(best["Relative drop"].loc[idx])), _esc(r.get("Note", ""))])
    return rf"""\begin{{table*}}[!t]
\caption{{Ablation results.}}
\label{{tab:5_8}}
\centering\small
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabularx}}{{\textwidth}}{{|p{{0.22\textwidth}}|Y|Y|Y|Y|p{{0.20\textwidth}}|}}
\hline
Model & Macro-F1 & HRR & FAR & Relative drop & Note \\
\hline
{_rows_to_latex(rows)}
\hline
\end{{tabularx}}
\end{{adjustbox}}
\end{{table*}}"""


TABLE_RENDERERS = {
    "tab:5_1": _render_table_5_1, "tab:5_2": _render_table_5_2,
    "tab:5_3": _render_table_5_3, "tab:5_4": _render_table_5_4,
    "tab:5_5": _render_table_5_5, "tab:5_6": _render_table_5_6,
    "tab:5_7": _render_table_5_7, "tab:5_8": _render_table_5_8,
}


def _figure_block(label: str, stem: str, caption: str, fig_dir: str) -> str:
    width = FIGURE_WIDTHS.get(label, r"0.90\textwidth")
    return rf"""\begin{{figure*}}[!t]
\centering
\includegraphics[width={width}]{{{fig_dir}/{stem}.pdf}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure*}}"""


def _extract_caption(block: str) -> str:
    m = re.search(r"\\caption\{([^}]*)\}", block)
    return m.group(1).strip() if m else "Figure."


# Unified LaTeX operation entry

def fill_tables_and_figures(
    tex_path: str,
    report_data_dir: str,
    figure_source_dir: str,
    figure_dest_dir: str,
    latex_figure_dir: str = "figures_auto",
    compact: bool = False,
) -> dict[str, int]:
    tex_path = Path(tex_path)
    report_dir = Path(report_data_dir)
    fig_src = Path(figure_source_dir)
    fig_dst = Path(figure_dest_dir)
    fig_dst.mkdir(parents=True, exist_ok=True)

    for src in fig_src.glob("*"):
        if src.suffix.lower() in {".pdf", ".png", ".csv"}:
            shutil.copy2(src, fig_dst / src.name)

    text = tex_path.read_text(encoding="utf-8")
    backup = tex_path.with_name(tex_path.name + ".bak_before_fill")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    tables_replaced = 0
    for label, (csv_name, env) in TABLE_SPECS.items():
        if compact and label in COMPACT_REMOVE_TABLES:
            text, ok = remove_env_by_label(text, label, env)
            if ok:
                tables_replaced += 1
            continue
        df = pd.read_csv(report_dir / csv_name)
        block = TABLE_RENDERERS[label](df)
        text, ok = replace_env_by_label(text, label, env, block)
        if ok:
            tables_replaced += 1

    figures_replaced = 0
    for placeholder, (label, stem, caption) in FIGURE_PLACEHOLDERS.items():
        block = _figure_block(label, stem, caption, latex_figure_dir)
        if placeholder in text:
            text = text.replace(placeholder, block)
            figures_replaced += 1

    captions_updated = 0
    for old, new in CAPTION_REPLACEMENTS.items():
        if old in text:
            text = text.replace(old, new)
            captions_updated += 1

    for old_ref, new_ref in {
        "Figure~5-1": r"Figure~\ref{fig:5_1}",
        "Figure~5-2": r"Figure~\ref{fig:5_2}",
        "Figure~5-3": r"Figure~\ref{fig:5_3}",
        "Figure~5-4": r"Figure~\ref{fig:5_4}",
        "Figure~5-5": r"Figure~\ref{fig:5_5}",
    }.items():
        if old_ref in text:
            text = text.replace(old_ref, new_ref)

    if compact:
        for label, env in COMPACT_REMOVE_FIGURES.items():
            text, _ = remove_env_by_label(text, label, env)

    for phrase in ["(placeholder)", "(placeholder figure); ", " (placeholder figure)", " (placeholder)."]:
        text = text.replace(phrase, "")
    text = text.replace("  ", " ")

    tex_path.write_text(text, encoding="utf-8")
    return {"tables_replaced": tables_replaced, "figures_replaced": figures_replaced, "captions_updated": captions_updated}


def insert_figures(
    tex_path: str,
    figure_dir: str,
    ext: str = "pdf",
) -> int:
    tex_path = Path(tex_path)
    text = tex_path.read_text(encoding="utf-8")

    backup = tex_path.with_name(tex_path.name + ".bak_before_insert")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    replacements = {
        "fig:intro_fig1": ("figure", "fig_intro_fig1"),
        "fig:method_fig2": ("figure*", "fig_method_fig2"),
        "fig:method_fig3": ("figure", "fig_method_fig3"),
        "fig:method_fig4": ("figure", "fig_method_fig4"),
        "fig:5_9": ("figure*", "fig_5_9"),
        "fig:5_10": ("figure*", "fig_5_10"),
        "fig:5_11": ("figure*", "fig_5_11"),
        "fig:5_12": ("figure*", "fig_5_12"),
    }

    count = 0
    for label, (env, stem) in replacements.items():
        span = find_env_block(text, label, env)
        if span is None:
            continue
        begin, end = span
        old_block = text[begin:end]
        caption = _extract_caption(old_block)
        width = r"0.98\textwidth" if env == "figure*" else r"0.96\columnwidth"
        rel = f"{figure_dir.rstrip('/')}/{stem}.{ext}"
        new_block = (
            f"\\begin{{{env}}}[!t]\n"
            "\\centering\n"
            f"\\includegraphics[width={width}]{{{rel}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\end{{{env}}}"
        )
        text = text[:begin] + new_block + text[end:]
        count += 1

    tex_path.write_text(text, encoding="utf-8")
    return count
