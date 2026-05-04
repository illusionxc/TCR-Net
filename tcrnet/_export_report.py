"""
Experiment report export -- internal module, not exposed via tcrnet/__init__.py.

Generates paper-table CSV files from trained model outputs (metrics + detail CSVs).
Called by tcrnet_paper_pipeline.py, not as a standalone entry point.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from tcrnet.config import load_config


MODEL_ORDER = [
    "Rule-Based", "Content-Only", "Content+Semantic",
    "Sequence+Semantic", "Unified-Multimodal", "w/o Rule", "TCR-Net",
]

ABLATION_ORDER = [
    "Full", "w/o Consistency", "w/o Sequence",
    "w/o Rule", "w/o Prototype", "w/o transformer",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_detail_csv(output_dir: str | Path) -> Path:
    p = Path(output_dir)
    for cand in [p / "details" / "test_details.csv", p / "details_tcr" / "test_details.csv"]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"test_details.csv not found under {output_dir}")


def _metrics_from_detail(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["risk_label_true"].to_numpy()
    y_pred = df["risk_label_pred"].to_numpy()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    p_macro, r_macro, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    normal = y_true == 0
    far = float((((y_pred[normal] == 1) | (y_pred[normal] == 2)).sum()) / max(int(normal.sum()), 1))
    return {
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "low_precision": float(p[0]), "low_recall": float(r[0]),
        "mid_precision": float(p[1]), "mid_recall": float(r[1]),
        "high_precision": float(p[2]), "high_recall": float(r[2]),
        "high_f1": float(f[2]), "far": far,
    }


def _mcnemar_p_value(reference: pd.DataFrame, candidate: pd.DataFrame) -> float | None:
    if len(reference) != len(candidate):
        return None
    ref_ok = reference["risk_label_pred"].to_numpy() == reference["risk_label_true"].to_numpy()
    cand_ok = candidate["risk_label_pred"].to_numpy() == candidate["risk_label_true"].to_numpy()
    b = int((ref_ok & ~cand_ok).sum())
    c = int((~ref_ok & cand_ok).sum())
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / max(b + c, 1)
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


def export_table_5_1(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    metadata = _load_json(data_dir / "metadata.json")
    all_rows = []
    for split in ["train", "val", "test"]:
        all_rows.extend(_load_jsonl(data_dir / f"{split}.jsonl"))
    risk = pd.Series([int(r["risk_label"]) for r in all_rows])
    df = pd.DataFrame([
        {"Data Dimension": "Total samples", "Statistic": len(all_rows), "Description": "Train + validation + test samples."},
        {"Data Dimension": "Task-type count", "Statistic": int(metadata["num_intents"]), "Description": "Number of intent / task classes."},
        {"Data Dimension": "Equipment-object classes", "Statistic": int(metadata["num_objects"]), "Description": "Number of equipment object categories."},
        {"Data Dimension": "Low/Medium/High samples", "Statistic": f"{int((risk==0).sum())}/{int((risk==1).sum())}/{int((risk==2).sum())}", "Description": "Risk-label distribution over the exported dataset."},
        {"Data Dimension": "Average history length", "Statistic": int(metadata["history_length"]), "Description": "Fixed history window length K."},
    ])
    df.to_csv(out_dir / "tab_5_1_data_source_label_distribution.csv", index=False)
    return df


def export_table_5_2(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test", "ood"]:
        sp = data_dir / f"{split}.jsonl"
        if not sp.exists():
            rows.append({"Split": split, "Total": None, "Low": None, "Medium": None, "High": None, "Description": f"{split}.jsonl not found.", "Available": False})
            continue
        split_rows = _load_jsonl(sp)
        risk = pd.Series([int(r["risk_label"]) for r in split_rows])
        rows.append({"Split": split, "Total": len(split_rows), "Low": int((risk == 0).sum()), "Medium": int((risk == 1).sum()), "High": int((risk == 2).sum()), "Description": "Station-jsonl split." if split != "ood" else "Domain-shift split for OOD robustness.", "Available": True})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tab_5_2_split_statistics.csv", index=False)
    return df


def export_table_5_3(out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame([
        {"Category": "File-structure features", "Main Content": "initial_state, response_mask, amplitude bounds, steady-state bounds, delta_t_window, settle_steps", "Target Module": "Shared encoder / Rule branch"},
        {"Category": "Business-semantic features", "Main Content": "intent_id, object_type, context, control_params-derived semantics", "Target Module": "Shared encoder / Consistency branch"},
        {"Category": "Transmission-context features", "Main Content": "source_type, time_features, history_seq, source / task transition context", "Target Module": "Shared encoder / Sequence branch / Rule branch"},
        {"Category": "Rule items", "Main Content": "rule_type_id, rule_dst_id, rule_role_id, rule_hour, rule_size, rule_depth", "Target Module": "Rule branch"},
        {"Category": "Observation set Obs(.)", "Main Content": "intent_norm, object_norm, source_norm, |control_param_0|, |control_param_1|", "Target Module": "Consistency branch"},
    ])
    df.to_csv(out_dir / "tab_5_3_input_fields_and_modules.csv", index=False)
    return df


def export_table_5_4(config: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    pm = config["paper_model"]
    pt = config["paper_train"]
    pd_cfg = config["data"]
    rows = [
        {"Configuration": "Shared dimension", "Symbol": "d", "Value": pm["d_model"], "Description": "Shared embedding dimension."},
        {"Configuration": "History length", "Symbol": "K", "Value": pd_cfg["history_length"], "Description": "History window length."},
        {"Configuration": "Learning rate", "Symbol": "xi", "Value": pt["lr"], "Description": "AdamW learning rate."},
        {"Configuration": "Weight decay", "Symbol": "omega", "Value": pt["weight_decay"], "Description": "AdamW weight decay."},
        {"Configuration": "Batch size", "Symbol": "B", "Value": pd_cfg["batch_size"], "Description": "Mini-batch size."},
        {"Configuration": "Max epochs", "Symbol": "E_max", "Value": pt["epochs"], "Description": "Maximum epochs."},
        {"Configuration": "Early-stop rounds", "Symbol": "E_pat", "Value": pt.get("early_stop_patience", 0), "Description": "Patience-based stop on validation Macro-F1."},
        {"Configuration": "Seed set", "Symbol": "--", "Value": config["seed"], "Description": "Single seed in current run config."},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tab_5_4_training_settings.csv", index=False)
    return df


def export_table_5_5(out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame([
        {"Method": "Rule-Based", "Struct": False, "Sem": False, "Context": True, "History": False, "Rule_feat": True, "Three_comp": False},
        {"Method": "Content-Only", "Struct": True, "Sem": False, "Context": False, "History": False, "Rule_feat": False, "Three_comp": False},
        {"Method": "Content+Semantic", "Struct": True, "Sem": True, "Context": False, "History": False, "Rule_feat": False, "Three_comp": False},
        {"Method": "Sequence+Semantic", "Struct": True, "Sem": True, "Context": False, "History": True, "Rule_feat": False, "Three_comp": False},
        {"Method": "Unified-Multimodal", "Struct": True, "Sem": True, "Context": True, "History": True, "Rule_feat": False, "Three_comp": False},
        {"Method": "TCR-Net", "Struct": True, "Sem": True, "Context": True, "History": True, "Rule_feat": True, "Three_comp": True},
    ])
    df.to_csv(out_dir / "tab_5_5_baseline_input_scope.csv", index=False)
    return df


def export_table_5_6(summary_path: Path, out_dir: Path) -> pd.DataFrame:
    summary = _load_json(summary_path)
    rows = []
    ref_payload = summary["experiments"].get("TCR-Net")
    ref_detail = pd.read_csv(_find_detail_csv(ref_payload["output_dir"])) if ref_payload else None
    for model in MODEL_ORDER:
        payload = summary["experiments"].get(model)
        if payload is None:
            continue
        detail = pd.read_csv(_find_detail_csv(payload["output_dir"]))
        extra = _metrics_from_detail(detail)
        test_metrics = payload["metrics"]["test"]
        rows.append({
            "Method": model,
            "Macro-F1": float(test_metrics["macro_f1"]),
            "Weighted-F1": float(test_metrics["weighted_f1"]),
            "Precision": extra["macro_precision"],
            "Recall": extra["macro_recall"],
            "HRR": float(test_metrics["high_recall"]),
            "FAR": extra["far"],
            "p-value": None if model == "TCR-Net" or ref_detail is None else _mcnemar_p_value(ref_detail, detail),
            "DetailCSV": str(_find_detail_csv(payload["output_dir"])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tab_5_6_main_overall_performance.csv", index=False)
    return df


def export_table_5_7(summary_path: Path, out_dir: Path) -> pd.DataFrame:
    summary = _load_json(summary_path)
    rows = []
    for model in MODEL_ORDER:
        payload = summary["experiments"].get(model)
        if payload is None:
            continue
        detail = pd.read_csv(_find_detail_csv(payload["output_dir"]))
        extra = _metrics_from_detail(detail)
        rows.append({"Method": model, "Low-P": extra["low_precision"], "Low-R": extra["low_recall"], "Mid-P": extra["mid_precision"], "Mid-R": extra["mid_recall"], "High-P": extra["high_precision"], "High-R": extra["high_recall"], "High-F1": extra["high_f1"], "DetailCSV": str(_find_detail_csv(payload["output_dir"]))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tab_5_7_per_risk_class_results.csv", index=False)
    return df


def export_table_5_8(summary_path: Path, out_dir: Path) -> pd.DataFrame:
    summary = _load_json(summary_path)
    rows = []
    full_payload = summary["experiments"].get("Full")
    full_macro = None
    if full_payload is not None:
        full_detail = pd.read_csv(_find_detail_csv(full_payload["output_dir"]))
        full_extra = _metrics_from_detail(full_detail)
        full_macro = float(full_payload["metrics"]["test"]["macro_f1"])
    for variant in ABLATION_ORDER:
        payload = summary["experiments"].get(variant)
        if payload is None:
            rows.append({"Model": variant, "Macro-F1": None, "HRR": None, "FAR": None, "Relative drop": None, "Note": "Not found.", "Available": False})
            continue
        detail = pd.read_csv(_find_detail_csv(payload["output_dir"]))
        extra = _metrics_from_detail(detail)
        macro = float(payload["metrics"]["test"]["macro_f1"])
        rows.append({"Model": variant, "Macro-F1": macro, "HRR": float(payload["metrics"]["test"]["high_recall"]), "FAR": extra["far"], "Relative drop": None if full_macro is None else full_macro - macro, "Note": "Full model" if variant == "Full" else "", "Available": True})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tab_5_8_ablation_results.csv", index=False)
    return df


def _external_subset_masks(detail: pd.DataFrame):
    rule_mask = (detail["V_type"] + detail["V_dst"] + detail["V_role"] + detail["V_time"] > 0) | (detail["V_size"] > 0)
    if "anomaly_type" not in detail.columns:
        return None, None, rule_mask
    seq_types = {7}
    cons_types = {5, 6}
    seq_mask = (detail["anomaly_type"].isin(seq_types)) & (~rule_mask)
    cons_mask = (detail["anomaly_type"].isin(cons_types)) & (~rule_mask)
    return cons_mask, seq_mask, rule_mask


def _f1_on_subset(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(precision_recall_fscore_support(df["risk_label_true"], df["risk_label_pred"], average="macro", zero_division=0)[2])


def export_fig_5_9_data(summary_path: Path, out_dir: Path) -> pd.DataFrame:
    summary = _load_json(summary_path)
    experiments = summary.get("experiments", {})
    if not experiments:
        return pd.DataFrame()
    ref_name = "Full" if "Full" in experiments else next(iter(experiments))
    ref_detail = pd.read_csv(_find_detail_csv(experiments[ref_name]["output_dir"]))
    cons_mask_ref, seq_mask_ref, rule_mask_ref = _external_subset_masks(ref_detail)
    if cons_mask_ref is None:
        cons_mask_ref = (ref_detail["R_cons"] >= ref_detail[["R_seq", "R_rule"]].max(axis=1)) & (~rule_mask_ref)
        seq_mask_ref = (ref_detail["R_seq"] >= ref_detail[["R_cons", "R_rule"]].max(axis=1)) & (~rule_mask_ref)
    subset_lookup = {
        "S_rule": set(ref_detail.loc[rule_mask_ref, "sample_id"].astype(int).tolist()),
        "S_cons": set(ref_detail.loc[cons_mask_ref, "sample_id"].astype(int).tolist()),
        "S_seq": set(ref_detail.loc[seq_mask_ref, "sample_id"].astype(int).tolist()),
    }

    rows = []
    for variant, payload in experiments.items():
        detail = pd.read_csv(_find_detail_csv(payload["output_dir"]))
        for subset_name, ids in subset_lookup.items():
            mask = detail["sample_id"].astype(int).isin(ids)
            rows.append({"Variant": variant, "Subset": subset_name, "Macro-F1": _f1_on_subset(detail[mask]), "Samples": int(mask.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fig_5_9_mechanism_subset_data.csv", index=False)
    return df


def export_fig_5_10_data(summary_path: Path, out_dir: Path) -> pd.DataFrame:
    summary = _load_json(summary_path)
    rows = []
    for model in MODEL_ORDER:
        payload = summary.get("experiments", {}).get(model)
        if payload is None:
            continue
        detail = pd.read_csv(_find_detail_csv(payload["output_dir"]))
        for key in ["object_type", "source_type", "intent_id"]:
            for bucket, sub in detail.groupby(key):
                rows.append({"Model": model, "ScenarioKey": key, "Bucket": bucket, "Macro-F1": _f1_on_subset(sub), "Samples": len(sub)})
        rule_active = ((detail["V_type"] + detail["V_dst"] + detail["V_role"] + detail["V_time"] > 0) | (detail["V_size"] > 0)).astype(int)
        d2 = detail.copy()
        d2["rule_active"] = rule_active
        for bucket, sub in d2.groupby("rule_active"):
            rows.append({"Model": model, "ScenarioKey": "rule_active", "Bucket": int(bucket), "Macro-F1": _f1_on_subset(sub), "Samples": len(sub)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fig_5_10_cross_scenario_data.csv", index=False)
    return df


def _predict_with_eta(scores: np.ndarray, eta_1: float, eta_2: float) -> np.ndarray:
    pred = np.zeros_like(scores, dtype=np.int64)
    pred[scores >= eta_1] = 1
    pred[scores >= eta_2] = 2
    return pred


def export_fig_5_11_data(summary_path: Path, out_dir: Path) -> pd.DataFrame:
    summary = _load_json(summary_path)
    rows = []
    sweep = np.linspace(0.68, 0.96, 8)
    for model in MODEL_ORDER:
        payload = summary.get("experiments", {}).get(model)
        if payload is None:
            continue
        detail = pd.read_csv(_find_detail_csv(payload["output_dir"]))
        hard_rule = (detail["V_type"] + detail["V_dst"] + detail["V_role"] + detail["V_time"] > 0) | (detail["V_size"] > 0)
        for subset_name, sub in {"all": detail, "rule_active": detail[hard_rule], "clean": detail[~hard_rule]}.items():
            if sub.empty:
                continue
            eta_1 = float(sub["eta_1"].iloc[0])
            for q in sweep:
                eta_2 = float(np.quantile(sub["R_total"], q))
                if eta_2 <= eta_1:
                    eta_2 = eta_1 + 1.0e-6
                pred = _predict_with_eta(sub["R_total"].to_numpy(), eta_1, eta_2)
                _, _, f_macro, _ = precision_recall_fscore_support(sub["risk_label_true"], pred, average="macro", zero_division=0)
                rows.append({"Model": model, "Subset": subset_name, "Eta2Quantile": float(q), "Eta1": eta_1, "Eta2": eta_2, "Macro-F1": float(f_macro), "Samples": len(sub)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fig_5_11_distribution_perturbation_data.csv", index=False)
    return df


def export_fig_5_12_data(detail_csv: Path, out_dir: Path) -> pd.DataFrame:
    if not detail_csv or not detail_csv.exists():
        return pd.DataFrame()
    detail = pd.read_csv(detail_csv)
    idx_cons = int((detail["R_cons"] - detail[["R_seq", "R_rule"]].max(axis=1)).idxmax())
    idx_seq = int((detail["R_seq"] - detail[["R_cons", "R_rule"]].max(axis=1)).idxmax())
    idx_rule = int((detail["R_rule"] - detail[["R_cons", "R_seq"]].max(axis=1)).idxmax())
    rows = []
    for case_name, idx in [("semantic_mismatch", idx_cons), ("temporal_drift", idx_seq), ("rule_conflict", idx_rule)]:
        row = detail.loc[idx]
        rows.append({"Case": case_name, "sample_id": int(row["sample_id"]), "intent_id": int(row["intent_id"]), "object_type": int(row["object_type"]), "source_type": int(row["source_type"]), "risk_label_true": int(row["risk_label_true"]), "risk_label_pred": int(row["risk_label_pred"]), "R_cons": float(row["R_cons"]), "R_seq": float(row["R_seq"]), "R_rule": float(row["R_rule"]), "R_total": float(row["R_total"]), "V_type": float(row["V_type"]), "V_dst": float(row["V_dst"]), "V_role": float(row["V_role"]), "V_time": float(row["V_time"]), "V_size": float(row["V_size"]), "eta_1": float(row["eta_1"]), "eta_2": float(row["eta_2"])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fig_5_12_representative_cases.csv", index=False)
    return df


def export_manifest(out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame([
        {"Item": "tab_5_1", "Status": "available", "Comment": "From mock_station_v1 dataset."},
        {"Item": "tab_5_2", "Status": "available", "Comment": "Train/val/test/ood split statistics."},
        {"Item": "tab_5_3", "Status": "available", "Comment": "From current data schema."},
        {"Item": "tab_5_4", "Status": "available", "Comment": "Training settings with early-stop patience."},
        {"Item": "tab_5_5", "Status": "available", "Comment": "From paper design."},
        {"Item": "tab_5_6", "Status": "available", "Comment": "Main metrics + McNemar p-values."},
        {"Item": "tab_5_7", "Status": "available", "Comment": "Per-class metrics from detail CSVs."},
        {"Item": "tab_5_8", "Status": "available", "Comment": "Ablation results."},
        {"Item": "fig_5_9", "Status": "available", "Comment": "Mechanism subset data."},
        {"Item": "fig_5_10", "Status": "available", "Comment": "Cross-scenario data."},
        {"Item": "fig_5_11", "Status": "available", "Comment": "Threshold-sweep data."},
        {"Item": "fig_5_12", "Status": "available", "Comment": "Representative cases."},
    ])
    df.to_csv(out_dir / "report_data_manifest.csv", index=False)
    return df


def run_all_exports(
    config_path: str,
    output_root: str,
    detail_csv: str,
    mock_data_dir: str | None = None,
) -> dict[str, str]:
    config = load_config(config_path)
    output_root_path = Path(output_root)
    out_dir = output_root_path / "report_data"
    _ensure_dir(out_dir)

    if mock_data_dir:
        data_dir = Path(mock_data_dir)
    else:
        data_dir = Path(config["data"]["data_dir"])
        if not data_dir.is_absolute():
            data_dir = output_root_path.resolve().parent / data_dir

    export_table_5_1(data_dir, out_dir)
    export_table_5_2(data_dir, out_dir)
    export_table_5_3(out_dir)
    export_table_5_4(config, out_dir)
    export_table_5_5(out_dir)
    export_table_5_6(output_root_path / "baselines" / "summary.json", out_dir)
    export_table_5_7(output_root_path / "baselines" / "summary.json", out_dir)

    abl_path = output_root_path / "ablations" / "summary.json"
    if abl_path.exists():
        export_table_5_8(abl_path, out_dir)
        export_fig_5_9_data(abl_path, out_dir)

    export_fig_5_10_data(output_root_path / "baselines" / "summary.json", out_dir)
    export_fig_5_11_data(output_root_path / "baselines" / "summary.json", out_dir)
    if detail_csv:
        export_fig_5_12_data(Path(detail_csv), out_dir)
    else:
        print("[export] Skipping fig_5_12 (no detail_csv provided)")
    export_manifest(out_dir)

    return {"report_data_dir": str(out_dir.resolve())}
