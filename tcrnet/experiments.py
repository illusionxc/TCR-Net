


from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


BASELINE_ORDER = [
    "Rule-Based",
    "Content-Only",
    "Content+Semantic",
    "Sequence+Semantic",
    "Unified-Multimodal",
    "w/o Rule",
    "TCR-Net",
]

ABLATION_ORDER = [
    "Full",
    "w/o Consistency",
    "w/o Sequence",
    "w/o Rule",
    "w/o Prototype",
    "w/o transformer",
]

REVISION_CORE_ORDER = [
    "TCR-Net",
    "Unified-Multimodal",
    "Sequence+Semantic",
    "Content+Semantic",
    "Content-Only",
    "Rule-Based",
    "w/o Rule",
]

REVISION_ABLATION_ORDER = [
    "Full",
    "w/o Consistency",
    "w/o Sequence",
    "w/o Rule",
    "w/o Prototype",
    "w/o transformer",
]

ATTENTION_DIRECTION_ORDER = [
    "Bidirectional",
    "Current-to-History",
    "History-to-Current",
    "No Cross-Attention",
]


def _apply_optimized_tcr_calibration(cfg: dict) -> dict:


    model_cfg = cfg.setdefault("paper_model", {})
    model_cfg["use_typed_rule_features"] = True
    model_cfg["stabilize_aux_evidence"] = True
    optimization_cfg = cfg.get("revision_optimization", {})
    if "lambda_proto" in optimization_cfg:
        lambda_proto = float(optimization_cfg["lambda_proto"])
        if lambda_proto < 0.0:
            raise ValueError("revision_optimization.lambda_proto must be nonnegative")
        cfg.setdefault("paper_loss", {})["lambda_proto"] = lambda_proto
    score_cfg = cfg.setdefault("paper_score", {})
    score_cfg["hybrid_mix_grid"] = [1.0]
    thresholds = score_cfg.setdefault("thresholds", {})
    thresholds["eta1_quantiles"] = [
        0.450, 0.475, 0.500, 0.525, 0.550,
        0.575, 0.600, 0.625, 0.650,
    ]
    thresholds["eta2_quantiles"] = [
        0.700, 0.725, 0.750, 0.775, 0.800,
        0.825, 0.850, 0.875, 0.900,
    ]
    score_cfg["optimization_protocol"] = (
        "typed rule evidence and log1p-stabilized component evidence; "
        "prototype-loss weight and global fusion design selected on validation runs; "
        "per-run eta1/eta2 selected on validation only"
    )
    return cfg


def prepare_station_config(config: dict, data_dir: str | Path) -> dict:

    cfg = deepcopy(config)
    cfg["data"]["source"] = "station_jsonl"
    cfg["data"]["data_dir"] = str(data_dir)
    return cfg


def _set_output(cfg, path: str):
    cfg["output_dir"] = path


def baseline_variants(base_cfg: dict) -> dict[str, dict]:

    variants = {}

    def _base():
        return deepcopy(base_cfg)


    c = _base()
    _set_output(c, "runs/baselines/tcr_net")
    c["paper_model"]["decision_mode"] = "decomposed"
    _apply_optimized_tcr_calibration(c)
    variants["TCR-Net"] = c


    c = _base()
    _set_output(c, "runs/baselines/rule_only")
    c["paper_model"].update(decision_mode="decomposed",
        use_structure_token=False, use_semantic_token=False, use_context_token=True,
        use_consistency=False, use_sequence=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Rule-Based"] = c


    c = _base()
    _set_output(c, "runs/baselines/content_only")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=False, use_context_token=False,
        use_consistency=False, use_sequence=False, use_rule=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Content-Only"] = c


    c = _base()
    _set_output(c, "runs/baselines/content_semantic")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=True, use_context_token=False,
        use_consistency=False, use_sequence=False, use_rule=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Content+Semantic"] = c


    c = _base()
    _set_output(c, "runs/baselines/sequence_semantic")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=True, use_context_token=False,
        use_consistency=False, use_rule=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Sequence+Semantic"] = c


    c = _base()
    _set_output(c, "runs/baselines/unified_multimodal")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=True, use_context_token=True,
        use_rule=False, use_consistency=False, use_prototype=False,
        use_transformer=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Unified-Multimodal"] = c


    c = _base()
    _set_output(c, "runs/baselines/wo_rule")
    c["paper_model"].update(decision_mode="decomposed", use_rule=False)
    _apply_optimized_tcr_calibration(c)
    variants["w/o Rule"] = c

    return {name: variants[name] for name in BASELINE_ORDER}


def ablation_variants(base_cfg: dict) -> dict[str, dict]:

    variants = {}

    def _base():
        return _apply_optimized_tcr_calibration(deepcopy(base_cfg))

    c = _base(); _set_output(c, "runs/ablations/full"); variants["Full"] = c

    c = _base(); _set_output(c, "runs/ablations/wo_consistency")
    c["paper_model"]["use_consistency"] = False
    c["paper_loss"]["lambda_pred"] = c["paper_loss"]["lambda_proto"] = 0.0
    variants["w/o Consistency"] = c

    c = _base(); _set_output(c, "runs/ablations/wo_sequence")
    c["paper_model"]["use_sequence"] = False
    c["paper_loss"]["lambda_seq"] = 0.0
    variants["w/o Sequence"] = c

    c = _base(); _set_output(c, "runs/ablations/wo_rule")
    c["paper_model"]["use_rule"] = False
    variants["w/o Rule"] = c

    c = _base(); _set_output(c, "runs/ablations/wo_prototype")
    c["paper_model"]["use_prototype"] = False
    c["paper_loss"]["lambda_proto"] = 0.0
    variants["w/o Prototype"] = c

    c = _base(); _set_output(c, "runs/ablations/wo_transformer")
    c["paper_model"]["use_transformer"] = False
    variants["w/o transformer"] = c

    return {name: variants[name] for name in ABLATION_ORDER}


def revision_core_variants(base_cfg: dict) -> dict[str, dict]:

    variants = baseline_variants(base_cfg)
    return {name: variants[name] for name in REVISION_CORE_ORDER}


def revision_ablation_variants(base_cfg: dict) -> dict[str, dict]:

    variants = ablation_variants(base_cfg)
    return {name: variants[name] for name in REVISION_ABLATION_ORDER}


def attention_direction_variants(base_cfg: dict) -> dict[str, dict]:

    modes = {
        "Bidirectional": "bidirectional",
        "Current-to-History": "current_to_history",
        "History-to-Current": "history_to_current",
        "No Cross-Attention": "none",
    }
    variants = {}
    for name, mode in modes.items():
        cfg = deepcopy(base_cfg)
        cfg["paper_model"]["decision_mode"] = "decomposed"
        cfg["paper_model"]["use_cross_attention"] = mode != "none"
        cfg["paper_model"]["cross_attention_mode"] = mode
        _apply_optimized_tcr_calibration(cfg)
        slug = name.lower().replace(" ", "_").replace("-", "_")
        _set_output(cfg, f"runs/attention_direction/{slug}")
        variants[name] = cfg
    return {name: variants[name] for name in ATTENTION_DIRECTION_ORDER}


def history_window_variants(
    base_cfg: dict, windows: tuple[int, ...] = (4, 8, 12, 16),
) -> dict[str, dict]:

    variants = {}
    for window in windows:
        if window <= 0:
            raise ValueError(f"history window must be a positive integer, got={window}")
        cfg = deepcopy(base_cfg)
        _apply_optimized_tcr_calibration(cfg)
        cfg["data"]["history_crop_length"] = int(window)
        _set_output(cfg, f"runs/history_window/k_{window}")
        variants[f"K={window}"] = cfg
    return variants
