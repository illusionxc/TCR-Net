"""
 ——  baseline  ablation 。


-------
    from tcrnet.experiments import baseline_variants, ablation_variants
    variants = baseline_variants(base_config)
    for name, cfg in variants.items():
        train(cfg)

Baseline （7 ）
--------------------
1. Rule-Based        :  +  token
2. Content-Only      :  token + unified 
3. Content+Semantic  :  +  token + unified 
4. Sequence+Semantic :  + unified 
5. Unified-Multimodal:  + unified （）
6. w/o Rule          :  TCR-Net 
7. TCR-Net           : 

Ablation （6 ）
--------------------
1. Full              :  TCR-Net
2. w/o Consistency   : 
3. w/o Sequence      : 
4. w/o Rule          : 
5. w/o Prototype     : 
6. w/o transformer   : 
"""

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


def prepare_station_config(config: dict, data_dir: str | Path) -> dict:
    """..."""
    cfg = deepcopy(config)
    cfg["data"]["source"] = "station_jsonl"
    cfg["data"]["data_dir"] = str(data_dir)
    return cfg


def _set_output(cfg, path: str):
    cfg["output_dir"] = path


def baseline_variants(base_cfg: dict) -> dict[str, dict]:
    """..."""
    variants = {}

    def _base():
        return deepcopy(base_cfg)

    # 1) TCR-Net 
    c = _base()
    _set_output(c, "runs/baselines/tcr_net")
    c["paper_model"]["decision_mode"] = "decomposed"
    variants["TCR-Net"] = c

    # 2) Rule-Based
    c = _base()
    _set_output(c, "runs/baselines/rule_only")
    c["paper_model"].update(decision_mode="decomposed",
        use_structure_token=False, use_semantic_token=False, use_context_token=True,
        use_consistency=False, use_sequence=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Rule-Based"] = c

    # 3) Content-Only
    c = _base()
    _set_output(c, "runs/baselines/content_only")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=False, use_context_token=False,
        use_consistency=False, use_sequence=False, use_rule=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Content-Only"] = c

    # 4) Content+Semantic
    c = _base()
    _set_output(c, "runs/baselines/content_semantic")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=True, use_context_token=False,
        use_consistency=False, use_sequence=False, use_rule=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Content+Semantic"] = c

    # 5) Sequence+Semantic
    c = _base()
    _set_output(c, "runs/baselines/sequence_semantic")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=True, use_context_token=False,
        use_consistency=False, use_rule=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Sequence+Semantic"] = c

    # 6) Unified-Multimodal
    c = _base()
    _set_output(c, "runs/baselines/unified_multimodal")
    c["paper_model"].update(decision_mode="unified",
        use_structure_token=True, use_semantic_token=True, use_context_token=True,
        use_rule=False, use_consistency=False, use_prototype=False)
    for k in ["lambda_pred", "lambda_proto", "lambda_seq"]:
        c["paper_loss"][k] = 0.0
    variants["Unified-Multimodal"] = c

    # 7) w/o Rule
    c = _base()
    _set_output(c, "runs/baselines/wo_rule")
    c["paper_model"].update(decision_mode="decomposed", use_rule=False)
    variants["w/o Rule"] = c

    return {name: variants[name] for name in BASELINE_ORDER}


def ablation_variants(base_cfg: dict) -> dict[str, dict]:
    """..."""
    variants = {}

    def _base():
        return deepcopy(base_cfg)

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
