


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


def prepare_station_config(config: dict[str, Any], data_dir: str | Path) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg["data"]["source"] = "station_jsonl"
    cfg["data"]["data_dir"] = str(data_dir)
    return cfg


def baseline_variants(base_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:


    variants = {}


    full = deepcopy(base_cfg)
    full["output_dir"] = "outputs/tcr_paper/baselines/tcr_net_full"
    full["paper_model"]["decision_mode"] = "decomposed"
    variants["TCR-Net"] = full


    rule_only = deepcopy(base_cfg)
    rule_only["output_dir"] = "outputs/tcr_paper/baselines/rule_only"
    rule_only["paper_model"]["decision_mode"] = "decomposed"
    rule_only["paper_model"]["use_structure_token"] = False
    rule_only["paper_model"]["use_semantic_token"] = False
    rule_only["paper_model"]["use_context_token"] = True
    rule_only["paper_model"]["use_consistency"] = False
    rule_only["paper_model"]["use_sequence"] = False
    rule_only["paper_model"]["use_prototype"] = False
    rule_only["paper_loss"]["lambda_pred"] = 0.0
    rule_only["paper_loss"]["lambda_proto"] = 0.0
    rule_only["paper_loss"]["lambda_seq"] = 0.0
    variants["Rule-Based"] = rule_only


    content_only = deepcopy(base_cfg)
    content_only["output_dir"] = "outputs/tcr_paper/baselines/content_only"
    content_only["paper_model"]["decision_mode"] = "unified"
    content_only["paper_model"]["use_structure_token"] = True
    content_only["paper_model"]["use_semantic_token"] = False
    content_only["paper_model"]["use_context_token"] = False
    content_only["paper_model"]["use_consistency"] = False
    content_only["paper_model"]["use_sequence"] = False
    content_only["paper_model"]["use_rule"] = False
    content_only["paper_model"]["use_prototype"] = False
    content_only["paper_loss"]["lambda_pred"] = 0.0
    content_only["paper_loss"]["lambda_proto"] = 0.0
    content_only["paper_loss"]["lambda_seq"] = 0.0
    variants["Content-Only"] = content_only


    content_sem = deepcopy(base_cfg)
    content_sem["output_dir"] = "outputs/tcr_paper/baselines/content_semantic"
    content_sem["paper_model"]["decision_mode"] = "unified"
    content_sem["paper_model"]["use_structure_token"] = True
    content_sem["paper_model"]["use_semantic_token"] = True
    content_sem["paper_model"]["use_context_token"] = False
    content_sem["paper_model"]["use_consistency"] = False
    content_sem["paper_model"]["use_sequence"] = False
    content_sem["paper_model"]["use_rule"] = False
    content_sem["paper_model"]["use_prototype"] = False
    content_sem["paper_loss"]["lambda_pred"] = 0.0
    content_sem["paper_loss"]["lambda_proto"] = 0.0
    content_sem["paper_loss"]["lambda_seq"] = 0.0
    variants["Content+Semantic"] = content_sem


    seq_sem = deepcopy(base_cfg)
    seq_sem["output_dir"] = "outputs/tcr_paper/baselines/sequence_semantic"
    seq_sem["paper_model"]["decision_mode"] = "unified"
    seq_sem["paper_model"]["use_structure_token"] = True
    seq_sem["paper_model"]["use_semantic_token"] = True
    seq_sem["paper_model"]["use_context_token"] = False
    seq_sem["paper_model"]["use_consistency"] = False
    seq_sem["paper_model"]["use_rule"] = False
    seq_sem["paper_model"]["use_prototype"] = False
    seq_sem["paper_loss"]["lambda_pred"] = 0.0
    seq_sem["paper_loss"]["lambda_proto"] = 0.0
    seq_sem["paper_loss"]["lambda_seq"] = 0.0
    variants["Sequence+Semantic"] = seq_sem


    unified = deepcopy(base_cfg)
    unified["output_dir"] = "outputs/tcr_paper/baselines/unified_multimodal"
    unified["paper_model"]["decision_mode"] = "unified"
    unified["paper_model"]["use_structure_token"] = True
    unified["paper_model"]["use_semantic_token"] = True
    unified["paper_model"]["use_context_token"] = True
    unified["paper_model"]["use_rule"] = False
    unified["paper_model"]["use_consistency"] = False
    unified["paper_model"]["use_prototype"] = False
    unified["paper_loss"]["lambda_pred"] = 0.0
    unified["paper_loss"]["lambda_proto"] = 0.0
    unified["paper_loss"]["lambda_seq"] = 0.0
    variants["Unified-Multimodal"] = unified


    wo_rule = deepcopy(base_cfg)
    wo_rule["output_dir"] = "outputs/tcr_paper/baselines/wo_rule"
    wo_rule["paper_model"]["decision_mode"] = "decomposed"
    wo_rule["paper_model"]["use_rule"] = False
    variants["w/o Rule"] = wo_rule

    return {name: variants[name] for name in BASELINE_ORDER}


def ablation_variants(base_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:

    variants = {}

    full = deepcopy(base_cfg)
    full["output_dir"] = "outputs/tcr_paper/ablations/full"
    variants["Full"] = full

    no_cons = deepcopy(base_cfg)
    no_cons["output_dir"] = "outputs/tcr_paper/ablations/wo_consistency"
    no_cons["paper_model"]["use_consistency"] = False
    no_cons["paper_loss"]["lambda_pred"] = 0.0
    no_cons["paper_loss"]["lambda_proto"] = 0.0
    variants["w/o Consistency"] = no_cons

    no_seq = deepcopy(base_cfg)
    no_seq["output_dir"] = "outputs/tcr_paper/ablations/wo_sequence"
    no_seq["paper_model"]["use_sequence"] = False
    no_seq["paper_loss"]["lambda_seq"] = 0.0
    variants["w/o Sequence"] = no_seq

    no_rule = deepcopy(base_cfg)
    no_rule["output_dir"] = "outputs/tcr_paper/ablations/wo_rule"
    no_rule["paper_model"]["use_rule"] = False
    variants["w/o Rule"] = no_rule

    no_proto = deepcopy(base_cfg)
    no_proto["output_dir"] = "outputs/tcr_paper/ablations/wo_prototype"
    no_proto["paper_model"]["use_prototype"] = False
    no_proto["paper_loss"]["lambda_proto"] = 0.0
    variants["w/o Prototype"] = no_proto

    no_transformer = deepcopy(base_cfg)
    no_transformer["output_dir"] = "outputs/tcr_paper/ablations/wo_transformer"
    no_transformer["paper_model"]["use_transformer"] = False
    variants["w/o transformer"] = no_transformer

    return {name: variants[name] for name in ABLATION_ORDER}
