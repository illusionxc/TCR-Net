


from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TASK_NAMES = [
    "inspection_photo",
    "defect_form",
    "maintenance_report",
    "handover_doc",
    "routine_record",
    "equipment_log",
]

EQUIPMENT_NAMES = [
    "transformer",
    "breaker",
    "line",
    "protection_device",
]

TERMINAL_NAMES = [
    "mobile_device",
    "laptop",
    "workstation",
]

ANOMALY_NAMES = [
    "normal",
    "extension_spoofing",
    "compression_anomaly",
    "source_anomaly",
    "destination_anomaly",
    "task_stage_mismatch",
    "role_mismatch",
    "sequence_deviation",
]


TASK_FILE_PROFILES = [

    {"ext_allowed": [0, 1], "size_mean": 2.0, "size_std": 1.0, "entropy_mean": 7.4, "entropy_std": 0.3,
     "has_macros": False, "max_size": 8.0, "min_size": 0.05},

    {"ext_allowed": [3, 4], "size_mean": 0.5, "size_std": 0.3, "entropy_mean": 4.8, "entropy_std": 0.5,
     "has_macros": True, "max_size": 2.0, "min_size": 0.02},

    {"ext_allowed": [2, 3], "size_mean": 1.5, "size_std": 1.0, "entropy_mean": 5.8, "entropy_std": 0.4,
     "has_macros": False, "max_size": 5.0, "min_size": 0.05},

    {"ext_allowed": [2, 5], "size_mean": 3.0, "size_std": 2.0, "entropy_mean": 6.2, "entropy_std": 0.5,
     "has_macros": False, "max_size": 12.0, "min_size": 0.1},

    {"ext_allowed": [3, 8], "size_mean": 0.2, "size_std": 0.15, "entropy_mean": 4.0, "entropy_std": 0.6,
     "has_macros": True, "max_size": 0.8, "min_size": 0.01},

    {"ext_allowed": [6, 7, 8], "size_mean": 0.3, "size_std": 0.2, "entropy_mean": 5.0, "entropy_std": 0.5,
     "has_macros": False, "max_size": 1.5, "min_size": 0.005},
]

EXTENSION_NAMES = [".jpg", ".png", ".pdf", ".docx", ".xlsx", ".zip", ".xml", ".json", ".csv", ".mp4"]


EXTENSION_MAGIC = {
    0: b"\xff\xd8\xff",
    1: b"\x89PNG",
    2: b"%PDF",
    3: b"PK\x03\x04",
    4: b"PK\x03\x04",
    5: b"PK\x03\x04",
    6: b"<?xml",
    7: b"{",
    8: b"",
    9: b"\x00\x00\x00",
}

RECORD_CATEGORIES = ["photo", "form", "document", "spreadsheet", "archive", "log"]
OPERATION_STAGES = ["pre_task", "on_site", "post_task", "review"]
CONFIDENTIALITY_LEVELS = [0, 1, 2, 3]

DESTINATION_TYPES = [
    "team_share",
    "archive",
    "review_station",
    "external",
]

OPERATOR_ROLES = [
    "field_worker",
    "supervisor",
    "engineer",
    "admin",
]


@dataclass
class SyntheticConfig:
    source: str
    data_dir: str | None
    train_samples: int
    val_samples: int
    test_samples: int
    train_normal_only: bool
    anomaly_ratio: float
    batch_size: int
    num_workers: int
    sequence_length: int
    history_length: int
    state_dim: int
    context_dim: int
    control_param_dim: int
    time_feature_dim: int
    history_feature_dim: int
    num_intents: int
    num_objects: int
    num_sources: int
    num_modes: int

    @classmethod
    def from_dict(cls, cfg: Dict) -> "SyntheticConfig":
        merged = dict(cfg)
        merged.setdefault("source", "synthetic")
        merged.setdefault("data_dir", None)
        return cls(**merged)


class SmartGridDataset(Dataset):
    def __init__(self, samples: List[Dict[str, np.ndarray | int | float]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        result: Dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                dtype = torch.long if np.issubdtype(value.dtype, np.integer) else torch.float32
                result[key] = torch.as_tensor(value, dtype=dtype)
            elif isinstance(value, (int, np.integer)):
                result[key] = torch.tensor(value, dtype=torch.long)
            elif isinstance(value, (float, np.floating)):
                result[key] = torch.tensor(float(value), dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported value type for key {key}: {type(value)}")
        return result


def _first_order_response(
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
    delay_steps: int,
    tau: float,
) -> np.ndarray:

    traj = np.zeros((steps, start.shape[0]), dtype=np.float32)
    for t in range(steps):
        if t <= delay_steps:
            traj[t] = start
            continue
        alpha = 1.0 - math.exp(-(t - delay_steps) / max(tau, 1e-3))
        traj[t] = start + alpha * (target - start)
    return traj


def _build_history_sequence(
    rng: np.random.Generator,
    task_id: int,
    equip_id: int,
    term_id: int,
    history_len: int,
    anomaly_type: int,
) -> np.ndarray:

    seq = np.zeros((history_len, 8), dtype=np.float32)
    base_hour = rng.integers(0, 24)
    for t in range(history_len):
        hist_task = task_id
        hist_size_dev = rng.normal(0.0, 0.12)
        hist_ent_dev = rng.normal(0.0, 0.08)


        if anomaly_type == 7 and t >= history_len // 2:
            hist_task = rng.integers(0, len(TASK_NAMES))
            hist_size_dev += rng.normal(0.6, 0.20)
            hist_ent_dev += rng.normal(0.4, 0.15)

        hour = (base_hour - history_len + t) % 24
        seq[t] = np.array(
            [
                hist_task / max(len(TASK_NAMES) - 1, 1),
                equip_id / max(len(EQUIPMENT_NAMES) - 1, 1),
                term_id / max(len(TERMINAL_NAMES) - 1, 1),
                hist_size_dev,
                hist_ent_dev,
                math.sin(2 * math.pi * hour / 24.0),
                math.cos(2 * math.pi * hour / 24.0),
                1.0 if anomaly_type == 7 and t >= history_len // 2 else 0.0,
            ],
            dtype=np.float32,
        )
    return seq


def _task_defaults(task_id: int) -> Tuple[int, int]:

    equip = [0, 2, 2, 0, 3, 1][task_id]
    term = [0, 0, 0, 1, 1, 2][task_id]
    return equip, term


def _task_file_profile(task_id: int) -> Dict:
    if 0 <= task_id < len(TASK_FILE_PROFILES):
        return dict(TASK_FILE_PROFILES[task_id])

    return {"ext_allowed": [0, 1], "size_mean": 1.0, "size_std": 1.0, "entropy_mean": 5.0, "entropy_std": 1.0,
            "has_macros": False, "max_size": 10.0, "min_size": 0.01}


def _infer_expected_file_fields(
    task_id: int,
    initial_state: np.ndarray,
    context: np.ndarray,
    file_params: np.ndarray,
    sequence_length: int,
) -> Dict[str, np.ndarray | float]:

    state_dim = int(initial_state.shape[0])
    file_size_limit = float(context[5]) if len(context) > 5 else 1.0
    H = int(sequence_length)

    target = initial_state.copy()
    response_mask = np.zeros(state_dim, dtype=np.float32)
    settle_steps = float(max(H // 2, 1))
    delta_t_window = float(max(H - 1, 1))

    quality_dev = float(file_params[0])
    severity = float(file_params[1])


    if task_id == 0:

        delta = quality_dev * file_size_limit * 0.3
        target[0] = np.clip(initial_state[0] + delta, 0.0, file_size_limit * 1.5)
        target[3] = np.clip(initial_state[3] + severity * 0.1, 0.0, 1.0)
        response_mask[[0, 3, 4]] = 1.0
    elif task_id == 1:

        target[4] = np.clip(initial_state[4] + severity * 0.2, 0.0, 1.0)
        response_mask[[4]] = 1.0
    elif task_id == 2:

        target[1] = np.clip(initial_state[1] + quality_dev * 0.15, 0.0, 1.0)
        target[4] = np.clip(initial_state[4] + severity * 0.1, 0.0, 1.0)
        response_mask[[1, 4]] = 1.0
    elif task_id == 3:

        target[0] = np.clip(initial_state[0] + quality_dev * file_size_limit * 0.5, 0.0, file_size_limit * 1.5)
        target[1] = np.clip(initial_state[1] + quality_dev * 0.1, 0.0, 1.0)
        target[4] = np.clip(initial_state[4] + severity * 0.15, 0.0, 1.0)
        response_mask[[0, 1, 4]] = 1.0
    elif task_id == 4:

        target[4] = np.clip(initial_state[4] + severity * 0.1, 0.0, 1.0)
        response_mask[[4]] = 1.0
    else:

        response_mask[[0]] = 1.0
        target[0] = np.clip(initial_state[0] + quality_dev * 0.05, 0.0, file_size_limit)

    delta_state = target - initial_state
    allowed_dir = np.sign(delta_state).astype(np.float32)
    amp_abs = np.maximum(np.abs(delta_state), 0.02)
    amp_low = 0.8 * amp_abs
    amp_high = 1.2 * amp_abs + 1e-3
    steady_low = np.zeros(state_dim, dtype=np.float32)
    steady_high = np.full(state_dim, 0.03, dtype=np.float32)
    steady_high[4] = 0.15
    steady_high[5] = 0.10

    return {
        "response_mask": response_mask.astype(np.float32),
        "allowed_dir": allowed_dir.astype(np.float32),
        "amp_low": amp_low.astype(np.float32),
        "amp_high": amp_high.astype(np.float32),
        "steady_low": steady_low.astype(np.float32),
        "steady_high": steady_high.astype(np.float32),
        "delta_t_window": np.float32(delta_t_window),
        "settle_steps": np.float32(settle_steps),
        "size_bounds": np.array([0.0, file_size_limit], dtype=np.float32),
        "entropy_bounds": np.array([0.0, 1.0], dtype=np.float32),
        "header_bounds": np.array([0.0, 0.3], dtype=np.float32),
        "metadata_bounds": np.array([0.0, 0.5], dtype=np.float32),
        "max_exec_flag": np.float32(0.5),
    }


def _build_default_history_sequence(
    task_id: int,
    equip_id: int,
    term_id: int,
    history_length: int,
) -> np.ndarray:

    seq = np.zeros((history_length, 8), dtype=np.float32)
    for t in range(history_length):
        seq[t] = np.array(
            [
                task_id / max(len(TASK_NAMES) - 1, 1),
                equip_id / max(len(EQUIPMENT_NAMES) - 1, 1),
                term_id / max(len(TERMINAL_NAMES) - 1, 1),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )
    return seq


def _generate_file_properties(
    rng: np.random.Generator,
    task_id: int,
    num_modes: int,
    domain_shift: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:


    profile = _task_file_profile(task_id)
    hour = int(rng.integers(0, 24))


    workload = rng.uniform(0.1, 0.9)
    file_volume = rng.uniform(0.0, 1.0)
    storage = rng.uniform(0.15, 0.85)
    network_quality = rng.uniform(0.8, 1.0)
    time_pressure = rng.uniform(0.0, 1.0)
    size_limit = rng.uniform(0.6, 1.0)

    if domain_shift:

        workload = np.clip(workload * rng.uniform(1.3, 2.0), 0.0, 1.0)
        file_volume = np.clip(file_volume * rng.uniform(0.3, 0.7), 0.0, 1.0)
        network_quality = np.clip(network_quality + rng.normal(-0.1, 0.05), 0.6, 1.0)
        time_pressure = np.clip(time_pressure + rng.uniform(0.1, 0.3), 0.0, 1.0)

    context = np.array([workload, file_volume, storage, network_quality, time_pressure, size_limit],
                       dtype=np.float32)


    raw_size = max(rng.normal(profile["size_mean"], profile["size_std"]), profile["min_size"])
    raw_size = min(raw_size, profile["max_size"])
    file_size_norm = raw_size / max(size_limit * 10.0, 1.0)


    if task_id in {2, 3}:
        comp_ratio = rng.uniform(0.3, 0.7)
    elif task_id in {0, 5}:
        comp_ratio = rng.uniform(0.8, 1.0)
    else:
        comp_ratio = rng.uniform(0.5, 0.9)


    entropy = rng.normal(profile["entropy_mean"], profile["entropy_std"])
    entropy = np.clip(entropy, 0.5, 7.9)
    entropy_norm = entropy / 8.0


    header_consistency = 0.0


    metadata_gap = rng.beta(2, 8) if rng.random() < 0.15 else rng.beta(1, 30)
    metadata_completeness = float(np.clip(metadata_gap, 0.0, 1.0))


    has_macros = 1.0 if (profile["has_macros"] and rng.random() < 0.1) else 0.0

    ext_id = int(rng.choice(profile["ext_allowed"]))
    file_params = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)], dtype=np.float32)

    initial_state = np.array(
        [file_size_norm, comp_ratio, entropy_norm, header_consistency, metadata_completeness, has_macros],
        dtype=np.float32,
    )

    exec_flag_steady = rng.beta(1, 25)
    if profile["has_macros"]:
        exec_flag_steady = rng.beta(2, 15)

    return initial_state, context, file_params, ext_id, hour


def _generate_event(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    allow_anomaly: bool,
    domain_shift: bool = False,
) -> Dict[str, np.ndarray | int | float]:


    H = cfg.sequence_length
    S = cfg.state_dim
    delta_t_window = float(max(H - 1, 1))

    task_id = int(rng.integers(0, cfg.num_intents))
    equip_id, term_id = _task_defaults(task_id)
    mode0 = int(rng.integers(0, cfg.num_modes))


    initial_state, context, file_params, ext_id, hour = _generate_file_properties(
        rng, task_id, cfg.num_modes, domain_shift
    )

    time_features = np.array(
        [math.sin(2.0 * math.pi * hour / 24.0), math.cos(2.0 * math.pi * hour / 24.0)],
        dtype=np.float32,
    )

    target = initial_state.copy()
    response_mask = np.zeros(S, dtype=np.float32)
    settle_steps = float(rng.integers(H // 3, max(H // 2, 2)))

    file_size_limit = float(context[5])
    quality_dev = float(file_params[0])
    severity = float(file_params[1])


    if task_id == 0:
        delta = abs(quality_dev) * file_size_limit * 0.3
        target[0] = np.clip(initial_state[0] + delta, 0.0, file_size_limit * 1.5)
        target[3] = np.clip(initial_state[3] + severity * 0.1, 0.0, 1.0)
        response_mask[[0, 3, 4]] = 1.0
    elif task_id == 1:
        target[4] = np.clip(initial_state[4] + severity * 0.2, 0.0, 1.0)
        response_mask[[4]] = 1.0
    elif task_id == 2:
        target[1] = np.clip(initial_state[1] + quality_dev * 0.15, 0.0, 1.0)
        target[4] = np.clip(initial_state[4] + severity * 0.1, 0.0, 1.0)
        response_mask[[1, 4]] = 1.0
    elif task_id == 3:
        target[0] = np.clip(initial_state[0] + abs(quality_dev) * file_size_limit * 0.5, 0.0, file_size_limit * 1.5)
        target[1] = np.clip(initial_state[1] + quality_dev * 0.1, 0.0, 1.0)
        target[4] = np.clip(initial_state[4] + severity * 0.15, 0.0, 1.0)
        response_mask[[0, 1, 4]] = 1.0
    elif task_id == 4:
        target[4] = np.clip(initial_state[4] + severity * 0.1, 0.0, 1.0)
        response_mask[[4]] = 1.0
    else:
        target[0] = np.clip(initial_state[0] + quality_dev * 0.05, 0.0, file_size_limit)
        response_mask[[0]] = 1.0

    delta_state = target - initial_state
    allowed_dir = np.sign(delta_state).astype(np.float32)
    amp_abs = np.maximum(np.abs(delta_state), 0.02)
    amp_low = 0.8 * amp_abs
    amp_high = 1.2 * amp_abs + 1e-3
    steady_low = np.zeros(S, dtype=np.float32)
    steady_high = np.full(S, 0.03, dtype=np.float32)
    steady_high[4] = 0.15
    steady_high[5] = 0.10

    max_exec_flag = 0.5

    delay_steps = int(rng.integers(1, max(H // 4, 2)))
    tau = float(rng.uniform(1.5, 4.0))
    expected_traj = _first_order_response(initial_state, target, H, delay_steps, tau)
    expected_traj[:, 4] = initial_state[4] if initial_state[4] > 0.5 else target[4]
    expected_traj[:, 5] = initial_state[5]

    anomaly_type = 0
    anomaly_label = 0
    actual_traj = expected_traj.copy()
    observed_equip_id = equip_id
    observed_term_id = term_id


    if allow_anomaly and rng.random() < cfg.anomaly_ratio:
        anomaly_label = 1

        type_weights = [0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.10, 0.20]
        anomaly_type = int(rng.choice(len(ANOMALY_NAMES), p=type_weights))

        if anomaly_type == 1:

            initial_state[3] = 1.0
            wrong_ext = int(rng.choice([e for e in range(len(EXTENSION_NAMES)) if e != ext_id]))
            actual_traj[:, 3] = 1.0
            target[3] = 1.0
        elif anomaly_type == 2:

            wrong_comp = rng.uniform(1.2, 2.0)
            actual_traj[delay_steps:, 1] = np.clip(
                actual_traj[delay_steps:, 1] * wrong_comp, 0.0, 1.0
            )
            target[1] = actual_traj[-1, 1]
        elif anomaly_type == 3:

            observed_term_id = int((term_id + rng.integers(1, max(cfg.num_sources, 2))) % max(cfg.num_sources, 1))
        elif anomaly_type == 4:

            observed_equip_id = int((equip_id + rng.integers(1, max(cfg.num_objects, 2))) % max(cfg.num_objects, 1))
        elif anomaly_type == 5:


            wrong_mode = int((mode0 + rng.integers(1, cfg.num_modes)) % cfg.num_modes)

            file_params = np.array([quality_dev + 0.5, severity + 0.3], dtype=np.float32)
        elif anomaly_type == 6:


            wrong_term = int((term_id + 2) % max(cfg.num_sources, 1))
            observed_term_id = wrong_term
        elif anomaly_type == 7:


            actual_traj[delay_steps:, 0] += rng.normal(0.0, 0.05, size=(H - delay_steps,))


    noise = rng.normal(0.0, 0.005, size=(H, S)).astype(np.float32)
    noise[:, 3] = 0.0
    noise[:, 5] = 0.0
    actual_traj += noise
    actual_traj[:, 0] = np.clip(actual_traj[:, 0], 0.0, 1.5)
    actual_traj[:, 1] = np.clip(actual_traj[:, 1], 0.0, 1.0)
    actual_traj[:, 2] = np.clip(actual_traj[:, 2], 0.0, 1.0)
    actual_traj[:, 3] = np.round(np.clip(actual_traj[:, 3], 0.0, 1.0))
    actual_traj[:, 4] = np.clip(actual_traj[:, 4], 0.0, 1.0)
    actual_traj[:, 5] = np.round(np.clip(actual_traj[:, 5], 0.0, 1.0))


    rule_type_id = int(task_id)
    rule_dst_id = int(equip_id)
    rule_role_id = int(term_id)
    rule_hour = float(hour)

    rule_size = float(
        40.0 + abs(float(file_params[0])) * 160.0 + abs(float(file_params[1])) * 90.0 + max(float(initial_state[0]), 0.0) * 40.0
    )
    rule_depth = float(mode0)


    if anomaly_label == 1 and anomaly_type in {1, 4}:

        rule_size = rule_size * 2.5
        rule_depth = rule_depth + 3.0
    if anomaly_label == 1 and anomaly_type == 2:

        rule_hour = (rule_hour + 12.0) % 24.0
    if anomaly_label == 1 and anomaly_type == 3:

        rule_dst_id = int((equip_id + 1) % max(cfg.num_objects, 1))
    if anomaly_label == 1 and anomaly_type == 6:

        rule_role_id = int((term_id + 2) % max(cfg.num_sources, 1))
        rule_size = rule_size * 1.5


    if anomaly_label == 0:

        if rng.random() < 0.03:
            rule_hour = (rule_hour + rng.choice([-2.0, 2.0])) % 24.0

        if rng.random() < 0.03:
            rule_size = rule_size * rng.uniform(1.3, 1.6)

        if rng.random() < 0.02:
            rule_dst_id = int((equip_id + 1) % max(cfg.num_objects, 1))


    if anomaly_label == 0:
        risk_label = 0
    elif anomaly_type in {1, 4}:
        risk_label = 2
    elif anomaly_type == 2:
        risk_label = 1 if rng.random() < 0.4 else 2
    elif anomaly_type == 5:
        if abs(float(file_params[1])) > 0.3 or abs(float(file_params[0])) > 0.5:
            risk_label = 2 if rng.random() < 0.4 else 1
        else:
            risk_label = 1
    elif anomaly_type == 6:
        if abs(severity) > 0.2 or abs(quality_dev) > 0.2:
            risk_label = 2 if rng.random() < 0.3 else 1
        else:
            risk_label = 1
    elif anomaly_type == 7:
        risk_label = 1
    elif anomaly_type == 3:
        risk_label = 2 if rng.random() < 0.6 else 1
    else:
        risk_label = 1


    history_seq = _build_history_sequence(
        rng=rng,
        task_id=task_id,
        equip_id=equip_id,
        term_id=term_id,
        history_len=cfg.history_length,
        anomaly_type=anomaly_type,
    )

    return {
        "intent_id": task_id,
        "control_type": task_id,
        "object_type": observed_equip_id,
        "source_type": observed_term_id,
        "control_params": file_params.astype(np.float32),
        "time_features": time_features.astype(np.float32),
        "context": context.astype(np.float32),
        "initial_state": initial_state.astype(np.float32),
        "actual_traj": actual_traj.astype(np.float32),
        "expected_traj": expected_traj.astype(np.float32),
        "history_seq": history_seq.astype(np.float32),
        "response_mask": response_mask.astype(np.float32),
        "allowed_dir": allowed_dir.astype(np.float32),
        "amp_low": amp_low.astype(np.float32),
        "amp_high": amp_high.astype(np.float32),
        "steady_low": steady_low.astype(np.float32),
        "steady_high": steady_high.astype(np.float32),
        "delta_t_window": np.float32(delta_t_window),
        "settle_steps": np.float32(settle_steps),
        "p_bounds": np.array([0.0, file_size_limit], dtype=np.float32),
        "q_bounds": np.array([0.0, 1.0], dtype=np.float32),
        "u_bounds": np.array([0.0, 1.0], dtype=np.float32),
        "f_bounds": np.array([0.0, 1.0], dtype=np.float32),
        "max_ramp": np.float32(0.10),
        "anomaly_label": anomaly_label,
        "anomaly_type": anomaly_type,
        "risk_label": risk_label,
        "rule_type_id": int(rule_type_id),
        "rule_dst_id": int(rule_dst_id),
        "rule_role_id": int(rule_role_id),
        "rule_hour": float(rule_hour),
        "rule_size": float(rule_size),
        "rule_depth": float(rule_depth),
    }


def _make_samples(
    count: int, cfg: SyntheticConfig, seed: int, allow_anomaly: bool, domain_shift: bool = False
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    return [_generate_event(rng, cfg, allow_anomaly=allow_anomaly, domain_shift=domain_shift) for _ in range(count)]


def _to_serializable(sample: Dict[str, np.ndarray | int | float]) -> Dict:
    serializable = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, int)):
            serializable[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            serializable[key] = float(value)
        else:
            serializable[key] = value
    return serializable


def _from_serializable(sample: Dict) -> Dict[str, np.ndarray | int | float]:
    array_keys = {
        "control_params", "time_features", "context", "initial_state",
        "actual_traj", "expected_traj", "history_seq",
        "response_mask", "allowed_dir", "amp_low", "amp_high",
        "steady_low", "steady_high", "p_bounds", "q_bounds", "u_bounds", "f_bounds",
    }
    int_keys = {"intent_id", "control_type", "object_type", "source_type",
                "anomaly_label", "anomaly_type"}
    int_keys = int_keys | {"risk_label", "rule_type_id", "rule_dst_id", "rule_role_id"}
    result: Dict[str, np.ndarray | int | float] = {}
    for key, value in sample.items():
        if key in array_keys:
            result[key] = np.asarray(value, dtype=np.float32)
        elif key in int_keys:
            result[key] = int(value)
        elif isinstance(value, list):

            result[key] = np.asarray(value, dtype=np.float32)
        else:
            result[key] = float(value)
    return result


def _normalize_loaded_sample(sample: Dict, meta: Dict[str, int]) -> Dict[str, np.ndarray | int | float]:

    normalized = dict(sample)

    if "intent_id" not in normalized:
        if "control_type" in normalized:
            normalized["intent_id"] = int(normalized["control_type"])
        else:
            raise KeyError("Each sample must provide either 'intent_id' or 'control_type'.")

    task_id = int(normalized["intent_id"])
    normalized.setdefault("control_type", task_id)

    equip_id, term_id = _task_defaults(task_id)
    normalized.setdefault("object_type", equip_id)
    normalized.setdefault("source_type", term_id)
    normalized.setdefault("anomaly_label", 0)
    normalized.setdefault("anomaly_type", 0)
    normalized.setdefault("risk_label", 0)

    if "control_params" not in normalized:
        normalized["control_params"] = np.zeros(2, dtype=np.float32)
    if "time_features" not in normalized:
        normalized["time_features"] = np.zeros(2, dtype=np.float32)

    initial_state = np.asarray(normalized["initial_state"], dtype=np.float32)
    context = np.asarray(normalized["context"], dtype=np.float32)
    file_params = np.asarray(normalized["control_params"], dtype=np.float32)
    actual_traj = np.asarray(normalized["actual_traj"], dtype=np.float32)
    sequence_length = int(actual_traj.shape[0])
    num_modes = int(meta.get("num_modes", 4))

    if "history_seq" not in normalized:
        normalized["history_seq"] = _build_default_history_sequence(
            task_id=task_id,
            equip_id=int(normalized["object_type"]),
            term_id=int(normalized["source_type"]),
            history_length=max(int(meta.get("history_length", 12)), 1),
        )

    expected_fields = _infer_expected_file_fields(
        task_id=task_id,
        initial_state=initial_state,
        context=context,
        file_params=file_params,
        sequence_length=sequence_length,
    )
    for key, value in expected_fields.items():
        normalized.setdefault(key, value)

    if "expected_traj" not in normalized:
        normalized["expected_traj"] = actual_traj.copy()


    cp_abs = np.abs(file_params)
    hour = float(
        ((math.atan2(float(normalized["time_features"][0]), float(normalized["time_features"][1])) + 2.0 * math.pi) % (2.0 * math.pi))
        * (24.0 / (2.0 * math.pi))
    )
    normalized.setdefault("rule_type_id", int(normalized["control_type"]))
    normalized.setdefault("rule_dst_id", int(normalized["object_type"]))
    normalized.setdefault("rule_role_id", int(normalized["source_type"]))
    normalized.setdefault("rule_hour", float(hour))
    normalized.setdefault(
        "rule_size",
        float(40.0 + cp_abs[0] * 160.0 + cp_abs[1] * 90.0 + max(float(initial_state[0]), 0.0) * 40.0),
    )
    normalized.setdefault("rule_depth", float(round(float(initial_state[4]) * 5.0)))

    return _from_serializable(_to_serializable(normalized))


def export_mock_station_dataset(config: Dict, seed: int, output_dir: str | Path) -> Dict[str, str]:

    cfg = SyntheticConfig.from_dict(config["data"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    split_defs = {
        "train": (cfg.train_samples, not cfg.train_normal_only),
        "val": (cfg.val_samples, True),
        "test": (cfg.test_samples, True),
        "ood": (cfg.test_samples, True),
    }
    file_map = {}
    for offset, (split, (count, allow_anomaly)) in enumerate(split_defs.items(), start=1):
        samples = _make_samples(count, cfg, seed + 100 * offset, allow_anomaly=allow_anomaly, domain_shift=(split == "ood"))
        if split == "train" and cfg.train_normal_only:
            samples = [s for s in samples if s["anomaly_label"] == 0]
            while len(samples) < count:
                extra = _make_samples(count // 2, cfg, seed + 1000 + len(samples), allow_anomaly=False)
                samples.extend([s for s in extra if s["anomaly_label"] == 0])
            samples = samples[:count]

        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(_to_serializable(sample), ensure_ascii=False) + "\n")
        file_map[split] = str(path)

    metadata = {
        "source": "mock_station_jsonl",
        "seed": seed,
        "train_file": file_map["train"],
        "val_file": file_map["val"],
        "test_file": file_map["test"],
        "ood_file": file_map["ood"],
        "num_intents": cfg.num_intents,
        "num_objects": cfg.num_objects,
        "num_sources": cfg.num_sources,
        "num_modes": cfg.num_modes,
        "state_dim": cfg.state_dim,
        "context_dim": cfg.context_dim,
        "history_feature_dim": cfg.history_feature_dim,
        "history_length": cfg.history_length,
        "sequence_length": cfg.sequence_length,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_map


def _load_jsonl(path: str | Path) -> List[Dict[str, np.ndarray | int | float]]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(_from_serializable(json.loads(line)))
    return samples


def load_station_dataset(data_dir: str | Path) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int]]:

    data_dir = Path(data_dir)
    train_samples = _load_jsonl(data_dir / "train.jsonl")
    val_samples = _load_jsonl(data_dir / "val.jsonl")
    test_samples = _load_jsonl(data_dir / "test.jsonl")

    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        sample0 = train_samples[0]
        meta = {
            "num_intents": len(TASK_NAMES),
            "num_objects": len(EQUIPMENT_NAMES),
            "num_sources": len(TERMINAL_NAMES),
            "state_dim": int(np.asarray(sample0["initial_state"]).shape[0]),
            "context_dim": int(np.asarray(sample0["context"]).shape[0]),
            "history_feature_dim": int(np.asarray(sample0["history_seq"]).shape[-1]),
            "sequence_length": int(np.asarray(sample0["actual_traj"]).shape[0]),
            "num_modes": 4,
        }

    meta = {
        "num_intents": int(meta["num_intents"]),
        "num_objects": int(meta["num_objects"]),
        "num_sources": int(meta["num_sources"]),
        "state_dim": int(meta["state_dim"]),
        "context_dim": int(meta["context_dim"]),
        "history_feature_dim": int(meta["history_feature_dim"]),
        "sequence_length": int(meta["sequence_length"]),
        "num_modes": int(meta["num_modes"]),
        "history_length": int(meta.get("history_length", 12)),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
    }
    train_samples = [_normalize_loaded_sample(sample, meta) for sample in train_samples]
    val_samples = [_normalize_loaded_sample(sample, meta) for sample in val_samples]
    test_samples = [_normalize_loaded_sample(sample, meta) for sample in test_samples]
    return train_samples, val_samples, test_samples, meta


def build_dataloaders(config: Dict, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:

    cfg = SyntheticConfig.from_dict(config["data"])

    if cfg.source in {"station_jsonl", "mock_station_jsonl"}:
        if cfg.data_dir is None:
            raise ValueError("data.data_dir must be provided when data.source uses station_jsonl.")
        train_samples, val_samples, test_samples, meta = load_station_dataset(cfg.data_dir)
    else:
        train_samples = _make_samples(cfg.train_samples, cfg, seed + 11, allow_anomaly=not cfg.train_normal_only)
        if cfg.train_normal_only:
            train_samples = [sample for sample in train_samples if sample["anomaly_label"] == 0]

        while len(train_samples) < max(64, cfg.batch_size):
            extra = _make_samples(cfg.train_samples // 2, cfg, seed + len(train_samples), allow_anomaly=False)
            train_samples.extend([sample for sample in extra if sample["anomaly_label"] == 0])

        val_samples = _make_samples(cfg.val_samples, cfg, seed + 23, allow_anomaly=True)
        test_samples = _make_samples(cfg.test_samples, cfg, seed + 37, allow_anomaly=True)
        meta = {
            "num_intents": cfg.num_intents,
            "num_objects": cfg.num_objects,
            "num_sources": cfg.num_sources,
            "state_dim": cfg.state_dim,
            "context_dim": cfg.context_dim,
            "history_feature_dim": cfg.history_feature_dim,
            "sequence_length": cfg.sequence_length,
            "num_modes": cfg.num_modes,
            "history_length": cfg.history_length,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "test_size": len(test_samples),
        }

    train_dataset = SmartGridDataset(train_samples)
    val_dataset = SmartGridDataset(val_samples)
    test_dataset = SmartGridDataset(test_samples)

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, meta
