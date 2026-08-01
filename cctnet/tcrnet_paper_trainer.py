from __future__ import annotations

import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import optim
from tqdm import tqdm

from .synthetic_data import build_dataloaders
from .tcrnet_paper import TCRNetPaper


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: Dict) -> torch.device:
    name = config.get("device", "auto")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def infer_three_level_labels(
    anomaly_label: torch.Tensor,
    violations: torch.Tensor,
    size_violation_gate: float,
) -> torch.Tensor:
    hard_rule = (violations[:, :4].sum(dim=1) > 0.0) | (violations[:, 4] > size_violation_gate)
    y = torch.zeros_like(anomaly_label, dtype=torch.long)
    anomaly = anomaly_label > 0
    y[anomaly] = 1
    y[anomaly & hard_rule] = 2
    return y


def get_supervised_labels(
    batch: Dict[str, torch.Tensor],
    violations: torch.Tensor,
    size_violation_gate: float,
) -> torch.Tensor:
    if "risk_label" in batch:
        return batch["risk_label"].long().clamp(min=0, max=2)
    return infer_three_level_labels(batch["anomaly_label"], violations, size_violation_gate)


def _compute_loss(
    model: TCRNetPaper,
    outputs: Dict[str, torch.Tensor],
    intent_id: torch.Tensor,
    y3: torch.Tensor,
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = config["paper_loss"]
    l_pred = torch.abs(outputs["r_obs"] - outputs["r_hat"]).mean()
    l_proto = outputs["d_proto"].mean()
    mu = model.seq_mu_bank[intent_id.long()]
    l_seq = ((outputs["h_seq"] - mu) ** 2).mean()

    class_weights = torch.as_tensor(loss_cfg.get("class_weights", [1.0, 1.2, 1.5]), device=y3.device, dtype=torch.float32)
    l_cls = torch.nn.functional.cross_entropy(outputs["logits"], y3, weight=class_weights)

    l_reg = torch.zeros((), device=y3.device)
    for p in model.parameters():
        l_reg = l_reg + (p ** 2).sum()

    total = (
        float(loss_cfg["lambda_pred"]) * l_pred
        + float(loss_cfg["lambda_proto"]) * l_proto
        + float(loss_cfg["lambda_seq"]) * l_seq
        + float(loss_cfg["lambda_cls"]) * l_cls
        + float(loss_cfg["lambda_reg"]) * l_reg
    )
    logs = {
        "loss": float(total.item()),
        "l_pred": float(l_pred.item()),
        "l_proto": float(l_proto.item()),
        "l_seq": float(l_seq.item()),
        "l_cls": float(l_cls.item()),
    }
    return total, logs


def _fit_norm_stats(values: Dict[str, np.ndarray], eps: float) -> Dict[str, Dict[str, float]]:
    stats = {}
    for key, arr in values.items():
        stats[key] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "eps": float(eps),
        }
    return stats


def _normalize(arr: np.ndarray, stat: Dict[str, float]) -> np.ndarray:
    return (arr - stat["min"]) / max(stat["max"] - stat["min"], stat["eps"])


def _compute_total_scores(
    model: TCRNetPaper,
    r_cons: np.ndarray,
    r_seq: np.ndarray,
    r_rule: np.ndarray,
    stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    cons_n = _normalize(r_cons, stats["R_cons"])
    seq_n = _normalize(r_seq, stats["R_seq"])
    rule_n = _normalize(r_rule, stats["R_rule"])
    beta = torch.softmax(model.beta_logits.detach().cpu(), dim=0).numpy()
    return beta[0] * cons_n + beta[1] * seq_n + beta[2] * rule_n


def _normalize_with_train(arr: np.ndarray, train_arr: np.ndarray, eps: float) -> np.ndarray:
    return (arr - float(np.min(train_arr))) / max(float(np.max(train_arr) - np.min(train_arr)), eps)


def _logit_risk_score(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-12, None)
    levels = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    return probs @ levels


def _calibrated_component_score(
    payload: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    weights: Tuple[float, float, float],
) -> np.ndarray:
    cons_n = _normalize(payload["R_cons"], stats["R_cons"])
    seq_n = _normalize(payload["R_seq"], stats["R_seq"])
    rule_n = _normalize(payload["R_rule"], stats["R_rule"])
    return float(weights[0]) * cons_n + float(weights[1]) * seq_n + float(weights[2]) * rule_n


def _predict_from_score(scores: np.ndarray, eta_1: float, eta_2: float) -> np.ndarray:
    y = np.zeros(scores.shape[0], dtype=np.int64)
    y[scores >= eta_1] = 1
    y[scores >= eta_2] = 2
    return y


def _select_decision_calibration(
    val_payload: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    config: Dict,
) -> Dict[str, object]:
    threshold_cfg = config.get("paper_score", {}).get("thresholds", {})
    q1 = threshold_cfg.get("eta1_quantiles", [0.55, 0.60, 0.65, 0.70, 0.75])
    q2 = threshold_cfg.get("eta2_quantiles", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    weight_grid = config.get("paper_score", {}).get(
        "component_weight_grid",
        [[0.34, 0.33, 0.33], [0.50, 0.25, 0.25], [0.25, 0.50, 0.25], [0.25, 0.25, 0.50], [0.40, 0.30, 0.30], [0.30, 0.40, 0.30], [0.30, 0.30, 0.40]],
    )
    mix_grid = config.get("paper_score", {}).get("hybrid_mix_grid", [0.0, 0.25, 0.50, 0.75, 1.0])
    logits_score = _logit_risk_score(val_payload["logits"])
    y3 = val_payload["y3"]

    best = {
        "macro_f1": -1.0,
        "high_recall": -1.0,
        "mix_logit": 0.0,
        "component_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        "eta_1": 0.0,
        "eta_2": 1.0,
    }
    for raw_weights in weight_grid:
        weight_sum = max(float(sum(raw_weights)), 1.0e-12)
        weights = tuple(float(w) / weight_sum for w in raw_weights)
        component_score = _calibrated_component_score(val_payload, stats, weights)
        for mix_logit in mix_grid:
            mix_logit = float(mix_logit)
            score = mix_logit * logits_score + (1.0 - mix_logit) * component_score
            cand_1 = [float(np.quantile(score, float(q))) for q in q1]
            cand_2 = [float(np.quantile(score, float(q))) for q in q2]
            for eta_1 in cand_1:
                for eta_2 in cand_2:
                    if eta_2 <= eta_1:
                        continue
                    pred = _predict_from_score(score, eta_1, eta_2)
                    macro = float(f1_score(y3, pred, average="macro", zero_division=0))
                    _, recall, _, _ = precision_recall_fscore_support(y3, pred, labels=[0, 1, 2], zero_division=0)
                    high_recall = float(recall[2])
                    if macro > best["macro_f1"] or (abs(macro - best["macro_f1"]) <= 1.0e-9 and high_recall > best["high_recall"]):
                        best = {
                            "macro_f1": macro,
                            "high_recall": high_recall,
                            "mix_logit": mix_logit,
                            "component_weights": list(weights),
                            "eta_1": float(eta_1),
                            "eta_2": float(eta_2),
                        }
    return best


def _apply_decision_calibration(
    payload: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    calibration: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    weights = tuple(float(x) for x in calibration["component_weights"])
    component_score = _calibrated_component_score(payload, stats, weights)
    logits_score = _logit_risk_score(payload["logits"])
    score = float(calibration["mix_logit"]) * logits_score + (1.0 - float(calibration["mix_logit"])) * component_score
    pred = _predict_from_score(score, float(calibration["eta_1"]), float(calibration["eta_2"]))
    return score, pred


def _select_eta(total_scores: np.ndarray, y3: np.ndarray, config: Dict) -> Dict[str, float]:
    threshold_cfg = config.get("paper_score", {}).get("thresholds", {})
    mode = str(threshold_cfg.get("mode", "grid_search")).lower()
    if mode == "manual":
        eta_1 = float(threshold_cfg["eta_1"])
        eta_2 = float(threshold_cfg["eta_2"])
        if eta_2 <= eta_1:
            eta_2 = eta_1 + 1.0e-6
        return {"eta_1": eta_1, "eta_2": eta_2}


    q1 = threshold_cfg.get("eta1_quantiles", [0.55, 0.60, 0.65, 0.70, 0.75])
    q2 = threshold_cfg.get("eta2_quantiles", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    cand_1 = [float(np.quantile(total_scores, float(q))) for q in q1]
    cand_2 = [float(np.quantile(total_scores, float(q))) for q in q2]
    best = {"eta_1": float(np.quantile(total_scores, 0.65)), "eta_2": float(np.quantile(total_scores, 0.85))}
    best_score = -1.0
    best_high_recall = -1.0
    for eta_1 in cand_1:
        for eta_2 in cand_2:
            if eta_2 <= eta_1:
                continue
            pred = _predict_level(total_scores, {"eta_1": eta_1, "eta_2": eta_2})
            macro = float(f1_score(y3, pred, average="macro", zero_division=0))
            _, recall, _, _ = precision_recall_fscore_support(y3, pred, labels=[0, 1, 2], zero_division=0)
            high_recall = float(recall[2])
            if macro > best_score or (abs(macro - best_score) <= 1.0e-9 and high_recall > best_high_recall):
                best_score = macro
                best_high_recall = high_recall
                best = {"eta_1": float(eta_1), "eta_2": float(eta_2)}
    return best


def _predict_level(total_scores: np.ndarray, eta: Dict[str, float]) -> np.ndarray:
    y = np.zeros(total_scores.shape[0], dtype=np.int64)
    y[total_scores >= eta["eta_1"]] = 1
    y[total_scores >= eta["eta_2"]] = 2
    return y


def _collect_split_outputs(
    model: TCRNetPaper,
    loader,
    device: torch.device,
    config: Dict,
) -> Dict[str, np.ndarray]:
    model.eval()
    gate = float(config["paper_rules"].get("size_violation_gate", 0.20))
    out = {
        "R_cons": [],
        "R_seq": [],
        "R_rule": [],
        "anomaly_label": [],
        "anomaly_type": [],
        "y3": [],
        "intent_id": [],
        "object_type": [],
        "source_type": [],
        "v_type": [],
        "v_dst": [],
        "v_role": [],
        "v_time": [],
        "v_size": [],
        "logits": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch)
            y3 = get_supervised_labels(batch, outputs["violations"], gate)
            out["R_cons"].append(outputs["R_cons"].detach().cpu().numpy())
            out["R_seq"].append(outputs["R_seq"].detach().cpu().numpy())
            out["R_rule"].append(outputs["R_rule"].detach().cpu().numpy())
            out["anomaly_label"].append(batch["anomaly_label"].detach().cpu().numpy())
            if "anomaly_type" in batch:
                out["anomaly_type"].append(batch["anomaly_type"].detach().cpu().numpy())
            else:
                out["anomaly_type"].append(np.zeros_like(batch["anomaly_label"].detach().cpu().numpy()))
            out["y3"].append(y3.detach().cpu().numpy())
            out["intent_id"].append(batch["intent_id"].detach().cpu().numpy())
            out["object_type"].append(batch["object_type"].detach().cpu().numpy())
            out["source_type"].append(batch["source_type"].detach().cpu().numpy())
            v = outputs["violations"].detach().cpu().numpy()
            out["v_type"].append(v[:, 0])
            out["v_dst"].append(v[:, 1])
            out["v_role"].append(v[:, 2])
            out["v_time"].append(v[:, 3])
            out["v_size"].append(v[:, 4])
            out["logits"].append(outputs["logits"].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _summarize(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "low_precision": float(p[0]),
        "low_recall": float(r[0]),
        "mid_precision": float(p[1]),
        "mid_recall": float(r[1]),
        "high_precision": float(p[2]),
        "high_recall": float(r[2]),
        "high_f1": float(f[2]),
    }


def _export_details(
    split_name: str,
    out_dir: Path,
    payload: Dict[str, np.ndarray],
    total_scores: np.ndarray,
    y_pred: np.ndarray,
    eta: Dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(total_scores)
    for i in range(n):
        rows.append(
            {
                "sample_id": i,
                "intent_id": int(payload["intent_id"][i]),
                "object_type": int(payload["object_type"][i]),
                "source_type": int(payload["source_type"][i]),
                "anomaly_label": int(payload["anomaly_label"][i]),
                "anomaly_type": int(payload["anomaly_type"][i]),
                "risk_label_true": int(payload["y3"][i]),
                "risk_label_pred": int(y_pred[i]),
                "R_cons": float(payload["R_cons"][i]),
                "R_seq": float(payload["R_seq"][i]),
                "R_rule": float(payload["R_rule"][i]),
                "R_total": float(total_scores[i]),
                "logit_low": float(payload["logits"][i, 0]),
                "logit_mid": float(payload["logits"][i, 1]),
                "logit_high": float(payload["logits"][i, 2]),
                "V_type": float(payload["v_type"][i]),
                "V_dst": float(payload["v_dst"][i]),
                "V_role": float(payload["v_role"][i]),
                "V_time": float(payload["v_time"][i]),
                "V_size": float(payload["v_size"][i]),
                "eta_1": float(eta["eta_1"]),
                "eta_2": float(eta["eta_2"]),
            }
        )
    csv_path = out_dir / f"{split_name}_details.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_tcrnet_paper(config: Dict) -> Dict:
    started_at = time.time()
    seed_everything(int(config["seed"]))
    device = resolve_device(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, meta = build_dataloaders(config, int(config["seed"]))
    model = TCRNetPaper(config, meta).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["paper_train"]["lr"]),
        weight_decay=float(config["paper_train"]["weight_decay"]),
    )
    epochs = int(config["paper_train"]["epochs"])
    momentum = float(config["paper_train"].get("prototype_momentum", 0.95))
    gate = float(config["paper_rules"].get("size_violation_gate", 0.20))

    best_macro = -1.0
    best_path = output_dir / "best_tcr.pt"
    history = []
    patience = int(config["paper_train"].get("early_stop_patience", 0) or 0)
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "l_pred": 0.0, "l_proto": 0.0, "l_seq": 0.0, "l_cls": 0.0}
        pbar = tqdm(train_loader, desc=f"TCR Epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch(batch, device)
            outputs = model(batch)
            y3 = get_supervised_labels(batch, outputs["violations"], gate)
            loss, logs = _compute_loss(model, outputs, batch["intent_id"], y3, config)


            with torch.no_grad():
                normal_mask = y3 == 0
                model.update_reference_banks(
                    z=outputs["z"],
                    h_seq=outputs["h_seq"],
                    intent_id=batch["intent_id"],
                    normal_mask=normal_mask,
                    momentum=momentum,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["paper_train"]["grad_clip_norm"]))
            optimizer.step()

            for key in running:
                running[key] += logs[key]
            if step % int(config["paper_train"]["log_interval"]) == 0 or step == len(train_loader):
                pbar.set_postfix(
                    {
                        "loss": f"{running['loss'] / step:.4f}",
                        "cls": f"{running['l_cls'] / step:.4f}",
                        "proto": f"{running['l_proto'] / step:.4f}",
                    }
                )

        train_out = _collect_split_outputs(model, train_loader, device, config)
        score_stats = _fit_norm_stats(
            {
                "R_cons": train_out["R_cons"],
                "R_seq": train_out["R_seq"],
                "R_rule": train_out["R_rule"],
            },
            eps=float(config["paper_score"]["stats_epsilon"]),
        )

        val_out = _collect_split_outputs(model, val_loader, device, config)
        decision_calibration = _select_decision_calibration(val_out, score_stats, config)
        val_total, val_pred = _apply_decision_calibration(val_out, score_stats, decision_calibration)
        val_metrics = _summarize(val_out["y3"], val_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running["loss"] / max(len(train_loader), 1),
                "val_macro_f1": val_metrics["macro_f1"],
                "eta_1": float(decision_calibration["eta_1"]),
                "eta_2": float(decision_calibration["eta_2"]),
                "mix_logit": float(decision_calibration["mix_logit"]),
                "component_weights": decision_calibration["component_weights"],
            }
        )

        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            stale_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "meta": meta,
                "score_stats": score_stats,
                "thresholds": {"eta_1": float(decision_calibration["eta_1"]), "eta_2": float(decision_calibration["eta_2"])},
                "decision_calibration": decision_calibration,
                "history": history,
            }
            torch.save(checkpoint, best_path)
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                break

    results = evaluate_tcrnet_paper(config, str(best_path), external_loaders=(train_loader, val_loader, test_loader, meta))
    results["runtime"] = {
        "seconds": float(time.time() - started_at),
        "epochs_ran": int(history[-1]["epoch"] if history else 0),
        "early_stop_patience": patience,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "torch": str(torch.__version__),
    }
    with (output_dir / "metrics_tcr.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def evaluate_tcrnet_paper(
    config: Dict,
    checkpoint_path: str,
    external_loaders=None,
) -> Dict:
    seed = int(config["seed"])
    device = resolve_device(config)
    if external_loaders is None:
        train_loader, val_loader, test_loader, meta = build_dataloaders(config, seed)
    else:
        train_loader, val_loader, test_loader, meta = external_loaders

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TCRNetPaper(config, meta).to(device)
    model.load_state_dict(checkpoint["model_state"])
    score_stats = checkpoint["score_stats"]
    decision_calibration = checkpoint.get(
        "decision_calibration",
        {
            "mix_logit": 0.0,
            "component_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "eta_1": float(checkpoint["thresholds"]["eta_1"]),
            "eta_2": float(checkpoint["thresholds"]["eta_2"]),
        },
    )
    eta = {"eta_1": float(decision_calibration["eta_1"]), "eta_2": float(decision_calibration["eta_2"])}

    split_results = {}
    details_dir = Path(config["output_dir"]) / "details_tcr"
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        payload = _collect_split_outputs(model, loader, device, config)
        total, pred = _apply_decision_calibration(payload, score_stats, decision_calibration)
        metrics = _summarize(payload["y3"], pred)
        metrics["avg_R_cons"] = float(payload["R_cons"].mean())
        metrics["avg_R_seq"] = float(payload["R_seq"].mean())
        metrics["avg_R_rule"] = float(payload["R_rule"].mean())
        metrics["avg_R_total"] = float(total.mean())
        metrics["thresholds"] = {"eta_1": float(eta["eta_1"]), "eta_2": float(eta["eta_2"])}
        metrics["decision_calibration"] = {
            "mix_logit": float(decision_calibration["mix_logit"]),
            "component_weights": [float(x) for x in decision_calibration["component_weights"]],
        }
        split_results[split_name] = metrics
        _export_details(split_name, details_dir, payload, total, pred, eta)

    split_results["checkpoint"] = checkpoint_path
    split_results["thresholds"] = {"eta_1": float(eta["eta_1"]), "eta_2": float(eta["eta_2"])}
    split_results["decision_calibration"] = {
        "mix_logit": float(decision_calibration["mix_logit"]),
        "component_weights": [float(x) for x in decision_calibration["component_weights"]],
    }
    split_results["score_stats"] = score_stats
    return split_results
