"""
TCR-Net trainer: training loop, threshold calibration, evaluation pipeline.

Flow:
    1. Each epoch: train one step -> update reference banks
    2. Validation set: grid search for eta1/eta2 thresholds
    3. Save best checkpoint (by Macro-F1)
    4. Final evaluation on test split with CSV details.

Threshold calibration:
    eta1, eta2 are determined via quantile grid search on the validation set.
    Search space is strictly limited to validation; test set never participates.
"""

from __future__ import annotations

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

from ..data.synthetic import build_dataloaders
from ..models.tcrnet import TCRNet


# -- Environment setup -----------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: Dict) -> torch.device:
    name = config.get("device", "auto")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


# -- Label generation ------------------------------------------------

def get_supervised_labels(
    batch: Dict[str, torch.Tensor],
    violations: torch.Tensor,
    size_violation_gate: float,
) -> torch.Tensor:
    if "risk_label" in batch:
        return batch["risk_label"].long().clamp(min=0, max=2)
    hard_rule = (violations[:, :4].sum(dim=1) > 0.0) | (violations[:, 4] > size_violation_gate)
    y = torch.zeros_like(batch["anomaly_label"], dtype=torch.long)
    anomaly = batch["anomaly_label"] > 0
    y[anomaly] = 1
    y[anomaly & hard_rule] = 2
    return y


# -- Loss function ---------------------------------------------------

def _compute_loss(
    model: TCRNet, outputs: Dict[str, torch.Tensor],
    intent_id: torch.Tensor, y3: torch.Tensor, config: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    lc = config["paper_loss"]
    l_pred  = torch.abs(outputs["r_obs"] - outputs["r_hat"]).mean()

    # Only constrain normal samples with L_seq,
    # allowing abnormal samples to drift away from class centers
    # so that R_seq can distinguish them at inference.
    normal_mask = (y3 == 0)
    mu = model.seq_mu_bank[intent_id.long()]
    l_proto = torch.tensor(0.0, device=y3.device)
    l_seq = ((outputs["h_seq"] - mu) ** 2)[normal_mask].mean() if normal_mask.any() else torch.tensor(0.0, device=y3.device)

    weights = torch.as_tensor(lc.get("class_weights", [1.0, 1.2, 1.5]),
                               device=y3.device, dtype=torch.float32)
    l_cls = torch.nn.functional.cross_entropy(outputs["logits"], y3, weight=weights)
    l_reg = sum((p ** 2).sum() for p in model.parameters())

    total = (float(lc["lambda_pred"])  * l_pred
           + float(lc["lambda_seq"])   * l_seq
           + float(lc["lambda_cls"])   * l_cls
           + float(lc["lambda_reg"])   * l_reg)
    return total, {"loss": float(total.item()), "l_pred": float(l_pred.item()),
                   "l_proto": float(l_proto.item()), "l_seq": float(l_seq.item()),
                   "l_cls": float(l_cls.item())}


# -- Calibration and decision ----------------------------------------

def _fit_norm_stats(values: Dict[str, np.ndarray], eps: float) -> Dict[str, Dict[str, float]]:
    return {k: {"min": float(np.min(v)), "max": float(np.max(v)), "eps": eps}
            for k, v in values.items()}


def _normalize(arr: np.ndarray, stat: Dict[str, float]) -> np.ndarray:
    return (arr - stat["min"]) / max(stat["max"] - stat["min"], stat["eps"])


def _logit_risk_score(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.clip(np.exp(logits).sum(axis=1, keepdims=True), 1e-12, None)
    return probs @ np.asarray([0.0, 0.5, 1.0], dtype=np.float32)


def _predict_from_score(scores: np.ndarray, eta1: float, eta2: float) -> np.ndarray:
    y = np.zeros(scores.shape[0], dtype=np.int64)
    y[scores >= eta1] = 1
    y[scores >= eta2] = 2
    return y


def _select_decision_calibration(
    val_payload: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    config: Dict,
) -> Dict:
    tc = config.get("paper_score", {}).get("thresholds", {})
    q1 = tc.get("eta1_quantiles", [0.55, 0.60, 0.65, 0.70, 0.75])
    q2 = tc.get("eta2_quantiles", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    wgrid = config.get("paper_score", {}).get("component_weight_grid", [
        [0.34, 0.33, 0.33], [0.50, 0.25, 0.25], [0.25, 0.50, 0.25],
        [0.25, 0.25, 0.50], [0.40, 0.30, 0.30], [0.30, 0.40, 0.30], [0.30, 0.30, 0.40],
    ])
    mgrid = [0.5]
    logits_score = _logit_risk_score(val_payload["logits"])
    y3 = val_payload["y3"]

    best = {"macro_f1": -1.0, "high_recall": -1.0, "mix_logit": 0.0,
            "component_weights": [1/3, 1/3, 1/3], "eta_1": 0.0, "eta_2": 1.0}
    for rw in wgrid:
        ws = tuple(float(w) / max(float(sum(rw)), 1e-12) for w in rw)
        comp_score = (
            ws[0] * _normalize(val_payload["R_cons"], stats["R_cons"])
          + ws[1] * _normalize(val_payload["R_seq"],  stats["R_seq"])
          + ws[2] * _normalize(val_payload["R_rule"], stats["R_rule"]))
        for mix in mgrid:
            score = float(mix) * logits_score + (1 - float(mix)) * comp_score
            for eta1 in [float(np.quantile(score, q)) for q in q1]:
                for eta2 in [float(np.quantile(score, q)) for q in q2]:
                    if eta2 <= eta1:
                        continue
                    pred = _predict_from_score(score, eta1, eta2)
                    macro = float(f1_score(y3, pred, average="macro", zero_division=0))
                    hr = float(precision_recall_fscore_support(
                        y3, pred, labels=[0, 1, 2], zero_division=0)[2][2])
                    if macro > best["macro_f1"] or (abs(macro - best["macro_f1"]) <= 1e-9 and hr > best["high_recall"]):
                        best.update({"macro_f1": macro, "high_recall": hr, "mix_logit": mix,
                                      "component_weights": list(ws), "eta_1": eta1, "eta_2": eta2})
    return best


def _apply_decision_calibration(
    payload: Dict[str, np.ndarray], stats, calib,
) -> Tuple[np.ndarray, np.ndarray]:
    ws = tuple(float(x) for x in calib["component_weights"])
    comp = (ws[0] * _normalize(payload["R_cons"], stats["R_cons"])
          + ws[1] * _normalize(payload["R_seq"],  stats["R_seq"])
          + ws[2] * _normalize(payload["R_rule"], stats["R_rule"]))
    ls = _logit_risk_score(payload["logits"])
    score = float(calib["mix_logit"]) * ls + (1 - float(calib["mix_logit"])) * comp
    return score, _predict_from_score(score, float(calib["eta_1"]), float(calib["eta_2"]))


# -- Data collection -------------------------------------------------

def _collect_split_outputs(
    model: TCRNet, loader, device: torch.device, config: Dict,
) -> Dict[str, np.ndarray]:
    model.eval()
    gate = float(config["paper_rules"].get("size_violation_gate", 0.20))
    out = {k: [] for k in ["R_cons", "R_seq", "R_rule", "anomaly_label", "anomaly_type",
                            "y3", "intent_id", "object_type", "source_type",
                            "v_type", "v_dst", "v_role", "v_time", "v_size", "logits"]}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            o = model(batch)
            y3 = get_supervised_labels(batch, o["violations"], gate)
            out["R_cons"].append(o["R_cons"].cpu().numpy())
            out["R_seq"].append(o["R_seq"].cpu().numpy())
            out["R_rule"].append(o["R_rule"].cpu().numpy())
            out["anomaly_label"].append(batch["anomaly_label"].cpu().numpy())
            out["anomaly_type"].append(batch.get("anomaly_type",
                torch.zeros_like(batch["anomaly_label"])).cpu().numpy())
            out["y3"].append(y3.cpu().numpy())
            out["intent_id"].append(batch["intent_id"].cpu().numpy())
            out["object_type"].append(batch["object_type"].cpu().numpy())
            out["source_type"].append(batch["source_type"].cpu().numpy())
            v = o["violations"].cpu().numpy()
            for k_, i_ in [("v_type",0),("v_dst",1),("v_role",2),("v_time",3),("v_size",4)]:
                out[k_].append(v[:, i_])
            out["logits"].append(o["logits"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _summarize(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    wf1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {"macro_f1": macro, "weighted_f1": wf1,
            "low_precision": float(p[0]), "low_recall": float(r[0]),
            "mid_precision": float(p[1]), "mid_recall": float(r[1]),
            "high_precision": float(p[2]), "high_recall": float(r[2]), "high_f1": float(f[2])}


def _export_details(split, out_dir, payload, total_scores, y_pred, eta):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(total_scores)):
        rows.append({"sample_id": i,
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
            "eta_1": float(eta["eta_1"]), "eta_2": float(eta["eta_2"]),
        })
    import csv
    with (out_dir / f"{split}_details.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# -- Training entry point --------------------------------------------

def train(config: Dict) -> Dict:
    started = time.time()
    seed_everything(int(config["seed"]))
    device = resolve_device(config)
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_l, val_l, test_l, meta = build_dataloaders(config, int(config["seed"]))
    model = TCRNet(config, meta).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(config["paper_train"]["lr"]),
                      weight_decay=float(config["paper_train"]["weight_decay"]))
    epochs = int(config["paper_train"]["epochs"])
    momentum = float(config["paper_train"].get("prototype_momentum", 0.95))
    gate = float(config["paper_rules"].get("size_violation_gate", 0.20))

    best_macro = -1.0
    best_path = out_dir / "best.pt"
    history = []
    patience = int(config["paper_train"].get("early_stop_patience", 0) or 0)
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "l_pred": 0.0, "l_seq": 0.0, "l_cls": 0.0}
        pbar = tqdm(train_l, desc=f"Epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, 1):
            batch = move_batch(batch, device)
            # Single forward pass: outputs used for both loss and reference bank update
            o = model(batch)
            y3 = get_supervised_labels(batch, o["violations"], gate)

            # Update reference banks BEFORE backward (using current-param outputs)
            with torch.no_grad():
                model.update_reference_banks(o["z"], o["h_seq"],
                    batch["intent_id"], normal_mask=(y3 == 0), momentum=momentum)

            loss, logs = _compute_loss(model, o, batch["intent_id"], y3, config)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(config["paper_train"]["grad_clip_norm"]))
            opt.step()

            for k in running: running[k] += logs[k]
            if step % int(config["paper_train"]["log_interval"]) == 0 or step == len(train_l):
                pbar.set_postfix({"loss": f"{running['loss']/step:.4f}"})

        # Validation
        train_out = _collect_split_outputs(model, train_l, device, config)
        stats = _fit_norm_stats({"R_cons": train_out["R_cons"], "R_seq": train_out["R_seq"],
                                  "R_rule": train_out["R_rule"]},
                                 eps=float(config["paper_score"]["stats_epsilon"]))
        val_out = _collect_split_outputs(model, val_l, device, config)
        calib = _select_decision_calibration(val_out, stats, config)
        val_score, val_pred = _apply_decision_calibration(val_out, stats, calib)
        vm = _summarize(val_out["y3"], val_pred)
        history.append({"epoch": epoch, "train_loss": running["loss"] / max(len(train_l), 1),
                        "val_macro_f1": vm["macro_f1"], "eta_1": calib["eta_1"],
                        "eta_2": calib["eta_2"], "mix_logit": calib["mix_logit"],
                        "component_weights": calib["component_weights"]})

        if vm["macro_f1"] > best_macro:
            best_macro = vm["macro_f1"]; stale = 0
            torch.save({"model_state": model.state_dict(), "config": config, "meta": meta,
                        "score_stats": stats, "decision_calibration": calib, "history": history,
                        "thresholds": {"eta_1": calib["eta_1"], "eta_2": calib["eta_2"]}}, best_path)
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break

    # Final evaluation
    results = evaluate(config, str(best_path),
                       external_loaders=(train_l, val_l, test_l, meta))
    results["runtime"] = {"seconds": float(time.time() - started),
                          "epochs_ran": int(history[-1]["epoch"] if history else 0),
                          "python": platform.python_version(), "device": str(device)}
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def evaluate(
    config: Dict, checkpoint_path: str, external_loaders=None,
) -> Dict:
    seed_everything(int(config["seed"]))
    device = resolve_device(config)
    if external_loaders is None:
        train_l, val_l, test_l, meta = build_dataloaders(config, int(config["seed"]))
    else:
        train_l, val_l, test_l, meta = external_loaders

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TCRNet(config, meta).to(device)
    model.load_state_dict(ckpt["model_state"])
    stats = ckpt["score_stats"]
    calib = ckpt.get("decision_calibration",
        {"mix_logit": 0.0, "component_weights": [1/3, 1/3, 1/3],
         "eta_1": float(ckpt["thresholds"]["eta_1"]),
         "eta_2": float(ckpt["thresholds"]["eta_2"])})
    eta = {"eta_1": float(calib["eta_1"]), "eta_2": float(calib["eta_2"])}

    results = {}
    details_dir = Path(config["output_dir"]) / "details"
    for split, loader in [("train", train_l), ("val", val_l), ("test", test_l)]:
        payload = _collect_split_outputs(model, loader, device, config)
        score, pred = _apply_decision_calibration(payload, stats, calib)
        m = _summarize(payload["y3"], pred)
        m.update({"avg_R_cons": float(payload["R_cons"].mean()),
                   "avg_R_seq": float(payload["R_seq"].mean()),
                   "avg_R_rule": float(payload["R_rule"].mean()),
                   "avg_R_total": float(score.mean()),
                   "thresholds": {"eta_1": eta["eta_1"], "eta_2": eta["eta_2"]}})
        results[split] = m
        _export_details(split, details_dir, payload, score, pred, eta)

    results["thresholds"] = {"eta_1": eta["eta_1"], "eta_2": eta["eta_2"]}
    results["decision_calibration"] = {"mix_logit": float(calib["mix_logit"]),
        "component_weights": [float(x) for x in calib["component_weights"]]}
    return results
