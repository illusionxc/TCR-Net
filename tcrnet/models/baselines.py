


from __future__ import annotations

import copy
import math
import random
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def _typed_rule_features(
    batch: Dict[str, torch.Tensor], config: Dict,
) -> torch.Tensor:


    rules = config.get("paper_rules", {})
    num_intents = int(config.get("data", {}).get("num_intents", 6))
    num_objects = int(config.get("data", {}).get("num_objects", 4))
    num_sources = int(config.get("data", {}).get("num_sources", 3))
    device = batch["intent_id"].device

    def allowed_matrix(rows, width):
        matrix = torch.zeros(num_intents, width, device=device)
        for task_index in range(num_intents):
            values = rows[task_index] if task_index < len(rows) else []
            for value in values:
                if 0 <= int(value) < width:
                    matrix[task_index, int(value)] = 1.0
        return matrix

    task = batch["intent_id"].long().clamp(0, num_intents - 1)
    allowed_types = allowed_matrix(
        rules.get("allowed_types", [[i] for i in range(num_intents)]),
        num_intents,
    )
    allowed_dst = allowed_matrix(
        rules.get(
            "allowed_dst",
            [[i % max(num_objects, 1)] for i in range(num_intents)],
        ),
        num_objects,
    )
    allowed_role = allowed_matrix(
        rules.get(
            "allowed_role",
            [[i % max(num_sources, 1)] for i in range(num_intents)],
        ),
        num_sources,
    )

    type_id = batch.get("rule_type_id", batch["control_type"]).long()
    type_id = type_id.clamp(0, num_intents - 1)
    dst_id = batch.get("rule_dst_id", batch["object_type"]).long()
    dst_id = dst_id.clamp(0, num_objects - 1)
    role_id = batch.get("rule_role_id", batch["source_type"]).long()
    role_id = role_id.clamp(0, num_sources - 1)
    v_type = (allowed_types[task, type_id] <= 0.0).float()
    v_dst = (allowed_dst[task, dst_id] <= 0.0).float()
    v_role = (allowed_role[task, role_id] <= 0.0).float()

    windows = rules.get(
        "allowed_time_windows",
        [[0, 23] for _ in range(num_intents)],
    )
    window_tensor = torch.as_tensor(
        [
            windows[i] if i < len(windows) else [0, 23]
            for i in range(num_intents)
        ],
        dtype=torch.float32,
        device=device,
    )
    hour = batch.get("rule_hour")
    if hour is None:
        angle = torch.atan2(
            batch["time_features"][:, 0],
            batch["time_features"][:, 1],
        )
        hour = torch.remainder(angle * (12.0 / math.pi), 24.0)
    hour = hour.float()
    selected_window = window_tensor[task]
    v_time = (
        (hour < selected_window[:, 0]) | (hour > selected_window[:, 1])
    ).float()

    size_bounds = rules.get(
        "size_bounds",
        [[0.0, 9999.0] for _ in range(num_intents)],
    )
    size_tensor = torch.as_tensor(
        [
            size_bounds[i] if i < len(size_bounds) else [0.0, 9999.0]
            for i in range(num_intents)
        ],
        dtype=torch.float32,
        device=device,
    )
    stage_max = rules.get("depth_max", [3.0 for _ in range(num_intents)])
    stage_tensor = torch.as_tensor(
        [
            stage_max[i] if i < len(stage_max) else 3.0
            for i in range(num_intents)
        ],
        dtype=torch.float32,
        device=device,
    )
    size_proxy = batch.get(
        "rule_size",
        40.0
        + torch.abs(batch["control_params"][:, 0]) * 160.0
        + torch.abs(batch["control_params"][:, 1]) * 90.0
        + torch.relu(batch["initial_state"][:, 0]) * 40.0,
    ).float()
    stage_proxy = batch.get(
        "rule_depth",
        torch.round(batch["initial_state"][:, 4]),
    ).float()
    selected_bounds = size_tensor[task]
    magnitude = (
        torch.relu(size_proxy - selected_bounds[:, 1])
        + torch.relu(selected_bounds[:, 0] - size_proxy)
        + torch.relu(stage_proxy - stage_tensor[task])
    )
    magnitude = torch.clamp(
        torch.log1p(magnitude) / math.log1p(100.0),
        min=0.0,
        max=1.0,
    )
    return torch.stack(
        [v_type, v_dst, v_role, v_time, magnitude],
        dim=-1,
    )


def _flatten_batch(
    batch: Dict[str, torch.Tensor],
    config: Dict | None = None,
    include_rule: bool = False,
) -> torch.Tensor:


    B, K, Hf = batch["history_seq"].shape
    category_features = torch.stack([
        batch["intent_id"].float() / 5.0,
        batch["object_type"].float() / 3.0,
        batch["source_type"].float() / 2.0,
    ], dim=-1)
    fields = [
        batch["initial_state"].float().reshape(B, -1),
        batch["control_params"].float().reshape(B, -1),
        batch["context"].float().reshape(B, -1),
        batch["time_features"].float().reshape(B, -1),
        category_features,
        batch["history_seq"].float().reshape(B, K * Hf),
    ]
    if include_rule:
        if config is None:
            raise ValueError("include_rule=True requires a configuration")
        fields.append(_typed_rule_features(batch, config))
    return torch.cat(fields, dim=-1)


def _current_event_batch(
    batch: Dict[str, torch.Tensor],
    config: Dict | None = None,
    include_rule: bool = False,
) -> torch.Tensor:

    batch_size = batch["initial_state"].shape[0]
    category_features = torch.stack([
        batch["intent_id"].float() / 5.0,
        batch["object_type"].float() / 3.0,
        batch["source_type"].float() / 2.0,
    ], dim=-1)
    fields = [
        batch["initial_state"].float().reshape(batch_size, -1),
        batch["control_params"].float().reshape(batch_size, -1),
        batch["context"].float().reshape(batch_size, -1),
        batch["time_features"].float().reshape(batch_size, -1),
        category_features,
    ]
    if include_rule:
        if config is None:
            raise ValueError("include_rule=True requires a configuration")
        fields.append(_typed_rule_features(batch, config))
    return torch.cat(fields, dim=-1)


def _batch_labels(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return batch.get("risk_label", batch["anomaly_label"]).long().clamp(0, 2)


def _seed_baseline(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_features(loader, device, config=None, include_rule=False):

    feats, labels = [], []
    for batch in loader:
        feats.append(
            _flatten_batch(
                batch,
                config=config,
                include_rule=include_rule,
            ).cpu()
        )
        labels.append(batch.get("risk_label", batch["anomaly_label"]).cpu().numpy())
    return torch.cat(feats, dim=0), np.concatenate(labels, axis=0)


def _collect_scores(model, loader, device, score_fn):

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = _flatten_batch(batch).to(device)
            s = score_fn(model, x).cpu().numpy()
            scores.append(s)
            labels.append(batch.get("risk_label", batch["anomaly_label"]).cpu().numpy())
    return {"score": np.concatenate(scores), "y_true": np.concatenate(labels)}


def _calibrate_thresholds(val_scores, val_y, config):

    from sklearn.metrics import f1_score, precision_recall_fscore_support
    tc = config.get("paper_score", {}).get("thresholds", {})
    q1 = tc.get("eta1_quantiles", [0.55, 0.60, 0.65, 0.70, 0.75])
    q2 = tc.get("eta2_quantiles", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    best = {"eta_1": float(np.quantile(val_scores, 0.65)), "eta_2": float(np.quantile(val_scores, 0.85)),
            "macro_f1": -1.0, "high_recall": -1.0}
    for eta1 in [float(np.quantile(val_scores, q)) for q in q1]:
        for eta2 in [float(np.quantile(val_scores, q)) for q in q2]:
            if eta2 <= eta1: continue
            pred = _to_3class(val_scores, eta1, eta2)
            m = float(f1_score(val_y, pred, average="macro", zero_division=0))
            hr = float(precision_recall_fscore_support(val_y, pred, labels=[0, 1, 2], zero_division=0)[2][2])
            if m > best["macro_f1"] or (abs(m - best["macro_f1"]) <= 1e-9 and hr > best["high_recall"]):
                best = {"eta_1": eta1, "eta_2": eta2, "macro_f1": m, "high_recall": hr}
    return best["eta_1"], best["eta_2"]


def _to_3class(scores, eta1, eta2):
    y = np.zeros(scores.shape[0], dtype=np.int64)
    y[scores >= eta1] = 1
    y[scores >= eta2] = 2
    return y


def _ordered_risk_score(probabilities: np.ndarray) -> np.ndarray:

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"Expected [n, 3] class probabilities, found {values.shape}"
        )
    return values @ np.asarray([0.0, 0.5, 1.0], dtype=np.float64)


def _calibrated_ordered_predictions(
    val_probabilities: np.ndarray,
    val_y: np.ndarray,
    test_probabilities: np.ndarray,
    config: Dict,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

    val_score = _ordered_risk_score(val_probabilities)
    test_score = _ordered_risk_score(test_probabilities)
    eta_1, eta_2 = _calibrate_thresholds(val_score, val_y, config)
    calibrated_val = _to_3class(val_score, eta_1, eta_2)
    calibrated_test = _to_3class(test_score, eta_1, eta_2)
    native_val = np.asarray(val_probabilities).argmax(axis=1)
    native_test = np.asarray(test_probabilities).argmax(axis=1)
    calibrated_metrics = _summarize(val_y, calibrated_val)
    native_metrics = _summarize(val_y, native_val)
    calibrated_key = (
        calibrated_metrics["macro_f1"],
        calibrated_metrics["high_recall"],
    )
    native_key = (
        native_metrics["macro_f1"],
        native_metrics["high_recall"],
    )
    if native_key > calibrated_key:
        return (
            native_val,
            native_test,
            {
                "mode": "native_argmax",
                "eta_1": None,
                "eta_2": None,
            },
        )
    return (
        calibrated_val,
        calibrated_test,
        {
            "mode": "ordered_thresholds",
            "eta_1": float(eta_1),
            "eta_2": float(eta_2),
        },
    )


def _summarize(y_true, y_pred):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    normal_count = int(np.sum(y_true == 0))
    false_alarms = int(np.sum((y_true == 0) & (y_pred != 0)))
    far = float(false_alarms / normal_count) if normal_count else 0.0
    return {"macro_f1": m, "weighted_f1": w,
            "precision": float(np.mean(p)), "recall": float(np.mean(r)),
            "low_precision": float(p[0]), "low_recall": float(r[0]),
            "mid_precision": float(p[1]), "mid_recall": float(r[1]),
            "high_precision": float(p[2]), "high_recall": float(r[2]),
            "high_f1": float(f[2]), "far": far}


def _prediction_payload(y_val, pred_val, y_test, pred_test):

    return {
        "val": {
            "y_true": np.asarray(y_val, dtype=np.int64).tolist(),
            "y_pred": np.asarray(pred_val, dtype=np.int64).tolist(),
        },
        "test": {
            "y_true": np.asarray(y_test, dtype=np.int64).tolist(),
            "y_pred": np.asarray(pred_test, dtype=np.int64).tolist(),
        },
    }


def _run_isolation_forest(config, train_loader, val_loader, test_loader, device):
    _seed_baseline(int(config["seed"]))
    from sklearn.ensemble import IsolationForest
    X_tr, _ = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    clf = IsolationForest(n_estimators=200, contamination=0.15, random_state=int(config["seed"]), n_jobs=-1)
    clf.fit(X_tr.numpy())
    vs = -clf.score_samples(X_val.numpy())
    ts = -clf.score_samples(X_te.numpy())
    e1, e2 = _calibrate_thresholds(vs, y_val, config)
    return {"name": "IsolationForest",
            "val_metrics": _summarize(y_val, _to_3class(vs, e1, e2)),
            "test_metrics": _summarize(y_te, _to_3class(ts, e1, e2)),
            "thresholds": {"eta_1": e1, "eta_2": e2}}


class _SVDDNet(nn.Module):
    def __init__(self, in_dim, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, latent),
        )
    def forward(self, x): return self.net(x)


def _run_deep_svdd(config, train_loader, val_loader, test_loader, device):
    _seed_baseline(int(config["seed"]))
    latent = int(config.get("baseline", {}).get("svdd_latent_dim", 32))
    X_full, y_full = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)

    X_norm = X_full[y_full == 0] if (y_full == 0).sum() > 64 else X_full
    model = _SVDDNet(X_full.shape[1], latent).to(device)
    opt = optim.Adam(model.parameters(), lr=float(config.get("baseline", {}).get("svdd_lr", 1e-3)))

    with torch.no_grad():
        center = model(X_norm[:1024].to(device)).mean(dim=0).detach()
        if center.norm() < 0.1: center += 0.1

    model.train()
    loader = DataLoader(TensorDataset(X_norm), batch_size=min(256, len(X_norm)), shuffle=True)
    for _ in range(int(config.get("baseline", {}).get("svdd_epochs", 20))):
        for (xb,) in loader:
            z = model(xb.to(device))
            opt.zero_grad()
            torch.mean((z - center.unsqueeze(0)).pow(2)).backward()
            opt.step()

    def sf(m, x): return torch.sqrt(torch.sum((m(x) - center.unsqueeze(0)).pow(2), dim=1))
    vo = _collect_scores(model, val_loader, device, sf)
    to = _collect_scores(model, test_loader, device, sf)
    e1, e2 = _calibrate_thresholds(vo["score"], vo["y_true"], config)
    return {"name": "DeepSVDD",
            "val_metrics": _summarize(vo["y_true"], _to_3class(vo["score"], e1, e2)),
            "test_metrics": _summarize(to["y_true"], _to_3class(to["score"], e1, e2)),
            "thresholds": {"eta_1": e1, "eta_2": e2}}


class _DAGMM(nn.Module):
    def __init__(self, in_dim, latent=8, hidden=64, n_gmm=3):
        super().__init__()
        self.n_gmm = n_gmm
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh())
        self.mu_enc = nn.Linear(hidden // 2, latent)
        self.lv_enc = nn.Linear(hidden // 2, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, hidden), nn.Tanh(),
            nn.Linear(hidden, in_dim))
        self.est = nn.Sequential(
            nn.Linear(latent + 2, hidden), nn.Tanh(),
            nn.Dropout(0.1), nn.Linear(hidden, n_gmm), nn.Softmax(dim=1))

    def forward(self, x):
        h = self.enc(x)
        muz, lvz = self.mu_enc(h), self.lv_enc(h)
        z = muz + torch.randn_like(lvz) * torch.exp(0.5 * lvz)
        xr = self.dec(z)
        err = torch.cat([((x - xr) ** 2).sum(1, keepdim=True),
                         (x - xr).abs().sum(1, keepdim=True)], dim=1)
        gam = self.est(torch.cat([z, err], dim=1))
        phi = gam.sum(0) / gam.sum(0).sum()
        mu = (gam.unsqueeze(-1) * z.unsqueeze(1)).sum(0) / gam.sum(0).unsqueeze(-1).clamp(1e-6)
        return z, xr, gam, phi, mu


def _run_dagmm(config, train_loader, val_loader, test_loader, device):
    _seed_baseline(int(config["seed"]))
    bl = config.get("baseline", {})
    latent = int(bl.get("dagmm_latent_dim", 8))
    hidden = int(bl.get("dagmm_hidden_dim", 64))
    ngmm   = int(bl.get("dagmm_n_gmm", 3))
    X_full, _ = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)

    model = _DAGMM(X_full.shape[1], latent, hidden, ngmm).to(device)
    opt = optim.Adam(model.parameters(), lr=float(bl.get("dagmm_lr", 1e-3)))
    lam_e = float(bl.get("dagmm_lambda_energy", 0.1))
    lam_c = float(bl.get("dagmm_lambda_cov", 0.005))

    loader = DataLoader(TensorDataset(X_full), batch_size=min(256, len(X_full)), shuffle=True)
    model.train()
    for _ in range(int(bl.get("dagmm_epochs", 20))):
        for (xb,) in loader:
            xb = xb.to(device)
            z, xr, gam, phi, mu = model(xb)
            recon = torch.mean((xb - xr).pow(2).sum(1))
            ze = z.unsqueeze(1).expand(-1, ngmm, -1)
            me = mu.unsqueeze(0)
            energy = torch.mean(torch.min(-torch.log(phi.unsqueeze(0) + 1e-8)
                                          + 0.5 * (ze - me).pow(2).sum(2), dim=1)[0])
            cov_reg = lam_c * torch.sum(phi ** 2)
            opt.zero_grad()
            (recon + lam_e * energy + cov_reg).backward()
            opt.step()

    def sf(m, x):
        z, _, gam, phi, mu = m(x)
        ze = z.unsqueeze(1).expand(-1, ngmm, -1)
        me = mu.unsqueeze(0)
        return torch.min(-torch.log(phi.unsqueeze(0) + 1e-8) + 0.5 * (ze - me).pow(2).sum(2), dim=1)[0]
    vo = _collect_scores(model, val_loader, device, sf)
    to = _collect_scores(model, test_loader, device, sf)
    e1, e2 = _calibrate_thresholds(vo["score"], vo["y_true"], config)
    return {"name": "DAGMM",
            "val_metrics": _summarize(vo["y_true"], _to_3class(vo["score"], e1, e2)),
            "test_metrics": _summarize(to["y_true"], _to_3class(to["score"], e1, e2)),
            "thresholds": {"eta_1": e1, "eta_2": e2}}


class _OCNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _run_ocnn(config, train_loader, val_loader, test_loader, device):
    _seed_baseline(int(config["seed"]))
    bl = config.get("baseline", {})
    X_full, y_full = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)

    X_norm = X_full[y_full == 0]
    X_anom = X_full[~ (y_full == 0)][:len(X_norm) // 4] if (~ (y_full == 0)).sum() > 0 else X_full[:len(X_norm) // 10]
    if len(X_anom) < 8: X_anom = X_full[:min(len(X_full) // 5, 64)]

    X_all = torch.cat([X_norm, X_anom])
    y_all = torch.cat([torch.ones(len(X_norm), dtype=torch.float32),
                       torch.zeros(len(X_anom), dtype=torch.float32)])

    model = _OCNN(X_full.shape[1], int(bl.get("ocnn_hidden_dim", 64))).to(device)
    opt = optim.Adam(model.parameters(), lr=float(bl.get("ocnn_lr", 1e-3)))
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(X_all, y_all), batch_size=min(128, len(X_all)), shuffle=True)
    model.train()
    for _ in range(int(bl.get("ocnn_epochs", 20))):
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb.to(device)), yb.to(device)).backward()
            opt.step()

    def sf(m, x): return -m(x)
    vo = _collect_scores(model, val_loader, device, sf)
    to = _collect_scores(model, test_loader, device, sf)
    e1, e2 = _calibrate_thresholds(vo["score"], vo["y_true"], config)
    return {"name": "OCNN",
            "val_metrics": _summarize(vo["y_true"], _to_3class(vo["score"], e1, e2)),
            "test_metrics": _summarize(to["y_true"], _to_3class(to["score"], e1, e2)),
            "thresholds": {"eta_1": e1, "eta_2": e2}}


def _run_xgboost_common(
    config,
    train_loader,
    val_loader,
    test_loader,
    device,
    *,
    include_rule: bool,
    method_name: str,
):
    import xgboost as xgb
    _seed_baseline(int(config["seed"]))
    X_tr, y_tr = _extract_features(
        train_loader, device, config, include_rule,
    )
    X_val, y_val = _extract_features(
        val_loader, device, config, include_rule,
    )
    X_te, y_te = _extract_features(
        test_loader, device, config, include_rule,
    )
    bl = config.get("baseline", {})
    x_tr = np.ascontiguousarray(X_tr.numpy(), dtype=np.float32)
    x_val = np.ascontiguousarray(X_val.numpy(), dtype=np.float32)
    x_test = np.ascontiguousarray(X_te.numpy(), dtype=np.float32)
    clf = xgb.XGBClassifier(
        n_estimators=int(bl.get("xgb_n_estimators", 300)),
        max_depth=int(bl.get("xgb_max_depth", 6)),
        learning_rate=float(bl.get("xgb_learning_rate", 0.05)),
        subsample=float(bl.get("xgb_subsample", 0.9)),
        colsample_bytree=float(bl.get("xgb_colsample_bytree", 0.9)),
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=int(config["seed"]),
        n_jobs=int(bl.get("xgb_n_jobs", -1)),
        verbosity=0,
        use_label_encoder=False,
    )
    try:
        clf.fit(x_tr, np.asarray(y_tr, dtype=np.int64))
    except ValueError as exc:
        if "Unable to avoid copy" in str(exc):
            raise RuntimeError(
                "The current XGBoost version is incompatible with NumPy. Install xgboost>=2.1 "
                "or run in a NumPy 1.26 environment."
            ) from exc
        raise
    val_prob = clf.predict_proba(x_val)
    test_prob = clf.predict_proba(x_test)
    val_pred, test_pred, thresholds = _calibrated_ordered_predictions(
        val_prob,
        np.asarray(y_val, dtype=np.int64),
        test_prob,
        config,
    )
    return {"name": method_name,
            "val_metrics": _summarize(y_val, val_pred),
            "test_metrics": _summarize(y_te, test_pred),
            "predictions": _prediction_payload(
                y_val, val_pred, y_te, test_pred,
            ),
            "thresholds": thresholds,
            "model_config": clf.get_params()}


def _run_xgboost(config, train_loader, val_loader, test_loader, device):
    return _run_xgboost_common(
        config,
        train_loader,
        val_loader,
        test_loader,
        device,
        include_rule=False,
        method_name="XGBoost",
    )


def _run_xgboost_rule(config, train_loader, val_loader, test_loader, device):
    return _run_xgboost_common(
        config,
        train_loader,
        val_loader,
        test_loader,
        device,
        include_rule=True,
        method_name="XGBoost+Rule",
    )

def _run_random_forest(config, train_loader, val_loader, test_loader, device):
    from sklearn.ensemble import RandomForestClassifier
    _seed_baseline(int(config["seed"]))
    X_tr, y_tr = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    bl = config.get("baseline", {})
    clf = RandomForestClassifier(
        n_estimators=int(bl.get("rf_n_estimators", 300)),
        max_depth=bl.get("rf_max_depth", None),
        min_samples_leaf=int(bl.get("rf_min_samples_leaf", 1)),
        class_weight="balanced",
        random_state=int(config["seed"]),
        n_jobs=int(bl.get("rf_n_jobs", -1)),
    )
    clf.fit(np.ascontiguousarray(X_tr.numpy()), y_tr)
    val_pred = clf.predict(X_val.numpy())
    test_pred = clf.predict(X_te.numpy())
    return {"name": "RandomForest",
            "val_metrics": _summarize(y_val, val_pred),
            "test_metrics": _summarize(y_te, test_pred),
            "predictions": _prediction_payload(
                y_val, val_pred, y_te, test_pred,
            )}

def _run_svm(config, train_loader, val_loader, test_loader, device):
    from sklearn.svm import SVC
    _seed_baseline(int(config["seed"]))
    X_tr, y_tr = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    clf = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", random_state=int(config["seed"]),
        probability=False,
    )
    clf.fit(np.ascontiguousarray(X_tr.numpy()), y_tr)
    val_pred = clf.predict(X_val.numpy())
    test_pred = clf.predict(X_te.numpy())
    return {"name": "SVM",
            "val_metrics": _summarize(y_val, val_pred),
            "test_metrics": _summarize(y_te, test_pred),
            "predictions": _prediction_payload(
                y_val, val_pred, y_te, test_pred,
            )}

def _run_logistic_regression(config, train_loader, val_loader, test_loader, device):
    from sklearn.linear_model import LogisticRegression
    _seed_baseline(int(config["seed"]))
    X_tr, y_tr = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs",
        class_weight="balanced", random_state=int(config["seed"]),
        n_jobs=1,
    )
    clf.fit(np.ascontiguousarray(X_tr.numpy()), y_tr)
    val_pred = clf.predict(X_val.numpy())
    test_pred = clf.predict(X_te.numpy())
    return {"name": "LogisticRegression",
            "val_metrics": _summarize(y_val, val_pred),
            "test_metrics": _summarize(y_te, test_pred),
            "predictions": _prediction_payload(
                y_val, val_pred, y_te, test_pred,
            )}


class _GRUClassifier(nn.Module):


    def __init__(
        self, current_dim: int, history_dim: int, hidden_dim: int,
        num_layers: int, dropout: float,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.current_encoder = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.history_encoder = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self, current_features: torch.Tensor, history_seq: torch.Tensor,
    ) -> torch.Tensor:
        current_repr = self.current_encoder(current_features)
        _, hidden = self.history_encoder(history_seq)
        history_repr = hidden[-1]
        return self.classifier(
            torch.cat([current_repr, history_repr], dim=-1)
        )


def _collect_gru_probabilities(
    model,
    loader,
    device,
    config,
    include_rule,
):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            current = _current_event_batch(
                batch,
                config=config,
                include_rule=include_rule,
            ).to(device)
            history = batch["history_seq"].float().to(device)
            logits = model(current, history)
            labels.append(_batch_labels(batch).cpu().numpy())
            probabilities.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )
    return np.concatenate(labels), np.concatenate(probabilities)


def _run_gru_common(
    config,
    train_loader,
    val_loader,
    test_loader,
    device,
    *,
    include_rule: bool,
    method_name: str,
):

    seed = int(config["seed"])
    _seed_baseline(seed)
    bl = config.get("baseline", {})
    first_batch = next(iter(train_loader))
    current_dim = int(
        _current_event_batch(
            first_batch,
            config=config,
            include_rule=include_rule,
        ).shape[1]
    )
    history_dim = int(first_batch["history_seq"].shape[-1])
    hidden_dim = int(bl.get("gru_hidden_dim", 96))
    num_layers = int(bl.get("gru_num_layers", 1))
    dropout = float(bl.get("gru_dropout", 0.1))
    model = _GRUClassifier(
        current_dim, history_dim, hidden_dim, num_layers, dropout,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(bl.get("gru_lr", 8e-4)),
        weight_decay=float(bl.get("gru_weight_decay", 1e-4)),
    )
    class_weights = torch.as_tensor(
        config.get("paper_loss", {}).get(
            "class_weights", [1.0, 1.2, 1.5]
        ),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    epochs = int(bl.get("gru_epochs", config.get("paper_train", {}).get("epochs", 24)))
    patience = int(bl.get(
        "gru_early_stop_patience",
        config.get("paper_train", {}).get("early_stop_patience", 8),
    ))
    grad_clip = float(config.get("paper_train", {}).get("grad_clip_norm", 5.0))
    best_macro = -1.0
    stale = 0
    best_state = None
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        for batch in train_loader:
            current = _current_event_batch(
                batch,
                config=config,
                include_rule=include_rule,
            ).to(device)
            history = batch["history_seq"].float().to(device)
            labels = _batch_labels(batch).to(device)
            optimizer.zero_grad()
            loss = criterion(model(current, history), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        val_y, val_prob = _collect_gru_probabilities(
            model,
            val_loader,
            device,
            config,
            include_rule,
        )
        val_pred, _, _ = _calibrated_ordered_predictions(
            val_prob,
            val_y,
            val_prob,
            config,
        )
        val_macro = _summarize(val_y, val_pred)["macro_f1"]
        if val_macro > best_macro:
            best_macro = val_macro
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("The GRU baseline did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    val_y, val_prob = _collect_gru_probabilities(
        model,
        val_loader,
        device,
        config,
        include_rule,
    )
    test_y, test_prob = _collect_gru_probabilities(
        model,
        test_loader,
        device,
        config,
        include_rule,
    )
    val_pred, test_pred, thresholds = _calibrated_ordered_predictions(
        val_prob,
        val_y,
        test_prob,
        config,
    )
    return {
        "name": method_name,
        "val_metrics": _summarize(val_y, val_pred),
        "test_metrics": _summarize(test_y, test_pred),
        "predictions": _prediction_payload(
            val_y, val_pred, test_y, test_pred,
        ),
        "runtime": {"epochs_ran": epochs_ran, "device": str(device)},
        "thresholds": thresholds,
        "model_config": {
            "current_dim": current_dim,
            "history_dim": history_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "include_typed_rule_features": bool(include_rule),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
    }


def _run_gru(config, train_loader, val_loader, test_loader, device):
    return _run_gru_common(
        config,
        train_loader,
        val_loader,
        test_loader,
        device,
        include_rule=False,
        method_name="GRU",
    )


def _run_gru_rule(config, train_loader, val_loader, test_loader, device):
    return _run_gru_common(
        config,
        train_loader,
        val_loader,
        test_loader,
        device,
        include_rule=True,
        method_name="GRU+Rule",
    )


REGISTRY = {
    "IsolationForest":      _run_isolation_forest,
    "DeepSVDD":             _run_deep_svdd,
    "DAGMM":                _run_dagmm,
    "OCNN":                 _run_ocnn,
    "XGBoost":              _run_xgboost,
    "XGBoost+Rule":         _run_xgboost_rule,
    "RandomForest":         _run_random_forest,
    "SVM":                  _run_svm,
    "LogisticRegression":   _run_logistic_regression,
    "GRU":                  _run_gru,
    "GRU+Rule":             _run_gru_rule,
}


def run_baseline(name, config, train_loader, val_loader, test_loader, device):
    if name not in REGISTRY:
        raise ValueError(f"Unknown baseline: {name}, choices: {list(REGISTRY)}")
    return REGISTRY[name](config, train_loader, val_loader, test_loader, device)


def run_all_baselines(config, train_loader, val_loader, test_loader, device):
    results = {}
    for name, fn in REGISTRY.items():
        print(f"  [baseline] {name} ...")
        try:
            results[name] = fn(config, train_loader, val_loader, test_loader, device)
        except Exception as e:
            results[name] = {"name": name, "error": str(e)}
    return results
