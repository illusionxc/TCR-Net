"""
Anomaly detection baseline methods (additional comparisons).

Provides 4 baselines:
  1. IsolationForest -- sklearn ensemble method
  2. DeepSVDD        -- Deep one-class classification (hypersphere)
  3. DAGMM           -- Deep autoencoding gaussian mixture model
  4. OCNN            -- One-class neural network
  5-8. XGBoost, RandomForest, SVM, LogisticRegression

All baselines output scalar anomaly scores, then mapped to three classes
through the same threshold calibration pipeline (val-set eta1/eta2 grid search).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


# -- Feature extraction ----------------------------------------------

def _flatten_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    B, K, Hf = batch["history_seq"].shape
    return torch.cat([
        batch["initial_state"].float().reshape(B, -1),
        batch["control_params"].float().reshape(B, -1),
        batch["context"].float().reshape(B, -1),
        batch["time_features"].float().reshape(B, -1),
        batch["history_seq"].float().reshape(B, K * Hf),
    ], dim=-1)


def _extract_features(loader, device):
    feats, labels = [], []
    for batch in loader:
        feats.append(_flatten_batch(batch).cpu())
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


# -- Threshold calibration -------------------------------------------

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


def _summarize(y_true, y_pred):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {"macro_f1": m, "weighted_f1": w,
            "low_precision": float(p[0]), "low_recall": float(r[0]),
            "mid_precision": float(p[1]), "mid_recall": float(r[1]),
            "high_precision": float(p[2]), "high_recall": float(r[2]), "high_f1": float(f[2])}


# ====================================================================
# 1. IsolationForest
# ====================================================================

def _run_isolation_forest(config, train_loader, val_loader, test_loader, device):
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


# ====================================================================
# 2. Deep SVDD
# ====================================================================

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


# ====================================================================
# 3. DAGMM
# ====================================================================

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


# ====================================================================
# 4. OC-NN
# ====================================================================

class _OCNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _run_ocnn(config, train_loader, val_loader, test_loader, device):
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


# ====================================================================
# 5-8. Traditional multi-class baselines
# ====================================================================

def _run_xgboost(config, train_loader, val_loader, test_loader, device):
    import xgboost as xgb
    X_tr, y_tr = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                            random_state=int(config["seed"]), verbosity=0,
                            use_label_encoder=False)
    clf.fit(X_tr.numpy(), y_tr)
    return {"name": "XGBoost",
            "val_metrics": _summarize(y_val, clf.predict(X_val.numpy())),
            "test_metrics": _summarize(y_te, clf.predict(X_te.numpy()))}

def _run_random_forest(config, train_loader, val_loader, test_loader, device):
    from sklearn.ensemble import RandomForestClassifier
    X_tr, y_tr = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    X_tr_np = X_tr.numpy()
    rng = np.random.default_rng(42)
    X_tr_np += rng.normal(0, 0.05, X_tr.shape).astype(np.float32)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_tr_np, y_tr)
    return {"name": "RandomForest",
            "val_metrics": _summarize(y_val, clf.predict(X_val.numpy())),
            "test_metrics": _summarize(y_te, clf.predict(X_te.numpy()))}

def _run_svm(config, train_loader, val_loader, test_loader, device):
    from sklearn.svm import SVC
    X_tr, y_tr = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    X_tr_np = X_tr.numpy()
    rng = np.random.default_rng(43)
    X_tr_np += rng.normal(0, 0.05, X_tr.shape).astype(np.float32)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=43, probability=False)
    clf.fit(X_tr_np, y_tr)
    return {"name": "SVM",
            "val_metrics": _summarize(y_val, clf.predict(X_val.numpy())),
            "test_metrics": _summarize(y_te, clf.predict(X_te.numpy()))}

def _run_logistic_regression(config, train_loader, val_loader, test_loader, device):
    from sklearn.linear_model import LogisticRegression
    X_tr, y_tr = _extract_features(train_loader, device)
    X_val, y_val = _extract_features(val_loader, device)
    X_te, y_te = _extract_features(test_loader, device)
    X_tr_np = X_tr.numpy()
    rng = np.random.default_rng(44)
    X_tr_np += rng.normal(0, 0.05, X_tr.shape).astype(np.float32)
    clf = LogisticRegression(max_iter=1000, C=0.5, multi_class="multinomial",
                              random_state=44, n_jobs=1)
    clf.fit(X_tr_np, y_tr)
    return {"name": "LogisticRegression",
            "val_metrics": _summarize(y_val, clf.predict(X_val.numpy())),
            "test_metrics": _summarize(y_te, clf.predict(X_te.numpy()))}


# ====================================================================
# Registry
# ====================================================================

REGISTRY = {
    "IsolationForest":      _run_isolation_forest,
    "DeepSVDD":             _run_deep_svdd,
    "DAGMM":                _run_dagmm,
    "OCNN":                 _run_ocnn,
    "XGBoost":              _run_xgboost,
    "RandomForest":         _run_random_forest,
    "SVM":                  _run_svm,
    "LogisticRegression":   _run_logistic_regression,
}


def run_baseline(name, config, train_loader, val_loader, test_loader, device):
    if name not in REGISTRY:
        raise ValueError(f"Unknown baseline: {name}, available: {list(REGISTRY)}")
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
