


from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def _flatten_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:

    parts = [
        batch["initial_state"].float().reshape(batch["initial_state"].shape[0], -1),
        batch["control_params"].float().reshape(batch["control_params"].shape[0], -1),
        batch["context"].float().reshape(batch["context"].shape[0], -1),
        batch["time_features"].float().reshape(batch["time_features"].shape[0], -1),
    ]

    B, K, H = batch["history_seq"].shape
    parts.append(batch["history_seq"].float().reshape(B, K * H))
    return torch.cat(parts, dim=-1)


def _extract_features_from_loader(loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:

    feats, labels = [], []
    for batch in loader:
        feats.append(_flatten_batch(batch).cpu())
        labels.append(batch.get("risk_label", batch["anomaly_label"]).cpu().numpy())
    return torch.cat(feats, dim=0), np.concatenate(labels, axis=0)


def _collect_anomaly_scores(model, loader: DataLoader, device: torch.device,
                            score_fn: Callable) -> Dict[str, np.ndarray]:

    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            x = _flatten_batch(batch).to(device)
            s = score_fn(model, x).cpu().numpy()
            scores.append(s)
            labels.append(batch.get("risk_label", batch["anomaly_label"]).cpu().numpy())
    return {
        "score": np.concatenate(scores, axis=0),
        "y_true": np.concatenate(labels, axis=0),
    }


def _calibrate_thresholds(val_scores: np.ndarray, val_y: np.ndarray,
                          config: Dict[str, Any]) -> Tuple[float, float]:

    threshold_cfg = config.get("paper_score", {}).get("thresholds", {})
    q1 = threshold_cfg.get("eta1_quantiles", [0.55, 0.60, 0.65, 0.70, 0.75])
    q2 = threshold_cfg.get("eta2_quantiles", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    cand_1 = [float(np.quantile(val_scores, float(q))) for q in q1]
    cand_2 = [float(np.quantile(val_scores, float(q))) for q in q2]

    from sklearn.metrics import f1_score

    best = {"eta_1": float(np.quantile(val_scores, 0.65)), "eta_2": float(np.quantile(val_scores, 0.85)),
            "macro_f1": -1.0, "high_recall": -1.0}
    for eta_1 in cand_1:
        for eta_2 in cand_2:
            if eta_2 <= eta_1:
                continue
            pred = _score_to_3class(val_scores, eta_1, eta_2)
            macro = float(f1_score(val_y, pred, average="macro", zero_division=0))
            _, recall, _, _ = _precision_recall_fscore_support(val_y, pred, labels=[0, 1, 2])
            high_recall = float(recall[2])
            if macro > best["macro_f1"] or (abs(macro - best["macro_f1"]) <= 1e-9 and high_recall > best["high_recall"]):
                best = {"eta_1": eta_1, "eta_2": eta_2, "macro_f1": macro, "high_recall": high_recall}
    return best["eta_1"], best["eta_2"]


def _score_to_3class(scores: np.ndarray, eta_1: float, eta_2: float) -> np.ndarray:

    y = np.zeros(scores.shape[0], dtype=np.int64)
    y[scores >= eta_1] = 1
    y[scores >= eta_2] = 2
    return y


def _precision_recall_fscore_support(y_true, y_pred, labels):

    from sklearn.metrics import precision_recall_fscore_support
    return precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)


def _summarize(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = _precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])
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


def train_eval_isolation_forest(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:

    from sklearn.ensemble import IsolationForest

    X_train, _ = _extract_features_from_loader(train_loader, device)
    X_val, y_val = _extract_features_from_loader(val_loader, device)
    X_test, y_test = _extract_features_from_loader(test_loader, device)


    clf = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=int(config["seed"]),
        n_jobs=-1,
    )
    clf.fit(X_train.numpy())


    val_score = -clf.score_samples(X_val.numpy())
    test_score = -clf.score_samples(X_test.numpy())


    eta_1, eta_2 = _calibrate_thresholds(val_score, y_val, config)


    val_pred = _score_to_3class(val_score, eta_1, eta_2)
    test_pred = _score_to_3class(test_score, eta_1, eta_2)

    return {
        "name": "IsolationForest",
        "val_metrics": _summarize(y_val, val_pred),
        "test_metrics": _summarize(y_test, test_pred),
        "thresholds": {"eta_1": eta_1, "eta_2": eta_2},
    }


class _DeepSVDDNet(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_eval_deep_svdd(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:

    X_train_full, y_train_full = _extract_features_from_loader(train_loader, device)
    X_val, y_val = _extract_features_from_loader(val_loader, device)
    X_test, y_test = _extract_features_from_loader(test_loader, device)

    input_dim = X_train_full.shape[1]
    latent_dim = int(config.get("baseline", {}).get("svdd_latent_dim", 32))
    model = _DeepSVDDNet(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("baseline", {}).get("svdd_lr", 1e-3)))


    normal_mask = y_train_full == 0
    X_train = X_train_full[normal_mask]
    if len(X_train) < 64:
        X_train = X_train_full

    train_dataset = TensorDataset(X_train)
    train_loader_svdd = DataLoader(train_dataset, batch_size=min(256, len(X_train)), shuffle=True)


    model.eval()
    with torch.no_grad():
        sample_feats = X_train[:min(1024, len(X_train))].to(device)
        init_z = model(sample_feats)
        center = init_z.mean(dim=0)

        if center.norm() < 0.1:
            center = center + 0.1
    center = center.detach()


    model.train()
    n_epochs = int(config.get("baseline", {}).get("svdd_epochs", 20))
    for epoch in range(n_epochs):
        total_loss = 0.0
        for (x_batch,) in train_loader_svdd:
            x_batch = x_batch.to(device)
            z = model(x_batch)
            loss = torch.mean((z - center.unsqueeze(0)).pow(2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


    def score_fn(m, x):
        z = m(x)
        return torch.sqrt(torch.sum((z - center.unsqueeze(0)).pow(2), dim=1))

    val_out = _collect_anomaly_scores(model, val_loader, device, score_fn)
    test_out = _collect_anomaly_scores(model, test_loader, device, score_fn)

    eta_1, eta_2 = _calibrate_thresholds(val_out["score"], val_out["y_true"], config)
    val_pred = _score_to_3class(val_out["score"], eta_1, eta_2)
    test_pred = _score_to_3class(test_out["score"], eta_1, eta_2)

    return {
        "name": "DeepSVDD",
        "val_metrics": _summarize(val_out["y_true"], val_pred),
        "test_metrics": _summarize(test_out["y_true"], test_pred),
        "thresholds": {"eta_1": eta_1, "eta_2": eta_2},
    }


class _DAGMMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class _DAGMMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class _DAGMMEstimation(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, n_gmm: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_gmm),
            nn.Softmax(dim=1),
        )

    def forward(self, z, recon_error):
        return self.net(torch.cat([z, recon_error], dim=1))


class _DAGMM(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64, n_gmm: int = 3):
        super().__init__()
        self.encoder = _DAGMMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = _DAGMMDecoder(latent_dim, hidden_dim, input_dim)
        self.estimation = _DAGMMEstimation(latent_dim, hidden_dim, n_gmm)
        self.n_gmm = n_gmm

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = mu_z + torch.randn_like(logvar_z) * torch.exp(0.5 * logvar_z)
        x_recon = self.decoder(z)
        recon_error = torch.cat([
            torch.sum((x - x_recon) ** 2, dim=1, keepdim=True),
            torch.sum((x - x_recon).abs(), dim=1, keepdim=True),
        ], dim=1)
        gamma = self.estimation(z, recon_error)


        gamma_sum = gamma.sum(dim=0)
        phi = gamma_sum / gamma_sum.sum()
        mu = (gamma.unsqueeze(-1) * z.unsqueeze(1)).sum(dim=0) / gamma_sum.unsqueeze(-1).clamp(min=1e-6)
        return z, x_recon, gamma, phi, mu


def train_eval_dagmm(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:

    X_train_full, y_train_full = _extract_features_from_loader(train_loader, device)
    X_val, y_val = _extract_features_from_loader(val_loader, device)
    X_test, y_test = _extract_features_from_loader(test_loader, device)

    input_dim = X_train_full.shape[1]
    latent_dim = int(config.get("baseline", {}).get("dagmm_latent_dim", 8))
    hidden_dim = int(config.get("baseline", {}).get("dagmm_hidden_dim", 64))
    n_gmm = int(config.get("baseline", {}).get("dagmm_n_gmm", 3))

    model = _DAGMM(input_dim, latent_dim, hidden_dim, n_gmm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("baseline", {}).get("dagmm_lr", 1e-3)))

    train_dataset = TensorDataset(X_train_full)
    train_loader_dagmm = DataLoader(train_dataset, batch_size=min(256, len(X_train_full)), shuffle=True)

    lambda_energy = float(config.get("baseline", {}).get("dagmm_lambda_energy", 0.1))
    lambda_cov = float(config.get("baseline", {}).get("dagmm_lambda_cov", 0.005))

    n_epochs = int(config.get("baseline", {}).get("dagmm_epochs", 20))
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for (x_batch,) in train_loader_dagmm:
            x_batch = x_batch.to(device)
            z, x_recon, gamma, phi, mu = model(x_batch)


            recon_loss = torch.mean(torch.sum((x_batch - x_recon) ** 2, dim=1))


            z_expand = z.unsqueeze(1).expand(-1, n_gmm, -1)
            mu_expand = mu.unsqueeze(0)
            diff = z_expand - mu_expand

            energy = -torch.log(phi.unsqueeze(0) + 1e-8) + 0.5 * torch.sum(diff ** 2, dim=2)
            min_energy = torch.min(energy, dim=1)[0]
            energy_loss = torch.mean(min_energy)


            cov_loss = lambda_cov * torch.sum(phi ** 2)

            loss = recon_loss + lambda_energy * energy_loss + cov_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


    def score_fn(m, x):
        z, _, gamma, phi, mu = m(x)
        z_expand = z.unsqueeze(1).expand(-1, n_gmm, -1)
        mu_expand = mu.unsqueeze(0)
        diff = z_expand - mu_expand
        energy = -torch.log(phi.unsqueeze(0) + 1e-8) + 0.5 * torch.sum(diff ** 2, dim=2)
        return torch.min(energy, dim=1)[0]

    val_out = _collect_anomaly_scores(model, val_loader, device, score_fn)
    test_out = _collect_anomaly_scores(model, test_loader, device, score_fn)

    eta_1, eta_2 = _calibrate_thresholds(val_out["score"], val_out["y_true"], config)
    val_pred = _score_to_3class(val_out["score"], eta_1, eta_2)
    test_pred = _score_to_3class(test_out["score"], eta_1, eta_2)

    return {
        "name": "DAGMM",
        "val_metrics": _summarize(val_out["y_true"], val_pred),
        "test_metrics": _summarize(test_out["y_true"], test_pred),
        "thresholds": {"eta_1": eta_1, "eta_2": eta_2},
    }


class _OCNN(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval_ocnn(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:


    X_train_full, y_train_full = _extract_features_from_loader(train_loader, device)
    X_val, y_val = _extract_features_from_loader(val_loader, device)
    X_test, y_test = _extract_features_from_loader(test_loader, device)

    input_dim = X_train_full.shape[1]
    hidden_dim = int(config.get("baseline", {}).get("ocnn_hidden_dim", 64))
    model = _OCNN(input_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("baseline", {}).get("ocnn_lr", 1e-3)))


    normal_mask = y_train_full == 0
    X_normal = X_train_full[normal_mask]

    if (~normal_mask).sum() > 0:
        X_anom = X_train_full[~normal_mask][:len(X_normal) // 4]
    else:
        X_anom = X_train_full[:len(X_normal) // 10]

    if len(X_anom) < 8:
        X_anom = X_train_full[:min(len(X_train_full) // 5, 64)]


    X_pos = X_normal
    y_pos = torch.ones(len(X_pos), dtype=torch.float32)
    X_neg = X_anom
    y_neg = torch.zeros(len(X_neg), dtype=torch.float32)

    X_all = torch.cat([X_pos, X_neg], dim=0)
    y_all = torch.cat([y_pos, y_neg], dim=0)

    dataset = TensorDataset(X_all, y_all)
    loader = DataLoader(dataset, batch_size=min(128, len(X_all)), shuffle=True)

    n_epochs = int(config.get("baseline", {}).get("ocnn_epochs", 20))
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            score = model(x_batch)
            loss = criterion(score, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


    def score_fn(m, x):
        return -m(x)

    val_out = _collect_anomaly_scores(model, val_loader, device, score_fn)
    test_out = _collect_anomaly_scores(model, test_loader, device, score_fn)

    eta_1, eta_2 = _calibrate_thresholds(val_out["score"], val_out["y_true"], config)
    val_pred = _score_to_3class(val_out["score"], eta_1, eta_2)
    test_pred = _score_to_3class(test_out["score"], eta_1, eta_2)

    return {
        "name": "OCNN",
        "val_metrics": _summarize(val_out["y_true"], val_pred),
        "test_metrics": _summarize(test_out["y_true"], test_pred),
        "thresholds": {"eta_1": eta_1, "eta_2": eta_2},
    }


BASELINE_REGISTRY = {
    "IsolationForest": train_eval_isolation_forest,
    "DeepSVDD": train_eval_deep_svdd,
    "DAGMM": train_eval_dagmm,
    "OCNN": train_eval_ocnn,
}


def run_baseline(
    name: str,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:

    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    fn = BASELINE_REGISTRY[name]
    return fn(config, train_loader, val_loader, test_loader, device)


def run_all_baselines(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:

    results = {}
    for name, fn in BASELINE_REGISTRY.items():
        print(f"[Baseline] Running {name} ...")
        try:
            result = fn(config, train_loader, val_loader, test_loader, device)
            results[name] = result
        except Exception as e:
            print(f"[Baseline] {name} failed: {e}")
            results[name] = {"name": name, "error": str(e)}
    return results
