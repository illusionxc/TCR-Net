from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:

    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


def _safe_inverse(cov: torch.Tensor, eps: float) -> torch.Tensor:
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    return torch.linalg.pinv(cov + eps * eye)


def _hour_from_time_features(time_features: torch.Tensor) -> torch.Tensor:

    sin_v = time_features[:, 0]
    cos_v = time_features[:, 1]
    angle = torch.atan2(sin_v, cos_v)
    angle = torch.where(angle < 0.0, angle + 2.0 * math.pi, angle)
    return angle * (24.0 / (2.0 * math.pi))


@dataclass
class PaperModelMeta:
    num_intents: int
    num_objects: int
    num_sources: int
    state_dim: int
    context_dim: int
    history_feature_dim: int
    sequence_length: int
    num_modes: int


class TCRNetPaper(nn.Module):
    def __init__(self, config: Dict, meta: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = config
        self.model_cfg = config["paper_model"]
        self.rule_cfg = config.get("paper_rules", {})
        self.meta = PaperModelMeta(**{k: meta[k] for k in PaperModelMeta.__annotations__.keys()})
        self.d_model = int(self.model_cfg["d_model"])
        self.obs_dim = int(self.model_cfg.get("obs_dim", 5))
        self.lambda_sigma = float(self.model_cfg.get("lambda_sigma", 1.0e-2))
        self.cov_eps = float(self.model_cfg.get("cov_epsilon", 1.0e-4))
        self.use_prototype = bool(self.model_cfg.get("use_prototype", True))
        self.use_consistency = bool(self.model_cfg.get("use_consistency", True))
        self.use_sequence = bool(self.model_cfg.get("use_sequence", True))
        self.use_rule = bool(self.model_cfg.get("use_rule", True))
        self.use_structure_token = bool(self.model_cfg.get("use_structure_token", True))
        self.use_semantic_token = bool(self.model_cfg.get("use_semantic_token", True))
        self.use_context_token = bool(self.model_cfg.get("use_context_token", True))
        self.use_transformer = bool(self.model_cfg.get("use_transformer", True))
        self.decision_mode = str(self.model_cfg.get("decision_mode", "decomposed")).lower()

        self.intent_embed = nn.Embedding(self.meta.num_intents, int(self.model_cfg.get("intent_embed_dim", 16)))
        self.object_embed = nn.Embedding(self.meta.num_objects, int(self.model_cfg.get("object_embed_dim", 12)))
        self.source_embed = nn.Embedding(self.meta.num_sources, int(self.model_cfg.get("source_embed_dim", 12)))

        self.x_proj = nn.Linear(self.meta.state_dim * 4 + 4, self.d_model)
        self.s_proj = nn.Linear(self.meta.context_dim + self.intent_embed.embedding_dim + self.object_embed.embedding_dim, self.d_model)
        self.c_proj = nn.Linear(2 + self.source_embed.embedding_dim + 2, self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.modal_pos = nn.Parameter(torch.zeros(1, 4, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(self.model_cfg["nhead"]),
            dim_feedforward=int(self.model_cfg["ffn_dim"]),
            dropout=float(self.model_cfg["dropout"]),
            batch_first=True,
            activation="gelu",
        )
        self.shared_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(self.model_cfg["shared_layers"]))

        self.z_proj = nn.Linear(self.d_model, self.d_model)
        self.obs_predictor = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(float(self.model_cfg["dropout"])),
            nn.Linear(self.d_model, self.obs_dim),
        )
        self.alpha_raw = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

        self.hist_proj = nn.Linear(self.meta.history_feature_dim, self.d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, int(self.model_cfg["max_history_length"]), self.d_model))
        seq_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(self.model_cfg["nhead"]),
            dim_feedforward=int(self.model_cfg["ffn_dim"]),
            dropout=float(self.model_cfg["dropout"]),
            batch_first=True,
            activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(seq_layer, num_layers=int(self.model_cfg["sequence_layers"]))

        self.gamma_raw = nn.Parameter(torch.zeros(5, dtype=torch.float32))
        self.beta_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.aux_head = nn.Sequential(
            nn.Linear(self.d_model * 2 + 3, self.d_model),
            nn.GELU(),
            nn.Dropout(float(self.model_cfg["dropout"])),
            nn.Linear(self.d_model, 3),
        )
        self.unified_head = nn.Sequential(
            nn.Linear(self.d_model * 2 + 5, self.d_model),
            nn.GELU(),
            nn.Dropout(float(self.model_cfg["dropout"])),
            nn.Linear(self.d_model, 3),
        )

        self.register_buffer("prototype_bank", torch.zeros(self.meta.num_intents, self.d_model))
        self.register_buffer("prototype_counts", torch.zeros(self.meta.num_intents))
        self.register_buffer("seq_mu_bank", torch.zeros(self.meta.num_intents, self.d_model))
        self.register_buffer("seq_cov_bank", torch.stack([torch.eye(self.d_model) for _ in range(self.meta.num_intents)], dim=0))
        self.register_buffer("seq_counts", torch.zeros(self.meta.num_intents))

        self._init_rule_templates()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modal_pos, std=0.02)
        nn.init.normal_(self.time_pos, std=0.02)

    def _init_rule_templates(self) -> None:
        max_intents = max(self.meta.num_intents, 1)
        default_allowed_types = [[i] for i in range(max_intents)]
        default_allowed_dst = [[i % max(self.meta.num_objects, 1)] for i in range(max_intents)]
        default_allowed_role = [[i % max(self.meta.num_sources, 1)] for i in range(max_intents)]

        allowed_types = self.rule_cfg.get("allowed_types", default_allowed_types)
        allowed_dst = self.rule_cfg.get("allowed_dst", default_allowed_dst)
        allowed_role = self.rule_cfg.get("allowed_role", default_allowed_role)
        time_windows = self.rule_cfg.get(
            "allowed_time_windows",
            [[0, 7], [6, 13], [12, 19], [18, 23], [0, 11], [8, 17]],
        )
        size_bounds = self.rule_cfg.get(
            "size_bounds",
            [[20.0, 220.0], [25.0, 240.0], [20.0, 200.0], [10.0, 140.0], [8.0, 110.0], [15.0, 180.0]],
        )
        depth_max = self.rule_cfg.get("depth_max", [2.0, 2.0, 2.0, 1.0, 1.0, 2.0])

        self.register_buffer("rule_allowed_type", self._to_rule_matrix(allowed_types, max(self.meta.num_intents, 1)))
        self.register_buffer("rule_allowed_dst", self._to_rule_matrix(allowed_dst, max(self.meta.num_objects, 1)))
        self.register_buffer("rule_allowed_role", self._to_rule_matrix(allowed_role, max(self.meta.num_sources, 1)))
        self.register_buffer("rule_time_window", self._to_rule_window(time_windows))
        self.register_buffer("rule_size_bounds", self._to_rule_bounds(size_bounds))
        self.register_buffer("rule_depth_max", self._to_rule_depth(depth_max))

    def _to_rule_matrix(self, rows, width: int) -> torch.Tensor:
        mat = torch.zeros(self.meta.num_intents, width, dtype=torch.float32)
        for i in range(self.meta.num_intents):
            if i < len(rows):
                for idx in rows[i]:
                    if 0 <= int(idx) < width:
                        mat[i, int(idx)] = 1.0
        return mat

    def _to_rule_window(self, rows) -> torch.Tensor:
        arr = torch.zeros(self.meta.num_intents, 2, dtype=torch.float32)
        for i in range(self.meta.num_intents):
            if i < len(rows):
                start, end = rows[i]
            else:
                start, end = 0.0, 23.0
            arr[i, 0] = float(start)
            arr[i, 1] = float(end)
        return arr

    def _to_rule_bounds(self, rows) -> torch.Tensor:
        arr = torch.zeros(self.meta.num_intents, 2, dtype=torch.float32)
        for i in range(self.meta.num_intents):
            if i < len(rows):
                low, high = rows[i]
            else:
                low, high = 0.0, 9999.0
            arr[i, 0] = float(low)
            arr[i, 1] = float(high)
        return arr

    def _to_rule_depth(self, rows) -> torch.Tensor:
        arr = torch.zeros(self.meta.num_intents, dtype=torch.float32)
        for i in range(self.meta.num_intents):
            arr[i] = float(rows[i] if i < len(rows) else 3.0)
        return arr

    def _build_tokens(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        intent_emb = self.intent_embed(batch["intent_id"])
        object_emb = self.object_embed(batch["object_type"])
        source_emb = self.source_embed(batch["source_type"])

        x_input = torch.cat(
            [
                batch["initial_state"],
                batch["control_params"],
                batch["response_mask"],
                batch["amp_high"] - batch["amp_low"],
                batch["steady_high"] - batch["steady_low"],
                batch["delta_t_window"].unsqueeze(-1),
                batch["settle_steps"].unsqueeze(-1),
            ],
            dim=-1,
        )
        s_input = torch.cat([batch["context"], intent_emb, object_emb], dim=-1)
        c_input = torch.cat(
            [
                batch["time_features"],
                source_emb,
                batch["control_params"],
            ],
            dim=-1,
        )

        x_tok = self.x_proj(x_input)
        s_tok = self.s_proj(s_input)
        c_tok = self.c_proj(c_input)

        if not self.use_structure_token:
            x_tok = torch.zeros_like(x_tok)
        if not self.use_semantic_token:
            s_tok = torch.zeros_like(s_tok)
        if not self.use_context_token:
            c_tok = torch.zeros_like(c_tok)

        batch_size = x_tok.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.stack([x_tok, s_tok, c_tok], dim=1)
        tokens = torch.cat([cls, tokens], dim=1) + self.modal_pos
        return tokens, source_emb

    def _shared_encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, _ = self._build_tokens(batch)
        if self.use_transformer:
            enc = self.shared_encoder(tokens)
            h = enc[:, 0]
        else:

            h = tokens[:, 1:].mean(dim=1)
        z = self.z_proj(h)
        return h, z

    def _obs_vector(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        intent_norm = batch["intent_id"].float() / max(self.meta.num_intents - 1, 1)
        object_norm = batch["object_type"].float() / max(self.meta.num_objects - 1, 1)
        source_norm = batch["source_type"].float() / max(self.meta.num_sources - 1, 1)
        cp = torch.abs(batch["control_params"])
        r = torch.stack([intent_norm, object_norm, source_norm, cp[:, 0], cp[:, 1]], dim=-1)
        return r

    def _sequence_encode(self, history_seq: torch.Tensor) -> torch.Tensor:
        if not self.use_sequence:
            return torch.zeros(history_seq.shape[0], self.d_model, device=history_seq.device, dtype=history_seq.dtype)
        bsz, hist_len, _ = history_seq.shape
        hist_len = min(hist_len, self.time_pos.shape[1])
        x = self.hist_proj(history_seq[:, :hist_len, :]) + self.time_pos[:, :hist_len, :]
        if not self.use_transformer:


            return x[:, 0]
        mask = _build_causal_mask(hist_len, x.device)
        out = self.sequence_encoder(x, mask=mask)
        return out[:, -1]

    def _rule_violations(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_rule:
            return torch.zeros(batch["intent_id"].shape[0], 5, device=batch["intent_id"].device)
        g = batch["intent_id"].long()
        type_id = batch.get("rule_type_id", batch["control_type"]).long().clamp(min=0, max=self.meta.num_intents - 1)
        dst_id = batch.get("rule_dst_id", batch["object_type"]).long().clamp(min=0, max=self.meta.num_objects - 1)
        role_id = batch.get("rule_role_id", batch["source_type"]).long().clamp(min=0, max=self.meta.num_sources - 1)

        allow_type = self.rule_allowed_type[g, type_id]
        allow_dst = self.rule_allowed_dst[g, dst_id]
        allow_role = self.rule_allowed_role[g, role_id]
        v_type = (allow_type <= 0.0).float()
        v_dst = (allow_dst <= 0.0).float()
        v_role = (allow_role <= 0.0).float()

        if "rule_hour" in batch:
            hour = batch["rule_hour"].float()
        else:
            hour = _hour_from_time_features(batch["time_features"])
        win = self.rule_time_window[g]
        v_time = ((hour < win[:, 0]) | (hour > win[:, 1])).float()

        if "rule_size" in batch:
            size_proxy = batch["rule_size"].float()
        else:
            cp = batch["control_params"]
            size_proxy = 40.0 + torch.abs(cp[:, 0]) * 160.0 + torch.abs(cp[:, 1]) * 90.0 + torch.relu(batch["initial_state"][:, 0]) * 40.0
        if "rule_depth" in batch:
            depth_proxy = batch["rule_depth"].float()
        else:
            depth_proxy = torch.round(batch["initial_state"][:, 4])
        bounds = self.rule_size_bounds[g]
        depth_max = self.rule_depth_max[g]
        v_size = torch.relu(size_proxy - bounds[:, 1]) + torch.relu(bounds[:, 0] - size_proxy) + torch.relu(depth_proxy - depth_max)
        return torch.stack([v_type, v_dst, v_role, v_time, v_size], dim=-1)

    def _mahalanobis(self, h_seq: torch.Tensor, intent_id: torch.Tensor) -> torch.Tensor:
        if not self.use_sequence:
            return torch.zeros(h_seq.shape[0], device=h_seq.device, dtype=h_seq.dtype)
        result = torch.zeros(h_seq.shape[0], device=h_seq.device, dtype=h_seq.dtype)
        for cls in intent_id.unique():
            cls_idx = int(cls.item())
            mask = intent_id == cls
            delta = h_seq[mask] - self.seq_mu_bank[cls_idx]
            cov = self.seq_cov_bank[cls_idx]
            precision = _safe_inverse(cov, self.cov_eps)
            dist = torch.sqrt(torch.clamp((delta @ precision * delta).sum(dim=1), min=0.0))
            result[mask] = dist
        return result

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h, z = self._shared_encode(batch)
        intent = batch["intent_id"].long()
        p = self.prototype_bank[intent] if self.use_prototype else torch.zeros_like(z)
        r = self._obs_vector(batch)
        r_hat = self.obs_predictor(torch.cat([z, p], dim=-1))

        d_proto = (z - p).pow(2).sum(dim=-1)
        d_attr = torch.abs(r - r_hat).sum(dim=-1) / float(self.obs_dim)
        alpha = torch.nn.functional.softplus(self.alpha_raw)
        r_cons = alpha[0] * d_proto + alpha[1] * d_attr
        if not self.use_consistency:
            r_cons = torch.zeros_like(r_cons)

        h_seq = self._sequence_encode(batch["history_seq"])
        r_seq = self._mahalanobis(h_seq, intent)

        violations = self._rule_violations(batch)
        gamma = torch.nn.functional.softplus(self.gamma_raw)
        r_rule = (violations * gamma.unsqueeze(0)).sum(dim=-1)
        if not self.use_rule:
            r_rule = torch.zeros_like(r_rule)

        if self.decision_mode == "unified":
            logits = self.unified_head(torch.cat([h, h_seq, violations], dim=-1))
            probs = torch.softmax(logits, dim=-1)
            risk_levels = torch.tensor([0.0, 0.5, 1.0], device=probs.device, dtype=probs.dtype)
            unified_score = probs @ risk_levels
            r_cons = unified_score
            r_seq = torch.zeros_like(unified_score)
            r_rule = torch.zeros_like(unified_score)
        else:
            logits = self.aux_head(torch.cat([h, h_seq, r_cons.unsqueeze(-1), r_seq.unsqueeze(-1), r_rule.unsqueeze(-1)], dim=-1))
            unified_score = torch.zeros_like(r_cons)

        return {
            "h_shared": h,
            "z": z,
            "prototype": p,
            "r_obs": r,
            "r_hat": r_hat,
            "d_proto": d_proto,
            "d_attr": d_attr,
            "R_cons": r_cons,
            "h_seq": h_seq,
            "R_seq": r_seq,
            "violations": violations,
            "R_rule": r_rule,
            "R_unified": unified_score,
            "logits": logits,
        }

    @torch.no_grad()
    def update_reference_banks(
        self,
        z: torch.Tensor,
        h_seq: torch.Tensor,
        intent_id: torch.Tensor,
        normal_mask: torch.Tensor,
        momentum: float,
    ) -> None:
        if normal_mask.sum() == 0:
            return
        if not self.use_prototype and not self.use_sequence:
            return
        z_n = z[normal_mask]
        h_n = h_seq[normal_mask]
        g_n = intent_id[normal_mask]

        for cls in g_n.unique():
            idx = int(cls.item())
            cls_mask = g_n == cls
            z_mean = z_n[cls_mask].mean(dim=0)
            h_cls = h_n[cls_mask]
            h_mean = h_cls.mean(dim=0)
            centered = h_cls - h_mean
            if centered.shape[0] <= 1:
                cov = torch.eye(self.d_model, device=h_cls.device, dtype=h_cls.dtype) * self.lambda_sigma
            else:
                cov = (centered.transpose(0, 1) @ centered) / float(centered.shape[0] - 1)
                cov = cov + self.lambda_sigma * torch.eye(self.d_model, device=h_cls.device, dtype=h_cls.dtype)

            if self.use_prototype:
                if self.prototype_counts[idx] <= 0:
                    self.prototype_bank[idx] = z_mean
                else:
                    self.prototype_bank[idx] = momentum * self.prototype_bank[idx] + (1.0 - momentum) * z_mean
                self.prototype_counts[idx] += cls_mask.sum().float()

            if self.use_sequence:
                if self.seq_counts[idx] <= 0:
                    self.seq_mu_bank[idx] = h_mean
                    self.seq_cov_bank[idx] = cov
                else:
                    self.seq_mu_bank[idx] = momentum * self.seq_mu_bank[idx] + (1.0 - momentum) * h_mean
                    self.seq_cov_bank[idx] = momentum * self.seq_cov_bank[idx] + (1.0 - momentum) * cov
                self.seq_counts[idx] += cls_mask.sum().float()

    @staticmethod
    def normalize_component(values: torch.Tensor, low: float, high: float, eps: float) -> torch.Tensor:
        return (values - low) / max(high - low, eps)
