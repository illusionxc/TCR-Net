"""
TCR-Net 。


--------
TCR-Net ：

1. （Cross-modal Transformer）：
2. R_cons（）：
3. R_seq（）：
4. R_rule（）：
5.  + ：β  → η₁, η₂  → {Low, Medium, High}


--------
2026-05:
  -  unified mode  aux_head （ violations[5]， aux_head  3 ）
  -  unified_head  unified mode（ cctnet ）
  -  _mahalanobis  for ， GPU 
  - 
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """
"""
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


def _safe_inverse(cov: torch.Tensor, eps: float) -> torch.Tensor:
    """
"""
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    return torch.linalg.pinv(cov + eps * eye)


def _hour_from_time_features(time_features: torch.Tensor) -> torch.Tensor:
    """
"""
    angle = torch.atan2(time_features[:, 0], time_features[:, 1])
    angle = torch.where(angle < 0.0, angle + 2.0 * math.pi, angle)
    return angle * (24.0 / (2.0 * math.pi))


@dataclass
class ModelMeta:
    """
"""
    num_intents: int          # //
    num_objects: int          # ///
    num_sources: int          # //
    state_dim: int            # 6
    context_dim: int          # 6
    history_feature_dim: int  # 8
    sequence_length: int
    num_modes: int            #  pre_task/on_site/post_task/review


class TCRNet(nn.Module):
    """
    TCR-Net：。

    : (x_i, s_i, c_i, H_i) — 、、 + 
    : (R_cons, R_seq, R_rule, R_total, y_hat) —  +  + 

    
    --------
    - ：4-token  Transformer（[CLS, , , ]）
    - R_cons： + 
    - R_seq： Transformer  → 
    - R_rule：
    - ：min-max  β  → η₁, η₂  → 
    """

    def __init__(self, config: Dict, meta: Dict[str, int]) -> None:
        super().__init__()
        self.cfg = config
        self.model_cfg = config["paper_model"]
        self.rule_cfg = config.get("paper_rules", {})
        self.meta = ModelMeta(**{k: meta[k] for k in ModelMeta.__annotations__})
        self.d_model = int(self.model_cfg["d_model"])
        self.obs_dim = int(self.model_cfg.get("obs_dim", 5))
        self.lambda_sigma = float(self.model_cfg.get("lambda_sigma", 1.0e-2))
        self.cov_eps = float(self.model_cfg.get("cov_epsilon", 1.0e-4))


        #  ablation_variants 
        # ""
        self.use_prototype   = bool(self.model_cfg.get("use_prototype", True))
        self.use_consistency = bool(self.model_cfg.get("use_consistency", True))
        self.use_sequence    = bool(self.model_cfg.get("use_sequence", True))
        self.use_rule        = bool(self.model_cfg.get("use_rule", True))
        self.use_structure   = bool(self.model_cfg.get("use_structure_token", True))
        self.use_semantic    = bool(self.model_cfg.get("use_semantic_token", True))
        self.use_context     = bool(self.model_cfg.get("use_context_token", True))
        self.use_transformer = bool(self.model_cfg.get("use_transformer", True))
        self.decision_mode   = str(self.model_cfg.get("decision_mode", "decomposed")).lower()
        # decision_mode: "decomposed"() = , "unified" =  logits


        # intent/object/source
        self.intent_embed  = nn.Embedding(self.meta.num_intents,
                                          int(self.model_cfg.get("intent_embed_dim", 16)))
        self.object_embed  = nn.Embedding(self.meta.num_objects,
                                          int(self.model_cfg.get("object_embed_dim", 12)))
        self.source_embed  = nn.Embedding(self.meta.num_sources,
                                          int(self.model_cfg.get("source_embed_dim", 12)))

        #  Cross-modal Transformer 
        #  d_model 
        # x_proj : state_dim*4(state + mask + amp_range + steady_range) +
        #              2(control_params) + 1(delta_t_window) + 1(settle_steps) = 28
        self.x_proj = nn.Linear(self.meta.state_dim * 4 + 4, self.d_model)
        self.s_proj = nn.Linear(
            self.meta.context_dim + self.intent_embed.embedding_dim + self.object_embed.embedding_dim,
            self.d_model,
        )
        self.c_proj = nn.Linear(2 + self.source_embed.embedding_dim + 2, self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))  # [CLS] token
        self.modal_pos  = nn.Parameter(torch.zeros(1, 4, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(self.model_cfg["nhead"]),
            dim_feedforward=int(self.model_cfg["ffn_dim"]),
            dropout=float(self.model_cfg["dropout"]),
            batch_first=True, activation="gelu",
        )
        self.shared_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=int(self.model_cfg["shared_layers"]),
        )

        #  R_cons 
        # z_proj:  h 
        # obs_predictor: 
        self.z_proj = nn.Linear(self.d_model, self.d_model)
        self.obs_predictor = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model), nn.GELU(),
            nn.Dropout(float(self.model_cfg["dropout"])),
            nn.Linear(self.d_model, self.obs_dim),
        )
        self.alpha_raw = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        # alpha = softplus(alpha_raw)  [, ] 

        #  R_seq 
        self.hist_proj = nn.Linear(self.meta.history_feature_dim, self.d_model)
        self.time_pos  = nn.Parameter(torch.zeros(1, int(self.model_cfg["max_history_length"]), self.d_model))
        seq_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(self.model_cfg["nhead"]),
            dim_feedforward=int(self.model_cfg["ffn_dim"]),
            dropout=float(self.model_cfg["dropout"]),
            batch_first=True, activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(
            seq_layer, num_layers=int(self.model_cfg["sequence_layers"]),
        )

        #  R_rule+  
        self.gamma_raw = nn.Parameter(torch.zeros(5, dtype=torch.float32))
        # gamma = softplus(gamma_raw)  [..] 
        self.beta_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # beta = softmax(beta_logits)  [, , ] 


        # decomposed  [h, h_seq, R_cons, R_seq, R_rule] (2d+3 )
        self.aux_head = nn.Sequential(
            nn.Linear(self.d_model * 2 + 3, self.d_model), nn.GELU(),
            nn.Dropout(float(self.model_cfg["dropout"])),
            nn.Linear(self.d_model, 3),  # 3  logits  low/medium/high
        )

        #   unified  
        #  [h, h_seq, violations] (2d+5 )
        # unified  Unified-Multimodal baseline
        #  head
        self.unified_head = nn.Sequential(
            nn.Linear(self.d_model * 2 + 5, self.d_model), nn.GELU(),
            nn.Dropout(float(self.model_cfg["dropout"])),
            nn.Linear(self.d_model, 3),
        )

        #  prototype +  
        #  buffer 
        #  update_reference_banks  EMA 
        # register_buffer  .to(device) 
        self.register_buffer("prototype_bank", torch.zeros(self.meta.num_intents, self.d_model))
        self.register_buffer("prototype_counts", torch.zeros(self.meta.num_intents))
        self.register_buffer("seq_mu_bank",  torch.zeros(self.meta.num_intents, self.d_model))
        self.register_buffer("seq_cov_bank", torch.stack(
            [torch.eye(self.d_model) for _ in range(self.meta.num_intents)], dim=0))
        self.register_buffer("seq_counts", torch.zeros(self.meta.num_intents))

        self._init_rule_templates()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modal_pos,  std=0.02)
        nn.init.normal_(self.time_pos,   std=0.02)



    def _init_rule_templates(self) -> None:
        """
"""
        max_i = max(self.meta.num_intents, 1)
        default_allowed = {"types": [[i] for i in range(max_i)],
                           "dst":   [[i % max(self.meta.num_objects, 1)] for i in range(max_i)],
                           "role":  [[i % max(self.meta.num_sources, 1)] for i in range(max_i)]}

        allowed_types = self.rule_cfg.get("allowed_types", default_allowed["types"])
        allowed_dst   = self.rule_cfg.get("allowed_dst",   default_allowed["dst"])
        allowed_role  = self.rule_cfg.get("allowed_role",  default_allowed["role"])
        time_windows  = self.rule_cfg.get("allowed_time_windows",
            [[0, 7], [6, 13], [12, 19], [18, 23], [0, 11], [8, 17]])
        size_bounds   = self.rule_cfg.get("size_bounds",
            [[20, 220], [25, 240], [20, 200], [10, 140], [8, 110], [15, 180]])
        depth_max     = self.rule_cfg.get("depth_max", [2, 2, 2, 1, 1, 2])

        def _mat(rows, w):
            """
"""
            m = torch.zeros(self.meta.num_intents, w, dtype=torch.float32)
            for i in range(self.meta.num_intents):
                if i < len(rows):
                    for idx in rows[i]:
                        if 0 <= int(idx) < w:
                            m[i, int(idx)] = 1.0
            return m

        def _win(rows):
            """
"""
            a = torch.zeros(self.meta.num_intents, 2, dtype=torch.float32)
            for i in range(self.meta.num_intents):
                s, e = rows[i] if i < len(rows) else (0.0, 23.0)
                a[i, 0], a[i, 1] = float(s), float(e)
            return a

        def _bnd(rows):
            """
"""
            a = torch.zeros(self.meta.num_intents, 2, dtype=torch.float32)
            for i in range(self.meta.num_intents):
                l, h = rows[i] if i < len(rows) else (0.0, 9999.0)
                a[i, 0], a[i, 1] = float(l), float(h)
            return a

        self.register_buffer("rule_allowed_type", _mat(allowed_types, max(self.meta.num_intents, 1)))
        self.register_buffer("rule_allowed_dst",  _mat(allowed_dst,   max(self.meta.num_objects, 1)))
        self.register_buffer("rule_allowed_role", _mat(allowed_role,  max(self.meta.num_sources, 1)))
        self.register_buffer("rule_time_window",  _win(time_windows))
        self.register_buffer("rule_size_bounds",  _bnd(size_bounds))
        self.register_buffer("rule_depth_max",
            torch.tensor([float(depth_max[i] if i < len(depth_max) else 3.0) for i in range(self.meta.num_intents)],
                         dtype=torch.float32))



    def _build_tokens(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        intent_emb = self.intent_embed(batch["intent_id"])
        object_emb = self.object_embed(batch["object_type"])
        source_emb = self.source_embed(batch["source_type"])

        x_input = torch.cat([
            batch["initial_state"], batch["control_params"],
            batch["response_mask"], batch["amp_high"] - batch["amp_low"],
            batch["steady_high"] - batch["steady_low"],
            batch["delta_t_window"].unsqueeze(-1),
            batch["settle_steps"].unsqueeze(-1),
        ], dim=-1)
        s_input = torch.cat([batch["context"], intent_emb, object_emb], dim=-1)
        c_input = torch.cat([batch["time_features"], source_emb, batch["control_params"]], dim=-1)

        x_tok = self.x_proj(x_input)
        s_tok = self.s_proj(s_input)
        c_tok = self.c_proj(c_input)

        if not self.use_structure:
            x_tok = torch.zeros_like(x_tok)
        if not self.use_semantic:
            s_tok = torch.zeros_like(s_tok)
        if not self.use_context:
            c_tok = torch.zeros_like(c_tok)

        B = x_tok.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.stack([x_tok, s_tok, c_tok], dim=1)
        tokens = torch.cat([cls, tokens], dim=1) + self.modal_pos
        return tokens, source_emb

    def _shared_encode(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
"""
        tokens, _ = self._build_tokens(batch)
        if self.use_transformer:
            enc = self.shared_encoder(tokens)
            h = enc[:, 0]       #  [CLS] token 
        else:
            # Transformer-free  [CLS] 3  token 
            h = tokens[:, 1:].mean(dim=1)
        z = self.z_proj(h)
        return h, z

    def _obs_vector(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
"""
        i_norm = batch["intent_id"].float() / max(self.meta.num_intents - 1, 1)
        o_norm = batch["object_type"].float() / max(self.meta.num_objects - 1, 1)
        s_norm = batch["source_type"].float() / max(self.meta.num_sources - 1, 1)
        cp = torch.abs(batch["control_params"])
        return torch.stack([i_norm, o_norm, s_norm, cp[:, 0], cp[:, 1]], dim=-1)

    def _sequence_encode(self, history_seq: torch.Tensor) -> torch.Tensor:
        """
"""
        if not self.use_sequence:
            return torch.zeros(history_seq.shape[0], self.d_model,
                               device=history_seq.device, dtype=history_seq.dtype)
        B, L, _ = history_seq.shape
        L = min(L, self.time_pos.shape[1])
        x = self.hist_proj(history_seq[:, :L, :]) + self.time_pos[:, :L, :]
        if not self.use_transformer:
            # Transformer-free 
            return x[:, 0]
        mask = _build_causal_mask(L, x.device)
        out = self.sequence_encoder(x, mask=mask)
        return out[:, -1]

    def _rule_violations(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
"""
        if not self.use_rule:
            return torch.zeros(batch["intent_id"].shape[0], 5,
                               device=batch["intent_id"].device)
        g = batch["intent_id"].long()
        type_id = batch.get("rule_type_id", batch["control_type"]).long().clamp(min=0, max=self.meta.num_intents - 1)
        dst_id  = batch.get("rule_dst_id",  batch["object_type"]).long().clamp(min=0, max=self.meta.num_objects - 1)
        role_id = batch.get("rule_role_id", batch["source_type"]).long().clamp(min=0, max=self.meta.num_sources - 1)

        v_type = (self.rule_allowed_type[g, type_id] <= 0.0).float()
        v_dst  = (self.rule_allowed_dst[g, dst_id]  <= 0.0).float()
        v_role = (self.rule_allowed_role[g, role_id] <= 0.0).float()

        hour = batch.get("rule_hour", _hour_from_time_features(batch["time_features"])).float()
        win = self.rule_time_window[g]
        v_time = ((hour < win[:, 0]) | (hour > win[:, 1])).float()

        size_proxy = batch.get("rule_size", 40.0 + torch.abs(batch["control_params"][:, 0]) * 160.0
                               + torch.abs(batch["control_params"][:, 1]) * 90.0
                               + torch.relu(batch["initial_state"][:, 0]) * 40.0).float()
        depth_proxy = batch.get("rule_depth", torch.round(batch["initial_state"][:, 4])).float()
        bounds = self.rule_size_bounds[g]
        v_size = (torch.relu(size_proxy - bounds[:, 1])
                  + torch.relu(bounds[:, 0] - size_proxy)
                  + torch.relu(depth_proxy - self.rule_depth_max[g]))
        return torch.stack([v_type, v_dst, v_role, v_time, v_size], dim=-1)

    def _mahalanobis(self, h_seq: torch.Tensor, intent_id: torch.Tensor) -> torch.Tensor:
        """
"""
        if not self.use_sequence:
            return torch.zeros(h_seq.shape[0], device=h_seq.device, dtype=h_seq.dtype)
        result = torch.zeros(h_seq.shape[0], device=h_seq.device, dtype=h_seq.dtype)
        # 6 
        for cls in intent_id.unique():
            idx = int(cls.item())
            mask = intent_id == cls
            delta = h_seq[mask] - self.seq_mu_bank[idx]                     # [n_cls, d]
            precision = _safe_inverse(self.seq_cov_bank[idx], self.cov_eps)  # [d, d]
            # delta @ precision * delta  sum(dim=1)  sqrt
            dist = torch.sqrt(torch.clamp((delta @ precision * delta).sum(dim=1), min=0.0))
            result[mask] = dist
        return result

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
"""

        h, z = self._shared_encode(batch)
        intent = batch["intent_id"].long()

        #  R_cons 

        # 1. d_proto:  z  p 
        # 2. d_attr:  L1 
        p = self.prototype_bank[intent] if self.use_prototype else torch.zeros_like(z)
        r = self._obs_vector(batch)
        r_hat = self.obs_predictor(torch.cat([z, p], dim=-1))
        d_proto = (z - p).pow(2).sum(dim=-1)
        d_attr  = torch.abs(r - r_hat).sum(dim=-1) / float(self.obs_dim)
        alpha = torch.nn.functional.softplus(self.alpha_raw)
        r_cons = alpha[0] * d_proto + alpha[1] * d_attr
        if not self.use_consistency:
            r_cons = torch.zeros_like(r_cons)

        #  R_seq 
        h_seq = self._sequence_encode(batch["history_seq"])
        r_seq = self._mahalanobis(h_seq, intent)

        #  R_rule 
        violations = self._rule_violations(batch)
        gamma = torch.nn.functional.softplus(self.gamma_raw)
        r_rule = (violations * gamma.unsqueeze(0)).sum(dim=-1)
        if not self.use_rule:
            r_rule = torch.zeros_like(r_rule)


        if self.decision_mode == "unified":
            # Unified mode:  logits
            #  Unified-Multimodal baseline
            logits = self.unified_head(torch.cat([h, h_seq, violations], dim=-1))
            probs  = torch.softmax(logits, dim=-1)
            levels = torch.tensor([0.0, 0.5, 1.0], device=probs.device, dtype=probs.dtype)
            unified_score = probs @ levels
            r_cons, r_seq, r_rule = unified_score, torch.zeros_like(unified_score), torch.zeros_like(unified_score)
        else:
            # Decomposed mode: 
            #  = [, , ]
            #  r_cons/r_seq/r_rule3
            #  violations5 aux_head 
            logits = self.aux_head(torch.cat([
                h, h_seq,
                r_cons.unsqueeze(-1), r_seq.unsqueeze(-1), r_rule.unsqueeze(-1),
            ], dim=-1))
            unified_score = torch.zeros_like(r_cons)

        return {
            "h_shared": h, "z": z, "prototype": p,
            "r_obs": r, "r_hat": r_hat, "d_proto": d_proto, "d_attr": d_attr,
            "R_cons": r_cons, "h_seq": h_seq, "R_seq": r_seq,
            "violations": violations, "R_rule": r_rule,
            "R_unified": unified_score, "logits": logits,
        }

    @torch.no_grad()
    def update_reference_banks(
        self, z: torch.Tensor, h_seq: torch.Tensor,
        intent_id: torch.Tensor, normal_mask: torch.Tensor,
        momentum: float,
    ) -> None:
        """
"""
        if normal_mask.sum() == 0 or (not self.use_prototype and not self.use_sequence):
            return
        z_n = z[normal_mask]
        h_n = h_seq[normal_mask]
        g_n = intent_id[normal_mask]

        for cls in g_n.unique():
            idx = int(cls.item())
            cm = g_n == cls
            z_mean = z_n[cm].mean(dim=0)
            h_cls  = h_n[cm]
            h_mean = h_cls.mean(dim=0)
            centered = h_cls - h_mean
            #  * lambda_sigma
            if centered.shape[0] > 1:
                cov = (centered.transpose(0, 1) @ centered) / float(centered.shape[0] - 1) \
                      + self.lambda_sigma * torch.eye(self.d_model, device=h_cls.device, dtype=h_cls.dtype)
            else:
                cov = torch.eye(self.d_model, device=h_cls.device, dtype=h_cls.dtype) * self.lambda_sigma

            if self.use_prototype:
                if self.prototype_counts[idx] <= 0:
                    self.prototype_bank[idx] = z_mean
                else:
                    self.prototype_bank[idx] = momentum * self.prototype_bank[idx] + (1 - momentum) * z_mean
                self.prototype_counts[idx] += cm.sum().float()

            if self.use_sequence:
                if self.seq_counts[idx] <= 0:
                    self.seq_mu_bank[idx] = h_mean
                    self.seq_cov_bank[idx] = cov
                else:
                    self.seq_mu_bank[idx]  = momentum * self.seq_mu_bank[idx]  + (1 - momentum) * h_mean
                    self.seq_cov_bank[idx] = momentum * self.seq_cov_bank[idx] + (1 - momentum) * cov
                self.seq_counts[idx] += cm.sum().float()

    @staticmethod
    def normalize_component(values: torch.Tensor, low: float, high: float, eps: float) -> torch.Tensor:
        """
"""
        return (values - low) / max(high - low, eps)
