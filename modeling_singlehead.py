from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


def edl_mse_loss(output: torch.Tensor, target: torch.Tensor, num_classes: int = 2, kl_weight: float = 0.1) -> torch.Tensor:
    """
    Evidential Deep Learning (EDL) MSE loss; matches `edl_mse_loss` in the v1 project file
    250822_finetuning_EDL_nofocal_loss.py exactly.
    """
    evidence = F.softplus(output)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S
    one_hot = F.one_hot(target, num_classes=num_classes).float().to(output.device)
    mse = torch.sum((one_hot - probs) ** 2, dim=1, keepdim=True)
    var = torch.sum(probs * (1.0 - probs), dim=1, keepdim=True)
    loglikelihood = mse + var

    kl_div = _kl_divergence_cpu_safe(alpha, num_classes)
    loss = loglikelihood + kl_weight * kl_div
    return loss.mean()


def _kl_divergence_cpu_safe(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Same as v1 `_kl_divergence_cpu_safe`: compute KL on CPU, then move the result back to the original device.
    """
    device = alpha.device
    alpha_cpu = alpha.cpu()
    beta_cpu = torch.ones((1, num_classes), dtype=torch.float32)

    S_alpha = torch.sum(alpha_cpu, dim=1, keepdim=True)
    S_beta = torch.sum(beta_cpu, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_cpu), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta_cpu), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha_cpu)

    kl = torch.sum((alpha_cpu - beta_cpu) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl.to(device)


class SingleHeadBinaryClassifier(nn.Module):
    """
    Stage-1: HF backbone + single prediction head; supports CE and EDL losses.
    - When loss_type='ce': equivalent to v1 cross-entropy training (F.cross_entropy + class_weights).
    - When loss_type='edl': uses the same EDL MSE loss as v1 250822_finetuning_EDL_nofocal_loss.py.

    Pooling: 'cls' | 'pooler' | 'mean' | 'last_hidden_state' | 'last_token'
    - 'cls': use the first token's representation ([CLS] for BERT-style models)
    - 'pooler': prefer pooler_output when the model provides it
    - 'mean' / 'last_hidden_state': masked mean pooling over last_hidden_state
    - 'last_token': representation of the last non-padding token in the sequence
    """

    def __init__(
        self,
        model_name: str,
        head_hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
        pooling: str = "cls",  # 'cls' | 'pooler' | 'mean' | 'last_hidden_state' | 'last_token'
        loss_type: str = "ce",  # 'ce' | 'edl'
        edl_kl_weight: float = 0.1,
        # Single-head L_cal (K=1 specialization)
        lambda_cal: float = 0.0,
        cal_alpha: float = 1.0,
        u_min: float = 0.05,
        u_max: float = 0.95,
        use_lcal: bool = False,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", None)
        if self.hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model config.")
        self.pooling = pooling
        self.loss_type = loss_type
        self.edl_kl_weight = float(edl_kl_weight)
        # L_cal toggle and hyperparameters
        self.use_lcal = bool(use_lcal)
        self.lambda_cal = float(lambda_cal)
        self.cal_alpha = float(cal_alpha)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        # Head structure: equivalent to BertForSequenceClassification single layer (optional two-layer MLP)
        self.dropout_layer = nn.Dropout(dropout)
        if head_hidden_dim is None or head_hidden_dim <= 0:
            self.head = nn.Linear(self.hidden_size, 2)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 2),
            )
        self.class_weights: Optional[torch.Tensor] = None

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = class_weights.detach().clone().float()

    def _pool(self, outputs, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # pooler_output branch
        if self.pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        last_hidden = outputs.last_hidden_state  # (b, L, H)
        # mean / last_hidden_state: masked mean pooling over last-layer hidden states
        if self.pooling in ("mean", "last_hidden_state"):
            if attention_mask is None:
                return last_hidden.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-8)
            return summed / denom
        # last_token: last non-padding token representation (BERT/GPT compatible)
        if self.pooling == "last_token":
            if attention_mask is not None:
                # attention_mask: (b, L); 1 = real token, 0 = padding
                lengths = attention_mask.long().sum(dim=1)  # (b,)
                # Defensive: if a row is all zeros, fall back to index 0
                lengths = lengths.clamp(min=1)
                idx = lengths - 1  # last non-padding position
                return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx, :]
            # Without attention_mask, use the last position
            return last_hidden[:, -1, :]
        # Default: CLS
        return last_hidden[:, 0, :]

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return pooled representation for embedding extraction. Shape: (batch, hidden)
        """
        if getattr(self.config, "model_type", None) == "longformer" and global_attention_mask is not None:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs, attention_mask)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        global_attention_mask = kwargs.get("global_attention_mask", None)
        if "num_items_in_batch" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "num_items_in_batch"}
        extra = {}
        if getattr(self.config, "model_type", None) == "longformer" and global_attention_mask is not None:
            extra["global_attention_mask"] = global_attention_mask
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **extra)
        pooled = self._pool(outputs, attention_mask)  # (b, h)
        pooled = self.dropout_layer(pooled)
        logits = self.head(pooled)  # (b, 2)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            labels = labels.long().view(-1)
            if self.loss_type == "edl":
                # Match v1 EDL (no class_weights)
                loss = edl_mse_loss(logits, labels, num_classes=2, kl_weight=self.edl_kl_weight)
                out["loss"] = loss
                out["loss_edl"] = loss.detach()
            else:
                w = self.class_weights.to(logits.device) if self.class_weights is not None else None
                ce = F.cross_entropy(logits, labels, weight=w, reduction="mean")
                total = ce
                out["loss_ce"] = ce.detach()
                # Single-head L_cal (K=1): only when using CE and use_lcal is on
                if self.use_lcal and self.lambda_cal > 0.0:
                    # p(x)
                    p = torch.softmax(logits, dim=-1)[:, 1]
                    # H(p)
                    eps = 1e-8
                    p_safe = p.clamp(min=eps, max=1.0 - eps)
                    H = -(p_safe * torch.log(p_safe) + (1.0 - p_safe) * torch.log(1.0 - p_safe))
                    # u(x) = clip(tanh(alpha * H), [u_min, u_max])
                    u_raw = torch.tanh(self.cal_alpha * H)
                    u = u_raw.clamp(min=self.u_min, max=self.u_max)
                    u_safe = u.clamp(min=eps, max=1.0 - eps)
                    # a(x) = y·p + (1−y)·(1−p)
                    y_float = labels.float()
                    a = y_float * p + (1.0 - y_float) * (1.0 - p)
                    # L_cal(x) = λ * [ −(1−a) log u − a log(1−u) ]
                    cal_ce = -((1.0 - a) * torch.log(u_safe) + a * torch.log(1.0 - u_safe))
                    lcal = self.lambda_cal * cal_ce.mean()
                    total = total + lcal
                    out["loss_lcal"] = lcal.detach()
                out["loss"] = total
        return out


class HFSingleHeadBinaryClassifier(nn.Module):
    """
    HuggingFace AutoModelForSequenceClassification (e.g. BertForSequenceClassification) as the
    Stage-1 single-head structure, with CE/EDL on logits matching v1.

    - Reuses HF backbone + classification head;
    - encode() calls the underlying encoder (e.g. .bert) and applies pooling for embeddings.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        loss_type: str = "ce",  # 'ce' | 'edl'
        edl_kl_weight: float = 0.1,
        # Single-head L_cal (K=1 specialization)
        lambda_cal: float = 0.0,
        cal_alpha: float = 1.0,
        u_min: float = 0.05,
        u_max: float = 0.95,
        use_lcal: bool = False,
    ) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.config = self.model.config
        self.hidden_size = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", None)
        if self.hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model config.")
        self.pooling = pooling
        self.loss_type = loss_type
        self.edl_kl_weight = float(edl_kl_weight)
        # L_cal
        self.use_lcal = bool(use_lcal)
        self.lambda_cal = float(lambda_cal)
        self.cal_alpha = float(cal_alpha)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.class_weights: Optional[torch.Tensor] = None

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = class_weights.detach().clone().float()

    def _base_encoder(self):
        # For BERT-style models prefer .bert; else fall back to base_model
        if hasattr(self.model, "bert"):
            return self.model.bert
        if hasattr(self.model, "base_model"):
            return self.model.base_model
        # Last resort: use model directly (may not suit all architectures)
        return self.model

    def _pool(self, outputs, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        last_hidden = outputs.last_hidden_state  # (b, L, H)
        # mean / last_hidden_state: masked mean pooling over last-layer hidden states
        if self.pooling in ("mean", "last_hidden_state"):
            if attention_mask is None:
                return last_hidden.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-8)
            return summed / denom
        # last_token: last non-padding token representation
        if self.pooling == "last_token":
            if attention_mask is not None:
                lengths = attention_mask.long().sum(dim=1)
                lengths = lengths.clamp(min=1)
                idx = lengths - 1
                return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx, :]
            return last_hidden[:, -1, :]
        # Default: CLS
        return last_hidden[:, 0, :]

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder = self._base_encoder()
        if getattr(self.config, "model_type", None) == "longformer" and global_attention_mask is not None:
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
        else:
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs, attention_mask)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        global_attention_mask = kwargs.get("global_attention_mask", None)
        if "num_items_in_batch" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "num_items_in_batch"}
        extra = {}
        if getattr(self.config, "model_type", None) == "longformer" and global_attention_mask is not None:
            extra["global_attention_mask"] = global_attention_mask
        # Do not use HF built-in loss; implement CE/EDL here to match v1
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            **extra,
        )
        logits = outputs.logits
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            labels = labels.long().view(-1)
            if self.loss_type == "edl":
                loss = edl_mse_loss(logits, labels, num_classes=2, kl_weight=self.edl_kl_weight)
                out["loss"] = loss
                out["loss_edl"] = loss.detach()
            else:
                w = self.class_weights.to(logits.device) if self.class_weights is not None else None
                ce = F.cross_entropy(logits, labels, weight=w, reduction="mean")
                total = ce
                out["loss_ce"] = ce.detach()
                if self.use_lcal and self.lambda_cal > 0.0:
                    p = torch.softmax(logits, dim=-1)[:, 1]
                    eps = 1e-8
                    p_safe = p.clamp(min=eps, max=1.0 - eps)
                    H = -(p_safe * torch.log(p_safe) + (1.0 - p_safe) * torch.log(1.0 - p_safe))
                    u_raw = torch.tanh(self.cal_alpha * H)
                    u = u_raw.clamp(min=self.u_min, max=self.u_max)
                    u_safe = u.clamp(min=eps, max=1.0 - eps)
                    y_float = labels.float()
                    a = y_float * p + (1.0 - y_float) * (1.0 - p)
                    cal_ce = -((1.0 - a) * torch.log(u_safe) + a * torch.log(1.0 - u_safe))
                    lcal = self.lambda_cal * cal_ce.mean()
                    total = total + lcal
                    out["loss_lcal"] = lcal.detach()
                out["loss"] = total
        return out
