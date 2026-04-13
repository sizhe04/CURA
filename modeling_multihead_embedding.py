from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            self.net = nn.Linear(input_dim, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadBinaryOnEmbedding(nn.Module):
    """
    Stage-2: Multi-head classifier operating on frozen embeddings.

    Loss = L_base + L_ind + L_coh  (Equation 6 in the paper)
      - L_base: class-weighted CE (or neighbor-aware soft-label CE when lambda_pop > 0)
      - L_ind:  individual uncertainty calibration (lambda_cal > 0)
      - L_coh:  cohort-aware risk alignment via neighbor soft labels (lambda_pop > 0)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        head_hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
        lambda_cal: float = 0.0,
        lambda_pop: float = 0.0,
        cal_alpha: float = 1.0,
        u_min: float = 0.05,
        u_max: float = 0.95,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.lambda_cal = float(lambda_cal)
        self.lambda_pop = float(lambda_pop)
        self.cal_alpha = float(cal_alpha)
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.heads = nn.ModuleList([MLPHead(self.input_dim, head_hidden_dim, dropout) for _ in range(self.num_heads)])
        self.class_weights: Optional[torch.Tensor] = None

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = class_weights.detach().clone().float()

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        neighbor_q: Optional[torch.Tensor] = None,
        neighbor_hhat: Optional[torch.Tensor] = None,
        neighbor_w: Optional[torch.Tensor] = None,
        neighbor_t: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        head_outputs: List[torch.Tensor] = []
        for head in self.heads:
            head_outputs.append(head(features))
        logits = torch.stack(head_outputs, dim=1)  # (b, K, 2)
        out: Dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            labels = labels.long().view(-1)

            # L_base: CE or neighbor-aware CE (L_base + L_coh combined)
            ce = logits.new_tensor(0.0)
            use_neighbor = (
                self.lambda_pop > 0.0
                and (neighbor_t is not None)
                and (neighbor_w is not None)
            )
            if use_neighbor:
                probs_heads = torch.softmax(logits, dim=-1)[..., 1]  # (b, K)
                p_bar = probs_heads.mean(dim=1)  # (b,)
                p_safe = p_bar.clamp(min=1e-8, max=1.0 - 1e-8)
                t = neighbor_t.float().view(-1).to(logits.device)
                w_pop = neighbor_w.float().view(-1).to(logits.device)
                if self.class_weights is not None:
                    cw = self.class_weights.to(logits.device)
                    w0 = cw[0]
                    w1 = cw[1]
                else:
                    w0 = logits.new_tensor(1.0)
                    w1 = logits.new_tensor(1.0)
                ce_vec = -(w1 * t * torch.log(p_safe) + w0 * (1.0 - t) * torch.log(1.0 - p_safe))
                ce = ((1.0 + w_pop) * ce_vec).mean()
            else:
                w = self.class_weights.to(logits.device) if self.class_weights is not None else None
                for k in range(self.num_heads):
                    ce = ce + F.cross_entropy(logits[:, k, :], labels, weight=w, reduction="mean")
                ce = ce / float(self.num_heads)

            # L_ind: individual uncertainty calibration
            cal_loss = logits.new_tensor(0.0)
            if self.lambda_cal > 0.0:
                probs_heads = torch.softmax(logits, dim=-1)[..., 1]  # (b, K)
                p_bar = probs_heads.mean(dim=1)  # (b,)
                y_float = labels.float()
                a = y_float * p_bar + (1.0 - y_float) * (1.0 - p_bar)
                eps = 1e-8
                p_safe = p_bar.clamp(min=eps, max=1 - eps)
                H = -(p_safe * torch.log(p_safe) + (1.0 - p_safe) * torch.log(1.0 - p_safe))
                u_raw = torch.tanh(self.cal_alpha * H)
                u = u_raw.clamp(min=self.u_min, max=self.u_max)
                u_safe = u.clamp(min=eps, max=1 - eps)
                cal_ce = -((1.0 - a) * torch.log(u_safe) + a * torch.log(1.0 - u_safe))
                cal_loss = self.lambda_cal * cal_ce.mean()

            out["loss"] = ce + cal_loss
            out["loss_ce"] = ce.detach()
            if self.lambda_cal > 0.0:
                out["loss_lcal"] = cal_loss.detach()

        return out

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.stack([head(features) for head in self.heads], dim=1)  # (b, K, 2)
        probs = torch.softmax(logits, dim=-1)[..., 1]
        mean = probs.mean(dim=1)
        var = probs.var(dim=1, unbiased=False)
        return mean, var
