from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCELoss(nn.Module):
    """
    Generalized Cross Entropy (GCE) loss.

    Paper: Zhang & Sabuncu (2018)
    L_q = (1 - p_y^q) / q, where p_y is predicted prob of true class.
    As q -> 0, approaches CE; for larger q, more robust to label noise.

    Args:
        q: robustness parameter in (0, 1]. Typical: 0.7
        reduction: 'mean'|'sum'|'none'
        weight: optional class weights (like nn.CrossEntropyLoss)
    """
    def __init__(self, q: float = 0.7, reduction: str = "mean", weight: Optional[torch.Tensor] = None):
        super().__init__()
        if q <= 0 or q > 1:
            raise ValueError("q must be in (0, 1].")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be mean|sum|none")
        self.q = float(q)
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B,C], targets: [B]
        probs = F.softmax(logits, dim=1)
        p = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # [B]
        # Clamp to avoid nan gradients at 0
        p = torch.clamp(p, min=1e-8, max=1.0)

        loss = (1.0 - p.pow(self.q)) / self.q  # [B]

        # Optional class weighting
        if self.weight is not None:
            w = self.weight.gather(0, targets)  # [B]
            loss = loss * w

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def effective_number_class_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    """
    Effective number of samples reweighting.
    w_c ∝ (1 - beta) / (1 - beta^{n_c})

    Args:
        counts: [C] tensor of class sample counts (must be >0)
        beta: close to 1.0; typical 0.9999 for large datasets
    """
    counts = counts.to(torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("All class counts must be > 0 for effective number weights.")

    eff_num = 1.0 - torch.pow(torch.tensor(beta, device=counts.device), counts)
    weights = (1.0 - beta) / eff_num
    # normalize weights to mean=1 for stable loss scale
    weights = weights / weights.mean()
    return weights
