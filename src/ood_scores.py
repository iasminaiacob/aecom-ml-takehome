from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

def ood_score_msp(logits: torch.Tensor) -> torch.Tensor:
    """
    MSP OOD score: lower max softmax prob => more OOD.
    Return a score where higher means "more OOD": 1 - max_softmax.
    """
    probs = torch.softmax(logits, dim=1)
    msp = probs.max(dim=1).values
    return 1.0 - msp

def ood_score_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Energy score (Liu et al.): E(x) = -T * logsumexp(logits/T)
    For OOD detection, higher energy tends to indicate OOD.
    Return 'energy' as higher=more OOD.
    """
    t = float(temperature)
    return torch.logsumexp(logits / t, dim=1)

@dataclass
class MahalanobisStats:
    class_means: torch.Tensor #[C, D]
    precision: torch.Tensor #[D, D] inverse covariance

class FeatureHook:
    def __init__(self):
        self.features: Optional[torch.Tensor] = None

    def __call__(self, module: nn.Module, inp, out):
        #flatten to [B, D]
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 2:
            out = torch.flatten(out, start_dim=1)
        self.features = out.detach()

def _get_penultimate_hook(model: nn.Module) -> Tuple[FeatureHook, torch.utils.hooks.RemovableHandle]:
    """
    Try to attach to a reasonable "penultimate" layer for common torchvision/timm models.
    - torchvision resnet: avgpool output
    - torchvision mobilenet: classifier pre-last is harder; we hook on features pooling if present
    - timm: often has global_pool; hook there if present
    """
    hook = FeatureHook()

    #torchvision ResNet
    if hasattr(model, "avgpool"):
        handle = model.avgpool.register_forward_hook(hook)
        return hook, handle

    #timm models often have global_pool
    if hasattr(model, "global_pool"):
        handle = model.global_pool.register_forward_hook(hook)
        return hook, handle

    #fallback: last module
    last = list(model.modules())[-1]
    handle = last.register_forward_hook(hook)
    return hook, handle

@torch.no_grad()
def fit_mahalanobis(
    model: nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    reg: float = 1e-4,
    max_samples: int = 8000,
) -> MahalanobisStats:
    """
    Fit class-conditional Gaussians in feature space with shared covariance.
    Uses penultimate features via a forward hook.
    """
    model.eval()
    hook, handle = _get_penultimate_hook(model)

    feats_list = []
    labels_list = []
    seen = 0

    for x, y in loader:
        x = x.to(device)
        _ = model(x)

        feats = hook.features
        if feats is None:
            raise RuntimeError("Feature hook did not capture features.")
        feats = feats.detach().cpu()

        y = y.detach().cpu()
        feats_list.append(feats)
        labels_list.append(y)

        seen += len(y)
        if seen >= max_samples:
            break

    handle.remove()

    feats_all = torch.cat(feats_list, dim=0)[:max_samples] #[N,D]
    y_all = torch.cat(labels_list, dim=0)[:max_samples] #[N]

    D = feats_all.shape[1]
    C = num_classes

    class_means = torch.zeros((C, D), dtype=torch.float32)
    counts = torch.zeros((C,), dtype=torch.int64)

    for c in range(C):
        mask = (y_all == c)
        if mask.any():
            class_means[c] = feats_all[mask].mean(dim=0)
            counts[c] = int(mask.sum())

    #shared covariance
    centered = feats_all - class_means[y_all]
    cov = (centered.T @ centered) / max(1, feats_all.shape[0] - 1)
    cov = cov + reg * torch.eye(D)

    precision = torch.linalg.inv(cov)

    return MahalanobisStats(class_means=class_means, precision=precision)

@torch.no_grad()
def ood_score_mahalanobis(
    model: nn.Module,
    loader,
    stats: MahalanobisStats,
    device: torch.device,
) -> np.ndarray:
    """
    Score OOD by negative minimum Mahalanobis distance:
    Higher score => more OOD.
    We compute min distance to any class mean, then return +min_dist as OOD score.
    """
    model.eval()
    hook, handle = _get_penultimate_hook(model)

    means = stats.class_means.to(device)      # [C,D]
    precision = stats.precision.to(device)    # [D,D]

    all_scores = []

    for x, _ in loader:
        x = x.to(device)
        _ = model(x)

        feats = hook.features
        if feats is None:
            raise RuntimeError("Feature hook did not capture features.")
        feats = feats.detach() #[B,D] on device

        #compute distances to each class mean
        #dist(c) = (f-mu_c)^T P (f-mu_c)
        diff = feats[:, None, :] - means[None, :, :] #[B,C,D]
        #(diff @ P) dot diff
        tmp = torch.einsum("bcd,dd->bcd", diff, precision)
        dist = torch.einsum("bcd,bcd->bc", tmp, diff) #[B,C]
        min_dist = dist.min(dim=1).values #[B]

        all_scores.append(min_dist.detach().cpu().numpy())

    handle.remove()
    return np.concatenate(all_scores, axis=0)