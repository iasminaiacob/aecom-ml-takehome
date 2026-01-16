from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TempScaleResult:
    temperature: float
    nll_before: float
    nll_after: float

class TemperatureScaler(nn.Module):
    """
    Learns a single scalar temperature T > 0 such that logits' = logits / T
    Fit by minimizing NLL on validation set.
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        if init_temp <= 0:
            raise ValueError("init_temp must be > 0")
        #optimize log(T) to keep T positive
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature().clamp(min=1e-6)

@torch.no_grad()
def nll_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    loss = F.cross_entropy(logits, y, reduction="mean")
    return float(loss.detach().cpu().item())

def fit_temperature(
    logits: torch.Tensor,
    y: torch.Tensor,
    device: torch.device | str = "cpu",
    init_temp: float = 1.0,
    max_iter: int = 200,
) -> TempScaleResult:
    """
    Fits temperature on given logits/labels by minimizing NLL
    logits: [N,C] (float tensor)
    y: [N] (long tensor)
    Returns fitted T and NLL before/after.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N,C]")
    if y.ndim != 1 or y.shape[0] != logits.shape[0]:
        raise ValueError("y must be [N] and match logits N")

    device = torch.device(device)
    logits = logits.to(device)
    y = y.to(device)

    nll_before = nll_from_logits(logits, y)

    scaler = TemperatureScaler(init_temp=init_temp).to(device)

    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        scaled = scaler(logits)
        loss = F.cross_entropy(scaled, y, reduction="mean")
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        t = float(scaler.temperature().detach().cpu().item())
        scaled_logits = scaler(logits)
        nll_after = nll_from_logits(scaled_logits, y)

    #avoid degenerate
    t = float(max(t, 1e-6))

    return TempScaleResult(temperature=t, nll_before=nll_before, nll_after=nll_after)

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / float(temperature)