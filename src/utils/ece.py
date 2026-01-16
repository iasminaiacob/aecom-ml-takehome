from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class ECEResult:
    ece: float
    bin_edges: np.ndarray #shape [M+1]
    bin_counts: np.ndarray #shape [M]
    bin_acc: np.ndarray #shape [M]
    bin_conf: np.ndarray #shape [M]

def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    m = np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits - m)
    return exps / np.sum(exps, axis=1, keepdims=True)

def ece_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> ECEResult:
    """
    Expected Calibration Error (ECE) with equal-width bins over confidence in [0,1].
    Uses:
      conf_i = max_k p(y=k | x_i)
      pred_i = argmax_k p(y=k | x_i)
      correct_i = 1[pred_i == y_true_i]

    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|

    Must handle empty bins (count=0) by skipping their contribution.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")

    logits = np.asarray(logits)
    y_true = np.asarray(y_true)

    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if y_true.ndim != 1 or y_true.shape[0] != logits.shape[0]:
        raise ValueError("y_true must have shape [N] and match logits N")

    probs = _softmax_np(logits)
    conf = np.max(probs, axis=1) #[N]
    pred = np.argmax(probs, axis=1) #[N]
    correct = (pred == y_true).astype(np.float64)

    #bins: [0,1] split into n_bins equal width
    #include 1.0 in last bin
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    #digitize returns bin index in 1..n_bins; right=True makes edge behavior sane
    bin_idx = np.digitize(conf, bin_edges[1:], right=True)  #values 0..n_bins-1
    #bin_idx in [0, n_bins-1]
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)

    N = float(len(conf))
    ece = 0.0

    for b in range(n_bins):
        mask = (bin_idx == b)
        cnt = int(np.sum(mask))
        bin_counts[b] = cnt
        if cnt == 0:
            bin_acc[b] = np.nan
            bin_conf[b] = np.nan
            continue

        acc_b = float(np.mean(correct[mask]))
        conf_b = float(np.mean(conf[mask]))
        bin_acc[b] = acc_b
        bin_conf[b] = conf_b

        ece += (cnt / N) * abs(acc_b - conf_b)

    #ece is guaranteed in [0,1]
    ece = float(np.clip(ece, 0.0, 1.0))

    return ECEResult(
        ece=ece,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bin_acc=bin_acc,
        bin_conf=bin_conf,
    )