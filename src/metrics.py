from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

def accuracy_top1(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    return float((preds == y_true).mean())

def macro_f1(logits: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    preds = logits.argmax(axis=1)
    return float(f1_score(y_true, preds, average="macro", labels=list(range(num_classes))))