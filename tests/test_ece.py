import numpy as np
import pytest
from src.utils.ece import ece_from_logits

def test_ece_handles_empty_bins():
    #construct logits so all confidences fall into a single high-confidence bin
    #most bins will be empty; function must not crash and must return finite ECE
    N, C = 200, 5
    logits = np.zeros((N, C), dtype=np.float64)
    logits[:, 0] = 20.0  # very confident class 0
    y_true = np.zeros(N, dtype=np.int64)      # all correct

    res = ece_from_logits(logits, y_true, n_bins=15)
    assert np.isfinite(res.ece)
    assert 0.0 <= res.ece <= 1.0
    #if all predictions are correct and confidence ~1, ECE should be near 0
    assert res.ece < 1e-3

def test_ece_extreme_logits_finite():
    #extreme logits shouldn't overflow or be NaN
    #mix correct and incorrect to make ECE non-trivial
    logits = np.array([
        [1000.0, -1000.0, -1000.0], #pred=0, conf~1
        [1000.0, -1000.0, -1000.0], #pred=0, conf~1
        [-1000.0, 1000.0, -1000.0], #pred=1, conf~1
        [-1000.0, 1000.0, -1000.0], #pred=1, conf~1
    ], dtype=np.float64)

    y_true = np.array([0, 1, 1, 0], dtype=np.int64) #2 correct, 2 wrong

    res = ece_from_logits(logits, y_true, n_bins=15)
    assert np.isfinite(res.ece)
    assert 0.0 <= res.ece <= 1.0
    #with conf~1 but only 50% accuracy, ECE should be ~0.5
    assert abs(res.ece - 0.5) < 0.05
