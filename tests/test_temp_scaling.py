import numpy as np
import torch
from src.utils.calibrate import fit_temperature, apply_temperature

def test_temperature_scaling_extreme_logits_finite():
    #logits with extreme magnitudes should not create NaNs during fit
    logits = torch.tensor([
        [1000.0, -1000.0, -1000.0],
        [1000.0, -1000.0, -1000.0],
        [-1000.0, 1000.0, -1000.0],
        [-1000.0, 1000.0, -1000.0],
    ], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    res = fit_temperature(logits, y, device="cpu", max_iter=50)
    assert np.isfinite(res.temperature)
    assert res.temperature > 0
    assert np.isfinite(res.nll_before)
    assert np.isfinite(res.nll_after)

def test_apply_temperature_changes_logits():
    logits = np.array([[1.0, 2.0]], dtype=np.float64)
    scaled = apply_temperature(logits, temperature=2.0)
    assert np.allclose(scaled, logits / 2.0)