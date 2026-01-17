import numpy as np
import torch
from src.ood_scores import ood_score_msp, ood_score_energy, fit_mahalanobis, ood_score_mahalanobis

def test_msp_score_higher_for_uncertain():
    # confident in-dist: max prob ~1 => score ~0
    logits_id = torch.tensor([[10.0, -10.0], [12.0, -12.0]])
    # uncertain/ood-ish: probs ~0.5 => score ~0.5
    logits_ood = torch.tensor([[0.0, 0.0], [0.1, -0.1]])

    s_id = ood_score_msp(logits_id).numpy()
    s_ood = ood_score_msp(logits_ood).numpy()
    assert s_ood.mean() > s_id.mean()

def test_energy_finite_extreme_logits():
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], dtype=torch.float32)
    s = ood_score_energy(logits, temperature=1.0).numpy()
    assert np.isfinite(s).all()

def test_energy_higher_for_uncertain():
    #confident ID
    logits_id = torch.tensor([[10.0, -10.0]])
    #uncertain
    logits_ood = torch.tensor([[0.0, 0.0]])

    s_id = ood_score_energy(logits_id, temperature=1.0).item()
    s_ood = ood_score_energy(logits_ood, temperature=1.0).item()

    #higher energy = more OOD
    assert s_ood > s_id

def test_msp_output_shape_matches_batch():
    logits = torch.randn(7, 5)
    scores = ood_score_msp(logits)
    assert scores.shape == (7,)

def test_mahalanobis_scores_finite(dummy_model, dummy_loader, device):
    stats = fit_mahalanobis(
        model=dummy_model,
        loader=dummy_loader,
        num_classes=3,
        device=device,
        max_samples=20,
    )

    scores = ood_score_mahalanobis(dummy_model, dummy_loader, stats, device)
    assert np.isfinite(scores).all()
    assert len(scores) == len(dummy_loader.dataset)