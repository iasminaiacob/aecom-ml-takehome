import pytest
import torch
from src.utils.losses import GCELoss, effective_number_class_weights

def test_gce_loss_invalid_q_raises():
    with pytest.raises(ValueError):
        GCELoss(q=0.0)
    with pytest.raises(ValueError):
        GCELoss(q=1.1)

def test_gce_loss_invalid_reduction_raises():
    with pytest.raises(ValueError):
        GCELoss(reduction="avg")

def test_gce_loss_output_shape_none_reduction():
    torch.manual_seed(0)
    loss_fn = GCELoss(q=0.7, reduction="none")
    logits = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(logits, targets)
    assert loss.shape == (4,)

def test_gce_loss_reduction_mean_is_scalar():
    torch.manual_seed(0)
    loss_fn = GCELoss(q=0.7, reduction="mean")
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  #scalar

def test_gce_loss_weight_increases_loss_for_weighted_class():
    """
    Iif we upweight one class and all samples are that class,
    the weighted loss should be larger than the unweighted loss.
    """
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    targets = torch.zeros(6, dtype=torch.long) #all class 0

    unweighted = GCELoss(q=0.7, reduction="mean")
    w = torch.tensor([10.0, 1.0, 1.0]) #upweight class 0
    weighted = GCELoss(q=0.7, reduction="mean", weight=w)

    loss_unw = unweighted(logits, targets).item()
    loss_w = weighted(logits, targets).item()

    assert loss_w > loss_unw

def test_effective_number_weights_shape_and_mean_one():
    counts = torch.tensor([100, 50, 10])
    w = effective_number_class_weights(counts, beta=0.9999)
    assert w.shape == counts.shape
    #normalized to mean ~ 1
    assert torch.isclose(w.mean(), torch.tensor(1.0), atol=1e-5)

def test_effective_number_weights_rare_class_gets_higher_weight():
    counts = torch.tensor([1000, 10])
    w = effective_number_class_weights(counts, beta=0.9999)
    #class with 10 samples should get larger weight than class with 1000
    assert w[1] > w[0]

def test_effective_number_weights_raises_on_nonpositive_counts():
    with pytest.raises(ValueError):
        effective_number_class_weights(torch.tensor([10, 0, 5]))
    with pytest.raises(ValueError):
        effective_number_class_weights(torch.tensor([-1, 5, 10]))