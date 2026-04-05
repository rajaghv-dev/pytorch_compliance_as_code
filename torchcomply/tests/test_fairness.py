"""Tests for FairnessGate and demographic parity — EU AI Act Article 10."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchcomply.core.fairness import (
    ComplianceViolation,
    FairnessGate,
    compute_demographic_parity,
)


def test_parity_computation_perfect():
    """Equal positive rates → 0 parity."""
    # Group 0 (attrs=0): indices 0,2,4 → preds [1,0,1] = 2/3 positive
    # Group 1 (attrs=1): indices 1,3,5 → preds [1,0,1] = 2/3 positive → disparity=0
    preds = torch.tensor([1, 1, 0, 0, 1, 1])
    attrs = torch.tensor([0, 1, 0, 1, 0, 1])
    assert abs(compute_demographic_parity(preds, attrs)) < 1e-6


def test_parity_computation_known():
    """Group 0: 2/3 positive rate. Group 1: 0/3 positive rate → disparity 2/3."""
    preds = torch.tensor([1, 1, 0, 0, 0, 0])
    attrs = torch.tensor([0, 0, 0, 1, 1, 1])
    dp = compute_demographic_parity(preds, attrs)
    assert abs(dp - 2 / 3) < 1e-5


def _make_loader(disparity: float = 0.0, n: int = 200):
    """Build a val loader with controllable disparity."""
    torch.manual_seed(0)
    x = torch.randn(n, 10)
    # group 0: 50% positive, group 1: (50% + disparity)% positive
    y0 = (torch.rand(n // 2) < 0.5).long()
    y1 = (torch.rand(n // 2) < (0.5 + disparity)).long()
    y = torch.cat([y0, y1])
    g = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)])
    # dataset returns (x, y, g)
    ds = TensorDataset(x, y, g)
    return DataLoader(ds, batch_size=64)


def _make_model():
    return nn.Sequential(nn.Linear(10, 2))


def test_gate_passes():
    model = _make_model()
    loader = _make_loader(disparity=0.0)
    gate = FairnessGate(threshold=0.20)
    gate.on_epoch_end(model, loader, epoch=0, protected_attr_idx=2)
    assert gate.log[-1]["status"] == "passed"


def test_gate_blocks():
    """Model that perfectly separates groups produces maximum disparity → gate blocks."""
    n = 100
    # Group 0: feature[:,0] strongly negative; Group 1: feature[:,0] strongly positive
    x0 = torch.randn(n, 10) * 0.01 - 10.0
    x1 = torch.randn(n, 10) * 0.01 + 10.0
    x = torch.cat([x0, x1])
    y = torch.zeros(2 * n, dtype=torch.long)
    g = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    from torch.utils.data import TensorDataset

    loader = DataLoader(TensorDataset(x, y, g), batch_size=64, shuffle=False)

    # Weight the model so class-1 score = x[:,0] → group 0 → pred 0, group 1 → pred 1
    model = _make_model()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[1, 0] = 1.0  # class-1 logit = x[:,0]
        model[0].bias.zero_()

    gate = FairnessGate(threshold=0.10)
    with pytest.raises(ComplianceViolation) as exc_info:
        gate.on_epoch_end(model, loader, epoch=0, protected_attr_idx=2)
    assert exc_info.value.disparity > exc_info.value.threshold


def test_log_recording():
    model = _make_model()
    loader_ok = _make_loader(disparity=0.0)
    gate = FairnessGate(threshold=0.90)
    gate.on_epoch_end(model, loader_ok, epoch=0, protected_attr_idx=2)
    gate.on_epoch_end(model, loader_ok, epoch=1, protected_attr_idx=2)
    assert len(gate.get_log()) == 2
    assert gate.get_log()[0]["epoch"] == 0
    assert gate.get_log()[1]["epoch"] == 1
