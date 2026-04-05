"""Tests for ComplianceTensor (__torch_function__ dispatcher) — EU AI Act Art. 12."""

import torch
import torch.nn as nn

from torchcomply.core.dispatcher_hooks import ComplianceTensor


def test_log_captures_matmul():
    ComplianceTensor.clear_log()
    a = ComplianceTensor(torch.randn(4, 8))
    b = torch.randn(8, 4)
    _ = a @ b
    log = ComplianceTensor.get_log()
    assert len(log) > 0


def test_log_has_required_fields():
    ComplianceTensor.clear_log()
    x = ComplianceTensor(torch.randn(2, 4))
    _ = x + x
    log = ComplianceTensor.get_log()
    entry = log[0]
    assert "operator" in entry
    assert "timestamp" in entry
    assert "input_shapes" in entry


def test_model_forward_logged():
    ComplianceTensor.clear_log()
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 4))
    x = ComplianceTensor(torch.randn(3, 10))
    _ = model(x)
    log = ComplianceTensor.get_log()
    assert len(log) >= 3  # at least one op per layer


def test_clear_log():
    x = ComplianceTensor(torch.randn(2, 2))
    _ = x + x
    assert len(ComplianceTensor.get_log()) > 0
    ComplianceTensor.clear_log()
    assert len(ComplianceTensor.get_log()) == 0


def test_output_numerically_correct():
    """ComplianceTensor must not alter computation results."""
    torch.manual_seed(0)
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    expected = a @ b
    ComplianceTensor.clear_log()
    ca = ComplianceTensor(a)
    result = ca @ b
    assert torch.allclose(result, expected, atol=1e-6)
