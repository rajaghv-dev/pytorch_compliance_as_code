"""Tests for ComplianceSecureInference (CrypTen) — GDPR Art. 25."""

import pytest
import torch
import torch.nn as nn

crypten = pytest.importorskip("crypten", reason="crypten not installed")

from torchcomply.integrations.crypten_bridge import ComplianceSecureInference  # noqa: E402


def _small_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 4))


def test_secure_predict_shape():
    model = _small_model()
    dummy = torch.randn(1, 10)
    secure = ComplianceSecureInference(model, dummy)
    x = torch.randn(1, 10)
    out = secure.secure_predict(x)
    assert out.shape == (1, 4)


def test_output_close_to_standard():
    """Encrypted output must be numerically close to plaintext output."""
    torch.manual_seed(1)
    model = _small_model().eval()
    dummy = torch.randn(1, 10)
    secure = ComplianceSecureInference(model, dummy)

    x = torch.randn(1, 10)
    with torch.no_grad():
        standard = model(x)
    encrypted = secure.secure_predict(x)

    # MPC introduces small numerical noise — within 1e-2 is acceptable
    assert torch.allclose(
        standard, encrypted, atol=1e-2
    ), f"Max diff: {(standard - encrypted).abs().max().item()}"


def test_inference_log_populated():
    model = _small_model()
    secure = ComplianceSecureInference(model, torch.randn(1, 10))
    secure.secure_predict(torch.randn(1, 10))
    log = secure.get_log()
    assert len(log) == 1
    assert "protocol" in log[0]
    assert log[0]["protocol"] == "SecretShare_MPC"
    assert "encryption_time_ms" in log[0]
