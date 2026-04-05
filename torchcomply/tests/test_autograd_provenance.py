"""Tests for ProvenanceLinear (custom Autograd) — GDPR Art. 17."""

import torch

from torchcomply.core.autograd_provenance import ProvenanceLinear


def _run_forward_backward(batch_size: int = 4, in_f: int = 8, out_f: int = 4):
    ProvenanceLinear.clear_log()
    weight = torch.randn(out_f, in_f, requires_grad=True)
    bias = torch.zeros(out_f, requires_grad=True)
    x = torch.randn(batch_size, in_f)
    subject_ids = torch.arange(batch_size, dtype=torch.long)
    out = ProvenanceLinear.apply(x, weight, bias, subject_ids)
    loss = out.sum()
    loss.backward()
    return ProvenanceLinear.get_provenance_log()


def test_provenance_log_populated():
    log = _run_forward_backward()
    assert len(log) == 1  # one backward call


def test_provenance_log_has_subject_ids():
    log = _run_forward_backward(batch_size=5)
    entry = log[0]
    assert "subject_ids" in entry
    assert len(entry["subject_ids"]) == 5


def test_provenance_log_has_grad_norm():
    log = _run_forward_backward()
    entry = log[0]
    assert "grad_norm" in entry
    assert entry["grad_norm"] >= 0.0


def test_forward_numerically_correct():
    """ProvenanceLinear must match nn.Linear output."""
    torch.manual_seed(42)
    weight = torch.randn(4, 8)
    bias = torch.zeros(4)
    x = torch.randn(3, 8)
    sids = torch.tensor([0, 1, 2])

    expected = x @ weight.t() + bias
    ProvenanceLinear.clear_log()
    result = ProvenanceLinear.apply(x, weight, bias, sids)
    assert torch.allclose(result, expected, atol=1e-5)


def test_clear_log():
    _run_forward_backward()
    assert len(ProvenanceLinear.get_provenance_log()) > 0
    ProvenanceLinear.clear_log()
    assert len(ProvenanceLinear.get_provenance_log()) == 0
