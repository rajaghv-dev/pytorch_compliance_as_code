"""Tests for AuditChain and compliance hooks — EU AI Act Article 12."""

import time

import pytest
import torch
import torch.nn as nn

from torchcomply.core.audit import (
    AuditChain,
    AuditEntry,
    IntegrityViolation,
    register_compliance_hooks,
)


def _make_entry(chain: AuditChain, name: str = "layer") -> AuditEntry:
    prev = chain.entries[-1].hash if chain.entries else ""
    e = AuditEntry(
        timestamp=time.time_ns(),
        module_name=name,
        operator_type="Linear",
        input_shapes=[[1, 10]],
        output_shape=(1, 5),
        output_hash="",
        device="cpu",
        prev_hash=prev,
    )
    chain.append(e)
    return e


def test_chain_creation():
    chain = AuditChain()
    for i in range(5):
        _make_entry(chain, f"layer_{i}")
    assert len(chain) == 5


def test_hook_registration():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    chain = AuditChain()
    handles = register_compliance_hooks(model, chain)
    assert len(handles) > 0
    x = torch.randn(2, 8)
    _ = model(x)
    # Should have 3 leaf-module entries (Linear, ReLU, Linear)
    assert len(chain) >= 3
    for h in handles:
        h.remove()


def test_chain_integrity():
    chain = AuditChain()
    for i in range(10):
        _make_entry(chain, f"l{i}")
    assert chain.verify() is True


def test_tamper_detection():
    chain = AuditChain()
    for i in range(8):
        _make_entry(chain, f"l{i}")
    # Tamper with entry #3
    chain.entries[3].timestamp = 0
    with pytest.raises(IntegrityViolation) as exc_info:
        chain.verify()
    assert exc_info.value.index == 3


def test_json_serialization():
    chain = AuditChain()
    for i in range(5):
        _make_entry(chain, f"layer_{i}")
    s = chain.to_json()
    chain2 = AuditChain.from_json(s)
    assert len(chain2) == len(chain)
    for a, b in zip(chain.entries, chain2.entries):
        assert a.hash == b.hash
        assert a.module_name == b.module_name


def test_summary_keys():
    chain = AuditChain()
    for i in range(3):
        _make_entry(chain)
    s = chain.summary()
    assert "total_entries" in s
    assert "chain_valid" in s
    assert s["total_entries"] == 3
    assert s["chain_valid"] is True
