"""Tests for AnnexIVReport and ModelIntrospector — EU AI Act Annex IV."""

import os
import tempfile

import pytest
import torch.nn as nn

pytest.importorskip("reportlab", reason="reportlab not installed")

from torchcomply.core.audit import AuditChain
from torchcomply.reports.annex_iv import AnnexIVReport, ModelIntrospector


def _mock_chain() -> AuditChain:
    import time
    from torchcomply.core.audit import AuditEntry

    chain = AuditChain()
    prev = ""
    for i in range(3):
        e = AuditEntry(
            timestamp=time.time_ns(),
            module_name=f"layer_{i}",
            operator_type="Linear",
            input_shapes=[[1, 10]],
            output_shape=(1, 5),
            output_hash="",
            device="cpu",
            prev_hash=prev,
        )
        chain.append(e)
        prev = e.hash
    return chain


def test_model_introspector():
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))
    mi = ModelIntrospector(model)
    assert mi.architecture == "Sequential"
    assert mi.total_params > 0
    assert mi.trainable_params > 0
    assert len(mi.layers) == 3  # 2 Linear + 1 ReLU (all leaves)


def test_pdf_generated():
    model = nn.Sequential(nn.Linear(10, 4), nn.ReLU())
    mi = ModelIntrospector(model)
    chain = _mock_chain()
    fairness_log = [{"epoch": 0, "disparity": 0.05, "threshold": 0.10, "status": "passed"}]
    training_config = {"optimizer": "Adam", "lr": 0.001, "epochs": 5}
    dataset_info = {"dataset_name": "IMDB", "size": 200}

    report = AnnexIVReport(
        model_introspection=mi,
        audit_chain=chain,
        fairness_log=fairness_log,
        training_config=training_config,
        dataset_info=dataset_info,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name

    try:
        report.save_pdf(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # non-trivial PDF
    finally:
        os.unlink(path)
