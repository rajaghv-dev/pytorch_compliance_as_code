"""Tests for CompliancePrivacyEngine (Opacus bridge) — GDPR Article 25."""

import pytest

pytest.importorskip("opacus", reason="opacus not installed")


class _MockPrivacyEngine:
    """Minimal mock that mimics opacus.PrivacyEngine interface."""

    def get_epsilon(self, delta: float) -> float:
        return 2.5


def test_wrapper_creation():
    from torchcomply.integrations.opacus_bridge import CompliancePrivacyEngine

    cpe = CompliancePrivacyEngine(_MockPrivacyEngine(), regulations=["gdpr_art_25"])
    summary = cpe.get_compliance_summary()
    assert "type" in summary
    assert summary["type"] == "differential_privacy"
    assert "gdpr_art_25" in summary["regulations"]


def test_epsilon_tracking():
    from torchcomply.integrations.opacus_bridge import CompliancePrivacyEngine

    cpe = CompliancePrivacyEngine(_MockPrivacyEngine())
    eps = cpe.get_epsilon(delta=1e-5)
    assert abs(eps - 2.5) < 1e-6


def test_step_counter():
    from torchcomply.integrations.opacus_bridge import CompliancePrivacyEngine

    cpe = CompliancePrivacyEngine(_MockPrivacyEngine())
    for _ in range(5):
        cpe.step()
    assert cpe.get_compliance_summary()["n_steps"] == 5


def test_log_to_dict():
    from torchcomply.integrations.opacus_bridge import CompliancePrivacyEngine

    cpe = CompliancePrivacyEngine(_MockPrivacyEngine())
    d = cpe.log_to_dict()
    assert "gdpr_article_25" in d
    assert "framework" in d
    assert d["framework"] == "opacus"
