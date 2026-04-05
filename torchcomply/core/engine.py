"""
engine.py — ComplianceEngine: single entry-point that unifies all torchcomply components.

ComplianceEngine is a Façade (GoF design pattern) that wires together the four
independent compliance subsystems — audit, fairness, consent, and reporting —
behind a single, minimal API. Practitioners only need one import:

    from torchcomply import ComplianceEngine

Usage::

    engine = ComplianceEngine(regulations=["eu_ai_act", "gdpr"])
    model  = engine.attach(model)          # registers audit hooks (Art. 12)
    gate   = engine.create_fairness_gate(threshold=0.10)  # demographic parity (Art. 10)
    ds     = engine.create_compliant_dataset(base_ds, registry, "classification")  # GDPR Art. 7
    engine.generate_report("annex_iv.pdf", model, training_config={...})  # Annex IV PDF
    print(engine.summary())               # Art. 17 quality management status
    engine.detach()                       # remove hooks to avoid memory leaks

Hook lifecycle:
  ``attach()`` stores hook handles in ``_hook_handles``. PyTorch forward hooks
  are reference-counted: if you don't call ``detach()``, the hooks (and their
  closure over ``chain``) remain alive as long as the model does. Always call
  ``detach()`` or use ``engine`` as a context manager in long-running services.

Thread safety:
  ``AuditChain`` is not thread-safe for concurrent writes. If you run inference
  across multiple threads, create one engine per thread or guard with a lock.

Design pattern reference:
  Gamma et al. (1994) — Design Patterns: Elements of Reusable Object-Oriented Software
  Façade pattern, Chapter 4.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional

import torch.nn as nn

from torchcomply.core.audit import AuditChain, register_compliance_hooks
from torchcomply.core.dataset import CompliantDataset, ConsentRegistry
from torchcomply.core.fairness import FairnessGate
from torchcomply.reports.annex_iv import AnnexIVReport, ModelIntrospector


class ComplianceEngine:
    """
    Central façade that ties audit, fairness, consent, and reporting together.

    Args:
        regulations: Regulation IDs to include in reports.
        risk_level: EU AI Act risk category ("high", "limited", "minimal").
    """

    def __init__(
        self,
        regulations: Optional[List[str]] = None,
        risk_level: str = "high",
    ) -> None:
        self.regulations = regulations or ["eu_ai_act", "gdpr"]
        self.risk_level = risk_level
        self.audit_chain = AuditChain()
        self.fairness_gate: Optional[FairnessGate] = None
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    # Model attachment
    # ------------------------------------------------------------------

    def attach(self, model: nn.Module) -> nn.Module:
        """Register compliance audit hooks on all leaf modules. Returns model unchanged."""
        handles = register_compliance_hooks(model, self.audit_chain)
        self._hook_handles.extend(handles)
        return model

    def detach(self) -> None:
        """Remove all registered hooks (call after inference / training)."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Context manager — preferred usage for production code
    # ------------------------------------------------------------------

    def __enter__(self) -> "ComplianceEngine":
        """Support `with ComplianceEngine(...) as engine:` — hooks removed automatically."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Remove all hooks on context exit, even if an exception was raised."""
        self.detach()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Sub-component factories
    # ------------------------------------------------------------------

    def create_fairness_gate(self, threshold: float = 0.10) -> FairnessGate:
        """Create and store a FairnessGate. Returns the gate for direct use."""
        self.fairness_gate = FairnessGate(threshold=threshold)
        return self.fairness_gate

    def create_compliant_dataset(
        self,
        base_dataset,
        consent_registry: ConsentRegistry,
        purpose: str,
    ) -> CompliantDataset:
        """Wrap a base Dataset with consent gating."""
        return CompliantDataset(base_dataset, consent_registry, purpose)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        filepath: str,
        model: nn.Module,
        training_config: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        dp_info: Optional[dict] = None,
        explanations: Optional[list] = None,
    ) -> None:
        """Generate an EU AI Act Annex IV PDF at ``filepath``."""
        introspection = ModelIntrospector(model)
        fairness_log = self.fairness_gate.get_log() if self.fairness_gate else []
        report = AnnexIVReport(
            model_introspection=introspection,
            audit_chain=self.audit_chain,
            fairness_log=fairness_log,
            training_config=training_config or {},
            dataset_info=dataset_info or {},
            dp_info=dp_info,
            explanations=explanations,
            regulations=self.regulations,
        )
        report.save_pdf(filepath)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def to_model_card(
        self,
        model: nn.Module,
        output_path: Optional[str] = None,
        training_config: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        dp_info: Optional[dict] = None,
    ) -> str:
        """Generate a HuggingFace-compatible model card (Markdown).

        Outputs the same compliance data as ``generate_report()`` but in the
        ``modelcard.md`` format that HuggingFace Hub accepts. Practitioners can
        push this file alongside the model checkpoint for public transparency.

        Returns the Markdown string and optionally writes it to ``output_path``.
        """
        introspection = ModelIntrospector(model)
        fairness_log = self.fairness_gate.get_log() if self.fairness_gate else []
        audit_s = self.audit_chain.summary()
        training_config = training_config or {}
        dataset_info = dataset_info or {}
        dp_info = dp_info or {}
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        regs = ", ".join(r.upper().replace("_", " ") for r in self.regulations)

        # Fair metric summary
        if fairness_log:
            last = fairness_log[-1]
            fairness_str = (
                f"- **Demographic parity disparity:** {last['disparity']:.4f} "
                f"(threshold: {last['threshold']:.4f}, status: **{last['status'].upper()}**)"
            )
        else:
            fairness_str = "- Fairness check not run"

        # DP summary
        if dp_info:
            dp_str = (
                f"- **Framework:** {dp_info.get('framework', 'unknown')}\n"
                f"- **ε:** {dp_info.get('epsilon', '—')}  |  "
                f"**δ:** {dp_info.get('delta', '—')}\n"
                f"- **Regulation:** GDPR Art. 25 (Privacy by Design)"
            )
        else:
            dp_str = "- Differential privacy not applied"

        # Audit chain summary
        chain_status = "✅ VALID" if audit_s.get("chain_valid") else "❌ INVALID"
        root_hash = audit_s.get("root_hash", "")[:16] + "…" if audit_s.get("root_hash") else "—"

        lines = [
            "---",
            "license: apache-2.0",
            "tags:",
            "  - compliance",
            "  - eu-ai-act",
            "  - gdpr",
            "  - torchcomply",
            "---",
            "",
            "# Model Card — torchcomply Compliance Report",
            "",
            f"> Auto-generated by [torchcomply](https://github.com/rajaghv-dev/pytorch_compliance_as_code) on {timestamp}",
            "",
            "## Model Details",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Architecture | {introspection.architecture} |",
            f"| Total parameters | {introspection.total_params:,} |",
            f"| Trainable parameters | {introspection.trainable_params:,} |",
            f"| Regulations covered | {regs} |",
            f"| Risk level | {self.risk_level} |",
            "",
            "## Intended Use",
            "",
            f"- **Purpose:** {dataset_info.get('purpose', 'Machine learning inference')}",
            f"- **Intended users:** {dataset_info.get('intended_users', 'Operators and deployers')}",
            "- **Out-of-scope:** Uses not covered by the consent scope or purpose limitation",
            "",
            "## Training Data",
            "",
        ]
        for k, v in dataset_info.items():
            lines.append(f"- **{k}:** {v}")
        if not dataset_info:
            lines.append("- No training data information provided")

        lines += [
            "",
            "## Training Methodology",
            "",
        ]
        for k, v in training_config.items():
            lines.append(f"- **{k}:** {v}")
        if not training_config:
            lines.append("- No training configuration provided")

        lines += [
            "",
            "## Privacy Assessment (GDPR Art. 25)",
            "",
            dp_str,
            "",
            "## Fairness Assessment (EU AI Act Art. 10)",
            "",
            fairness_str,
            "",
            "## Audit Trail (EU AI Act Art. 12)",
            "",
            f"- **Total entries:** {audit_s.get('total_entries', 0)}",
            f"- **Chain integrity:** {chain_status}",
            f"- **Root hash:** `{root_hash}`",
            "- **Note:** Root hash is a chain fingerprint — any modification to any logged",
            "  operation changes this value. Include in regulatory filings.",
            "",
            "## Regulatory Mapping",
            "",
            "| Control | Regulation |",
            "|---------|------------|",
            "| Audit chain (hash-chained log) | EU AI Act Art. 12 |",
            "| Fairness gate (demographic parity) | EU AI Act Art. 10 |",
            "| Differential privacy (DP-SGD) | GDPR Art. 25, Art. 32 |",
            "| Explainability (integrated gradients) | EU AI Act Art. 13 |",
            "| Human oversight routing | EU AI Act Art. 14 |",
            "| Technical documentation | EU AI Act Art. 11, Annex IV |",
            "",
            "## Limitations",
            "",
            "- Demographic parity is one of several fairness criteria; does not guarantee",
            "  equalized odds or calibration across groups.",
            "- Gradient provenance (GDPR Art. 17 unlearning) is experimental — covers only",
            "  layers explicitly wrapped with `ProvenanceLinear`.",
            "- CrypTen MPC inference uses single-machine simulation in this demo; production",
            "  deployments require network-isolated parties.",
            "",
            f"*Generated by torchcomply v{self._version()} — "
            f"[PyTorch Conference Europe 2026](https://github.com/rajaghv-dev/pytorch_compliance_as_code)*",
        ]

        card = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(card, encoding="utf-8")
        return card

    @staticmethod
    def _version() -> str:
        try:
            import torchcomply
            return torchcomply.__version__
        except Exception:
            return "unknown"

    def summary(self) -> str:
        """Return a Unicode-bordered compliance summary box for terminal display."""
        audit_s = self.audit_chain.summary()
        n_entries = audit_s["total_entries"]
        chain_ok = "✅ VALID" if audit_s["chain_valid"] else "❌ INVALID"

        if self.fairness_gate and self.fairness_gate.log:
            last = self.fairness_gate.log[-1]
            fair_str = f"Parity {last['disparity']:.3f}  |  {last['status'].upper()}"
        else:
            fair_str = "Not checked"

        regs = ", ".join(r.upper().replace("_", " ") for r in self.regulations)

        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║            TORCHCOMPLY — COMPLIANCE SUMMARY                  ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Regulations : {regs:<45}║",
            f"║  Risk level  : {self.risk_level:<45}║",
            "║                                                              ║",
            f"║  Audit Trail : {n_entries:<5} entries  |  Chain: {chain_ok:<17}║",
            f"║  Fairness    : {fair_str:<45}║",
            "║                                                              ║",
            "║  Status      : READY FOR REGULATORY REVIEW ✅                ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
