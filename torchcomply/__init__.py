"""
torchcomply — Compliance-as-Code library for PyTorch.

Embeds EU AI Act and GDPR controls as executable PyTorch logic.
Presented at PyTorch Conference Europe 2026, Station F, Paris.
"""

from torchcomply.core.audit import AuditChain, IntegrityViolation, register_compliance_hooks
from torchcomply.core.autograd_provenance import ProvenanceLinear
from torchcomply.core.dataset import CompliantDataset, ConsentRegistry, ConsentViolation
from torchcomply.core.diff import ComplianceDiff, ComplianceRegressionError, ComplianceSnapshot
from torchcomply.core.dispatcher_hooks import ComplianceTensor
from torchcomply.core.engine import ComplianceEngine
from torchcomply.core.fairness import ComplianceViolation, FairnessGate
from torchcomply.integrations.opacus_bridge import EpsilonBudgetExceeded
from torchcomply.reports.annex_iv import AnnexIVReport, ModelIntrospector

__version__ = "0.4.0"
__all__ = [
    "ComplianceEngine",
    "AuditChain",
    "IntegrityViolation",
    "register_compliance_hooks",
    "ComplianceTensor",
    "ProvenanceLinear",
    "FairnessGate",
    "ComplianceViolation",
    "CompliantDataset",
    "ConsentRegistry",
    "ConsentViolation",
    "AnnexIVReport",
    "ModelIntrospector",
    "ComplianceDiff",
    "ComplianceSnapshot",
    "ComplianceRegressionError",
    "EpsilonBudgetExceeded",
]
