"""
dataset.py — Consent-aware dataset wrapper for GDPR Articles 6 and 7 compliance.

CompliantDataset wraps any torch Dataset, gating every __getitem__ call through
a ConsentRegistry. Denied accesses raise ConsentViolation and are logged.
A DatasetProfile is computed at init to surface class imbalance warnings.

GDPR Consent Model
------------------
GDPR Article 7 defines conditions for valid consent:
  - Must be freely given, specific, informed, and unambiguous
  - Must cover the specific *purpose* for which data is processed
  - Can be withdrawn at any time

``ConsentRegistry`` models this as a per-subject, per-purpose lookup.
A subject may consent to "training" but not "analytics", or vice versa.
Revoked consent (``consent: False``) is immediately enforced: the subject's
samples raise ConsentViolation and are excluded from the DataLoader batch.

Class Imbalance Warning (EU AI Act Art. 10)
-------------------------------------------
Article 10(2)(f) requires providers to examine training data for possible biases.
``DatasetProfile`` computes the majority/minority class ratio and emits a warning
when it exceeds 5:1 — a commonly cited threshold in the fairness literature.

Regulatory references:
  GDPR Art. 6 — Lawfulness of processing
    https://gdpr-info.eu/art-6-gdpr/
  GDPR Art. 7 — Conditions for consent
    https://gdpr-info.eu/art-7-gdpr/
  EU AI Act Art. 10 — Data and data governance
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 10)

Further reading:
  ICO Guide to GDPR — Consent
    https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/consent/
  He & Garcia (2009) — Learning from Imbalanced Data (IEEE TKDE)
    https://doi.org/10.1109/TKDE.2008.239
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class ConsentViolation(Exception):
    """Raised when data access is attempted for a subject without consent."""

    def __init__(self, subject_id: str, purpose: str) -> None:
        self.subject_id = subject_id
        self.purpose = purpose
        super().__init__(f"Consent denied: subject={subject_id}, purpose={purpose}")


@dataclass
class DatasetProfile:
    num_samples: int
    num_classes: int
    class_distribution: Dict[Any, int]
    max_class_ratio: float
    warnings: List[str] = field(default_factory=list)


class ConsentRegistry:
    """
    Stores consent decisions per subject and purpose.

    ``records`` format::

        {
            "subject_001": {"consent": True,  "purposes": ["classification", "analytics"]},
            "subject_002": {"consent": False, "purposes": []},
        }
    """

    def __init__(self, records: Dict[str, dict]) -> None:
        self._records = records
        self.access_log: List[dict] = []

    def has_consent(self, subject_id: str, purpose: str) -> bool:
        record = self._records.get(subject_id, {})
        consented = record.get("consent", False)
        purpose_ok = purpose in record.get("purposes", [])
        granted = consented and purpose_ok

        if not consented:
            reason = "consent_withdrawn"
        elif not purpose_ok:
            reason = "purpose_not_listed"
        else:
            reason = "consent_granted"

        self.access_log.append(
            {
                "subject_id": subject_id,
                "purpose": purpose,
                "status": "granted" if granted else "denied",
                "reason": reason,
            }
        )
        return granted

    def access_log_summary(self) -> dict:
        granted = sum(1 for e in self.access_log if e["status"] == "granted")
        denied = len(self.access_log) - granted
        return {"granted": granted, "denied": denied}


class CompliantDataset(Dataset):
    """
    Wraps a base Dataset with consent gating and statistical profiling.

    The base dataset items must be tuples/lists of at least (data, label).
    If a third element is present it is used as the ``subject_id``; otherwise
    the sample index is used.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        consent_registry: ConsentRegistry,
        purpose: str,
    ) -> None:
        self._base = base_dataset
        self._registry = consent_registry
        self._purpose = purpose
        self._profile = self._compute_profile()

        # Surface imbalance warnings at construction time
        for w in self._profile.warnings:
            print(f"  ⚠️  WARNING: {w}")

    # ------------------------------------------------------------------
    # Profile computation
    # ------------------------------------------------------------------

    def _compute_profile(self) -> DatasetProfile:
        n = len(self._base)
        labels: List[Any] = []
        for i in range(n):
            item = self._base[i]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lbl = item[1]
                labels.append(int(lbl) if isinstance(lbl, (int, float, torch.Tensor)) else lbl)

        dist: Dict[Any, int] = dict(Counter(labels))
        num_classes = len(dist)
        if dist:
            counts = list(dist.values())
            ratio = max(counts) / max(min(counts), 1)
        else:
            ratio = 1.0

        warnings: List[str] = []
        if ratio > 5.0:
            warnings.append(
                f"Class imbalance {ratio:.1f}:1 exceeds 5:1 — EU AI Act Art. 10 data governance"
            )

        return DatasetProfile(
            num_samples=n,
            num_classes=num_classes,
            class_distribution=dist,
            max_class_ratio=ratio,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        item = self._base[idx]

        if isinstance(item, (list, tuple)) and len(item) >= 3:
            subject_id = str(item[2])
        else:
            subject_id = str(idx)

        if not self._registry.has_consent(subject_id, self._purpose):
            raise ConsentViolation(subject_id, self._purpose)

        return item

    def __len__(self) -> int:
        return len(self._base)

    @property
    def profile(self) -> DatasetProfile:
        return self._profile
