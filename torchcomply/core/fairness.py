"""
fairness.py — Demographic parity gate for EU AI Act Article 10 compliance.

FairnessGate wraps a validation loop: after each epoch it computes the
demographic parity disparity and raises ComplianceViolation if it exceeds
the configured threshold, blocking further training.

Demographic Parity (a.k.a. Statistical Parity)
-----------------------------------------------
Demographic parity requires that a classifier's positive prediction rate is
equal across protected groups (e.g. gender, ethnicity, age band):

    P(Ŷ=1 | A=0) ≈ P(Ŷ=1 | A=1)

We measure the *disparity* as max_rate - min_rate across all groups.
A disparity of 0 means perfect parity; a disparity of 1 means one group
receives all positive predictions and another receives none.

Note on metric choice: demographic parity is one of several fairness criteria.
Others include equalised odds, calibration within groups, and individual fairness.
The right metric depends on the deployment context. See the references below for
a thorough comparison.

Regulatory references:
  EU AI Act Art. 10(2)(f) — Examination of data for possible biases
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 10)
  EU Charter of Fundamental Rights Art. 21 — Non-discrimination
    https://www.europarl.europa.eu/charter/pdf/text_en.pdf

Further reading:
  Barocas, Hardt & Narayanan — Fairness and Machine Learning (textbook, free online)
    https://fairmlbook.org/
  Hardt, Price & Srebro (2016) — Equality of Opportunity in Supervised Learning
    https://arxiv.org/abs/1610.02413
  Microsoft Fairlearn — fairness assessment library for ML practitioners
    https://fairlearn.org/
  Google What-If Tool — visual exploration of model fairness
    https://pair-code.github.io/what-if-tool/
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader


class ComplianceViolation(Exception):
    """Raised when fairness disparity exceeds the configured threshold."""

    def __init__(self, disparity: float, threshold: float) -> None:
        self.disparity = disparity
        self.threshold = threshold
        super().__init__(
            f"Fairness violation: disparity={disparity:.4f} > threshold={threshold:.4f}"
        )


def compute_demographic_parity(predictions: Tensor, protected_attr: Tensor) -> float:
    """
    Compute demographic parity disparity between protected groups.

    Returns max(positive_rate_per_group) - min(positive_rate_per_group).
    A value of 0 means perfect parity; EU AI Act Art. 10 requires minimisation.
    """
    groups = protected_attr.unique()
    rates: List[float] = []
    for g in groups:
        mask = protected_attr == g
        rate = predictions[mask].float().mean().item()
        rates.append(rate)
    if len(rates) < 2:
        raise ValueError(
            f"Cannot compute demographic parity: only {len(rates)} protected group(s) found in data. "
            "At least 2 groups are required. Check that protected_attr_idx points to the correct "
            "batch element and that your validation set contains samples from multiple groups."
        )
    return float(max(rates) - min(rates))


class FairnessGate:
    """
    Monitors demographic parity per epoch and blocks non-compliant training.

    Usage::

        gate = FairnessGate(threshold=0.10)
        for epoch in range(N):
            train_one_epoch(model, optimizer, train_loader)
            gate.on_epoch_end(model, val_loader, epoch, protected_attr_idx=2)
    """

    def __init__(
        self,
        threshold: float = 0.10,
        metric_name: str = "demographic_parity",
    ) -> None:
        self.threshold = threshold
        self.metric_name = metric_name
        self.log: List[dict] = []

    def on_epoch_end(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        epoch: int,
        protected_attr_idx: Optional[int] = None,
    ) -> None:
        """
        Run model on val_loader, compute parity, log result.
        Raises ComplianceViolation if disparity > threshold.

        Args:
            protected_attr_idx: index in the batch tuple for the protected attribute tensor.
                                 If None the fairness check is skipped.
        """
        model.eval()
        device = next(model.parameters()).device
        all_preds: List[Tensor] = []
        all_attrs: List[Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                if protected_attr_idx is not None and 0 <= protected_attr_idx < len(batch):
                    x = batch[0].to(device)
                    attrs = batch[protected_attr_idx]
                    all_attrs.append(attrs)
                else:
                    x = batch[0].to(device)

                out = model(x)
                preds = out.argmax(dim=-1) if out.dim() > 1 else (out.squeeze() > 0.5).long()
                all_preds.append(preds.cpu())

        predictions = torch.cat(all_preds)

        if not all_attrs:
            self.log.append(
                {"epoch": epoch, "disparity": 0.0, "threshold": self.threshold, "status": "skipped"}
            )
            return

        attrs = torch.cat(all_attrs)
        disparity = compute_demographic_parity(predictions, attrs)
        status = "passed" if disparity <= self.threshold else "blocked"
        self.log.append(
            {"epoch": epoch, "disparity": disparity, "threshold": self.threshold, "status": status}
        )

        if disparity > self.threshold:
            raise ComplianceViolation(disparity, self.threshold)

    def get_log(self) -> List[dict]:
        return list(self.log)
