"""
Fairness gate — Art. 10 (Data Governance) + Art. 15 (Accuracy/Robustness).

WHAT THIS DEMONSTRATES
----------------------
Blocks model export if demographic disparity across protected groups exceeds
a configurable threshold.

This is a compliance control: it ensures that a model that exhibits unfair
behaviour toward a protected group cannot be exported for deployment until
the disparity is below the regulatory limit.

HOW IT WORKS
------------
1. A forward hook is registered on the classification head.
2. On each inference batch, the hook computes the per-group positive
   prediction rate (demographic parity).
3. If the disparity (max_rate - min_rate) exceeds DISPARITY_THRESHOLD,
   an internal flag is set.
4. The custom export guard checks this flag and raises ExportBlockedError
   if the model is not yet fair enough.

USAGE
-----
    python examples/fairness_gate.py

    # Or integrate into your training loop:
    from examples.fairness_gate import FairnessGate

    gate = FairnessGate(model, threshold=0.1)
    gate.attach()
    # ... run inference on validation set ...
    gate.export_if_fair("model_export.pt")

REGULATORY MAPPING
------------------
  Art. 10 §2(f): Training data must be free of biases that could lead to
                 discrimination.
  Art. 15 §1:    High-risk AI systems shall achieve appropriate levels of
                 accuracy, robustness, and cybersecurity.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("pct.example.fairness_gate")

# ----------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------- #

# Maximum allowed demographic disparity in positive prediction rate.
# 0.1 = 10 percentage-point gap between the highest and lowest group.
DISPARITY_THRESHOLD = 0.1

# Attribute index in the batch that identifies the protected group.
# In practice this comes from your dataset's metadata; 0 is a placeholder.
GROUP_ATTRIBUTE_INDEX = 0


# ----------------------------------------------------------------------- #
# FairnessGate
# ----------------------------------------------------------------------- #

class ExportBlockedError(RuntimeError):
    """Raised when export is blocked due to fairness violations."""


class FairnessGate:
    """
    Registers a forward hook on a classification module to monitor
    demographic parity and block export when disparity is too high.

    Parameters
    ----------
    module : nn.Module
        The classification head to monitor (e.g. the final linear layer).
    threshold : float
        Maximum allowed disparity in positive prediction rate (default: 0.1).
    """

    def __init__(
        self,
        module: nn.Module,
        threshold: float = DISPARITY_THRESHOLD,
    ) -> None:
        self.module    = module
        self.threshold = threshold

        # Running statistics per group label.
        self._group_positives: dict[int, int] = {}
        self._group_totals:    dict[int, int] = {}

        # Flag set to True when disparity exceeds threshold.
        self._export_blocked = False
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

        # Latest disparity reading.
        self._last_disparity: float = 0.0

    def attach(self) -> None:
        """Register the forward hook on the monitored module."""
        self._hook_handle = self.module.register_forward_hook(self._hook)
        logger.info(
            "FairnessGate: attached to %s (threshold=%.2f)",
            type(self.module).__name__,
            self.threshold,
        )

    def detach(self) -> None:
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.info("FairnessGate: detached")

    def export_if_fair(self, save_path: str) -> None:
        """
        Save the model with torch.save only if disparity is within threshold.

        Raises ExportBlockedError if the model is not yet fair enough.
        """
        if self._export_blocked:
            disparity = self._last_disparity
            raise ExportBlockedError(
                f"Export blocked: demographic disparity {disparity:.3f} "
                f"exceeds threshold {self.threshold:.3f}. "
                f"Art.10 §2(f) requires bias-free training data. "
                f"Reduce disparity before deploying this model."
            )
        torch.save(self.module.state_dict(), save_path)
        logger.info("FairnessGate: export allowed — saved to %s", save_path)

    def reset_stats(self) -> None:
        """Clear accumulated group statistics (call between epochs)."""
        self._group_positives.clear()
        self._group_totals.clear()
        self._export_blocked = False
        self._last_disparity = 0.0

    @property
    def disparity(self) -> float:
        """Current max-min positive prediction rate across groups."""
        return self._last_disparity

    # ------------------------------------------------------------------ #
    # Hook implementation
    # ------------------------------------------------------------------ #

    def _hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        """
        Forward hook: accumulate per-group positive predictions and
        check disparity after each batch.

        The hook expects `inputs[0]` to contain group labels as the first
        feature column (index GROUP_ATTRIBUTE_INDEX).  In a real system
        the group labels would be passed separately via a side channel.
        """
        # Guard: we need the inputs to extract group labels.
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return

        batch_input = inputs[0]   # shape: (batch_size, n_features)
        predictions = output      # shape: (batch_size, n_classes)

        # Extract positive predictions (class 1 argmax).
        with torch.no_grad():
            pred_labels = predictions.argmax(dim=-1).cpu()

        # Extract group attribute (column 0 of input features).
        if batch_input.dim() < 2 or batch_input.shape[1] <= GROUP_ATTRIBUTE_INDEX:
            return

        groups = batch_input[:, GROUP_ATTRIBUTE_INDEX].long().cpu()

        # Accumulate per-group statistics.
        for group_id, is_positive in zip(groups.tolist(), pred_labels.tolist()):
            group_id = int(group_id)
            self._group_totals[group_id] = self._group_totals.get(group_id, 0) + 1
            if is_positive == 1:
                self._group_positives[group_id] = (
                    self._group_positives.get(group_id, 0) + 1
                )

        # Compute demographic parity.
        disparity = self._compute_disparity()
        self._last_disparity = disparity

        if disparity > self.threshold:
            self._export_blocked = True
            logger.warning(
                "FairnessGate: disparity=%.3f exceeds threshold=%.3f — "
                "export blocked (Art.10 §2(f))",
                disparity,
                self.threshold,
            )
        else:
            logger.debug(
                "FairnessGate: disparity=%.3f within threshold=%.3f",
                disparity,
                self.threshold,
            )

    def _compute_disparity(self) -> float:
        """Compute max - min positive prediction rate across groups."""
        rates = []
        for group_id, total in self._group_totals.items():
            if total > 0:
                positives = self._group_positives.get(group_id, 0)
                rates.append(positives / total)

        if len(rates) < 2:
            return 0.0
        return max(rates) - min(rates)


# ----------------------------------------------------------------------- #
# Demo
# ----------------------------------------------------------------------- #

def _demo() -> None:
    """
    Demonstrate FairnessGate with a toy 2-class model and synthetic data.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    # Simple 2-class linear classifier.
    model = nn.Linear(10, 2)
    gate  = FairnessGate(model, threshold=0.1)
    gate.attach()

    print("\n── FairnessGate Demo ──────────────────────────────────────────")
    print("Art.10 + Art.15 compliance gate for demographic parity")
    print()

    # Simulate 5 batches.
    for batch_idx in range(5):
        batch_size = 32
        # Features: column 0 is group label (0 or 1), rest are random.
        x = torch.randn(batch_size, 10)
        x[:, 0] = (torch.rand(batch_size) > 0.5).float()

        # Run forward pass.
        with torch.no_grad():
            out = model(x)

        print(f"  Batch {batch_idx + 1}: disparity={gate.disparity:.3f}  "
              f"export_blocked={gate._export_blocked}")

    # Attempt export.
    print()
    try:
        gate.export_if_fair("/tmp/fair_model.pt")
        print("✓ Export succeeded — model is within fairness threshold")
    except ExportBlockedError as e:
        print(f"✗ Export blocked: {e}")

    gate.detach()


if __name__ == "__main__":
    _demo()
