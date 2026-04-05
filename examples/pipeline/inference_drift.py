"""
Inference drift monitor — Art. 61 (Post-Market Monitoring).

WHAT THIS DEMONSTRATES
----------------------
Registers a forward hook on the output layer to track the running
mean and variance of output logits.  Alerts when the KL divergence
from the baseline distribution exceeds a configurable threshold.

This is an Art. 61 compliance control: high-risk AI systems must be
monitored post-deployment to detect performance drift that could affect
safety or reliability.

HOW IT WORKS
------------
1. A forward hook is registered on the model's output layer.
2. The first BASELINE_BATCHES batches establish the baseline distribution
   (running mean and variance of output logits).
3. After the baseline window, the hook computes KL divergence between
   the current 100-sample window and the baseline.
4. If KL divergence exceeds DRIFT_THRESHOLD, an alert is emitted.

USAGE
-----
    from examples.inference_drift import InferenceDriftMonitor

    monitor = InferenceDriftMonitor(model.fc_out, threshold=0.1)
    monitor.attach()

    # Run inference ...
    for batch in loader:
        output = model(batch)

    # Check drift at any time.
    print(f"KL divergence: {monitor.current_kl_divergence:.4f}")

REGULATORY MAPPING
------------------
  Art. 61 §1: Providers of high-risk AI systems placed on the Union market
              shall establish and document a post-market monitoring system.
  Art. 61 §3: The post-market monitoring system shall actively and
              systematically collect, document and analyse relevant data.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import torch

logger = logging.getLogger("pct.example.inference_drift")

# ----------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------- #

# KL divergence above this value triggers an Art.61 alert.
DRIFT_THRESHOLD = 0.1

# How many batches to accumulate for the baseline distribution.
BASELINE_BATCHES = 20

# Size of the sliding window used for current distribution estimation.
WINDOW_SIZE = 100


# ----------------------------------------------------------------------- #
# InferenceDriftMonitor
# ----------------------------------------------------------------------- #

class InferenceDriftMonitor:
    """
    Monitors output logit distribution shift via KL divergence.

    Parameters
    ----------
    module : torch.nn.Module
        The output layer to monitor (e.g. the final classifier linear).
    threshold : float
        KL divergence from baseline that triggers an Art.61 alert.
    baseline_batches : int
        Number of batches used to establish the baseline distribution.
    window_size : int
        Sliding window size for computing the current distribution.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        threshold: float = DRIFT_THRESHOLD,
        baseline_batches: int = BASELINE_BATCHES,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self.module           = module
        self.threshold        = threshold
        self.baseline_batches = baseline_batches
        self.window_size      = window_size

        # Baseline statistics (computed from first N batches).
        self._baseline_mean: Optional[torch.Tensor] = None
        self._baseline_var:  Optional[torch.Tensor] = None
        self._baseline_done:  bool = False

        # Accumulator for baseline.
        self._baseline_outputs: list[torch.Tensor] = []
        self._baseline_batch_count = 0

        # Sliding window for current distribution.
        self._window: deque[torch.Tensor] = deque(maxlen=window_size)

        # Most recently computed KL divergence.
        self._current_kl: float = 0.0
        self._alert_count: int = 0

        self._hook_handle = None

    def attach(self) -> None:
        """Register the drift monitoring hook."""
        self._hook_handle = self.module.register_forward_hook(self._hook)
        logger.info(
            "InferenceDriftMonitor: attached to %s (threshold=%.3f)",
            type(self.module).__name__,
            self.threshold,
        )

    def detach(self) -> None:
        """Remove the hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.info(
                "InferenceDriftMonitor: detached (alerts=%d, final_kl=%.4f)",
                self._alert_count,
                self._current_kl,
            )

    @property
    def current_kl_divergence(self) -> float:
        """Latest KL divergence between current window and baseline."""
        return self._current_kl

    @property
    def alert_count(self) -> int:
        """Total number of drift alerts emitted."""
        return self._alert_count

    @property
    def baseline_ready(self) -> bool:
        """True once the baseline window has been completed."""
        return self._baseline_done

    # ------------------------------------------------------------------ #
    # Hook implementation
    # ------------------------------------------------------------------ #

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook: update baseline or check for drift."""
        with torch.no_grad():
            # Flatten output to 1D per sample and move to CPU.
            logits = output.detach().cpu().float()

        # Phase 1: accumulate baseline.
        if not self._baseline_done:
            self._baseline_outputs.append(logits)
            self._baseline_batch_count += 1

            if self._baseline_batch_count >= self.baseline_batches:
                self._compute_baseline()
        else:
            # Phase 2: check drift.
            self._window.extend(logits.unbind(0))
            if len(self._window) >= self.window_size:
                self._check_drift()

    def _compute_baseline(self) -> None:
        """Compute and store the baseline mean and variance."""
        all_outputs = torch.cat(self._baseline_outputs, dim=0)  # (N, C)
        self._baseline_mean = all_outputs.mean(dim=0)
        self._baseline_var  = all_outputs.var(dim=0).clamp(min=1e-8)
        self._baseline_done = True
        self._baseline_outputs.clear()

        logger.info(
            "InferenceDriftMonitor: baseline established from %d batches. "
            "Mean range: [%.4f, %.4f]  Var range: [%.4f, %.4f]",
            self._baseline_batch_count,
            float(self._baseline_mean.min()),
            float(self._baseline_mean.max()),
            float(self._baseline_var.min()),
            float(self._baseline_var.max()),
        )

    def _check_drift(self) -> None:
        """
        Compute KL divergence between the current window distribution and
        the baseline Gaussian approximation.

        KL(current || baseline) for diagonal Gaussians:
          0.5 * sum(log(σ₁²/σ₀²) + σ₀²/σ₁² + (μ₀-μ₁)²/σ₁² - 1)
        """
        current = torch.stack(list(self._window), dim=0)  # (W, C)
        curr_mean = current.mean(dim=0)
        curr_var  = current.var(dim=0).clamp(min=1e-8)

        # KL(baseline || current) — how much has the distribution shifted?
        kl = 0.5 * (
            (self._baseline_var / curr_var)
            + ((curr_mean - self._baseline_mean) ** 2 / curr_var)
            - 1.0
            + curr_var.log() - self._baseline_var.log()
        ).sum().item()

        self._current_kl = float(kl)

        if kl > self.threshold:
            self._alert_count += 1
            logger.warning(
                "COMPLIANCE ALERT [Art.61 §1]: Inference drift detected. "
                "KL divergence = %.4f (threshold = %.4f). "
                "Alert #%d. "
                "The output distribution has shifted from baseline — "
                "investigate model performance and data distribution.",
                kl,
                self.threshold,
                self._alert_count,
            )
        else:
            logger.debug(
                "InferenceDriftMonitor: KL=%.4f within threshold=%.4f",
                kl,
                self.threshold,
            )


# ----------------------------------------------------------------------- #
# Demo
# ----------------------------------------------------------------------- #

def _demo() -> None:
    """Demonstrate drift detection with a toy model and simulated drift."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    import torch.nn as nn

    print("\n── InferenceDrift Demo ────────────────────────────────────────")
    print("Art.61 (Post-Market Monitoring): KL divergence drift detection")
    print()

    model = nn.Linear(16, 4)   # 4-class classifier
    monitor = InferenceDriftMonitor(
        model, threshold=0.1, baseline_batches=10, window_size=50
    )
    monitor.attach()

    # Phase 1: normal distribution — establishes baseline.
    print("  Phase 1: establishing baseline (10 batches) …")
    for i in range(15):
        x = torch.randn(8, 16)
        with torch.no_grad():
            _ = model(x)

    # Phase 2: simulate distribution shift (mean shift by +2.0).
    print("  Phase 2: simulating distribution drift …")
    for i in range(20):
        x = torch.randn(8, 16) + 2.0   # shifted input distribution
        with torch.no_grad():
            _ = model(x)
        if i % 5 == 0:
            print(
                f"    Batch {i+1}: KL divergence = {monitor.current_kl_divergence:.4f}"
            )

    monitor.detach()
    print(f"\n  Baseline ready: {monitor.baseline_ready}")
    print(f"  Final KL divergence: {monitor.current_kl_divergence:.4f}")
    print(f"  Total drift alerts: {monitor.alert_count}")

    if monitor.alert_count > 0:
        print("\n  ✗ Art.61 §1 flag raised: output distribution drifted from baseline")
        print("    Recommended actions:")
        print("    - Investigate input data for distribution shift")
        print("    - Check for data pipeline changes or covariate shift")
        print("    - Consider model retraining or recalibration")
    else:
        print("\n  ✓ Art.61 §1 check passed: output distribution stable")


if __name__ == "__main__":
    _demo()
