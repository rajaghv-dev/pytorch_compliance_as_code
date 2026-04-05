"""
opacus_bridge.py — Opacus DP-SGD wrapper for GDPR Article 25 (Privacy by Design).

CompliancePrivacyEngine decorates an Opacus PrivacyEngine with compliance metadata
so that epsilon/delta values are surfaced in the Annex IV PDF report.

Differential Privacy Background
--------------------------------
DP-SGD (Differentially Private Stochastic Gradient Descent) provides a
formal mathematical guarantee: the trained model reveals at most ε bits of
information about any individual training sample, with probability (1 - δ).

The (ε, δ)-DP guarantee is computed by the Opacus privacy accountant using
the Rényi DP composition theorem (Mironov, 2017). Smaller ε = stronger privacy;
typical values for production systems range from ε=1 (strong) to ε=10 (moderate).

Mechanism:
  1. Per-sample gradient clipping: ‖∇L_i‖₂ ≤ C  (clips outlier contributions)
  2. Gaussian noise addition: ∇L += N(0, σ²C²I)  (masks individual samples)
  3. Batch gradient = (1/n) Σ noisy_clipped_gradients

The noise multiplier σ and clipping norm C together determine the ε budget
consumed per step. Opacus tracks this automatically.

Regulatory references:
  GDPR Art. 25 — Data protection by design and by default
    https://gdpr-info.eu/art-25-gdpr/
  GDPR Art. 32 — Security of processing
    https://gdpr-info.eu/art-32-gdpr/

Key papers:
  Abadi et al. (2016) — Deep Learning with Differential Privacy
    https://arxiv.org/abs/1607.00133
  Mironov (2017) — Rényi Differential Privacy of the Gaussian Mechanism
    https://arxiv.org/abs/1702.07476

Opacus documentation:
  https://opacus.ai/
  https://github.com/pytorch/opacus
"""

from __future__ import annotations

from typing import List, Optional


class EpsilonBudgetExceeded(Exception):
    """Raised when the privacy budget ε exceeds the configured maximum.

    Mirrors ``ComplianceViolation`` from the fairness gate — the same control
    that stops biased training also stops over-budget privacy training.
    """

    def __init__(self, epsilon: float, max_epsilon: float) -> None:
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        super().__init__(
            f"Privacy budget exceeded: ε={epsilon:.2f} > max_epsilon={max_epsilon:.2f}. "
            f"Training stopped. Lower target_epsilon or increase training data size."
        )


class CompliancePrivacyEngine:
    """
    Wraps an ``opacus.PrivacyEngine`` with compliance metadata tracking.

    Args:
        privacy_engine: An already-constructed Opacus PrivacyEngine.
        regulations: List of regulation IDs this engine satisfies.
    """

    def __init__(
        self,
        privacy_engine,
        regulations: Optional[List[str]] = None,
        max_epsilon: Optional[float] = None,
    ) -> None:
        """
        Args:
            privacy_engine: An already-constructed Opacus PrivacyEngine.
            regulations: List of regulation IDs this engine satisfies.
            max_epsilon: Hard upper bound on ε. If the consumed ε exceeds this
                value when ``check_epsilon()`` is called, raises
                ``EpsilonBudgetExceeded``. Recommended: ≤ 8 for GDPR Art.25.
                Set to ``None`` (default) to disable enforcement (advisory mode only).

        Example — enforced mode::

            cpe = CompliancePrivacyEngine(pe, max_epsilon=8.0)
            # after each epoch:
            cpe.check_epsilon(delta=1e-5)  # raises EpsilonBudgetExceeded if ε > 8
        """
        self._engine = privacy_engine
        self.regulations = regulations or ["gdpr_art_25", "gdpr_art_32"]
        self._n_steps: int = 0
        self.max_epsilon: Optional[float] = max_epsilon

    # ------------------------------------------------------------------
    # Privacy accounting
    # ------------------------------------------------------------------

    def get_epsilon(self, delta: float) -> float:
        """Delegate to wrapped engine's privacy accountant."""
        return float(self._engine.get_epsilon(delta))

    def check_epsilon(self, delta: float) -> float:
        """Compute current ε and enforce the budget if ``max_epsilon`` is set.

        Raises ``EpsilonBudgetExceeded`` if consumed ε > ``max_epsilon``.
        Always returns the current ε value so it can be logged.

        Call once per epoch (after the epoch's optimizer steps complete) to
        provide the same hard-stop semantics as ``FairnessGate.on_epoch_end()``.
        """
        eps = self.get_epsilon(delta)
        if self.max_epsilon is not None and eps > self.max_epsilon:
            raise EpsilonBudgetExceeded(eps, self.max_epsilon)
        return eps

    def step(self) -> None:
        """Increment internal step counter. Call once per optimizer.step()."""
        self._n_steps += 1

    # ------------------------------------------------------------------
    # Compliance metadata
    # ------------------------------------------------------------------

    def get_compliance_summary(self) -> dict:
        """Return a dict of key privacy metrics for compliance logging."""
        return {
            "type": "differential_privacy",
            "framework": "opacus",
            "regulations": self.regulations,
            "n_steps": self._n_steps,
        }

    def log_to_dict(self) -> dict:
        """Full metadata dict for inclusion in the Annex IV PDF report."""
        return {
            "type": "differential_privacy",
            "framework": "opacus",
            "regulations": self.regulations,
            "n_steps": self._n_steps,
            "max_epsilon_enforced": self.max_epsilon,
            "gdpr_article_25": "Privacy by Design — DP-SGD gradient noise injection",
            "gdpr_article_32": "Technical measures for personal data protection",
        }
