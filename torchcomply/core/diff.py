"""
diff.py — ComplianceDiff: compare two compliance snapshots across runs.

EU AI Act Article 9 (Risk Management) requires ongoing monitoring of deployed
AI systems. This module provides a structured diff between two compliance
states — answering "did we regress since the last run?"

Usage::

    from torchcomply.core.diff import ComplianceDiff

    diff = ComplianceDiff(snapshot_before, snapshot_after)
    print(diff.report())   # prints a comparison table
    diff.assert_no_regression()  # raises if any metric worsened beyond tolerance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComplianceSnapshot:
    """A point-in-time compliance measurement captured after a training/eval run.

    Create one snapshot at the end of each run and persist it (e.g. to MLflow
    or a JSONL file) so ``ComplianceDiff`` can compare across runs.

    Args:
        run_id: Unique identifier for this run (e.g. git SHA, timestamp, MLflow run ID).
        audit_root_hash: ``AuditChain.root_hash()`` — fingerprints the full log.
        audit_entries: Number of entries in the audit chain.
        fairness_parity: Demographic parity disparity (lower is better).
        fairness_passed: Whether the fairness gate passed.
        epsilon: Privacy budget consumed (lower is better). None if DP not used.
        delta: Target δ for DP. None if DP not used.
        drift_detected: Whether deployment drift was detected. None if not monitored.
        accuracy: Model accuracy on the evaluation set. None if not measured.
        extra: Any additional key-value pairs to include in the comparison.
    """

    run_id: str
    audit_root_hash: str = ""
    audit_entries: int = 0
    fairness_parity: Optional[float] = None
    fairness_passed: Optional[bool] = None
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    drift_detected: Optional[bool] = None
    accuracy: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "audit_root_hash": self.audit_root_hash,
            "audit_entries": self.audit_entries,
            "fairness_parity": self.fairness_parity,
            "fairness_passed": self.fairness_passed,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "drift_detected": self.drift_detected,
            "accuracy": self.accuracy,
            **self.extra,
        }

    @classmethod
    def from_engine(
        cls,
        run_id: str,
        engine,
        accuracy: Optional[float] = None,
        drift_detected: Optional[bool] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        fairness_parity: Optional[float] = None,
        fairness_passed: Optional[bool] = None,
    ) -> "ComplianceSnapshot":
        """Construct a snapshot from a live ``ComplianceEngine`` at the end of a run.

        ``fairness_parity`` and ``fairness_passed`` can be passed explicitly when
        the fairness gate was not attached to the engine (e.g. computed externally).
        If not provided, the engine's fairness gate log is used if available.
        """
        audit_s = engine.audit_chain.summary()
        if fairness_parity is None and engine.fairness_gate and engine.fairness_gate.log:
            last = engine.fairness_gate.log[-1]
            fairness_parity = last["disparity"]
            fairness_passed = last["status"] == "passed"
        return cls(
            run_id=run_id,
            audit_root_hash=audit_s.get("root_hash", ""),
            audit_entries=audit_s.get("total_entries", 0),
            fairness_parity=fairness_parity,
            fairness_passed=fairness_passed,
            epsilon=epsilon,
            delta=delta,
            drift_detected=drift_detected,
            accuracy=accuracy,
        )


@dataclass
class _DiffLine:
    metric: str
    before: Any
    after: Any
    direction: str  # "lower_better" | "higher_better" | "match" | "info"
    regression: bool = False
    tolerance: float = 0.0


class ComplianceDiff:
    """Compare two ``ComplianceSnapshot`` instances and report regressions.

    Args:
        before: Snapshot from the previous run (baseline).
        after: Snapshot from the current run (candidate).
        fairness_tolerance: Maximum allowed *increase* in fairness disparity
            before flagging as a regression. Default: 0.02 (2 percentage points).
        epsilon_tolerance: Maximum allowed *increase* in ε before flagging.
            Default: 5.0 (large, since ε scale varies by dataset/epochs).
        accuracy_tolerance: Maximum allowed *decrease* in accuracy before
            flagging. Default: 0.02 (2 percentage points).
    """

    def __init__(
        self,
        before: ComplianceSnapshot,
        after: ComplianceSnapshot,
        fairness_tolerance: float = 0.02,
        epsilon_tolerance: float = 5.0,
        accuracy_tolerance: float = 0.02,
    ) -> None:
        self.before = before
        self.after = after
        self.fairness_tolerance = fairness_tolerance
        self.epsilon_tolerance = epsilon_tolerance
        self.accuracy_tolerance = accuracy_tolerance
        self._lines: List[_DiffLine] = self._compute()

    def _compute(self) -> List[_DiffLine]:
        b, a = self.before, self.after
        lines: List[_DiffLine] = []

        # Run IDs
        lines.append(_DiffLine("run_id", b.run_id, a.run_id, "info"))

        # Audit chain
        lines.append(_DiffLine("audit_entries", b.audit_entries, a.audit_entries, "info"))
        lines.append(_DiffLine("audit_root_hash", b.audit_root_hash[:12] + "…", a.audit_root_hash[:12] + "…", "info"))

        # Fairness
        if b.fairness_parity is not None and a.fairness_parity is not None:
            parity_delta = a.fairness_parity - b.fairness_parity
            regression = parity_delta > self.fairness_tolerance
            lines.append(_DiffLine(
                "fairness_parity", b.fairness_parity, a.fairness_parity,
                "lower_better", regression=regression, tolerance=self.fairness_tolerance,
            ))
        if b.fairness_passed is not None and a.fairness_passed is not None:
            lines.append(_DiffLine(
                "fairness_passed", b.fairness_passed, a.fairness_passed,
                "match", regression=(b.fairness_passed and not a.fairness_passed),
            ))

        # Privacy
        if b.epsilon is not None and a.epsilon is not None:
            eps_delta = a.epsilon - b.epsilon
            regression = eps_delta > self.epsilon_tolerance
            lines.append(_DiffLine(
                "epsilon (ε)", b.epsilon, a.epsilon,
                "lower_better", regression=regression, tolerance=self.epsilon_tolerance,
            ))

        # Accuracy
        if b.accuracy is not None and a.accuracy is not None:
            acc_delta = b.accuracy - a.accuracy  # positive = drop
            regression = acc_delta > self.accuracy_tolerance
            lines.append(_DiffLine(
                "accuracy", b.accuracy, a.accuracy,
                "higher_better", regression=regression, tolerance=self.accuracy_tolerance,
            ))

        # Drift
        if b.drift_detected is not None and a.drift_detected is not None:
            regression = (not b.drift_detected) and a.drift_detected
            lines.append(_DiffLine(
                "drift_detected", b.drift_detected, a.drift_detected,
                "match", regression=regression,
            ))

        return lines

    @property
    def regressions(self) -> List[_DiffLine]:
        return [line for line in self._lines if line.regression]

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    def report(self) -> str:
        """Return a formatted comparison table as a string."""
        COL = 22
        header = (
            f"  {'Metric':<{COL}} │ {'Before':>18} │ {'After':>18} │ {'Status':>10}"
        )
        sep = "─" * len(header)
        rows = [
            "╔" + "═" * (len(header) - 2) + "╗",
            "║" + " COMPLIANCE DIFF ".center(len(header) - 2) + "║",
            "║" + f"  {self.before.run_id}  →  {self.after.run_id}".center(len(header) - 2) + "║",
            "╠" + "═" * (len(header) - 2) + "╣",
            header,
            sep,
        ]
        for line in self._lines:
            def fmt(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                if isinstance(v, bool):
                    return "✅" if v else "❌"
                return str(v)

            if line.regression:
                status = "⚠️  REGRESSED"
            elif line.direction == "info":
                status = ""
            else:
                status = "✅ OK"

            rows.append(
                f"  {line.metric:<{COL}} │ {fmt(line.before):>18} │ {fmt(line.after):>18} │ {status:>10}"
            )

        rows.append(sep)
        if self.has_regressions:
            rows.append(f"  ⚠️  {len(self.regressions)} regression(s) detected")
        else:
            rows.append("  ✅ No regressions detected")
        rows.append("╚" + "═" * (len(header) - 2) + "╝")
        return "\n".join(rows)

    def assert_no_regression(self) -> None:
        """Raise ``ComplianceRegressionError`` if any metric regressed.

        Use this in CI pipelines or deployment gates to block a new model
        version from being promoted if its compliance metrics degraded.
        """
        if self.has_regressions:
            metrics = ", ".join(line.metric for line in self.regressions)
            raise ComplianceRegressionError(
                f"Compliance regression detected in: {metrics}\n{self.report()}"
            )


class ComplianceRegressionError(Exception):
    """Raised by ``ComplianceDiff.assert_no_regression()`` when a metric degraded."""
    pass
