"""
mlflow_logger.py — MLflow metric logging for compliance audit integration.

Logs fairness disparity, audit chain stats, and DP parameters to an MLflow
experiment so compliance evidence is versioned alongside model artifacts.

Why version compliance evidence with MLflow?
--------------------------------------------
EU AI Act Article 12 requires that logs can be used post-hoc to verify
system behaviour. Storing compliance metrics in the same versioned experiment
as the model weights ensures:

  1. Reproducibility — a given model version is always paired with the
     fairness and DP metrics measured during *that* training run.
  2. Auditability — regulators can inspect the full training history,
     including epochs where the fairness gate was triggered.
  3. Traceability — MLflow run IDs can be cross-referenced against
     AuditChain JSON exports for full end-to-end evidence chains.

The default SQLite backend (``mlflow.db``) is suitable for local development.
For production, point ``tracking_uri`` at an MLflow Tracking Server backed
by PostgreSQL and S3 artifact storage.

Regulatory references:
  EU AI Act Art. 9 — Risk management system (continuous evidence logging)
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 9)
  EU AI Act Art. 12 — Record-keeping
  EU AI Act Art. 17 — Quality management system

MLflow documentation:
  https://mlflow.org/docs/latest/tracking.html
  Logging metrics: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric
"""

from __future__ import annotations

from typing import Optional

try:
    import mlflow as _mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class ComplianceMLflowLogger:
    """
    Logs torchcomply compliance metrics to MLflow.

    Args:
        experiment_name: MLflow experiment to log under.
        tracking_uri: Optional MLflow tracking server URI.
    """

    def __init__(
        self,
        experiment_name: str = "torchcomply",
        tracking_uri: Optional[str] = None,
    ) -> None:
        if not _MLFLOW_AVAILABLE:
            raise ImportError("mlflow is not installed. Run: pip install mlflow")
        if tracking_uri:
            _mlflow.set_tracking_uri(tracking_uri)
        _mlflow.set_experiment(experiment_name)
        self._run = None

    def start_run(self, run_name: Optional[str] = None) -> None:
        self._run = _mlflow.start_run(run_name=run_name)

    def log_fairness(
        self,
        epoch: int,
        disparity: float,
        threshold: float,
        status: str,
    ) -> None:
        _mlflow.log_metrics(
            {"fairness_disparity": disparity, "fairness_threshold": threshold},
            step=epoch,
        )
        _mlflow.log_param(f"fairness_status_epoch_{epoch}", status)

    def log_audit_summary(self, summary: dict) -> None:
        _mlflow.log_metrics(
            {
                "audit_total_entries": summary.get("total_entries", 0),
                "audit_chain_valid": int(summary.get("chain_valid", False)),
                "audit_unique_operators": summary.get("unique_operators", 0),
            }
        )

    def log_dp_params(self, epsilon: float, delta: float, n_steps: int) -> None:
        _mlflow.log_params({"dp_epsilon": epsilon, "dp_delta": delta, "dp_n_steps": n_steps})

    def end_run(self) -> None:
        if self._run:
            _mlflow.end_run()
            self._run = None
