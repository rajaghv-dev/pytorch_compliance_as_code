"""
PyTorch Compliance Toolkit -- annotators subpackage.

Annotators are pure in-memory transformations over lists of EntityRecord
objects.  They do NOT re-read source files.  Each annotator receives a list
of records, enriches them with compliance metadata, and returns the list.

Annotators are designed to be independently importable and composable.
The ``run_all_annotators`` function executes them in the canonical order.
"""

from __future__ import annotations

import logging
from typing import Optional

from .compliance_tagger import ComplianceTagger
from .confidence import ConfidenceAnnotator
from .dispatcher_level import DispatcherLevelAnnotator
from .distributed_safety import DistributedSafetyAnnotator
from .export_survival import ExportSurvivalAnnotator
from .hook_consumers import HookConsumersAnnotator
from .lifecycle import LifecycleAnnotator

__all__ = [
    "ComplianceTagger",
    "DispatcherLevelAnnotator",
    "DistributedSafetyAnnotator",
    "LifecycleAnnotator",
    "ExportSurvivalAnnotator",
    "ConfidenceAnnotator",
    "HookConsumersAnnotator",
    "run_all_annotators",
]

logger = logging.getLogger("pct.annotators")


def run_all_annotators(
    records: list,
    config: Optional[dict] = None,
    determinism_operators: Optional[set] = None,
) -> list:
    """
    Run all annotators in the correct order.

    Execution order:
    1. HookConsumersAnnotator — ensure hook consumers are tagged before
       the compliance tagger runs (BUG-12 fix).
    2. ComplianceTagger — 3-tier compliance tagging (name, structure, semantic).
    3. DispatcherLevelAnnotator — classify execution level.
    4. DistributedSafetyAnnotator — distributed safety classification.
    5. LifecycleAnnotator — lifecycle phase assignment.
    6. ExportSurvivalAnnotator — export format survival.
    7. ConfidenceAnnotator — final confidence calibration (must run last so
       it can see all tags and rationales).

    Each annotator is wrapped in a try/except so that a failure in one
    annotator does not prevent the others from running.

    Parameters
    ----------
    records : list
        List of EntityRecord objects to annotate.
    config : dict | None
        Optional configuration (reserved for future use).
    determinism_operators : set | None
        Set of operator names from Sphinx deterministic.md notes, passed
        to the ConfidenceAnnotator for confidence boosting.

    Returns
    -------
    list
        The annotated records.
    """
    # Define annotator pipeline in execution order
    annotators = [
        ("HookConsumersAnnotator", HookConsumersAnnotator()),
        ("ComplianceTagger", ComplianceTagger()),
        ("DispatcherLevelAnnotator", DispatcherLevelAnnotator()),
        ("DistributedSafetyAnnotator", DistributedSafetyAnnotator()),
        ("LifecycleAnnotator", LifecycleAnnotator()),
        ("ExportSurvivalAnnotator", ExportSurvivalAnnotator()),
        ("ConfidenceAnnotator", ConfidenceAnnotator(determinism_operators=determinism_operators)),
    ]

    logger.info(
        "Running %d annotators on %d records", len(annotators), len(records)
    )

    for name, annotator in annotators:
        logger.info("Starting annotator: %s", name)
        try:
            records = annotator.annotate(records)
            logger.info("Completed annotator: %s", name)
        except Exception:
            logger.error(
                "Annotator '%s' failed — skipping", name, exc_info=True
            )

    logger.info("All annotators complete. %d records processed.", len(records))
    return records
