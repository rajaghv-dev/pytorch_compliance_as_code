"""
Lifecycle phase annotator for PyTorch EntityRecords.

Assigns a lifecycle_phase to each entity:
  - training_only: only relevant during training (backward, grad, autograd)
  - inference_safe: safe for use during inference
  - data_preparation: related to data loading / preprocessing
  - deployment: related to model export / deployment
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.annotators.lifecycle")


class LifecycleAnnotator:
    """Assigns lifecycle_phase: training_only | inference_safe | data_preparation | deployment."""

    # Pre-compiled pattern for training-related entity names (case-insensitive)
    _TRAINING_PATTERN = re.compile(r"backward|grad|autograd", re.IGNORECASE)

    def annotate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Assign lifecycle_phase to each record.

        Rules are evaluated in priority order; first match wins.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records to classify.

        Returns
        -------
        list[EntityRecord]
            The same list, with lifecycle_phase populated.
        """
        logger.info("Annotating %d records with LifecycleAnnotator", len(records))
        tagged_count = 0
        new_count = 0

        for rec in records:
            had_phase = bool(rec.lifecycle_phase)
            phase = self._classify(rec)
            if phase:
                rec.lifecycle_phase = phase
                tagged_count += 1
                if not had_phase:
                    new_count += 1

                # Warn on potential conflicts
                self._check_conflicts(rec)

        logger.info(
            "Tagged %d records (%d new tags)", tagged_count, new_count
        )
        return records

    def _classify(self, rec: EntityRecord) -> str:
        """
        Determine lifecycle phase for a single record.

        Priority order (first match wins):
        1. backward / grad / autograd in name -> training_only
        2. hook_definition/hook_consumer + "forward" in name -> inference_safe
        3. source in torch/utils/data/ -> data_preparation
        4. source in torch/onnx/ or torch/export/ or torchscript -> deployment
        5. dispatch_key or profiler_hook -> inference_safe
        6. test_case -> training_only
        7. Fallback: has compliance_tags -> inference_safe

        Returns
        -------
        str
            Lifecycle phase or "" if no classification applies.
        """
        name = rec.entity_name or ""
        source = rec.source_file or ""
        subcat = rec.subcategory or ""

        # Rule 1: backward / grad / autograd in entity name
        if self._TRAINING_PATTERN.search(name):
            return "training_only"

        # Rule 2: hook definition or consumer with "forward" in name
        if subcat in ("hook_definition", "hook_consumer") and "forward" in name.lower():
            return "inference_safe"

        # Rule 3: data loading utilities
        if "torch/utils/data/" in source:
            return "data_preparation"

        # Rule 4: deployment / export paths
        if (
            "torch/onnx/" in source
            or "torch/export/" in source
            or subcat == "torchscript"
        ):
            return "deployment"

        # Rule 5: dispatch keys and profiler hooks are inference-safe
        if subcat in ("dispatch_key", "profiler_hook"):
            return "inference_safe"

        # Rule 6: test cases default to training_only
        if rec.entity_type == "test_case":
            return "training_only"

        # Rule 7: fallback — if entity has compliance tags, conservatively
        # classify as inference_safe (it likely matters at inference time too)
        if rec.compliance_tags:
            return "inference_safe"

        return ""

    def _check_conflicts(self, rec: EntityRecord) -> None:
        """Warn when lifecycle phase seems inconsistent with other metadata."""
        # Training-only entities that also have deployment-related tags
        if rec.lifecycle_phase == "training_only" and "eu_ai_act_art_11" in rec.compliance_tags:
            logger.warning(
                "Entity '%s' (%s) is training_only but tagged with Art 11 "
                "(technical documentation) — may need deployment-phase review",
                rec.entity_name,
                rec.id,
            )


def annotate_lifecycle(records: list[EntityRecord]) -> list[EntityRecord]:
    """Module-level convenience function for lifecycle annotation."""
    return LifecycleAnnotator().annotate(records)
