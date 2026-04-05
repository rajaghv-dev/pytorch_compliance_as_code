"""
Dispatcher-level annotator for PyTorch EntityRecords.

Classifies entities by execution level:
  - python_wrapper: Python-level hooks / wrappers
  - c_dispatch: C++ dispatch keys or torch.library usage
  - compiled_graph: Dynamo graph breaks, compiled dispatch keys
"""

from __future__ import annotations

import logging
from typing import Optional

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.annotators.dispatcher_level")


class DispatcherLevelAnnotator:
    """Classifies entities by execution level: python_wrapper | c_dispatch | compiled_graph."""

    # Dispatch keys associated with compiled graph execution
    COMPILED_DISPATCH_KEYS = {"CompositeExplicitAutograd", "FuncTorchBatched"}

    def annotate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Assign execution_level to each record.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records to classify.

        Returns
        -------
        list[EntityRecord]
            The same list, with execution_level populated where applicable.
        """
        logger.info("Annotating %d records with DispatcherLevelAnnotator", len(records))
        tagged_count = 0
        new_count = 0

        for rec in records:
            had_level = bool(rec.execution_level)
            level = self._classify(rec)
            if level:
                rec.execution_level = level
                tagged_count += 1
                if not had_level:
                    new_count += 1

        logger.info(
            "Tagged %d records (%d new tags)", tagged_count, new_count
        )
        return records

    def _classify(self, rec: EntityRecord) -> str:
        """
        Determine the execution level for a single record.

        Rules are evaluated in priority order; first match wins.

        Returns
        -------
        str
            One of "python_wrapper", "c_dispatch", "compiled_graph", or ""
            if no classification applies.
        """
        raw = rec.raw_text or ""

        # ---- compiled_graph ----
        # Rule: graph break reasons from torch._dynamo
        if rec.subcategory == "graph_break_reason":
            return "compiled_graph"

        # Rule: entity lives in _dynamo path
        if "_dynamo" in (rec.source_file or ""):
            return "compiled_graph"

        # Rule: dispatch key is CompositeExplicitAutograd or FuncTorchBatched
        if rec.entity_name in self.COMPILED_DISPATCH_KEYS:
            return "compiled_graph"
        if rec.metadata.get("dispatch_key") in self.COMPILED_DISPATCH_KEYS:
            return "compiled_graph"

        # ---- c_dispatch ----
        # Rule: C++ dispatch key entities
        if rec.language == "cpp" and rec.subcategory == "dispatch_key":
            return "c_dispatch"

        # Rule: entity uses torch.library in its raw text
        if "torch.library" in raw:
            return "c_dispatch"

        # ---- python_wrapper ----
        # Rule: entity registers hooks via register_*_hook in raw_text
        if "register_" in raw and "_hook" in raw:
            return "python_wrapper"

        # Rule: hook definitions are python-level wrappers
        if rec.subcategory == "hook_definition":
            return "python_wrapper"

        # ---- Fallback ----
        # If entity has compliance tags but no classification matched,
        # default conservatively to python_wrapper.
        if rec.compliance_tags:
            return "python_wrapper"

        return ""

    def _check_conflicts(self, rec: EntityRecord) -> None:
        """Warn if the assigned level seems inconsistent with other metadata."""
        if rec.execution_level == "c_dispatch" and rec.language == "python":
            logger.warning(
                "Entity '%s' (%s) classified as c_dispatch but language is Python",
                rec.entity_name,
                rec.id,
            )


def annotate_dispatcher_level(records: list[EntityRecord]) -> list[EntityRecord]:
    """Module-level convenience function for dispatcher-level annotation."""
    return DispatcherLevelAnnotator().annotate(records)
