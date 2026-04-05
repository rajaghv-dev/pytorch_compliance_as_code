"""
Export survival annotator for PyTorch EntityRecords.

Classifies entities by their survival across 5 export formats:
  - onnx
  - exported_program
  - torchscript
  - torch_compile
  - distributed_checkpoint

Each target is assigned "yes", "no", or "partial" indicating whether the
entity's compliance behaviour is preserved through that export path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.annotators.export_survival")


class ExportSurvivalAnnotator:
    """Classifies entities by survival across 5 export formats."""

    EXPORT_TARGETS = [
        "onnx",
        "exported_program",
        "torchscript",
        "torch_compile",
        "distributed_checkpoint",
    ]

    def annotate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Assign export_survival to each record with compliance_tags.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records to classify.

        Returns
        -------
        list[EntityRecord]
            The same list, with export_survival populated.
        """
        logger.info(
            "Annotating %d records with ExportSurvivalAnnotator", len(records)
        )
        tagged_count = 0
        new_count = 0

        EXPORT_SUBCATEGORIES = {
            "onnx_export", "torchscript", "dynamo_compile", "fx_graph",
            "serialization", "graph_break_reason", "torch_export",
            "hook_definition", "dispatch_key", "autograd_provenance",
            "profiler_hook", "data_loading",
        }

        for rec in records:
            # Only skip if record has no compliance tags AND is not an export-relevant subcategory
            if not rec.compliance_tags and rec.subcategory not in EXPORT_SUBCATEGORIES:
                continue

            had_survival = bool(rec.export_survival)
            survival = self._classify(rec)
            if survival:
                rec.export_survival = survival
                tagged_count += 1
                if not had_survival:
                    new_count += 1

        logger.info(
            "Tagged %d records (%d new tags)", tagged_count, new_count
        )
        return records

    def _classify(self, rec: EntityRecord) -> dict[str, str]:
        """
        Determine export survival for a single record.

        Patterns are checked in priority order; first match wins.

        Returns
        -------
        dict[str, str]
            Map of export target -> "yes" | "no" | "partial", or {}
            if no pattern matches.
        """
        name = rec.entity_name or ""
        name_lower = name.lower()
        qname = (rec.qualified_name or "").lower()
        subcat = rec.subcategory or ""

        # ---- State dict hooks: fire during DCP save/load ----
        # Check this BEFORE generic hook_definition to catch the specific case.
        if subcat == "hook_definition" and "state_dict" in name_lower:
            return {
                "onnx": "no",
                "exported_program": "no",
                "torchscript": "no",
                "torch_compile": "no",
                "distributed_checkpoint": "yes",
            }

        # ---- Python forward/backward hooks ----
        # Hooks are Python callbacks; most export formats strip them.
        if subcat == "hook_definition" and "forward" in name_lower:
            return {
                "onnx": "no",
                "exported_program": "no",
                "torchscript": "partial",
                "torch_compile": "no",
                "distributed_checkpoint": "yes",
            }

        # ---- Dispatch keys ----
        # Dispatch keys are a C++/compiler concept; survive in compile but
        # not in graph-based exports.
        if subcat == "dispatch_key":
            return {
                "onnx": "no",
                "exported_program": "partial",
                "torchscript": "no",
                "torch_compile": "yes",
                "distributed_checkpoint": "no",
            }

        # ---- Autograd Functions ----
        # Custom autograd.Function subclasses survive through most compiled paths.
        if rec.entity_type == "class" and (
            "autograd" in qname or subcat == "autograd_function"
        ):
            return {
                "onnx": "no",
                "exported_program": "yes",
                "torchscript": "yes",
                "torch_compile": "yes",
                "distributed_checkpoint": "no",
            }

        # ---- Profiler hooks ----
        # Profiling is mostly stripped during export.
        if subcat == "profiler_hook":
            return {
                "onnx": "no",
                "exported_program": "no",
                "torchscript": "no",
                "torch_compile": "partial",
                "distributed_checkpoint": "no",
            }

        # ---- Data loading entities ----
        # Data loading is a Python-only runtime concept; no export path preserves it.
        if subcat == "data_loading":
            return {
                "onnx": "no",
                "exported_program": "no",
                "torchscript": "no",
                "torch_compile": "no",
                "distributed_checkpoint": "no",
            }

        # ---- torch.export entities ----
        # These are designed to survive exported_program and torch.compile.
        if subcat == "torch_export":
            return {
                "onnx": "partial",
                "exported_program": "yes",
                "torchscript": "no",
                "torch_compile": "yes",
                "distributed_checkpoint": "no",
            }

        # ---- ONNX export entities ----
        if subcat == "onnx_export":
            return {
                "onnx": "yes",
                "exported_program": "no",
                "torchscript": "partial",
                "torch_compile": "no",
                "distributed_checkpoint": "no",
            }

        # ---- TorchScript entities ----
        if subcat == "torchscript":
            return {
                "onnx": "partial",
                "exported_program": "no",
                "torchscript": "yes",
                "torch_compile": "partial",
                "distributed_checkpoint": "no",
            }

        # ---- Dynamo/torch.compile entities ----
        if subcat == "dynamo_compile":
            return {
                "onnx": "partial",
                "exported_program": "partial",
                "torchscript": "no",
                "torch_compile": "yes",
                "distributed_checkpoint": "no",
            }

        # ---- FX graph entities ----
        # FX IR is the bridge between compile formats.
        if subcat == "fx_graph":
            return {
                "onnx": "partial",
                "exported_program": "partial",
                "torchscript": "partial",
                "torch_compile": "yes",
                "distributed_checkpoint": "no",
            }

        # ---- Serialization entities ----
        if subcat == "serialization":
            return {
                "onnx": "partial",
                "exported_program": "partial",
                "torchscript": "partial",
                "torch_compile": "no",
                "distributed_checkpoint": "yes",
            }

        # ---- Graph break reason annotations ----
        # These indicate torch.compile limitations.
        if subcat == "graph_break_reason":
            return {
                "onnx": "no",
                "exported_program": "no",
                "torchscript": "no",
                "torch_compile": "no",
                "distributed_checkpoint": "no",
            }

        # No pattern matched — return empty dict.
        return {}


def generate_survival_matrix(records: list[EntityRecord]) -> dict[str, Any]:
    """
    Build a summary survival matrix for the top 50 compliance entities.

    Entities are sorted by mapping_confidence descending.  The matrix shows
    each entity and its survival status across all export targets.

    Parameters
    ----------
    records : list[EntityRecord]
        All annotated records.

    Returns
    -------
    dict
        A dictionary with keys:
        - "entities": list of dicts with entity_name, id, mapping_confidence,
          and per-target survival status.
        - "summary": per-target counts of yes/no/partial.
        - "total_classified": number of entities with export_survival data.
    """
    # Filter to entities that have export_survival data
    classified = [r for r in records if r.export_survival]

    # Sort by mapping_confidence descending, take top 50
    classified.sort(key=lambda r: r.mapping_confidence, reverse=True)
    top_50 = classified[:50]

    targets = ExportSurvivalAnnotator.EXPORT_TARGETS

    # Build per-entity rows
    entities = []
    for rec in top_50:
        row = {
            "entity_name": rec.entity_name,
            "id": rec.id,
            "mapping_confidence": rec.mapping_confidence,
        }
        for target in targets:
            row[target] = rec.export_survival.get(target, "unknown")
        entities.append(row)

    # Build summary counts
    summary: dict[str, dict[str, int]] = {}
    for target in targets:
        summary[target] = {"yes": 0, "no": 0, "partial": 0}
    for rec in classified:
        for target in targets:
            status = rec.export_survival.get(target, "")
            if status in summary.get(target, {}):
                summary[target][status] += 1

    return {
        "entities": entities,
        "summary": summary,
        "total_classified": len(classified),
    }


def annotate_export_survival(records: list[EntityRecord]) -> list[EntityRecord]:
    """Module-level convenience function for export survival annotation."""
    return ExportSurvivalAnnotator().annotate(records)
