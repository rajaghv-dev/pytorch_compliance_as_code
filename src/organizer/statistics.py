"""
Compliance statistics computer for EntityRecords.

Computes aggregate statistics including counts by dimension, compliance
density, confidence distributions, per-module compliance breakdown,
export survival summaries, and top compliance entities.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.organizer.statistics")


class StatisticsComputer:
    """Computes compliance statistics over the entity set."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("pct.organizer.statistics")

    def compute(self, records: list[EntityRecord]) -> dict[str, Any]:
        """
        Compute all statistics from the given records.

        Parameters
        ----------
        records : list[EntityRecord]
            Deduplicated entity records.

        Returns
        -------
        dict[str, Any]
            Complete statistics dictionary.
        """
        total = len(records)
        self.logger.info("Computing statistics for %d records", total)

        stats: dict[str, Any] = {
            "total_entities": total,
            "by_language": self._count_by_field(records, "language"),
            "by_entity_type": self._count_by_field(records, "entity_type"),
            "by_subcategory": self._count_by_field(records, "subcategory"),
        }

        # Compliance density
        tagged = [r for r in records if r.compliance_tags]
        stats["compliance_tagged"] = len(tagged)
        stats["compliance_density"] = round(len(tagged) / total, 4) if total > 0 else 0.0

        # Per-article counts
        per_article: dict[str, int] = defaultdict(int)
        for rec in records:
            for tag in rec.compliance_tags:
                per_article[tag] += 1
        stats["per_article"] = dict(per_article)

        # Confidence distributions
        stats["confidence_distributions"] = {
            "extraction": self._confidence_histogram(
                [r.extraction_confidence for r in records]
            ),
            "mapping": self._confidence_histogram(
                [r.mapping_confidence for r in records]
            ),
        }

        # Per-module compliance (top-level module from module_path)
        per_module: dict[str, int] = defaultdict(int)
        for rec in records:
            if rec.module_path and rec.compliance_tags:
                # Extract top-level module (e.g. "torch.nn" from "torch.nn.modules.module")
                parts = rec.module_path.split(".")
                if len(parts) >= 2:
                    top_module = f"{parts[0]}.{parts[1]}"
                else:
                    top_module = parts[0]
                per_module[top_module] += 1
        stats["per_module_compliance"] = dict(per_module)

        # Export survival summary
        stats["export_survival_summary"] = self._export_survival_summary(records)

        # Top 20 compliance entities by mapping_confidence
        compliance_entities = [r for r in records if r.compliance_tags]
        compliance_entities.sort(key=lambda r: r.mapping_confidence, reverse=True)
        stats["top_compliance_entities"] = [
            {
                "id": r.id,
                "entity_name": r.entity_name,
                "compliance_tags": r.compliance_tags,
                "mapping_confidence": r.mapping_confidence,
            }
            for r in compliance_entities[:20]
        ]

        self.logger.info(
            "Statistics: %d total, %d tagged (density=%.4f), %d articles",
            total, len(tagged), stats["compliance_density"], len(per_article),
        )
        return stats

    def _count_by_field(self, records: list[EntityRecord], field_name: str) -> dict[str, int]:
        """Count records grouped by a string field value."""
        counts: dict[str, int] = defaultdict(int)
        for rec in records:
            val = getattr(rec, field_name, "")
            if val:
                counts[val] += 1
        return dict(counts)

    def _confidence_histogram(self, values: list[float]) -> dict[str, int]:
        """
        Bucket confidence values into 5 bins: 0.0-0.2, 0.2-0.4, etc.

        The last bin is inclusive on both ends: [0.8, 1.0].
        """
        bins = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }
        for v in values:
            if v < 0.2:
                bins["0.0-0.2"] += 1
            elif v < 0.4:
                bins["0.2-0.4"] += 1
            elif v < 0.6:
                bins["0.4-0.6"] += 1
            elif v < 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1
        return bins

    def _export_survival_summary(self, records: list[EntityRecord]) -> dict[str, dict[str, int]]:
        """
        Summarise export survival across all records.

        Groups by export format, counting yes/no/partial values.
        """
        summary: dict[str, dict[str, int]] = {}
        for rec in records:
            if not rec.export_survival:
                continue
            for fmt, status in rec.export_survival.items():
                if fmt not in summary:
                    summary[fmt] = {"yes": 0, "no": 0, "partial": 0}
                status_str = str(status).lower()
                if status_str in summary[fmt]:
                    summary[fmt][status_str] += 1
        return summary

    def write_results(self, stats: dict[str, Any], output_path: Path) -> None:
        """Write statistics to a JSON file atomically."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(output_path))
            self.logger.info("Wrote statistics to %s", output_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
