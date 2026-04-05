"""
CSV converter for EntityRecords.

Writes a flat CSV file where each row represents one compliance-relevant
entity extracted from the PyTorch source tree.

WHY CSV?
    CSV is the most portable output format.  A compliance reviewer who does
    not know Python can open it in Excel or Google Sheets to filter, sort, and
    annotate evidence without installing anything.

COLUMNS
    The CSV includes all human-readable fields but omits raw_text (too long)
    and relations (nested structure that doesn't flatten well).
    Export survival is exploded into one column per export format.

OUTPUT
    storage/output/compliance_evidence.csv
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.converters.csv")


class CsvConverter:
    """Converts a list of EntityRecords to a flat CSV file."""

    # Columns included in the output, in display order.
    # If you need to add a new column, add it here and in _to_row().
    COLUMNS: list[str] = [
        "id",
        "entity_name",
        "entity_type",
        "subcategory",
        "language",
        "source_file",
        "module_path",
        "qualified_name",
        "start_line",
        "end_line",
        "lifecycle_phase",
        "execution_level",
        "distributed_safety",
        "compliance_tags",               # comma-separated list
        "mapping_confidence",            # 0.000 – 1.000
        "extraction_confidence",         # 0.000 – 1.000
        "mapping_rationale",
        "docstring_excerpt",             # first 200 chars of docstring
        "export_onnx",
        "export_exported_program",
        "export_torchscript",
        "export_torch_compile",
        "export_distributed_checkpoint",
        "extractor",
        "timestamp",
    ]

    def convert(self, records: list[EntityRecord], output_path: Path) -> None:
        """
        Write *records* to a CSV file at *output_path*.

        Creates parent directories automatically.

        Parameters
        ----------
        records : list[EntityRecord]
            The entity records to serialise.
        output_path : Path
            Destination file path (should end with .csv).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "CSV: writing %d records → %s", len(records), output_path
        )

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as fh:
                # Plain utf-8 — no BOM, compatible with all tools.
                writer = csv.DictWriter(
                    fh,
                    fieldnames=self.COLUMNS,
                    extrasaction="ignore",  # ignore any extra keys silently
                )
                writer.writeheader()
                written = 0
                for rec in records:
                    writer.writerow(self._to_row(rec))
                    written += 1

            logger.info("CSV: done — %d rows written to %s", written, output_path)

        except OSError as exc:
            logger.error("CSV: failed to write '%s': %s", output_path, exc)
            raise

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _to_row(self, rec: EntityRecord) -> dict:
        """
        Flatten one EntityRecord into a CSV-safe dictionary.

        All values are plain strings so csv.DictWriter can write them
        without extra quoting logic.
        """
        export = rec.export_survival or {}

        return {
            "id":                           rec.id,
            "entity_name":                  rec.entity_name,
            "entity_type":                  rec.entity_type,
            "subcategory":                  rec.subcategory,
            "language":                     rec.language,
            "source_file":                  rec.source_file,
            "module_path":                  rec.module_path,
            "qualified_name":               rec.qualified_name,
            "start_line":                   str(rec.start_line),
            "end_line":                     str(rec.end_line),
            "lifecycle_phase":              rec.lifecycle_phase,
            "execution_level":              rec.execution_level,
            "distributed_safety":           rec.distributed_safety,
            # Join compliance tags as a comma-separated string.
            "compliance_tags":              "; ".join(rec.compliance_tags),
            "mapping_confidence":           f"{rec.mapping_confidence:.3f}",
            "extraction_confidence":        f"{rec.extraction_confidence:.3f}",
            "mapping_rationale":            rec.mapping_rationale,
            # Truncate docstring to keep the file scannable.
            "docstring_excerpt":            (rec.docstring or "")[:200],
            # Export survival columns (empty string if not applicable).
            "export_onnx":                  str(export.get("onnx", "")),
            "export_exported_program":      str(export.get("exported_program", "")),
            "export_torchscript":           str(export.get("torchscript", "")),
            "export_torch_compile":         str(export.get("torch_compile", "")),
            "export_distributed_checkpoint": str(
                export.get("distributed_checkpoint", "")
            ),
            "extractor":                    rec.extractor,
            "timestamp":                    rec.timestamp,
        }
