"""
Validator for EntityRecords.

Checks every record for required fields, data integrity, value ranges,
and duplicate IDs.  Issues are categorised as error, warning, or info.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.organizer.validation")

# Allowed values for constrained fields
ALLOWED_LANGUAGES = {"python", "cpp", "yaml", "rst", "markdown", "config"}
ALLOWED_ENTITY_TYPES = {
    "function", "method", "class", "operator", "enum",
    "doc_directive", "test_case", "config_entry",
    "commit", "dependency", "license",
    "import", "assignment", "module_class", "tool_reference",
}


class Validator:
    """Validates EntityRecords for required fields and data integrity."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("pct.organizer.validation")

    def validate(self, records: list[EntityRecord]) -> dict[str, Any]:
        """
        Validate all records and return a validation report.

        Parameters
        ----------
        records : list[EntityRecord]
            Records to validate.

        Returns
        -------
        dict[str, Any]
            Validation report with counts and issue details.
        """
        issues: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []

        error_count = 0
        warning_count = 0
        info_count = 0

        for i, rec in enumerate(records):
            # Progress logging — 25K step keeps it to ~10 lines for 227K records
            if (i + 1) % 25000 == 0:
                self.logger.info("  Validation progress: %d/%d records", i + 1, len(records))

            record_issues = self._validate_record(rec)

            # Check for duplicate IDs
            if rec.id in seen_ids:
                duplicate_ids.append(rec.id)
                record_issues.append({
                    "record_id": rec.id,
                    "entity_name": rec.entity_name,
                    "level": "error",
                    "field": "id",
                    "message": f"Duplicate ID: {rec.id}",
                })
            seen_ids.add(rec.id)

            # Count by level and accumulate
            for issue in record_issues:
                level = issue["level"]
                if level == "error":
                    error_count += 1
                elif level == "warning":
                    warning_count += 1
                else:
                    info_count += 1
            issues.extend(record_issues)

        report = {
            "total_checked": len(records),
            "errors": error_count,
            "warnings": warning_count,
            "infos": info_count,
            "duplicate_ids": duplicate_ids,
            "issues": issues,
        }

        self.logger.info(
            "Validation complete: %d checked, %d errors, %d warnings, %d infos",
            len(records), error_count, warning_count, info_count,
        )
        if duplicate_ids:
            self.logger.warning("Found %d duplicate IDs", len(duplicate_ids))

        return report

    def _validate_record(self, rec: EntityRecord) -> list[dict[str, str]]:
        """Validate a single record and return a list of issues."""
        issues: list[dict[str, str]] = []

        def _add(level: str, field_name: str, message: str) -> None:
            issues.append({
                "record_id": rec.id,
                "entity_name": rec.entity_name,
                "level": level,
                "field": field_name,
                "message": message,
            })

        # Required non-empty fields
        if not rec.id:
            _add("error", "id", "Missing required field: id")
        if not rec.source_file:
            _add("error", "source_file", "Missing required field: source_file")
        if not rec.entity_name:
            _add("error", "entity_name", "Missing required field: entity_name")
        if not rec.entity_type:
            _add("error", "entity_type", "Missing required field: entity_type")

        # raw_text checks
        if not rec.raw_text:
            _add("info", "raw_text", "raw_text is empty")
        elif len(rec.raw_text) > 5000:
            _add("warning", "raw_text", "raw_text truncated at 5000 chars")

        # Confidence range checks
        if not (0.0 <= rec.extraction_confidence <= 1.0):
            _add(
                "warning", "extraction_confidence",
                f"extraction_confidence out of range [0,1]: {rec.extraction_confidence}",
            )
        if not (0.0 <= rec.mapping_confidence <= 1.0):
            _add(
                "warning", "mapping_confidence",
                f"mapping_confidence out of range [0,1]: {rec.mapping_confidence}",
            )

        # Language validation
        if rec.language and rec.language not in ALLOWED_LANGUAGES:
            _add(
                "warning", "language",
                f"Unrecognised language: '{rec.language}' "
                f"(allowed: {', '.join(sorted(ALLOWED_LANGUAGES))})",
            )

        # Entity type validation
        if rec.entity_type and rec.entity_type not in ALLOWED_ENTITY_TYPES:
            _add(
                "warning", "entity_type",
                f"Unrecognised entity_type: '{rec.entity_type}' "
                f"(allowed: {', '.join(sorted(ALLOWED_ENTITY_TYPES))})",
            )

        # Data quality notes
        if not rec.docstring:
            _add("info", "docstring", "No docstring available")
        if not rec.compliance_tags:
            _add("info", "compliance_tags", "No compliance tags assigned")

        return issues

    def write_results(self, report: dict[str, Any], output_path: Path) -> None:
        """Write the validation report to a JSON file atomically."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(output_path))
            self.logger.info("Wrote validation report to %s", output_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
