"""
Multi-dimensional indexer for EntityRecords.

Builds lookup indexes keyed by name, file, article, lifecycle phase,
execution level, entity type, language, and subcategory.  Also produces
a reverse lookup from record ID to the full EntityRecord.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.organizer.entity_index")


class EntityIndexer:
    """Builds multi-dimensional indexes over EntityRecords."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("pct.organizer.entity_index")

    def build_indexes(self, records: list[EntityRecord]) -> dict[str, dict[str, list[str]]]:
        """
        Build all index dimensions from the given records.

        Parameters
        ----------
        records : list[EntityRecord]
            Deduplicated entity records.

        Returns
        -------
        dict[str, dict[str, list[str]]]
            Nested dict: dimension -> key -> list of record IDs.
        """
        indexes: dict[str, dict[str, list[str]]] = {
            "by_name": {},
            "by_file": {},
            "by_article": {},
            "by_lifecycle": {},
            "by_execution_level": {},
            "by_entity_type": {},
            "by_language": {},
            "by_subcategory": {},
        }

        for i, rec in enumerate(records):
            rid = rec.id

            # by_name
            if rec.entity_name:
                indexes["by_name"].setdefault(rec.entity_name, []).append(rid)

            # by_file
            if rec.source_file:
                indexes["by_file"].setdefault(rec.source_file, []).append(rid)

            # by_article (one entry per compliance tag)
            for tag in rec.compliance_tags:
                indexes["by_article"].setdefault(tag, []).append(rid)

            # by_lifecycle
            if rec.lifecycle_phase:
                indexes["by_lifecycle"].setdefault(rec.lifecycle_phase, []).append(rid)

            # by_execution_level
            if rec.execution_level:
                indexes["by_execution_level"].setdefault(rec.execution_level, []).append(rid)

            # by_entity_type
            if rec.entity_type:
                indexes["by_entity_type"].setdefault(rec.entity_type, []).append(rid)

            # by_language
            if rec.language:
                indexes["by_language"].setdefault(rec.language, []).append(rid)

            # by_subcategory
            if rec.subcategory:
                indexes["by_subcategory"].setdefault(rec.subcategory, []).append(rid)

            # Progress logging
            if (i + 1) % 50000 == 0:
                self.logger.info("  Indexing progress: %d/%d records", i + 1, len(records))

        self.logger.info(
            "Built indexes: %s",
            {dim: len(keys) for dim, keys in indexes.items()},
        )
        return indexes

    def build_id_lookup(self, records: list[EntityRecord]) -> dict[str, dict]:
        """
        Build a reverse lookup from record ID to the full EntityRecord dict.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records.

        Returns
        -------
        dict[str, dict]
            Mapping from record ID to its serialised dictionary.
        """
        lookup: dict[str, dict] = {}
        for rec in records:
            lookup[rec.id] = rec.to_dict()
        self.logger.info("Built ID lookup for %d records", len(lookup))
        return lookup

    def write_results(
        self,
        indexes: dict[str, dict[str, list[str]]],
        id_lookup: dict[str, dict],
        output_dir: Path,
    ) -> None:
        """
        Write index and reverse lookup files atomically.

        Parameters
        ----------
        indexes : dict
            Multi-dimensional index.
        id_lookup : dict
            ID-to-record mapping.
        output_dir : Path
            Directory to write files into.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(indexes, output_dir / "entity_index.json")
        self._write_json(id_lookup, output_dir / "entities_by_id.json")

    def _write_json(self, data: Any, output_path: Path) -> None:
        """Write a JSON file atomically."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(output_path))
            self.logger.info("Wrote %s", output_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
