"""
Deduplicator for EntityRecords across all JSONL sources.

Loads raw JSONL files from storage/raw/, identifies duplicate records using a
composite key, and merges them by keeping the richest version of each entity.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.organizer.dedup")


class Deduplicator:
    """Deduplicates EntityRecords across all JSONL sources."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("pct.organizer.dedup")

    def load_all_records(self, raw_dir: Path) -> list[EntityRecord]:
        """
        Load all EntityRecords from JSONL files in raw_dir.

        Each line of a .jsonl file is expected to be a JSON object representing
        an EntityRecord.  Plain .json files containing a JSON array of records
        are also supported for backward compatibility.

        Parameters
        ----------
        raw_dir : Path
            Directory containing JSONL (or JSON) output files.

        Returns
        -------
        list[EntityRecord]
            All loaded records.
        """
        records: list[EntityRecord] = []
        if not raw_dir.exists():
            self.logger.warning("Raw directory does not exist: %s", raw_dir)
            return records

        jsonl_files = sorted(raw_dir.glob("*.jsonl"))
        json_files = sorted(raw_dir.glob("*.json"))
        all_files = jsonl_files + json_files

        if not all_files:
            self.logger.warning("No JSONL/JSON files found in %s", raw_dir)
            return records

        for filepath in all_files:
            try:
                file_records = self._load_file(filepath)
                records.extend(file_records)
            except Exception as exc:
                self.logger.error("Failed to load %s: %s", filepath, exc)

        self.logger.info(
            "Loaded %d records from %d files", len(records), len(all_files)
        )
        return records

    def _load_file(self, filepath: Path) -> list[EntityRecord]:
        """Load records from a single JSONL or JSON file."""
        records: list[EntityRecord] = []
        suffix = filepath.suffix.lower()

        with open(filepath, "r", encoding="utf-8") as fh:
            if suffix == ".jsonl":
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        records.append(EntityRecord.from_dict(d))
                    except (json.JSONDecodeError, TypeError) as exc:
                        self.logger.warning(
                            "Skipping malformed line %d in %s: %s",
                            line_num, filepath.name, exc,
                        )
            else:
                # Plain JSON — expect a list of dicts
                data = json.load(fh)
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict):
                            records.append(EntityRecord.from_dict(d))
                elif isinstance(data, dict):
                    records.append(EntityRecord.from_dict(data))

        self.logger.info("  %s: %d records", filepath.name, len(records))
        return records

    def deduplicate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Deduplicate records using composite key:
        source_file + entity_name + entity_type + subcategory + start_line.

        When duplicates are found, the richer record is kept and fields are
        merged from the secondary record.

        Parameters
        ----------
        records : list[EntityRecord]
            Input records (may contain duplicates).

        Returns
        -------
        list[EntityRecord]
            Deduplicated records.
        """
        # Group by dedup key
        groups: dict[tuple, list[EntityRecord]] = {}
        for rec in records:
            key = (
                rec.source_file,
                rec.entity_name,
                rec.entity_type,
                rec.subcategory,
                rec.start_line,
            )
            groups.setdefault(key, []).append(rec)

        result: list[EntityRecord] = []
        duplicates_removed = 0

        for key, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Sort by richness descending; richest becomes primary
                group.sort(key=lambda r: count_populated_fields(r), reverse=True)
                merged = group[0]
                for secondary in group[1:]:
                    merged = merge_records(merged, secondary)
                    duplicates_removed += 1
                result.append(merged)

            # Progress logging for large datasets
            if len(result) % 25000 == 0 and len(result) > 0:
                self.logger.info("  Dedup progress: %d unique records so far", len(result))

        self.logger.info(
            "Deduplicated to %d unique records (removed %d duplicates)",
            len(result), duplicates_removed,
        )
        return result

    def write_results(self, records: list[EntityRecord], output_path: Path) -> None:
        """
        Write deduplicated records to a JSONL file atomically.

        Parameters
        ----------
        records : list[EntityRecord]
            Records to write.
        output_path : Path
            Destination JSONL file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp_path, str(output_path))
            self.logger.info("Wrote %d records to %s", len(records), output_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def count_populated_fields(record: EntityRecord) -> int:
    """
    Count the number of non-empty / non-default fields in a record.

    Used to determine which record is "richer" when merging duplicates.
    A field is considered populated if it is a non-empty string, a non-zero
    number, or a non-empty collection.
    """
    count = 0
    for f in fields(record):
        val = getattr(record, f.name)
        if isinstance(val, str) and val:
            count += 1
        elif isinstance(val, (list, dict)) and val:
            count += 1
        elif isinstance(val, float) and val != 0.0:
            count += 1
        elif isinstance(val, int) and val != 0:
            count += 1
    return count


def merge_records(primary: EntityRecord, secondary: EntityRecord) -> EntityRecord:
    """
    Merge two duplicate records, keeping the primary as the base.

    Rules:
    - Prefer specialized extractor's compliance_tags over empty tags
    - Merge metadata dicts (primary wins on conflicts)
    - Merge relations lists (deduplicate by type+target)
    - Keep highest extraction_confidence and mapping_confidence

    Parameters
    ----------
    primary : EntityRecord
        The richer record (base).
    secondary : EntityRecord
        The less-rich record to merge from.

    Returns
    -------
    EntityRecord
        Merged record.
    """
    # Merge compliance_tags (union, preserving order)
    if not primary.compliance_tags and secondary.compliance_tags:
        primary.compliance_tags = list(secondary.compliance_tags)
    elif secondary.compliance_tags:
        existing = set(primary.compliance_tags)
        for tag in secondary.compliance_tags:
            if tag not in existing:
                primary.compliance_tags.append(tag)
                existing.add(tag)

    # Merge metadata dicts — secondary fills gaps, primary wins conflicts
    if secondary.metadata:
        merged_meta = dict(secondary.metadata)
        merged_meta.update(primary.metadata)
        primary.metadata = merged_meta

    # Merge annotations similarly
    if secondary.annotations:
        merged_ann = dict(secondary.annotations)
        merged_ann.update(primary.annotations)
        primary.annotations = merged_ann

    # Merge relations — deduplicate by (type, target)
    if secondary.relations:
        existing_keys: set[tuple] = set()
        for rel in primary.relations:
            if isinstance(rel, dict):
                existing_keys.add((rel.get("type", ""), rel.get("target", "")))
        for rel in secondary.relations:
            if isinstance(rel, dict):
                key = (rel.get("type", ""), rel.get("target", ""))
                if key not in existing_keys:
                    primary.relations.append(rel)
                    existing_keys.add(key)

    # Keep highest confidence scores
    primary.extraction_confidence = max(
        primary.extraction_confidence, secondary.extraction_confidence
    )
    primary.mapping_confidence = max(
        primary.mapping_confidence, secondary.mapping_confidence
    )

    # Fill empty string fields from secondary
    for f in fields(primary):
        if f.name in ("id", "extractor", "timestamp"):
            continue
        val = getattr(primary, f.name)
        if isinstance(val, str) and not val:
            secondary_val = getattr(secondary, f.name)
            if isinstance(secondary_val, str) and secondary_val:
                setattr(primary, f.name, secondary_val)

    # Merge export_survival
    if secondary.export_survival and not primary.export_survival:
        primary.export_survival = dict(secondary.export_survival)
    elif secondary.export_survival and primary.export_survival:
        for k, v in secondary.export_survival.items():
            if k not in primary.export_survival:
                primary.export_survival[k] = v

    return primary
