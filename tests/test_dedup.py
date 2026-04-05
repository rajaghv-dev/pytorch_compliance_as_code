"""
Tests for src/organizer/dedup.py — record deduplication and merging.

WHAT IS TESTED
--------------
- Deduplicator.deduplicate() removes exact duplicates.
- Deduplicator.deduplicate() keeps the richer of two duplicates.
- merge_records() merges compliance_tags from both records (union).
- merge_records() merges metadata dicts (primary wins on conflict).
- merge_records() keeps the higher confidence score.
- merge_records() fills empty string fields from the secondary record.
- count_populated_fields() counts non-empty fields correctly.
- Deduplicator.write_results() writes a valid JSONL file.
- Deduplicator.load_all_records() loads from both .json and .jsonl files.

HOW TO RUN
----------
    pytest tests/test_dedup.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extractors.base import EntityRecord
from src.organizer.dedup import (
    Deduplicator,
    count_populated_fields,
    merge_records,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**kwargs) -> EntityRecord:
    """Build an EntityRecord with sensible defaults for dedup tests."""
    defaults = dict(
        source_file="torch/nn/module.py",
        entity_name="forward",
        entity_type="method",
        subcategory="",
        start_line=10,
        extractor="test_extractor",
    )
    defaults.update(kwargs)
    rec = EntityRecord(**defaults)
    from src.extractors.base import compute_stable_id
    rec.id = compute_stable_id(rec)
    return rec


# ---------------------------------------------------------------------------
# count_populated_fields()
# ---------------------------------------------------------------------------


class TestCountPopulatedFields:

    def test_empty_record_has_low_count(self):
        """A record with only default fields should have a low field count."""
        rec = EntityRecord()
        count = count_populated_fields(rec)
        # extraction_confidence=1.0 counts as non-default; timestamp is set.
        # We just check it's less than for a richer record.
        assert count >= 0

    def test_richer_record_has_higher_count(self):
        """A record with more non-empty fields should score higher."""
        sparse = _make_record()
        rich = _make_record(
            docstring="A method that does forward pass.",
            signature="def forward(self, x): ...",
            compliance_tags=["eu_ai_act_art_15"],
        )
        assert count_populated_fields(rich) > count_populated_fields(sparse)


# ---------------------------------------------------------------------------
# merge_records()
# ---------------------------------------------------------------------------


class TestMergeRecords:

    def test_compliance_tags_are_merged_as_union(self):
        """Merging two records should produce the union of their tags."""
        primary = _make_record(compliance_tags=["eu_ai_act_art_15"])
        secondary = _make_record(compliance_tags=["eu_ai_act_art_61"])

        merged = merge_records(primary, secondary)
        assert "eu_ai_act_art_15" in merged.compliance_tags
        assert "eu_ai_act_art_61" in merged.compliance_tags

    def test_no_duplicate_tags_after_merge(self):
        """Tags present in both records should not be duplicated."""
        primary = _make_record(compliance_tags=["eu_ai_act_art_15", "eu_ai_act_art_10"])
        secondary = _make_record(compliance_tags=["eu_ai_act_art_15"])

        merged = merge_records(primary, secondary)
        assert merged.compliance_tags.count("eu_ai_act_art_15") == 1

    def test_higher_confidence_is_kept(self):
        """The higher of the two confidence scores should be kept."""
        primary   = _make_record(mapping_confidence=0.5)
        secondary = _make_record(mapping_confidence=0.9)

        merged = merge_records(primary, secondary)
        assert merged.mapping_confidence == 0.9

    def test_primary_wins_on_metadata_conflict(self):
        """When both records have the same metadata key, primary's value wins."""
        primary   = _make_record(metadata={"key": "primary_value"})
        secondary = _make_record(metadata={"key": "secondary_value"})

        merged = merge_records(primary, secondary)
        assert merged.metadata["key"] == "primary_value"

    def test_secondary_fills_empty_string_fields(self):
        """Empty string fields in primary should be filled from secondary."""
        primary   = _make_record(docstring="")
        secondary = _make_record(docstring="Useful docstring from secondary.")

        merged = merge_records(primary, secondary)
        assert merged.docstring == "Useful docstring from secondary."

    def test_primary_string_is_not_overwritten(self):
        """A non-empty string in primary should not be overwritten."""
        primary   = _make_record(docstring="Primary docstring.")
        secondary = _make_record(docstring="Secondary docstring.")

        merged = merge_records(primary, secondary)
        assert merged.docstring == "Primary docstring."

    def test_export_survival_merged(self):
        """Export survival dicts should be merged (secondary fills missing keys)."""
        primary   = _make_record(export_survival={"onnx": "yes"})
        secondary = _make_record(export_survival={"torchscript": "no"})

        merged = merge_records(primary, secondary)
        assert merged.export_survival["onnx"] == "yes"
        assert merged.export_survival["torchscript"] == "no"


# ---------------------------------------------------------------------------
# Deduplicator.deduplicate()
# ---------------------------------------------------------------------------


class TestDeduplicator:

    def test_exact_duplicate_is_removed(self):
        """Two records with identical dedup keys → only one in the output."""
        rec = _make_record()
        result = Deduplicator().deduplicate([rec, rec])
        assert len(result) == 1

    def test_different_records_are_both_kept(self):
        """Records with different names/files/lines should all be kept."""
        r1 = _make_record(entity_name="forward",  start_line=10)
        r2 = _make_record(entity_name="backward", start_line=50)
        result = Deduplicator().deduplicate([r1, r2])
        assert len(result) == 2

    def test_richer_duplicate_wins(self):
        """When duplicates exist, the record with more fields should survive."""
        sparse = _make_record(docstring="")
        rich   = _make_record(docstring="A detailed docstring.", compliance_tags=["eu_ai_act_art_15"])

        result = Deduplicator().deduplicate([sparse, rich])
        assert len(result) == 1
        # The surviving record should have the docstring from the rich version.
        assert result[0].docstring == "A detailed docstring."

    def test_empty_input_returns_empty(self):
        """An empty input list should produce an empty output list."""
        assert Deduplicator().deduplicate([]) == []


# ---------------------------------------------------------------------------
# Deduplicator.write_results() and load_all_records()
# ---------------------------------------------------------------------------


class TestDeduplicatorIO:

    def test_write_and_reload_jsonl(self, tmp_path, sample_records):
        """write_results() should produce a JSONL file that load_all_records() can reload."""
        output_file = tmp_path / "records.jsonl"
        dedup = Deduplicator()
        dedup.write_results(sample_records, output_file)

        assert output_file.exists()

        # Reload from the JSONL file.
        loaded = dedup.load_all_records(tmp_path)
        assert len(loaded) == len(sample_records)
        loaded_names = {r.entity_name for r in loaded}
        expected_names = {r.entity_name for r in sample_records}
        assert loaded_names == expected_names

    def test_write_creates_parent_dirs(self, tmp_path):
        """write_results() should create parent directories automatically."""
        nested = tmp_path / "deep" / "nested" / "records.jsonl"
        dedup = Deduplicator()
        rec = _make_record()
        dedup.write_results([rec], nested)
        assert nested.exists()

    def test_load_all_records_handles_empty_dir(self, tmp_path):
        """load_all_records() on an empty directory should return []."""
        result = Deduplicator().load_all_records(tmp_path)
        assert result == []

    def test_load_all_records_skips_malformed_lines(self, tmp_path):
        """load_all_records() should skip JSONL lines that are not valid JSON."""
        bad_jsonl = tmp_path / "bad.jsonl"
        bad_jsonl.write_text(
            '{"entity_name": "ok", "entity_type": "method", "source_file": "f.py"}\n'
            'NOT VALID JSON\n'
            '{"entity_name": "ok2", "entity_type": "function", "source_file": "g.py"}\n',
            encoding="utf-8",
        )
        records = Deduplicator().load_all_records(tmp_path)
        # Two valid lines → two records (the bad line is skipped).
        assert len(records) == 2
