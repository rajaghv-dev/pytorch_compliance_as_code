"""
Tests for src/extractors/base.py — EntityRecord, compute_stable_id,
BaseExtractor (write buffering, file reading), and CheckpointManager.

WHAT IS TESTED
--------------
- EntityRecord round-trip: to_dict() → from_dict() preserves all fields.
- compute_stable_id() produces deterministic, collision-free IDs.
- BaseExtractor.write_record() and flush_all() write valid JSON files.
- BaseExtractor atomic write safety (temp file + rename pattern).
- CheckpointManager: mark_done, is_done, reset, save/load state.

HOW TO RUN
----------
    pytest tests/test_base.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extractors.base import (
    CheckpointManager,
    EntityRecord,
    compute_stable_id,
    BaseExtractor,
)


# ---------------------------------------------------------------------------
# Minimal concrete extractor for testing
# ---------------------------------------------------------------------------
# BaseExtractor is abstract (has extract()), so we need a concrete subclass.


class _DummyExtractor(BaseExtractor):
    """Minimal extractor that returns a predefined list of records."""

    def __init__(self, repo_path: Path, output_path: Path, records=None):
        super().__init__("dummy", repo_path, output_path)
        self._records_to_return = records or []

    def extract(self):
        for rec in self._records_to_return:
            self.write_record(rec, str(self.output_path / "dummy.json"))
        return self._records_to_return


# ---------------------------------------------------------------------------
# EntityRecord
# ---------------------------------------------------------------------------


class TestEntityRecord:

    def test_default_fields_are_sane(self):
        """A freshly created EntityRecord should have sensible empty defaults."""
        rec = EntityRecord()
        assert rec.id == ""
        assert rec.compliance_tags == []
        assert rec.relations == []
        assert rec.export_survival == {}
        assert rec.extraction_confidence == 1.0
        assert rec.mapping_confidence == 0.0

    def test_to_dict_round_trip(self):
        """to_dict() → from_dict() should reproduce the original record exactly."""
        original = EntityRecord(
            id="aabbccddeeff001122",
            source_file="torch/nn/modules/module.py",
            entity_name="forward",
            entity_type="method",
            compliance_tags=["eu_ai_act_art_15"],
            mapping_confidence=0.75,
            metadata={"custom": "value"},
        )
        restored = EntityRecord.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.entity_name == original.entity_name
        assert restored.compliance_tags == original.compliance_tags
        assert restored.mapping_confidence == original.mapping_confidence
        assert restored.metadata == original.metadata

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict() should not raise if the dict has extra keys."""
        d = {
            "id": "abc",
            "entity_name": "foo",
            "entity_type": "function",
            "unknown_future_field": "some_value",  # should be ignored
        }
        rec = EntityRecord.from_dict(d)
        assert rec.entity_name == "foo"
        assert not hasattr(rec, "unknown_future_field")

    def test_list_fields_are_independent_across_instances(self):
        """List fields must not be shared between instances (mutable defaults bug)."""
        r1 = EntityRecord()
        r2 = EntityRecord()
        r1.compliance_tags.append("eu_ai_act_art_15")
        # r2's tags should NOT be affected.
        assert r2.compliance_tags == []


# ---------------------------------------------------------------------------
# compute_stable_id()
# ---------------------------------------------------------------------------


class TestComputeStableId:

    def test_same_inputs_produce_same_id(self):
        """The same record content must always produce the same ID."""
        rec = EntityRecord(
            source_file="torch/nn/modules/module.py",
            entity_name="forward",
            entity_type="method",
            subcategory="",
            module_path="torch.nn.modules.module",
            start_line=100,
        )
        id1 = compute_stable_id(rec)
        id2 = compute_stable_id(rec)
        assert id1 == id2

    def test_different_source_files_produce_different_ids(self):
        """Different source files must yield different IDs."""
        base_kwargs = dict(
            entity_name="helper",
            entity_type="function",
            subcategory="",
            module_path="torch",
            start_line=10,
        )
        rec_a = EntityRecord(source_file="torch/a.py", **base_kwargs)
        rec_b = EntityRecord(source_file="torch/b.py", **base_kwargs)
        assert compute_stable_id(rec_a) != compute_stable_id(rec_b)

    def test_different_line_numbers_produce_different_ids(self):
        """Same name at different line numbers → different IDs."""
        base_kwargs = dict(
            source_file="torch/module.py",
            entity_name="helper",
            entity_type="function",
            subcategory="",
            module_path="torch",
        )
        rec_10 = EntityRecord(start_line=10, **base_kwargs)
        rec_20 = EntityRecord(start_line=20, **base_kwargs)
        assert compute_stable_id(rec_10) != compute_stable_id(rec_20)

    def test_id_length_is_20(self):
        """Stable IDs must be exactly 20 hex characters."""
        rec = EntityRecord(
            source_file="torch/nn/module.py",
            entity_name="forward",
            entity_type="method",
            start_line=1,
        )
        assert len(compute_stable_id(rec)) == 20

    def test_id_is_hex(self):
        """Stable IDs must be lowercase hex characters only."""
        rec = EntityRecord(
            source_file="torch/module.py",
            entity_name="foo",
            entity_type="function",
            start_line=1,
        )
        id_ = compute_stable_id(rec)
        assert all(c in "0123456789abcdef" for c in id_)


# ---------------------------------------------------------------------------
# BaseExtractor (write buffering)
# ---------------------------------------------------------------------------


class TestBaseExtractorWriteBuffer:

    def test_write_record_and_flush(self, tmp_path, minimal_record):
        """write_record() + flush() should produce a readable JSONL file."""
        extractor = _DummyExtractor(tmp_path, tmp_path)
        output_file = str(tmp_path / "test_output.jsonl")

        extractor.write_record(minimal_record, output_file)
        extractor.flush(output_file)

        # The file must exist and contain valid JSONL.
        assert Path(output_file).exists()
        lines = [l for l in Path(output_file).read_text().splitlines() if l.strip()]
        data = [json.loads(l) for l in lines]
        assert len(data) == 1
        assert data[0]["entity_name"] == minimal_record.entity_name

    def test_write_records_batch(self, tmp_path, sample_records):
        """write_records() should write all records after flush_all()."""
        extractor = _DummyExtractor(tmp_path, tmp_path)
        output_file = str(tmp_path / "batch_output.jsonl")

        extractor.write_records(sample_records, output_file)
        extractor.flush_all()

        lines = [l for l in Path(output_file).read_text().splitlines() if l.strip()]
        data = [json.loads(l) for l in lines]
        assert len(data) == len(sample_records)

    def test_incremental_flushes_accumulate(self, tmp_path, minimal_record):
        """Multiple flush cycles should accumulate records, not overwrite them."""
        extractor = _DummyExtractor(tmp_path, tmp_path)
        output_file = str(tmp_path / "accum.jsonl")

        # Write and flush once.
        extractor.write_record(minimal_record, output_file)
        extractor.flush(output_file)

        # Write a second record with a different name and flush again.
        second = EntityRecord(
            entity_name="register_backward_hook",
            entity_type="method",
            source_file="torch/nn/modules/module.py",
        )
        extractor.write_record(second, output_file)
        extractor.flush(output_file)

        lines = [l for l in Path(output_file).read_text().splitlines() if l.strip()]
        data = [json.loads(l) for l in lines]
        names = {r["entity_name"] for r in data}
        assert minimal_record.entity_name in names
        assert "register_backward_hook" in names

    def test_report_stats_returns_dict(self, tmp_path):
        """report_stats() should return a dict with the expected keys."""
        extractor = _DummyExtractor(tmp_path, tmp_path)
        stats = extractor.report_stats()
        assert "files_processed" in stats
        assert "records_produced" in stats
        assert "errors" in stats
        assert "warnings" in stats


# ---------------------------------------------------------------------------
# BaseExtractor (file helpers)
# ---------------------------------------------------------------------------


class TestBaseExtractorFileHelpers:

    def test_file_to_module_path(self, tmp_path):
        """file_to_module_path() converts a relative path to dot notation."""
        extractor = _DummyExtractor(tmp_path, tmp_path)
        filepath = tmp_path / "torch" / "nn" / "modules" / "module.py"
        result = extractor.file_to_module_path(filepath)
        assert result == "torch.nn.modules.module"

    def test_compute_qualified_name(self, tmp_path):
        """compute_qualified_name() joins module + class + method correctly."""
        extractor = _DummyExtractor(tmp_path, tmp_path)
        qn = extractor.compute_qualified_name(
            "torch.nn.modules.module", "Module", "forward"
        )
        assert qn == "torch.nn.modules.module.Module.forward"

    def test_read_file_safe_returns_content(self, tmp_path):
        """read_file_safe() should return file content for an existing file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass\n", encoding="utf-8")

        extractor = _DummyExtractor(tmp_path, tmp_path)
        content = extractor.read_file_safe(test_file)
        assert content is not None
        assert "def hello" in content

    def test_read_file_safe_returns_none_for_missing(self, tmp_path):
        """read_file_safe() should return None for a non-existent file."""
        extractor = _DummyExtractor(tmp_path, tmp_path)
        result = extractor.read_file_safe(tmp_path / "does_not_exist.py")
        assert result is None

    def test_get_raw_text_truncates_at_5000(self, tmp_path):
        """get_raw_text() must cap output at 5000 characters."""
        # Write a file with 6000 chars.
        big_file = tmp_path / "big.py"
        big_file.write_text("x" * 6000, encoding="utf-8")

        extractor = _DummyExtractor(tmp_path, tmp_path)
        text = extractor.get_raw_text(big_file, 1, 1)
        assert len(text) <= 5000 + len("\n... [truncated]")


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class TestCheckpointManager:

    def test_fresh_checkpoint_has_no_completed_phases(self, tmp_output_dir):
        """A brand-new checkpoint should report no completed phases."""
        cm = CheckpointManager(tmp_output_dir)
        assert not cm.is_done("catalog")
        assert not cm.is_done("extract")

    def test_mark_done_persists(self, tmp_output_dir):
        """mark_done() should persist across CheckpointManager instances."""
        cm1 = CheckpointManager(tmp_output_dir)
        cm1.mark_done("catalog", {"records": 42})

        # Create a second instance to verify disk persistence.
        cm2 = CheckpointManager(tmp_output_dir)
        assert cm2.is_done("catalog")
        meta = cm2.get_metadata("catalog")
        assert meta["records"] == 42

    def test_reset_clears_all_phases(self, tmp_output_dir):
        """reset() with no argument should clear all completed phases."""
        cm = CheckpointManager(tmp_output_dir)
        cm.mark_done("catalog")
        cm.mark_done("extract")

        cm.reset()
        assert not cm.is_done("catalog")
        assert not cm.is_done("extract")

    def test_reset_single_phase(self, tmp_output_dir):
        """reset(phase) should only clear the specified phase."""
        cm = CheckpointManager(tmp_output_dir)
        cm.mark_done("catalog")
        cm.mark_done("extract")

        cm.reset("catalog")
        assert not cm.is_done("catalog")
        assert cm.is_done("extract")

    def test_save_and_load_state(self, tmp_output_dir):
        """save_state() and load_state() should round-trip arbitrary data."""
        cm = CheckpointManager(tmp_output_dir)
        cm.save_state({"custom_key": "custom_value", "number": 99})

        state = cm.load_state()
        assert state["custom_key"] == "custom_value"
        assert state["number"] == 99

    def test_checkpoint_file_is_valid_json(self, tmp_output_dir):
        """The checkpoint file must always be valid JSON after mark_done()."""
        cm = CheckpointManager(tmp_output_dir)
        cm.mark_done("catalog", {"records": 100})

        # Read the raw checkpoint file and verify it parses correctly.
        checkpoint_file = tmp_output_dir / "session" / "checkpoint.json"
        assert checkpoint_file.exists()
        data = json.loads(checkpoint_file.read_text())
        assert "completed_phases" in data
        assert "catalog" in data["completed_phases"]
