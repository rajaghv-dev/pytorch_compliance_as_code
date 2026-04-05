"""
Shared pytest fixtures for the PyTorch Compliance Toolkit test suite.

WHAT IS A FIXTURE?
    A fixture is a function that sets up data or resources that multiple
    tests need.  Pytest injects them by name into test functions.

GOLDEN SUBSET
    Many tests can run against a small "golden subset" of PyTorch files
    (torch/nn/modules/module.py, etc.) instead of the full 1 GB repo.
    This makes tests fast and runnable on any machine.

    If the full PyTorch repo is available at PYTORCH_REPO_PATH, integration
    tests will use it.  Otherwise they fall back to the golden subset.

USAGE IN TESTS
    def test_something(tmp_output_dir, sample_records):
        ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.extractors.base import EntityRecord

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Default PyTorch repo path (overridable via environment variable).
PYTORCH_REPO_PATH = Path(
    os.environ.get("PYTORCH_REPO_PATH", "/home/raja/gemini-torch/pytorch")
)

# Golden subset directory — a small number of hand-picked PyTorch files
# that cover the most compliance-relevant entities.
GOLDEN_DIR = Path(__file__).parent / "golden"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """
    Return a fresh temporary directory for pipeline output.

    Each test that uses this fixture gets its own isolated directory,
    so tests cannot interfere with each other.
    """
    out = tmp_path / "storage"
    out.mkdir()
    return out


@pytest.fixture()
def pytorch_repo() -> Path:
    """
    Return the path to the PyTorch repository.

    Skips the test if the repo is not available (e.g. in CI without
    the full checkout).  Use this fixture only in integration tests.
    """
    if not PYTORCH_REPO_PATH.is_dir():
        pytest.skip(
            f"PyTorch repo not found at {PYTORCH_REPO_PATH}. "
            "Set PYTORCH_REPO_PATH env var to run integration tests."
        )
    return PYTORCH_REPO_PATH


@pytest.fixture()
def golden_dir() -> Path:
    """Return the path to the golden test fixtures directory."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    return GOLDEN_DIR


@pytest.fixture()
def minimal_record() -> EntityRecord:
    """
    Return a minimal but fully valid EntityRecord for use in unit tests.

    Covers the required fields only; annotators and other tests can add
    more fields on top of this baseline.
    """
    return EntityRecord(
        id="abc123def456789012",   # 18-char fake hex ID
        source_file="torch/nn/modules/module.py",
        language="python",
        entity_name="register_forward_hook",
        entity_type="method",
        subcategory="hook_definition",
        module_path="torch.nn.modules.module",
        qualified_name="torch.nn.modules.module.Module.register_forward_hook",
        start_line=100,
        end_line=140,
        raw_text="def register_forward_hook(self, hook): ...",
        docstring="Registers a forward hook on the module.",
        extractor="hookability",
    )


@pytest.fixture()
def sample_records() -> list[EntityRecord]:
    """
    Return a small list of EntityRecords covering several entity types.

    Useful for testing annotators, deduplication, and converters without
    running the full extractor pipeline.
    """
    return [
        # Hook definition — should get eu_ai_act_art_61 tag.
        EntityRecord(
            id="aaa000111222333444",
            source_file="torch/nn/modules/module.py",
            language="python",
            entity_name="register_forward_hook",
            entity_type="method",
            subcategory="hook_definition",
            module_path="torch.nn.modules.module",
            qualified_name="torch.nn.modules.module.Module.register_forward_hook",
            start_line=100,
            end_line=140,
            raw_text="def register_forward_hook(self, hook): ...",
            docstring="Registers a forward hook.",
            extraction_confidence=1.0,
            extractor="hookability",
        ),
        # Determinism API — should get eu_ai_act_art_15 tag.
        EntityRecord(
            id="bbb000111222333444",
            source_file="torch/__init__.py",
            language="python",
            entity_name="use_deterministic_algorithms",
            entity_type="function",
            subcategory="",
            module_path="torch",
            qualified_name="torch.use_deterministic_algorithms",
            start_line=200,
            end_line=230,
            raw_text="def use_deterministic_algorithms(mode): ...",
            docstring="Sets whether PyTorch operations must use deterministic algorithms.",
            extraction_confidence=1.0,
            extractor="operator_determinism",
        ),
        # Dataset class — should get eu_ai_act_art_10 tag.
        EntityRecord(
            id="ccc000111222333444",
            source_file="torch/utils/data/dataset.py",
            language="python",
            entity_name="Dataset",
            entity_type="class",
            subcategory="",
            module_path="torch.utils.data.dataset",
            qualified_name="torch.utils.data.dataset.Dataset",
            start_line=50,
            end_line=200,
            raw_text="class Dataset: ...",
            docstring="An abstract class representing a dataset.",
            extraction_confidence=1.0,
            extractor="data_provenance",
        ),
        # Record without compliance tags (should not be tagged).
        EntityRecord(
            id="ddd000111222333444",
            source_file="torch/utils/benchmark/utils.py",
            language="python",
            entity_name="_internal_helper",
            entity_type="function",
            subcategory="",
            module_path="torch.utils.benchmark.utils",
            qualified_name="torch.utils.benchmark.utils._internal_helper",
            start_line=10,
            end_line=20,
            raw_text="def _internal_helper(): ...",
            docstring="",
            extraction_confidence=0.8,
            extractor="api_documentation",
        ),
    ]


@pytest.fixture()
def golden_records_path(golden_dir: Path) -> Path:
    """
    Return the path to the golden records JSON file.

    If the file does not exist yet, write a minimal fixture so tests
    have something to load.  The golden file should be updated when the
    expected output changes.
    """
    path = golden_dir / "golden_records.json"
    if not path.exists():
        # Write a minimal golden file so tests can run before a full
        # pipeline has been executed.
        minimal = [
            {
                "id": "aaa000111222333444",
                "source_file": "torch/nn/modules/module.py",
                "language": "python",
                "entity_name": "register_forward_hook",
                "entity_type": "method",
                "subcategory": "hook_definition",
                "module_path": "torch.nn.modules.module",
                "qualified_name": (
                    "torch.nn.modules.module.Module.register_forward_hook"
                ),
                "start_line": 100,
                "end_line": 140,
                "raw_text": "def register_forward_hook(self, hook): ...",
                "docstring": "Registers a forward hook.",
                "relations": [],
                "compliance_tags": ["eu_ai_act_art_61"],
                "lifecycle_phase": "inference_safe",
                "execution_level": "",
                "distributed_safety": "",
                "distributed_safety_notes": "",
                "export_survival": {},
                "extraction_confidence": 1.0,
                "mapping_confidence": 0.8,
                "mapping_rationale": "Tier-1 name match",
                "extractor": "hookability",
                "annotations": {},
                "metadata": {},
                "timestamp": "2026-01-01T00:00:00+00:00",
                "signature": "",
            }
        ]
        path.write_text(json.dumps(minimal, indent=2), encoding="utf-8")

    return path
