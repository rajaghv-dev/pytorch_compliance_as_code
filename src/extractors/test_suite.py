"""
Test Suite Extractor — extracts test functions relevant to compliance evidence
from the PyTorch test suite.

Compliance relevance:
    - Tests touching backward/grad/autograd get ``lifecycle_phase = "training_only"``.
    - Tests with compliance-relevant keywords (deterministic, export, etc.) get
      ``subcategory = "compliance_test"``; others get ``"general_test"``.
    - Compliance tags are promoted to the top-level ``compliance_tags`` field
      (not buried in metadata).

Sources scanned:
    - ``test/test_autograd.py``, ``test/test_nn.py``, ``test/test_cuda.py``, etc.
    - ``test/test_compliance_*.py`` — any compliance-specific tests
    - ``test/functorch/test_*.py`` — functorch tests
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from .base import EntityRecord, BaseExtractor, compute_stable_id

logger = logging.getLogger("pct.extractors.test_suite")

# ---------------------------------------------------------------------------
# Compliance keyword detection
# ---------------------------------------------------------------------------
# Each key maps a tag name to a list of keyword fragments.  If any fragment
# appears in the lowercased test body, the tag is applied.

_COMPLIANCE_KEYWORD_MAP: dict[str, list[str]] = {
    "reproducibility":     ["deterministic", "reproducib", "seed"],
    "numerical_stability": ["nan", "inf", "overflow", "underflow"],
    "hookability":         ["hook", "register_"],
    "export":              ["export", "onnx", "serialize", "save", "load"],
    "provenance":          ["backward", "grad", "autograd"],
    "dispatch":            ["dispatch", "__torch_function__", "__torch_dispatch__"],
}

# Keywords in the test body that indicate training-phase relevance
_TRAINING_KEYWORDS: set[str] = {"backward", "grad", "autograd"}


class TestSuiteExtractor(BaseExtractor):
    """
    Extract test patterns relevant to compliance evidence from the PyTorch
    test suite.

    Walks known test file patterns, parses ASTs, and emits an EntityRecord
    for every ``test_*`` function definition found.  Tests are classified by
    compliance relevance based on keyword presence in the test body.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the test suite extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="test_suite", repo_path=repo_path, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Run the test suite extraction pass.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting test suite extraction")
        output_file = str(self.output_path / "test_suite.jsonl")

        # Patterns for test files to scan
        test_file_patterns: list[str] = [
            "test/test_autograd.py",
            "test/test_nn.py",
            "test/test_cuda.py",
            "test/test_deterministic.py",
            "test/onnx/test_*.py",
            "test/export/test_*.py",
            "test/test_serialization.py",
            "test/test_dispatch.py",
            "test/test_fx.py",
            "test/test_jit*.py",
            # Additional: compliance-specific tests
            "test/test_compliance_*.py",
            # Additional: functorch tests
            "test/functorch/test_*.py",
        ]

        for pattern in test_file_patterns:
            self._extract_tests_from_pattern(pattern, output_file)

        # Flush remaining buffered records
        self.flush(output_file)

        self.logger.info(
            "Test suite extraction complete — %d records produced",
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # Internal extraction logic
    # ------------------------------------------------------------------ #

    def _extract_tests_from_pattern(self, pattern: str, output_file: str) -> None:
        """
        Find files matching *pattern*, parse ASTs, and emit records for
        every ``test_*`` function found.

        Parameters
        ----------
        pattern : str
            Glob pattern relative to the repo root.
        output_file : str
            Destination JSONL path.
        """
        for filepath in self.find_files(pattern):
            source = self.read_file_safe(filepath)
            if source is None:
                self.logger.warning("Skipping unreadable test file: %s", filepath)
                continue

            try:
                tree = ast.parse(source, filename=str(filepath))
            except SyntaxError as exc:
                self.logger.error("SyntaxError parsing %s: %s", filepath, exc)
                self._errors += 1
                continue

            self._files_processed += 1
            module_path = self.file_to_module_path(filepath)
            rel_path = str(filepath.relative_to(self.repo_path))

            # Walk AST for test functions (``test_*`` naming convention)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not node.name.startswith("test_"):
                    continue

                start_line = node.lineno
                end_line = getattr(node, "end_lineno", node.lineno)

                # Extract decorators
                decorators = self.extract_decorators(node)

                # Get a preview of the test body for keyword matching
                test_text = self.get_raw_text(filepath, start_line, end_line).lower()

                # --- Classify compliance tags from test body keywords ---
                compliance_tags: list[str] = []
                for tag, keywords in _COMPLIANCE_KEYWORD_MAP.items():
                    if any(kw in test_text for kw in keywords):
                        compliance_tags.append(tag)

                # --- Determine subcategory ---
                subcategory = "compliance_test" if compliance_tags else "general_test"

                # --- Determine lifecycle phase ---
                # Tests that touch backward/grad/autograd are training-only
                lifecycle_phase = ""
                if any(kw in test_text for kw in _TRAINING_KEYWORDS):
                    lifecycle_phase = "training_only"

                # --- Detect skip conditions from decorators ---
                skip_conditions = [
                    d for d in decorators
                    if "skip" in d.lower() or "expectedfailure" in d.lower()
                ]

                # Extract docstring
                docstring = ast.get_docstring(node) or ""

                # Qualified name for the test function
                qualified_name = self.compute_qualified_name(module_path, "", node.name)

                # Raw text preview — capped at 30 lines for readability
                preview_end = min(start_line + 30, end_line)
                raw_text = self.get_raw_text(filepath, start_line, preview_end)

                # Build metadata
                metadata: dict[str, Any] = {
                    "decorators": decorators,
                    "skip_conditions": skip_conditions,
                    "has_skip": bool(skip_conditions),
                }

                record = self.make_record(
                    source_file=rel_path,
                    language="python",
                    entity_name=node.name,
                    entity_type="test_case",
                    subcategory=subcategory,
                    module_path=module_path,
                    qualified_name=qualified_name,
                    start_line=start_line,
                    end_line=end_line,
                    raw_text=raw_text,
                    docstring=docstring,
                    compliance_tags=compliance_tags,
                    lifecycle_phase=lifecycle_phase,
                    extraction_confidence=1.0,
                    metadata=metadata,
                )
                self.write_record(record, output_file)
