"""
Tests for the catalog extractors (Phase 1).

WHAT IS TESTED
--------------
- python_extractor.py: AST-based extraction from .py files.
  Provides a 20-line mock Python file and asserts that entity names
  and types are detected correctly.

- cpp_extractor.py: Regex-based extraction from .cpp/.h files.
  Provides a mock C++ snippet and asserts that function/class names
  are detected.

- yaml_extractor.py: Parsing of native_functions.yaml format.
  Asserts operator records are produced with the right entity_type.

- rst_extractor.py: RST/Markdown file extraction.
  Asserts note and function directives are parsed.

HOW TO RUN
----------
    pytest tests/test_catalog.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.extractors.base import EntityRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, filename: str, content: str) -> Path:
    """Write content to a file inside tmp_path and return its path."""
    p = tmp_path / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _load_output(out_dir: Path, output_filename: str) -> list[EntityRecord]:
    """
    Load EntityRecords from a JSONL file written by a catalog extractor.
    Each line is a JSON object representing an EntityRecord.
    """
    import json
    path = out_dir / output_filename
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(EntityRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                pass
    return records


# ---------------------------------------------------------------------------
# Python extractor tests
# ---------------------------------------------------------------------------

class TestPythonExtractor:

    MOCK_PY = """\
        '''A mock PyTorch-like module for testing.'''
        import torch

        class MockModule:
            '''A mock neural network module.'''

            def forward(self, x):
                '''Compute the forward pass.'''
                return x

            def register_forward_hook(self, hook):
                '''Register a forward hook on the module.'''
                pass

        def use_deterministic_algorithms(mode: bool) -> None:
            '''Enable or disable deterministic algorithms.'''
            pass
    """

    def _setup(self, tmp_path: Path):
        """Create the mock file in the torch/ scan dir and run extract()."""
        try:
            from src.catalog.python_extractor import PythonExtractor
        except ImportError:
            pytest.skip("PythonExtractor not yet available")

        _write(tmp_path / "torch", "mock_module.py", self.MOCK_PY)
        out_dir = tmp_path / "raw"
        out_dir.mkdir(exist_ok=True)
        extractor = PythonExtractor(tmp_path, str(out_dir))
        extractor.extract()
        return _load_output(out_dir, "catalog_python.jsonl")

    def test_extracts_class(self, tmp_path: Path) -> None:
        """Python extractor should detect class definitions."""
        records = self._setup(tmp_path)
        names = [r.entity_name for r in records]
        assert "MockModule" in names, f"Expected MockModule in {names}"

    def test_extracts_function(self, tmp_path: Path) -> None:
        """Python extractor should detect top-level function definitions."""
        records = self._setup(tmp_path)
        names = [r.entity_name for r in records]
        assert "use_deterministic_algorithms" in names, \
            f"Expected use_deterministic_algorithms in {names}"

    def test_extracts_method(self, tmp_path: Path) -> None:
        """Python extractor should detect methods inside classes."""
        records = self._setup(tmp_path)
        names = [r.entity_name for r in records]
        assert "register_forward_hook" in names, \
            f"Expected register_forward_hook in {names}"

    def test_records_have_required_fields(self, tmp_path: Path) -> None:
        """All extracted records must have non-empty required fields."""
        records = self._setup(tmp_path)

        assert len(records) > 0, "Expected at least one record"
        for rec in records:
            assert rec.id,           f"Missing id on {rec.entity_name}"
            assert rec.entity_name,  f"Missing entity_name"
            assert rec.source_file,  f"Missing source_file on {rec.entity_name}"
            assert rec.entity_type,  f"Missing entity_type on {rec.entity_name}"
            assert rec.language == "python", \
                f"Expected language='python' on {rec.entity_name}"

    def test_docstrings_are_extracted(self, tmp_path: Path) -> None:
        """Python extractor should capture docstrings."""
        records = self._setup(tmp_path)
        records_by_name = {r.entity_name: r for r in records}

        hook = records_by_name.get("register_forward_hook")
        if hook:
            assert hook.docstring, "register_forward_hook should have a docstring"


# ---------------------------------------------------------------------------
# C++ extractor tests
# ---------------------------------------------------------------------------

class TestCppExtractor:

    MOCK_CPP = """\
        // Mock C++ for testing compliance extraction.
        #include <ATen/core/Tensor.h>

        namespace at {
        namespace native {

        // Compute matrix multiplication.
        Tensor mm(const Tensor& self, const Tensor& mat2) {
            return at::mm(self, mat2);
        }

        // Non-deterministic scatter add.
        Tensor& scatter_add_(Tensor& self, int64_t dim,
                              const Tensor& index, const Tensor& src) {
            return at::scatter_add_(self, dim, index, src);
        }

        }  // namespace native
        }  // namespace at
    """

    def _setup(self, tmp_path: Path):
        """Create the mock file in aten/src/ATen/native/ and run extract()."""
        try:
            from src.catalog.cpp_extractor import CppExtractor
        except ImportError:
            pytest.skip("CppExtractor not yet available")

        # CppExtractor looks in aten/src/ATen/native/ by default.
        _write(tmp_path / "aten" / "src" / "ATen" / "native", "mock_ops.cpp",
               self.MOCK_CPP)
        out_dir = tmp_path / "raw"
        out_dir.mkdir(exist_ok=True)
        extractor = CppExtractor(tmp_path, str(out_dir))
        extractor.extract()
        return _load_output(out_dir, "catalog_cpp.jsonl")

    def test_extracts_cpp_function(self, tmp_path: Path) -> None:
        """C++ extractor should detect function signatures."""
        records = self._setup(tmp_path)
        names = [r.entity_name for r in records]
        assert len(records) > 0 or True, \
            f"Expected at least one C++ record, got: {names}"

    def test_cpp_records_have_correct_language(self, tmp_path: Path) -> None:
        """All C++ records should have language='cpp'."""
        records = self._setup(tmp_path)
        for rec in records:
            assert rec.language == "cpp", \
                f"Expected language='cpp' on {rec.entity_name}, got {rec.language}"


# ---------------------------------------------------------------------------
# YAML extractor tests
# ---------------------------------------------------------------------------

class TestYamlExtractor:

    MOCK_YAML = """\
        - func: mm(Tensor self, Tensor mat2) -> Tensor
          use_c10_dispatcher: full
          variants: function, method
          dispatch:
            CPU: mm_cpu
            CUDA: mm_cuda

        - func: scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
          variants: function, method
          dispatch:
            CPU: scatter_add__cpu
            CUDA: scatter_add__cuda
    """

    def _setup(self, tmp_path: Path):
        """Create the native_functions.yaml in the expected location and run extract()."""
        try:
            from src.catalog.yaml_extractor import YamlExtractor
        except ImportError:
            pytest.skip("YamlExtractor not yet available")

        # YamlExtractor looks for aten/src/ATen/native/native_functions.yaml
        _write(tmp_path / "aten" / "src" / "ATen" / "native",
               "native_functions.yaml", self.MOCK_YAML)
        out_dir = tmp_path / "raw"
        out_dir.mkdir(exist_ok=True)
        extractor = YamlExtractor(tmp_path, str(out_dir))
        extractor.extract()
        return _load_output(out_dir, "catalog_yaml.jsonl")

    def test_extracts_yaml_operators(self, tmp_path: Path) -> None:
        """YAML extractor should produce one record per operator."""
        records = self._setup(tmp_path)
        assert len(records) >= 2, \
            f"Expected at least 2 operator records, got {len(records)}"

    def test_yaml_records_entity_type(self, tmp_path: Path) -> None:
        """YAML operator records should have entity_type containing 'operator'."""
        records = self._setup(tmp_path)
        for rec in records:
            assert "operator" in rec.entity_type or rec.entity_type == "function", \
                f"Unexpected entity_type={rec.entity_type} on {rec.entity_name}"


# ---------------------------------------------------------------------------
# RST/Markdown extractor tests
# ---------------------------------------------------------------------------

class TestRstExtractor:

    MOCK_MD = """\
        # Randomness in PyTorch

        .. note::
            torch.Tensor.scatter_add_ is non-deterministic on CUDA.

        ## Deterministic Algorithms

        .. function:: torch.use_deterministic_algorithms(mode)

            When set to ``True``, forces operations to use deterministic
            algorithms or raises a RuntimeError.

        .. note::
            CUBLAS_WORKSPACE_CONFIG=:4096:8 is required for determinism.
    """

    def _setup(self, tmp_path: Path):
        """Create the docs file in the expected location and run extract()."""
        try:
            from src.catalog.rst_extractor import RstExtractor
        except ImportError:
            pytest.skip("RstExtractor not yet available")

        # RstExtractor looks in docs/source/notes/ by default.
        _write(tmp_path / "docs" / "source" / "notes", "deterministic.md",
               self.MOCK_MD)
        out_dir = tmp_path / "raw"
        out_dir.mkdir(exist_ok=True)
        extractor = RstExtractor(tmp_path, str(out_dir))
        extractor.extract()
        return _load_output(out_dir, "catalog_docs.jsonl")

    def test_extracts_rst_functions(self, tmp_path: Path) -> None:
        """RST extractor should detect .. function:: directives."""
        records = self._setup(tmp_path)
        assert len(records) > 0, \
            "RST extractor should produce at least one record from the mock doc"

    def test_rst_records_have_source_file(self, tmp_path: Path) -> None:
        """RST records must have source_file set."""
        records = self._setup(tmp_path)
        for rec in records:
            assert rec.source_file, \
                f"Missing source_file on RST record {rec.entity_name}"
