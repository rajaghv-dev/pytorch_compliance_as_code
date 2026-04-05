"""
C++ catalog extractor for the PyTorch Compliance Toolkit.

Walks ``.cpp``, ``.h``, ``.cu``, and ``.cuh`` files under ``aten/``, ``c10/``,
and ``torch/csrc/`` in the PyTorch repository.  Uses **regex** (not a proper
C++ AST parser) to extract:

* **Class / struct definitions**
* **Function definitions**
* **Enum definitions** (including ``enum class``)
* **Namespace scoping** via a simple brace-counting stack

.. note::

   Regex will miss templates, preprocessor conditionals, and multi-line
   signatures.  ``extraction_confidence`` is set to 0.7 to reflect this.

Output: ``storage/raw/catalog_cpp.jsonl``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ..extractors.base import BaseExtractor, EntityRecord, compute_stable_id

logger = logging.getLogger("pct.catalog.cpp")

# ---------------------------------------------------------------------------
# Directories and file extensions to scan
# ---------------------------------------------------------------------------

_SCAN_DIRS = ["aten", "c10", "torch/csrc"]
_CPP_EXTENSIONS = ("*.cpp", "*.h", "*.cu", "*.cuh")

_OUTPUT_FILE = "catalog_cpp.jsonl"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Class/struct: captures the name and optional base class specification.
_CLASS_RE = re.compile(
    r'(?:class|struct)\s+(?:TORCH_API\s+|C10_API\s+)?'
    r'(\w+)'
    r'(?:\s*:\s*(?:public|private|protected)\s+[\w:<>, ]+)?'
    r'\s*\{',
    re.MULTILINE,
)

# Function definition: return_type name(params) with optional qualifiers.
# This intentionally uses a broad pattern to capture C++/CUDA qualifiers.
_FUNCTION_RE = re.compile(
    r'(?:^|\n)\s*'
    r'(?:virtual\s+|static\s+|inline\s+|__device__\s+|__global__\s+|__host__\s+|TORCH_API\s+|C10_API\s+)*'
    r'(?:[\w:<>*& ]+\s+)?'   # return type (greedy but may over-match)
    r'(\w+)\s*'               # function name
    r'\(([^)]*)\)\s*'         # parameters
    r'(?:const\s*)?'
    r'(?:override\s*)?'
    r'(?:noexcept[^{;]*)?'
    r'(?:\{|;)',
    re.MULTILINE,
)

# Enum: ``enum [class] Name [: underlying] { values }``
_ENUM_RE = re.compile(
    r'enum\s+(?:class\s+)?(\w+)\s*(?::\s*\w+\s*)?\{([^}]+)\}',
    re.MULTILINE | re.DOTALL,
)

# Namespace: ``namespace Name {``
_NAMESPACE_OPEN_RE = re.compile(r'namespace\s+(\w+)\s*\{', re.MULTILINE)

# Names that are clearly not real function names (C++ keywords, macros, etc.).
_FUNCTION_NAME_BLACKLIST = frozenset({
    "if", "else", "for", "while", "do", "switch", "case", "return",
    "sizeof", "typeof", "alignof", "static_assert", "throw", "catch",
    "try", "delete", "new", "using", "typedef", "decltype",
    "TORCH_CHECK", "TORCH_WARN", "AT_ERROR", "AT_ASSERT", "C10_LOG",
    "AT_DISPATCH_ALL_TYPES", "AT_DISPATCH_FLOATING_TYPES",
})


class CppCatalogExtractor(BaseExtractor):
    """
    Catalog extractor for C++ / CUDA source files using regex-based parsing.

    Because we are not using a proper C++ parser, extraction confidence is
    set to 0.7 across the board.  Regex will miss templates, preprocessor
    conditionals, and multi-line signatures.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__(name="catalog_cpp", repo_path=repo_path, output_path=output_path)
        self._output_file = str(self.output_path / _OUTPUT_FILE)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Walk C++ files and extract entities via regex.

        Returns
        -------
        int
            Total number of records produced.
        """
        all_records: list[EntityRecord] = []

        for scan_dir in _SCAN_DIRS:
            target = self.repo_path / scan_dir
            if not target.is_dir():
                self.logger.warning("Scan directory does not exist: %s", target)
                self._warnings += 1
                continue

            self.logger.info("Starting C++ catalog extraction for %s", target)

            for ext_pattern in _CPP_EXTENSIONS:
                cpp_files = self.find_files(ext_pattern, root=target)

                for filepath in cpp_files:
                    try:
                        records = self._process_file(filepath)
                        if records:
                            all_records.extend(records)
                            self.write_records(records, self._output_file)
                        self._files_processed += 1
                    except Exception as exc:
                        self.logger.error(
                            "Unexpected error processing %s: %s", filepath, exc
                        )
                        self._errors += 1

        self.flush(self._output_file)
        self.logger.info(
            "Completed %d files, %d records",
            self._files_processed,
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # Per-file processing
    # ------------------------------------------------------------------ #

    def _process_file(self, filepath: Path) -> list[EntityRecord]:
        """
        Parse a single C++ file with regex and extract entities.

        Parameters
        ----------
        filepath : Path
            Absolute path to the source file.

        Returns
        -------
        list[EntityRecord]
        """
        source = self.read_file_safe(filepath)
        if source is None:
            return []

        try:
            rel_path = str(filepath.relative_to(self.repo_path))
        except ValueError:
            rel_path = str(filepath)

        # Build a simple namespace context by scanning the file.
        # This is approximate: we track ``namespace X {`` opens and count
        # braces to determine when we leave a namespace scope.
        namespace_context = self._resolve_namespace_at_line(source)

        records: list[EntityRecord] = []

        # --- Classes / structs ---
        for match in _CLASS_RE.finditer(source):
            name = match.group(1)
            line_no = source[:match.start()].count("\n") + 1
            ns = namespace_context.get(line_no, "")
            qualified = f"{ns}::{name}" if ns else name

            records.append(self.make_record(
                source_file=rel_path,
                language="cpp",
                entity_name=name,
                entity_type="class",
                subcategory="cpp_class",
                module_path=rel_path.replace("/", ".").rsplit(".", 1)[0],
                qualified_name=qualified,
                start_line=line_no,
                end_line=line_no,  # We don't track class end with regex.
                raw_text=self.get_raw_text(filepath, line_no, line_no + 30),
                extraction_confidence=0.7,
                metadata={"namespace": ns},
            ))

        # --- Functions ---
        for match in _FUNCTION_RE.finditer(source):
            name = match.group(1)
            params = match.group(2).strip()

            # Filter out false positives (keywords, macros).
            if name in _FUNCTION_NAME_BLACKLIST:
                continue
            # Skip names that are all uppercase (likely macros).
            if name.isupper() and len(name) > 2:
                continue

            line_no = source[:match.start()].count("\n") + 1
            ns = namespace_context.get(line_no, "")
            qualified = f"{ns}::{name}" if ns else name

            records.append(self.make_record(
                source_file=rel_path,
                language="cpp",
                entity_name=name,
                entity_type="function",
                subcategory="cpp_function",
                module_path=rel_path.replace("/", ".").rsplit(".", 1)[0],
                qualified_name=qualified,
                start_line=line_no,
                end_line=line_no,
                raw_text=self.get_raw_text(filepath, line_no, line_no + 20),
                signature=f"{name}({params})",
                extraction_confidence=0.7,
                metadata={"namespace": ns, "params": params},
            ))

        # --- Enums ---
        for match in _ENUM_RE.finditer(source):
            name = match.group(1)
            body = match.group(2)
            line_no = source[:match.start()].count("\n") + 1
            ns = namespace_context.get(line_no, "")
            qualified = f"{ns}::{name}" if ns else name

            # Parse enum values: split on commas, strip whitespace and
            # trailing ``= value`` initialisers.
            raw_values = [v.strip() for v in body.split(",") if v.strip()]
            enum_values = []
            for rv in raw_values:
                # Remove ``= initialiser`` and inline comments.
                val_name = rv.split("=")[0].split("//")[0].strip()
                if val_name and val_name.isidentifier():
                    enum_values.append(val_name)

            records.append(self.make_record(
                source_file=rel_path,
                language="cpp",
                entity_name=name,
                entity_type="enum",
                subcategory="cpp_enum",
                module_path=rel_path.replace("/", ".").rsplit(".", 1)[0],
                qualified_name=qualified,
                start_line=line_no,
                end_line=line_no,
                raw_text=self.get_raw_text(filepath, line_no, line_no + len(raw_values) + 2),
                extraction_confidence=0.7,
                metadata={"namespace": ns, "values": enum_values},
            ))

        return records

    # ------------------------------------------------------------------ #
    # Namespace tracking
    # ------------------------------------------------------------------ #

    def _resolve_namespace_at_line(self, source: str) -> dict[int, str]:
        """
        Build a mapping from line number to active namespace string.

        Uses a simple brace-counting heuristic: when we see
        ``namespace X {`` we push ``X`` onto a stack and record the brace
        depth.  When the brace depth drops back below the recorded level we
        pop the namespace.

        This is approximate -- it does not handle comments or strings that
        contain braces, and it ignores anonymous namespaces.

        Parameters
        ----------
        source : str
            Full file contents.

        Returns
        -------
        dict[int, str]
            Mapping of line number (1-based) to namespace string
            (e.g. ``"at::native"``).
        """
        # First pass: find all namespace openings with their positions.
        ns_opens: list[tuple[int, str]] = []  # (char_offset, name)
        for match in _NAMESPACE_OPEN_RE.finditer(source):
            ns_opens.append((match.start(), match.group(1)))

        if not ns_opens:
            return {}

        # Second pass: walk character by character tracking brace depth
        # and namespace stack.  For performance we only record namespace
        # transitions and then interpolate.
        ns_stack: list[tuple[str, int]] = []   # (name, brace_depth_at_open)
        brace_depth = 0
        ns_open_idx = 0
        line_no = 1
        ns_at_line: dict[int, str] = {}

        # Pre-sort namespace opens by offset.
        ns_opens.sort()

        current_ns = ""
        for i, ch in enumerate(source):
            if ch == "\n":
                ns_at_line[line_no] = current_ns
                line_no += 1

            # Check if we are at a namespace opening.
            if ns_open_idx < len(ns_opens) and i == ns_opens[ns_open_idx][0]:
                # The opening brace for this namespace will be encountered
                # shortly.  We record the intent to push.
                ns_name = ns_opens[ns_open_idx][1]
                # Find the next '{' after this position.
                brace_pos = source.find("{", i)
                if brace_pos >= 0:
                    ns_stack.append((ns_name, brace_depth + 1))
                ns_open_idx += 1

            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                # Check if we've closed a namespace.
                while ns_stack and brace_depth < ns_stack[-1][1]:
                    ns_stack.pop()

            # Recompute the current namespace string.
            current_ns = "::".join(ns[0] for ns in ns_stack)

        # Record the last line.
        ns_at_line[line_no] = current_ns

        return ns_at_line
