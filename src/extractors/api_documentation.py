"""
API Documentation Extractor — extracts structured RST directives and
docstring annotations from PyTorch documentation and source code.

Compliance relevance:
    - ``warning``, ``danger``, ``caution`` directives map to EU AI Act
      Article 13 (transparency obligations).
    - ``deprecated`` directives map to Article 13.
    - ``versionadded``, ``versionchanged`` directives map to Article 11
      (technical documentation).
    - ``seealso`` directives are informational cross-references.

Sources scanned:
    - ``docs/source/**/*.rst`` — reStructuredText documentation
    - Key Python source files — docstring warnings/notes
"""

from __future__ import annotations

import ast
import re
import logging
from pathlib import Path
from typing import Any

from .base import EntityRecord, BaseExtractor, compute_stable_id

logger = logging.getLogger("pct.extractors.api_documentation")

# ---------------------------------------------------------------------------
# RST directive patterns
# ---------------------------------------------------------------------------
# Each regex captures the directive body (indented block following the marker).

_RST_DIRECTIVES: dict[str, re.Pattern] = {
    "warning": re.compile(
        r'^\.\. warning::\s*\n((?:\s+.+\n)*)', re.MULTILINE
    ),
    "note": re.compile(
        r'^\.\. note::\s*\n((?:\s+.+\n)*)', re.MULTILINE
    ),
    "deprecated": re.compile(
        r'^\.\. deprecated::\s*(.+)\n((?:\s+.+\n)*)', re.MULTILINE
    ),
    "versionadded": re.compile(
        r'^\.\. versionadded::\s*(.+)', re.MULTILINE
    ),
    "versionchanged": re.compile(
        r'^\.\. versionchanged::\s*(.+)\n((?:\s+.+\n)*)', re.MULTILINE
    ),
    "danger": re.compile(
        r'^\.\. danger::\s*\n((?:\s+.+\n)*)', re.MULTILINE
    ),
    "caution": re.compile(
        r'^\.\. caution::\s*\n((?:\s+.+\n)*)', re.MULTILINE
    ),
    "seealso": re.compile(
        r'^\.\. seealso::\s*\n((?:\s+.+\n)*)', re.MULTILINE
    ),
}

# ---------------------------------------------------------------------------
# Compliance tag mapping for directive types
# ---------------------------------------------------------------------------

_DIRECTIVE_COMPLIANCE_TAGS: dict[str, list[str]] = {
    "warning":        ["eu_ai_act_art_13"],
    "danger":         ["eu_ai_act_art_13"],
    "caution":        ["eu_ai_act_art_13"],
    "deprecated":     ["eu_ai_act_art_13"],
    "versionadded":   ["eu_ai_act_art_11"],
    "versionchanged": ["eu_ai_act_art_11"],
    "note":           [],        # informational, no specific article
    "seealso":        [],        # cross-reference, informational
}

# ---------------------------------------------------------------------------
# Docstring warning pattern (used in Python source scanning)
# ---------------------------------------------------------------------------

_DOCSTRING_WARNING_RE = re.compile(
    r'(Warning|NOTE|DEPRECATED|Caution|TODO|FIXME|HACK|SECURITY|XXX)'
    r'[\s:]+(.+?)(?=\n\n|\n\s*(?:Args|Returns|Raises|Example|Note)|\Z)',
    re.DOTALL | re.IGNORECASE,
)


class APIDocumentationExtractor(BaseExtractor):
    """
    Extract structured documentation directives and docstring annotations.

    Produces EntityRecords for each RST directive found in the documentation
    tree and for docstring-level warnings/notes found in key Python source
    files.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the API documentation extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="api_documentation", repo_path=repo_path, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Run the API documentation extraction pass.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting API documentation extraction")
        output_file = str(self.output_path / "api_documentation.jsonl")

        # --- RST directives from docs/ ---
        self._extract_rst_directives(output_file)

        # --- Docstring warnings from key Python source files ---
        self._extract_docstring_warnings(output_file)

        # Flush remaining buffered records
        self.flush(output_file)

        self.logger.info(
            "API documentation extraction complete — %d records produced",
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # RST directive extraction
    # ------------------------------------------------------------------ #

    def _extract_rst_directives(self, output_file: str) -> None:
        """
        Scan ``docs/source/**/*.rst`` for structured RST directives.

        Each directive match (warning, note, deprecated, versionadded,
        versionchanged, danger, caution, seealso) becomes an EntityRecord
        with ``entity_type = "doc_directive"`` and ``language = "rst"``.
        """
        for filepath in self.find_files("docs/source/**/*.rst"):
            content = self.read_file_safe(filepath)
            if content is None:
                self.logger.warning("Skipping unreadable RST file: %s", filepath)
                continue

            self._files_processed += 1
            rel_path = str(filepath.relative_to(self.repo_path))

            # Try each directive pattern against the file content
            for directive_type, regex in _RST_DIRECTIVES.items():
                for match in regex.finditer(content):
                    # Compute the line number of the match
                    line_num = content[:match.start()].count("\n") + 1

                    # The matched text (directive body)
                    raw_text = match.group(0).strip()[:1000]

                    # Look up compliance tags for this directive type
                    compliance_tags = _DIRECTIVE_COMPLIANCE_TAGS.get(directive_type, [])

                    record = self.make_record(
                        source_file=rel_path,
                        language="rst",
                        entity_name=f"{filepath.stem}::{directive_type}:{line_num}",
                        entity_type="doc_directive",
                        subcategory=directive_type,
                        module_path=f"docs.{filepath.stem}",
                        qualified_name=f"docs.{filepath.stem}.{directive_type}.L{line_num}",
                        start_line=line_num,
                        end_line=line_num + raw_text.count("\n"),
                        raw_text=raw_text,
                        compliance_tags=list(compliance_tags),
                        extraction_confidence=1.0,
                        metadata={"directive_type": directive_type},
                    )
                    self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # Python docstring warning extraction
    # ------------------------------------------------------------------ #

    def _extract_docstring_warnings(self, output_file: str) -> None:
        """
        Scan key Python source files for warning/note/deprecated markers
        inside docstrings.

        Looks for patterns like ``Warning:``, ``DEPRECATED:``, ``Caution:``
        inside function/class docstrings and emits records with
        ``language = "python"``.
        """
        # Key source file patterns to scan
        source_patterns = [
            "torch/nn/modules/*.py",
            "torch/nn/functional.py",
            "torch/autograd/*.py",
            "torch/optim/*.py",
        ]

        for pattern in source_patterns:
            for filepath in self.find_files(pattern):
                source = self.read_file_safe(filepath)
                if source is None:
                    self.logger.warning("Skipping unreadable Python file: %s", filepath)
                    continue

                try:
                    tree = ast.parse(source, filename=str(filepath))
                except SyntaxError as exc:
                    self.logger.error("SyntaxError parsing %s: %s", filepath, exc)
                    self._errors += 1
                    continue

                self._files_processed += 1
                rel_path = str(filepath.relative_to(self.repo_path))
                module_path = self.file_to_module_path(filepath)

                # Walk the AST looking for function/class definitions with docstrings
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        continue

                    docstring = ast.get_docstring(node)
                    if not docstring:
                        continue

                    # Search the docstring for warning/note patterns
                    for match in _DOCSTRING_WARNING_RE.finditer(docstring):
                        annotation_type = match.group(1).upper()
                        raw_text = match.group(0).strip()[:1000]

                        # Map docstring annotation types to compliance tags
                        compliance_tags: list[str] = []
                        annotation_lower = annotation_type.lower()
                        if annotation_lower in ("warning", "caution", "security"):
                            compliance_tags = ["eu_ai_act_art_13"]
                        elif annotation_lower == "deprecated":
                            compliance_tags = ["eu_ai_act_art_13"]

                        # Derive subcategory from the annotation type
                        subcategory = f"docstring_{annotation_lower}"

                        record = self.make_record(
                            source_file=rel_path,
                            language="python",
                            entity_name=f"{node.name}::{match.group(1)}",
                            entity_type="doc_directive",
                            subcategory=subcategory,
                            module_path=module_path,
                            qualified_name=f"{module_path}.{node.name}.{annotation_lower}",
                            start_line=node.lineno,
                            end_line=node.lineno,
                            raw_text=raw_text,
                            compliance_tags=compliance_tags,
                            extraction_confidence=0.8,  # name-matched, not AST-found
                            metadata={
                                "parent_entity": node.name,
                                "annotation_type": annotation_type,
                            },
                        )
                        self.write_record(record, output_file)
