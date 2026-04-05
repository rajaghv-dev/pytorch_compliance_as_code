"""
RST / Markdown documentation extractor for the PyTorch Compliance Toolkit.

Parses ``docs/source/**/*.rst`` and ``docs/source/**/*.md`` to extract:

* **RST directives** -- ``.. warning::``, ``.. note::``, ``.. deprecated::``,
  ``.. versionadded::``, ``.. versionchanged::``, ``.. danger::``,
  ``.. caution::``
* **Auto-reference directives** -- ``.. automodule::``, ``.. autoclass::``,
  ``.. autofunction::``, ``.. automethod::``
* **Compliance notes** -- Full content of files under ``docs/source/notes/``
  treated as first-class compliance documents.
* **Determinism table entries** -- Structured extraction from
  ``deterministic.rst`` or ``deterministic.md``, parsing operator-level
  determinism data.

Output: ``storage/raw/catalog_docs.jsonl``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ..extractors.base import BaseExtractor, EntityRecord, compute_stable_id

logger = logging.getLogger("pct.catalog.rst")

_OUTPUT_FILE = "catalog_docs.jsonl"

# ---------------------------------------------------------------------------
# RST directive patterns
# ---------------------------------------------------------------------------

DIRECTIVE_PATTERNS = {
    "warning":        re.compile(r'\.\. warning::\s*\n((?:[ \t]+.+\n)*)', re.MULTILINE),
    "note":           re.compile(r'\.\. note::\s*\n((?:[ \t]+.+\n)*)', re.MULTILINE),
    "deprecated":     re.compile(r'\.\. deprecated::\s*([^\n]*)\n((?:[ \t]+.+\n)*)', re.MULTILINE),
    "versionadded":   re.compile(r'\.\. versionadded::\s*([^\n]+)', re.MULTILINE),
    "versionchanged": re.compile(r'\.\. versionchanged::\s*([^\n]*)\n((?:[ \t]+.+\n)*)', re.MULTILINE),
    "danger":         re.compile(r'\.\. danger::\s*\n((?:[ \t]+.+\n)*)', re.MULTILINE),
    "caution":        re.compile(r'\.\. caution::\s*\n((?:[ \t]+.+\n)*)', re.MULTILINE),
}

# Auto-reference directives: ``.. automodule:: torch.nn``
AUTOREF_PATTERN = re.compile(
    r'\.\. auto(?:module|class|function|method)::\s*(.+)', re.MULTILINE
)

# ---------------------------------------------------------------------------
# Determinism table patterns (markdown and RST)
# ---------------------------------------------------------------------------

# Markdown table row: ``| operator_name | yes/no | backend_list |``
_MD_TABLE_ROW_RE = re.compile(
    r'^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|',
    re.MULTILINE,
)

# RST simple table row (space-separated columns delimited by whitespace blocks).
_RST_TABLE_ROW_RE = re.compile(
    r'^(\S+.*?)\s{2,}(\S+.*?)\s{2,}(\S+.*?)$',
    re.MULTILINE,
)


class RstCatalogExtractor(BaseExtractor):
    """
    Catalog extractor for RST and Markdown documentation files.

    Extracts compliance-relevant directives, auto-references, full compliance
    notes, and the determinism operator table.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__(name="catalog_rst", repo_path=repo_path, output_path=output_path)
        self._output_file = str(self.output_path / _OUTPUT_FILE)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Walk documentation files and extract directives and notes.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting RST/Markdown catalog extraction")

        docs_dir = self.repo_path / "docs" / "source"
        if not docs_dir.is_dir():
            self.logger.warning("docs/source directory not found at %s", docs_dir)
            self._warnings += 1
            # Attempt to find docs elsewhere.
            docs_dir = self.repo_path / "docs"
            if not docs_dir.is_dir():
                self.logger.warning("No docs directory found; skipping RST extraction")
                return 0

        # Collect all RST and Markdown files.
        rst_files = self.find_files("*.rst", root=docs_dir)
        md_files = self.find_files("*.md", root=docs_dir)
        all_doc_files = rst_files + md_files

        self.logger.info("Found %d documentation files to process", len(all_doc_files))

        for filepath in all_doc_files:
            try:
                self._process_doc_file(filepath)
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

    def _process_doc_file(self, filepath: Path) -> None:
        """
        Process a single documentation file.

        Applies different extraction strategies depending on whether the
        file is under ``docs/source/notes/`` or is the determinism doc.

        Parameters
        ----------
        filepath : Path
            Absolute path to the documentation file.
        """
        try:
            rel_path = str(filepath.relative_to(self.repo_path))
        except ValueError:
            rel_path = str(filepath)

        content = self.read_file_safe(filepath)
        if content is None:
            return

        records: list[EntityRecord] = []

        # --- Special handling: compliance notes under docs/source/notes/ ---
        # These are first-class compliance documents. We extract the FULL
        # content as a single EntityRecord per file.
        notes_dir = self.repo_path / "docs" / "source" / "notes"
        is_compliance_note = False
        try:
            filepath.relative_to(notes_dir)
            is_compliance_note = True
        except ValueError:
            pass

        if is_compliance_note:
            records.append(self._extract_compliance_note(filepath, rel_path, content))

        # --- Special handling: determinism document ---
        # Look for the determinism operator table in deterministic.rst or
        # deterministic.md files.
        fname_lower = filepath.name.lower()
        if "deterministic" in fname_lower or "determinism" in fname_lower:
            det_records = self._extract_determinism_table(filepath, rel_path, content)
            records.extend(det_records)

        # --- General directive extraction ---
        # Extract RST directives (warnings, notes, deprecated, etc.)
        records.extend(self._extract_directives(filepath, rel_path, content))

        # --- Auto-reference extraction ---
        records.extend(self._extract_autorefs(filepath, rel_path, content))

        if records:
            self.write_records(records, self._output_file)

    # ------------------------------------------------------------------ #
    # Compliance notes (full-document extraction)
    # ------------------------------------------------------------------ #

    def _extract_compliance_note(
        self, filepath: Path, rel_path: str, content: str
    ) -> EntityRecord:
        """
        Extract the full content of a compliance note file.

        Files under ``docs/source/notes/`` are treated as first-class
        compliance documents.  The entire file content is stored as a
        single EntityRecord.

        Parameters
        ----------
        filepath : Path
            Absolute path.
        rel_path : str
            Repo-relative path.
        content : str
            Full file contents.

        Returns
        -------
        EntityRecord
        """
        name = filepath.stem
        # Cap raw_text at 5000 chars via get_raw_text (which reads from cache).
        total_lines = content.count("\n") + 1

        return self.make_record(
            source_file=rel_path,
            language="rst" if filepath.suffix == ".rst" else "markdown",
            entity_name=name,
            entity_type="doc_directive",
            subcategory="compliance_note",
            module_path=f"docs.source.notes.{name}",
            qualified_name=f"docs.source.notes.{name}",
            start_line=1,
            end_line=total_lines,
            raw_text=self.get_raw_text(filepath, 1, total_lines),
            extraction_confidence=0.95,
            metadata={
                "doc_type": "compliance_note",
                "total_lines": total_lines,
            },
        )

    # ------------------------------------------------------------------ #
    # RST directive extraction
    # ------------------------------------------------------------------ #

    def _extract_directives(
        self, filepath: Path, rel_path: str, content: str
    ) -> list[EntityRecord]:
        """
        Extract RST directives (warning, note, deprecated, etc.) from content.

        Each matched directive becomes a separate EntityRecord with the
        directive type as the subcategory.

        Parameters
        ----------
        filepath : Path
            Absolute path.
        rel_path : str
            Repo-relative path.
        content : str
            Full file contents.

        Returns
        -------
        list[EntityRecord]
        """
        records: list[EntityRecord] = []

        for directive_type, pattern in DIRECTIVE_PATTERNS.items():
            for match in pattern.finditer(content):
                # Compute the line number from the character offset.
                line_no = content[:match.start()].count("\n") + 1
                matched_text = match.group(0).strip()

                # For directives with version info (deprecated, versionadded, etc.)
                # extract the version string.
                version = ""
                body = matched_text
                if directive_type in ("deprecated", "versionadded", "versionchanged"):
                    # Group 1 is typically the version string.
                    version = match.group(1).strip() if match.group(1) else ""

                lang = "rst" if filepath.suffix == ".rst" else "markdown"
                name = f"{directive_type}_{line_no}"

                records.append(self.make_record(
                    source_file=rel_path,
                    language=lang,
                    entity_name=name,
                    entity_type="doc_directive",
                    subcategory=f"directive_{directive_type}",
                    module_path=self.file_to_module_path(filepath),
                    qualified_name=f"{self.file_to_module_path(filepath)}.{name}",
                    start_line=line_no,
                    end_line=line_no + matched_text.count("\n"),
                    raw_text=matched_text[:5000],
                    extraction_confidence=0.95,
                    metadata={
                        "directive_type": directive_type,
                        "version": version,
                    },
                ))

        return records

    # ------------------------------------------------------------------ #
    # Auto-reference extraction
    # ------------------------------------------------------------------ #

    def _extract_autorefs(
        self, filepath: Path, rel_path: str, content: str
    ) -> list[EntityRecord]:
        """
        Extract ``.. automodule::``, ``.. autoclass::``, etc. references.

        These directives indicate which Python entities are documented and
        help build the cross-reference graph between code and documentation.

        Parameters
        ----------
        filepath : Path
            Absolute path.
        rel_path : str
            Repo-relative path.
        content : str
            Full file contents.

        Returns
        -------
        list[EntityRecord]
        """
        records: list[EntityRecord] = []

        for match in AUTOREF_PATTERN.finditer(content):
            ref_target = match.group(1).strip()
            line_no = content[:match.start()].count("\n") + 1
            matched_text = match.group(0).strip()

            # Determine the auto-directive type from the matched text.
            if "automodule" in matched_text:
                auto_type = "automodule"
            elif "autoclass" in matched_text:
                auto_type = "autoclass"
            elif "autofunction" in matched_text:
                auto_type = "autofunction"
            elif "automethod" in matched_text:
                auto_type = "automethod"
            else:
                auto_type = "auto_unknown"

            lang = "rst" if filepath.suffix == ".rst" else "markdown"

            records.append(self.make_record(
                source_file=rel_path,
                language=lang,
                entity_name=ref_target,
                entity_type="doc_directive",
                subcategory=f"autoref_{auto_type}",
                module_path=self.file_to_module_path(filepath),
                qualified_name=f"{self.file_to_module_path(filepath)}.autoref.{ref_target}",
                start_line=line_no,
                end_line=line_no,
                raw_text=matched_text[:5000],
                extraction_confidence=0.95,
                metadata={
                    "auto_type": auto_type,
                    "ref_target": ref_target,
                },
                # Store a relation linking docs to the code entity.
                relations=[{
                    "type": "documents",
                    "target": ref_target,
                }],
            ))

        return records

    # ------------------------------------------------------------------ #
    # Determinism table extraction
    # ------------------------------------------------------------------ #

    def _extract_determinism_table(
        self, filepath: Path, rel_path: str, content: str
    ) -> list[EntityRecord]:
        """
        Parse the operator determinism table from the determinism doc.

        Looks for markdown tables (``| operator | ... |``) or RST simple
        tables.  For each table row, creates a structured EntityRecord with
        determinism metadata.

        Parameters
        ----------
        filepath : Path
            Absolute path.
        rel_path : str
            Repo-relative path.
        content : str
            Full file contents.

        Returns
        -------
        list[EntityRecord]
        """
        records: list[EntityRecord] = []
        lang = "rst" if filepath.suffix == ".rst" else "markdown"

        # Try markdown table format first.
        md_rows = list(_MD_TABLE_ROW_RE.finditer(content))
        if md_rows:
            self.logger.info(
                "Found %d markdown table rows in %s", len(md_rows), rel_path
            )
            for match in md_rows:
                col1 = match.group(1).strip()
                col2 = match.group(2).strip()
                col3 = match.group(3).strip()

                # Skip header rows and separator rows.
                if col1.startswith("-") or col1.startswith("="):
                    continue
                if col1.lower() in ("operator", "function", "name", "api"):
                    continue

                line_no = content[:match.start()].count("\n") + 1

                # Heuristic: col1 = operator name, col2 = deterministic flag,
                # col3 = affected backends.
                deterministic = self._parse_deterministic_flag(col2)

                records.append(self.make_record(
                    source_file=rel_path,
                    language=lang,
                    entity_name=col1,
                    entity_type="doc_directive",
                    subcategory="determinism_table_entry",
                    module_path=self.file_to_module_path(filepath),
                    qualified_name=f"{self.file_to_module_path(filepath)}.determinism.{col1}",
                    start_line=line_no,
                    end_line=line_no,
                    raw_text=match.group(0).strip(),
                    extraction_confidence=0.95,
                    metadata={
                        "operator_name": col1,
                        "deterministic_enforceable": deterministic,
                        "affected_backends": col3,
                    },
                ))

        # Try RST simple table format if no markdown rows were found.
        if not md_rows:
            rst_rows = list(_RST_TABLE_ROW_RE.finditer(content))
            if rst_rows:
                self.logger.info(
                    "Found %d RST table rows in %s", len(rst_rows), rel_path
                )
                for match in rst_rows:
                    col1 = match.group(1).strip()
                    col2 = match.group(2).strip()
                    col3 = match.group(3).strip()

                    # Skip header/separator rows.
                    if col1.startswith("-") or col1.startswith("="):
                        continue
                    if col1.lower() in ("operator", "function", "name", "api"):
                        continue

                    line_no = content[:match.start()].count("\n") + 1
                    deterministic = self._parse_deterministic_flag(col2)

                    records.append(self.make_record(
                        source_file=rel_path,
                        language=lang,
                        entity_name=col1,
                        entity_type="doc_directive",
                        subcategory="determinism_table_entry",
                        module_path=self.file_to_module_path(filepath),
                        qualified_name=f"{self.file_to_module_path(filepath)}.determinism.{col1}",
                        start_line=line_no,
                        end_line=line_no,
                        raw_text=match.group(0).strip(),
                        extraction_confidence=0.95,
                        metadata={
                            "operator_name": col1,
                            "deterministic_enforceable": deterministic,
                            "affected_backends": col3,
                        },
                    ))

        return records

    def _parse_deterministic_flag(self, text: str) -> bool:
        """
        Parse a deterministic flag from table text.

        Recognises ``yes``, ``true``, ``1`` (case-insensitive) as True;
        everything else as False.

        Parameters
        ----------
        text : str
            The cell text.

        Returns
        -------
        bool
        """
        return text.strip().lower() in ("yes", "true", "1", "y")
