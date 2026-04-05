"""
Sphinx notes extractor — parses ``docs/source/notes/`` for compliance-relevant
documentation, with special handling for the determinism operator table.

Passes
------
1. Full document extraction (one EntityRecord per .md/.rst file)
2. Determinism table parser (structured records per operator row)
3. Cross-reference helper ``get_determinism_operators()``
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import BaseExtractor, EntityRecord, compute_stable_id

OUTPUT_FILE = "sphinx_notes.jsonl"

# ---------------------------------------------------------------------------
# Compliance tag inference from filename stems
# ---------------------------------------------------------------------------

_STEM_TAG_MAP: list[tuple[str, list[str]]] = [
    ("determinism", ["eu_ai_act_art_15", "reproducibility"]),
    ("autograd", ["eu_ai_act_art_15"]),
    ("cuda", ["eu_ai_act_art_15"]),
    ("serialization", ["eu_ai_act_art_12"]),
    ("broadcasting", ["eu_ai_act_art_15"]),
    ("faq", ["eu_ai_act_art_13"]),
]


def _infer_compliance_tags(stem: str) -> list[str]:
    """Infer compliance tags from the filename stem."""
    lower = stem.lower()
    for prefix, tags in _STEM_TAG_MAP:
        if lower.startswith(prefix):
            return list(tags)
    return []


# ---------------------------------------------------------------------------
# RST cross-reference stripping
# ---------------------------------------------------------------------------

_RST_XREF_RE = re.compile(r":(?:func|class|meth|mod|attr|ref):`~?([^`]+)`")


def _strip_rst_xref(text: str) -> str:
    """Remove RST cross-reference markup, keeping the inner text."""
    return _RST_XREF_RE.sub(r"\1", text)


class SphinxNotesExtractor(BaseExtractor):
    """Extract compliance notes from docs/source/notes/."""

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__(
            name="sphinx_notes",
            repo_path=repo_path,
            output_path=output_path,
        )
        # Populated by the determinism table parser for cross-reference use
        self._determinism_operators: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> int:
        total = 0
        total += self._pass_1_full_documents()
        total += self._pass_2_determinism_table()

        out = str(self.output_path / OUTPUT_FILE)
        self.flush(out)
        self.report_stats()
        return total

    def get_determinism_operators(self) -> set[str]:
        """
        Return the set of operator names found in the determinism table.

        Must be called after :meth:`extract` to return meaningful results.
        """
        return set(self._determinism_operators)

    # ------------------------------------------------------------------
    # Pass 1 — Full document extraction
    # ------------------------------------------------------------------

    def _pass_1_full_documents(self) -> int:
        self.logger.info("Pass 1 — Full document extraction: starting")
        notes_dir = self.repo_path / "docs" / "source" / "notes"
        if not notes_dir.exists():
            self.logger.warning("docs/source/notes/ not found at %s", notes_dir)
            self._warnings += 1
            return 0

        count = 0
        extensions = ("*.md", "*.rst")
        files: list[Path] = []
        for ext in extensions:
            files.extend(sorted(notes_dir.glob(ext)))

        for fpath in files:
            try:
                content = self.read_file_safe(fpath)
                if content is None:
                    continue
                self._files_processed += 1
                rel = str(fpath.relative_to(self.repo_path))
                stem = fpath.stem
                full_length = len(content)
                # Cap raw_text at 5000 chars
                raw_text = content[:5000]

                lang = "rst" if fpath.suffix == ".rst" else "markdown"
                record = self.make_record(
                    source_file=rel,
                    language=lang,
                    entity_name=stem,
                    entity_type="doc_directive",
                    subcategory="compliance_note",
                    module_path=f"docs.source.notes.{stem}",
                    qualified_name=f"docs.source.notes.{stem}",
                    start_line=1,
                    end_line=content.count("\n") + 1,
                    raw_text=raw_text,
                    extraction_confidence=0.95,
                    compliance_tags=_infer_compliance_tags(stem),
                    lifecycle_phase="documentation",
                    metadata={"full_length": full_length},
                )
                self.write_record(record, str(self.output_path / OUTPUT_FILE))
                count += 1

            except Exception as exc:
                self.logger.error("Error processing note %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("Pass 1 — Full document extraction: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Pass 2 — Determinism table parser
    # ------------------------------------------------------------------

    def _pass_2_determinism_table(self) -> int:
        self.logger.info("Pass 2 — Determinism table parser: starting")
        count = 0

        # Try both .rst and .md variants
        candidates = [
            self.repo_path / "docs" / "source" / "notes" / "determinism.rst",
            self.repo_path / "docs" / "source" / "notes" / "deterministic.rst",
            self.repo_path / "docs" / "source" / "notes" / "determinism.md",
            self.repo_path / "docs" / "source" / "notes" / "deterministic.md",
        ]

        target: Path | None = None
        for c in candidates:
            if c.exists():
                target = c
                break

        if target is None:
            self.logger.warning(
                "No determinism note file found among candidates: %s",
                [str(c) for c in candidates],
            )
            self._warnings += 1
            return 0

        try:
            content = self.read_file_safe(target)
            if content is None:
                return 0
            self._files_processed += 1
            rel = str(target.relative_to(self.repo_path))

            if target.suffix == ".md":
                count += self._parse_markdown_tables(content, rel)
            else:
                count += self._parse_rst_tables(content, rel)

        except Exception as exc:
            self.logger.error(
                "Error parsing determinism table in %s: %s", target, exc
            )
            self._errors += 1

        self.logger.info("Pass 2 — Determinism table parser: %d records", count)
        return count

    # ---- Markdown table parsing ----

    _MD_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")

    def _parse_markdown_tables(self, content: str, rel: str) -> int:
        count = 0
        lines = content.splitlines()
        in_table = False
        header_cols: list[str] = []

        for lineno_0, line in enumerate(lines):
            row_match = self._MD_TABLE_ROW_RE.match(line)
            if row_match:
                cells = [c.strip() for c in row_match.group(1).split("|")]
                # Skip separator rows (all dashes/colons)
                if all(re.match(r"^[-:]+$", c) for c in cells if c):
                    continue
                # First real row is the header
                if not in_table:
                    header_cols = cells
                    in_table = True
                    continue
                # Data row — try to extract operator info
                rec = self._make_table_entry(cells, header_cols, rel, lineno_0 + 1)
                if rec is not None:
                    self.write_record(rec, str(self.output_path / OUTPUT_FILE))
                    count += 1
            else:
                in_table = False
                header_cols = []

        return count

    # ---- RST table parsing ----

    def _parse_rst_tables(self, content: str, rel: str) -> int:
        count = 0
        lines = content.splitlines()

        # RST simple tables: separated by lines of ===
        # Also try grid tables with | ... | ... |
        in_table = False
        header_cols: list[str] = []

        for lineno_0, line in enumerate(lines):
            # Check for markdown-style rows inside RST (common in Sphinx)
            row_match = self._MD_TABLE_ROW_RE.match(line)
            if row_match:
                cells = [c.strip() for c in row_match.group(1).split("|")]
                if all(re.match(r"^[-:=+]+$", c) for c in cells if c):
                    continue
                if not in_table:
                    header_cols = cells
                    in_table = True
                    continue
                rec = self._make_table_entry(cells, header_cols, rel, lineno_0 + 1)
                if rec is not None:
                    self.write_record(rec, str(self.output_path / OUTPUT_FILE))
                    count += 1
            elif re.match(r"^[=]+(\s+[=]+)+\s*$", line):
                # RST simple table separator — reset
                in_table = False
                header_cols = []
            else:
                # Could be a simple-table data row (space-separated columns)
                if in_table and line.strip():
                    # Try splitting on 2+ spaces
                    cells = re.split(r"\s{2,}", line.strip())
                    if len(cells) >= 2:
                        rec = self._make_table_entry(
                            cells, header_cols, rel, lineno_0 + 1
                        )
                        if rec is not None:
                            self.write_record(
                                rec, str(self.output_path / OUTPUT_FILE)
                            )
                            count += 1
                elif not line.strip():
                    pass  # blank line inside table is OK
                else:
                    in_table = False
                    header_cols = []

        return count

    # ---- Shared table-row → EntityRecord ----

    def _make_table_entry(
        self,
        cells: list[str],
        header_cols: list[str],
        rel: str,
        lineno: int,
    ) -> EntityRecord | None:
        """Convert a table row into a determinism_table_entry record."""
        if not cells:
            return None

        # First cell is the operator name
        raw_name = _strip_rst_xref(cells[0]).strip()
        if not raw_name or raw_name.startswith("-"):
            return None

        # Clean up backticks and other markup
        op_name = raw_name.strip("`").strip()
        if not op_name:
            return None

        # Try to extract deterministic_enforceable and affected_backends
        deterministic_enforceable: bool | str = "unknown"
        affected_backends: list[str] = []
        notes = ""

        if len(cells) > 1:
            det_str = cells[1].strip().lower()
            if det_str in ("yes", "true"):
                deterministic_enforceable = True
            elif det_str in ("no", "false"):
                deterministic_enforceable = False

        if len(cells) > 2:
            backend_str = _strip_rst_xref(cells[2]).strip()
            if backend_str:
                affected_backends = [
                    b.strip() for b in re.split(r"[,/]", backend_str) if b.strip()
                ]

        if len(cells) > 3:
            notes = _strip_rst_xref(cells[3]).strip()

        self._determinism_operators.add(op_name)

        record = self.make_record(
            source_file=rel,
            language="rst",
            entity_name=op_name,
            entity_type="doc_directive",
            subcategory="determinism_table_entry",
            module_path="docs.source.notes.determinism",
            qualified_name=f"docs.determinism_table.{op_name}",
            start_line=lineno,
            end_line=lineno,
            raw_text=" | ".join(cells)[:300],
            extraction_confidence=0.95,
            compliance_tags=["eu_ai_act_art_15", "reproducibility"],
            lifecycle_phase="documentation",
            metadata={
                "operator_name": op_name,
                "deterministic_enforceable": deterministic_enforceable,
                "affected_backends": affected_backends,
                "notes": notes,
            },
        )
        return record
