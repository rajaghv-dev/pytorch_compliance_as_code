"""
Compliance Tools Extractor — scans PyTorch core for references to external
compliance-related tools (Opacus, CrypTen, Captum) and internal analysis
utilities (torch.fx.passes, torch.utils.viz).

Compliance relevance:
    - Opacus / CrypTen references map to EU AI Act Article 10 + GDPR Article 5
      (privacy by design).
    - Captum references map to EU AI Act Article 14 (human oversight /
      explainability).
    - torch.fx.passes graph analysis tools map to Article 15 (accuracy,
      robustness).
    - torch.utils.viz visualization tools map to Article 13 (transparency).

Also extracts ``TorchDispatchMode`` and related dispatch-mode APIs that
underpin many of these tools.
"""

from __future__ import annotations

import ast
import re
import logging
from pathlib import Path
from typing import Any

from .base import EntityRecord, BaseExtractor, compute_stable_id

logger = logging.getLogger("pct.extractors.compliance_tools")

# ---------------------------------------------------------------------------
# Regex patterns for external tool references
# ---------------------------------------------------------------------------

_TOOL_PATTERNS: dict[str, re.Pattern] = {
    "opacus": re.compile(
        r'\b(opacus|differential.privacy|dp_sgd|privacy.engine)\b', re.IGNORECASE
    ),
    "crypten": re.compile(
        r'\b(crypten|secure.computation|mpc|secret.sharing)\b', re.IGNORECASE
    ),
    "captum": re.compile(
        r'\b(captum|attribution|integrated.gradients|saliency|shap)\b', re.IGNORECASE
    ),
}

# ---------------------------------------------------------------------------
# Compliance tags per tool
# ---------------------------------------------------------------------------

_TOOL_COMPLIANCE_TAGS: dict[str, list[str]] = {
    "opacus":  ["eu_ai_act_art_10", "gdpr_art_5"],
    "crypten": ["eu_ai_act_art_10", "gdpr_art_5"],
    "captum":  ["eu_ai_act_art_14"],
}

# ---------------------------------------------------------------------------
# Additional scan targets: internal analysis / viz utilities
# ---------------------------------------------------------------------------

_FX_PASSES_PATTERN = re.compile(r'\btorch\.fx\.passes\b')
_VIZ_PATTERN = re.compile(r'\btorch\.utils\.viz\b')

# Dispatch-mode targets extracted via AST
_DISPATCH_MODE_TARGETS: set[str] = {
    "TorchDispatchMode",
    "enable_torch_dispatch_mode",
    "BaseTorchDispatchMode",
}


class ComplianceToolsExtractor(BaseExtractor):
    """
    Scan PyTorch core for references to compliance-relevant external tools
    and internal analysis utilities.

    Produces EntityRecords for each regex match or AST-located entity,
    tagged with the appropriate compliance articles.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the compliance tools extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="compliance_tools", repo_path=repo_path, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Run the compliance tools extraction pass.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting compliance tools extraction")
        output_file = str(self.output_path / "compliance_tools.jsonl")

        # --- External tool references (opacus, crypten, captum) ---
        self._extract_tool_references(output_file)

        # --- Internal analysis utilities (torch.fx.passes, torch.utils.viz) ---
        self._extract_internal_utilities(output_file)

        # --- Dispatch-mode API used by these tools ---
        self._extract_dispatch_mode_api(output_file)

        # Flush remaining buffered records
        self.flush(output_file)

        self.logger.info(
            "Compliance tools extraction complete — %d records produced",
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # External tool reference scanning
    # ------------------------------------------------------------------ #

    def _extract_tool_references(self, output_file: str) -> None:
        """
        Scan all Python files under ``torch/`` for references to Opacus,
        CrypTen, and Captum.

        Each regex match is emitted as an EntityRecord with the appropriate
        compliance tags (privacy or explainability).
        """
        for filepath in self.find_files("torch/**/*.py"):
            content = self.read_file_safe(filepath)
            if content is None:
                self.logger.warning("Skipping unreadable file: %s", filepath)
                continue

            self._files_processed += 1
            rel_path = str(filepath.relative_to(self.repo_path))
            module_path = self.file_to_module_path(filepath)
            lines = content.splitlines()

            for tool, regex in _TOOL_PATTERNS.items():
                for match in regex.finditer(content):
                    # Compute line number of the match
                    line_num = content[:match.start()].count("\n") + 1

                    # Get surrounding context (3 lines before/after)
                    ctx_start = max(0, line_num - 4)
                    ctx_end = min(len(lines), line_num + 3)
                    context = "\n".join(lines[ctx_start:ctx_end])

                    # Look up compliance tags for this tool
                    compliance_tags = _TOOL_COMPLIANCE_TAGS.get(tool, [])

                    record = self.make_record(
                        source_file=rel_path,
                        language="python",
                        entity_name=f"{tool}::{match.group()}",
                        entity_type="tool_reference",
                        subcategory=tool,
                        module_path=module_path,
                        qualified_name=f"{module_path}.ref.{tool}",
                        start_line=line_num,
                        end_line=line_num,
                        raw_text=context,
                        compliance_tags=list(compliance_tags),
                        extraction_confidence=0.8,  # name-matched via regex
                        metadata={"tool": tool, "match": match.group()},
                    )
                    self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # Internal analysis / visualization utilities
    # ------------------------------------------------------------------ #

    def _extract_internal_utilities(self, output_file: str) -> None:
        """
        Scan for references to ``torch.fx.passes.*`` (graph analysis) and
        ``torch.utils.viz.*`` (visualization) across the codebase.

        - ``torch.fx.passes`` references get ``eu_ai_act_art_15`` tags.
        - ``torch.utils.viz`` references get ``eu_ai_act_art_13`` tags.
        """
        for filepath in self.find_files("torch/**/*.py"):
            content = self.read_file_safe(filepath)
            if content is None:
                continue

            # Already counted in _extract_tool_references, so don't double-count
            rel_path = str(filepath.relative_to(self.repo_path))
            module_path = self.file_to_module_path(filepath)
            lines = content.splitlines()

            # --- torch.fx.passes references → Article 15 ---
            for match in _FX_PASSES_PATTERN.finditer(content):
                line_num = content[:match.start()].count("\n") + 1
                ctx_start = max(0, line_num - 4)
                ctx_end = min(len(lines), line_num + 3)
                context = "\n".join(lines[ctx_start:ctx_end])

                record = self.make_record(
                    source_file=rel_path,
                    language="python",
                    entity_name=f"fx_passes::{match.group()}",
                    entity_type="tool_reference",
                    subcategory="fx_passes",
                    module_path=module_path,
                    qualified_name=f"{module_path}.ref.fx_passes",
                    start_line=line_num,
                    end_line=line_num,
                    raw_text=context,
                    compliance_tags=["eu_ai_act_art_15"],
                    extraction_confidence=0.8,
                    metadata={"tool": "torch.fx.passes", "match": match.group()},
                )
                self.write_record(record, output_file)

            # --- torch.utils.viz references → Article 13 ---
            for match in _VIZ_PATTERN.finditer(content):
                line_num = content[:match.start()].count("\n") + 1
                ctx_start = max(0, line_num - 4)
                ctx_end = min(len(lines), line_num + 3)
                context = "\n".join(lines[ctx_start:ctx_end])

                record = self.make_record(
                    source_file=rel_path,
                    language="python",
                    entity_name=f"viz::{match.group()}",
                    entity_type="tool_reference",
                    subcategory="visualization",
                    module_path=module_path,
                    qualified_name=f"{module_path}.ref.viz",
                    start_line=line_num,
                    end_line=line_num,
                    raw_text=context,
                    compliance_tags=["eu_ai_act_art_13"],
                    extraction_confidence=0.8,
                    metadata={"tool": "torch.utils.viz", "match": match.group()},
                )
                self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # Dispatch-mode API extraction
    # ------------------------------------------------------------------ #

    def _extract_dispatch_mode_api(self, output_file: str) -> None:
        """
        Extract ``TorchDispatchMode`` and related dispatch-mode API classes
        and functions from ``torch/utils/_python_dispatch.py``.

        These are the extension points that tools like Opacus/CrypTen/Captum
        use to hook into PyTorch's dispatch machinery.
        """
        for filepath in self.find_files("torch/utils/_python_dispatch.py"):
            source = self.read_file_safe(filepath)
            if source is None:
                self.logger.warning("Skipping unreadable file: %s", filepath)
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

            for node in ast.walk(tree):
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    continue
                if node.name not in _DISPATCH_MODE_TARGETS:
                    continue

                start_line = node.lineno
                end_line = getattr(node, "end_lineno", node.lineno)
                entity_type = "class" if isinstance(node, ast.ClassDef) else "function"
                docstring = ast.get_docstring(node) or ""
                qualified_name = self.compute_qualified_name(module_path, "", node.name)

                record = self.make_record(
                    source_file=rel_path,
                    language="python",
                    entity_name=node.name,
                    entity_type=entity_type,
                    subcategory="dispatch_mode_api",
                    module_path=module_path,
                    qualified_name=qualified_name,
                    start_line=start_line,
                    end_line=end_line,
                    raw_text=self.get_raw_text(filepath, start_line, end_line),
                    docstring=docstring,
                    compliance_tags=["eu_ai_act_art_15"],
                    extraction_confidence=1.0,  # AST-found
                    metadata={},
                )
                self.write_record(record, output_file)
