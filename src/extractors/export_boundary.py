"""
Export boundary extractor — discovers ONNX, torch.export, TorchScript,
torch.compile / Dynamo, FX, and serialization APIs across the PyTorch
repository.

Each export target represents a boundary where the model's representation
changes form, which has direct compliance implications (EU AI Act Art 11/12/15).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from .base import BaseExtractor, EntityRecord, compute_stable_id

OUTPUT_FILE = "export_boundary.jsonl"

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

ONNX_TARGETS = {"export", "dynamo_export", "register_custom_op_symbolic"}
TORCH_EXPORT_TARGETS = {"ExportedProgram", "Constraint", "dynamic_dim", "export", "unflatten"}
TORCHSCRIPT_TARGETS = {"trace", "script", "save", "load", "freeze", "optimize_for_inference"}
DYNAMO_FUNCTION_TARGETS = {"optimize", "reset", "explain", "assume_constant_result"}
FX_TARGETS = {"symbolic_trace", "Graph", "GraphModule", "Node", "Interpreter", "Transformer"}
SERIALIZATION_TARGETS = {"save", "load"}

_GRAPH_BREAK_RE = re.compile(
    r"""(?:graph_break|unimplemented)\s*\(.*?["\']([^"\']+)["\']""",
    re.DOTALL,
)


class ExportBoundaryExtractor(BaseExtractor):
    """Extract export-boundary entities from the PyTorch repository."""

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__(
            name="export_boundary",
            repo_path=repo_path,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> int:
        total = 0
        total += self._extract_onnx()
        total += self._extract_torch_export()
        total += self._extract_torchscript()
        total += self._extract_dynamo()
        total += self._extract_fx()
        total += self._extract_serialization()

        out = str(self.output_path / OUTPUT_FILE)
        self.flush(out)
        self.report_stats()
        return total

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def _extract_onnx(self) -> int:
        self.logger.info("ONNX extraction: starting")
        count = 0
        onnx_dir = self.repo_path / "torch" / "onnx"
        if not onnx_dir.exists():
            self.logger.warning("torch/onnx directory not found")
            self._warnings += 1
            return 0

        files = sorted(onnx_dir.rglob("*.py"))
        for fpath in files:
            try:
                source = self.read_file_safe(fpath)
                if source is None:
                    continue
                tree = ast.parse(source, filename=str(fpath))
                self._files_processed += 1
                rel = str(fpath.relative_to(self.repo_path))
                module_path = self.file_to_module_path(fpath)

                symbolic_count = 0
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name in ONNX_TARGETS:
                            record = self.make_record(
                                source_file=rel,
                                language="python",
                                entity_name=node.name,
                                entity_type="function",
                                subcategory="onnx_export",
                                module_path=module_path,
                                qualified_name=f"{module_path}.{node.name}",
                                start_line=node.lineno,
                                end_line=getattr(node, "end_lineno", node.lineno),
                                signature=self.extract_function_signature(node),
                                docstring=ast.get_docstring(node) or "",
                                extraction_confidence=1.0,
                                compliance_tags=["eu_ai_act_art_11"],
                                lifecycle_phase="export",
                            )
                            self.write_record(
                                record, str(self.output_path / OUTPUT_FILE)
                            )
                            count += 1

                        # Count symbolic_* functions
                        if node.name.startswith("symbolic_"):
                            symbolic_count += 1

                if symbolic_count > 0:
                    record = self.make_record(
                        source_file=rel,
                        language="python",
                        entity_name=f"symbolic_functions_in_{fpath.stem}",
                        entity_type="config_entry",
                        subcategory="onnx_export",
                        module_path=module_path,
                        qualified_name=f"{module_path}.symbolic_*",
                        extraction_confidence=0.9,
                        compliance_tags=["eu_ai_act_art_11"],
                        lifecycle_phase="export",
                        metadata={"symbolic_function_count": symbolic_count},
                    )
                    self.write_record(record, str(self.output_path / OUTPUT_FILE))
                    count += 1

            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Error processing ONNX file %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("ONNX extraction: %d records", count)
        return count

    # ------------------------------------------------------------------
    # torch.export
    # ------------------------------------------------------------------

    def _extract_torch_export(self) -> int:
        self.logger.info("torch.export extraction: starting")
        count = 0
        export_dir = self.repo_path / "torch" / "export"
        if not export_dir.exists():
            self.logger.warning("torch/export directory not found")
            self._warnings += 1
            return 0

        files = sorted(export_dir.rglob("*.py"))
        for fpath in files:
            try:
                count += self._walk_for_targets(
                    fpath,
                    TORCH_EXPORT_TARGETS,
                    subcategory="torch_export",
                    compliance_tags=["eu_ai_act_art_11"],
                    lifecycle_phase="export",
                )
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Error in torch.export for %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("torch.export extraction: %d records", count)
        return count

    # ------------------------------------------------------------------
    # TorchScript
    # ------------------------------------------------------------------

    def _extract_torchscript(self) -> int:
        self.logger.info("TorchScript extraction: starting")
        count = 0
        jit_dir = self.repo_path / "torch" / "jit"
        if not jit_dir.exists():
            self.logger.warning("torch/jit directory not found")
            self._warnings += 1
            return 0

        files = sorted(jit_dir.rglob("*.py"))
        for fpath in files:
            try:
                count += self._walk_for_targets(
                    fpath,
                    TORCHSCRIPT_TARGETS,
                    subcategory="torchscript",
                    compliance_tags=["eu_ai_act_art_11"],
                    lifecycle_phase="export",
                )
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Error in TorchScript for %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("TorchScript extraction: %d records", count)
        return count

    # ------------------------------------------------------------------
    # torch.compile / Dynamo
    # ------------------------------------------------------------------

    def _extract_dynamo(self) -> int:
        self.logger.info("Dynamo extraction: starting")
        count = 0
        dynamo_dir = self.repo_path / "torch" / "_dynamo"
        if not dynamo_dir.exists():
            self.logger.warning("torch/_dynamo directory not found")
            self._warnings += 1
            return 0

        files = sorted(dynamo_dir.rglob("*.py"))
        for fpath in files:
            try:
                # Extract named function targets
                count += self._walk_for_targets(
                    fpath,
                    DYNAMO_FUNCTION_TARGETS,
                    subcategory="dynamo_compile",
                    compliance_tags=["eu_ai_act_art_11"],
                    lifecycle_phase="compilation",
                )
                # Extract graph break reasons
                count += self._extract_graph_break_reasons(fpath)
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Error in Dynamo for %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("Dynamo extraction: %d records", count)
        return count

    def _extract_graph_break_reasons(self, fpath: Path) -> int:
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        count = 0
        rel = str(fpath.relative_to(self.repo_path))
        module_path = self.file_to_module_path(fpath)

        seen_reasons: set[str] = set()
        for match in _GRAPH_BREAK_RE.finditer(source):
            reason = match.group(1).strip()
            if reason in seen_reasons:
                continue
            seen_reasons.add(reason)

            lineno = source[: match.start()].count("\n") + 1
            record = self.make_record(
                source_file=rel,
                language="python",
                entity_name=reason[:120],
                entity_type="config_entry",
                subcategory="graph_break_reason",
                module_path=module_path,
                qualified_name=f"{module_path}:graph_break:{reason[:80]}",
                start_line=lineno,
                end_line=lineno,
                raw_text=source[match.start() : match.end()][:300],
                extraction_confidence=0.85,
                compliance_tags=["eu_ai_act_art_15"],
                lifecycle_phase="compilation",
                metadata={"reason": reason},
            )
            self.write_record(record, str(self.output_path / OUTPUT_FILE))
            count += 1

        return count

    # ------------------------------------------------------------------
    # FX
    # ------------------------------------------------------------------

    def _extract_fx(self) -> int:
        self.logger.info("FX extraction: starting")
        count = 0
        fx_dir = self.repo_path / "torch" / "fx"
        if not fx_dir.exists():
            self.logger.warning("torch/fx directory not found")
            self._warnings += 1
            return 0

        files = sorted(fx_dir.rglob("*.py"))
        for fpath in files:
            try:
                count += self._walk_for_targets(
                    fpath,
                    FX_TARGETS,
                    subcategory="fx_graph",
                    compliance_tags=["eu_ai_act_art_11"],
                    lifecycle_phase="compilation",
                )
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Error in FX for %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("FX extraction: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _extract_serialization(self) -> int:
        self.logger.info("Serialization extraction: starting")
        count = 0
        target = self.repo_path / "torch" / "serialization.py"
        if not target.exists():
            self.logger.warning("torch/serialization.py not found")
            self._warnings += 1
            return 0

        try:
            count += self._walk_for_targets(
                target,
                SERIALIZATION_TARGETS,
                subcategory="serialization",
                compliance_tags=["eu_ai_act_art_12"],
                lifecycle_phase="serialization",
            )
        except SyntaxError as exc:
            self.logger.warning("Syntax error in %s: %s", target, exc)
            self._warnings += 1
        except Exception as exc:
            self.logger.error("Error in serialization: %s", exc)
            self._errors += 1

        self.logger.info("Serialization extraction: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _walk_for_targets(
        self,
        fpath: Path,
        target_names: set[str],
        subcategory: str,
        compliance_tags: list[str],
        lifecycle_phase: str,
    ) -> int:
        """AST-walk *fpath* for functions and classes whose name is in *target_names*."""
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        tree = ast.parse(source, filename=str(fpath))
        self._files_processed += 1
        count = 0
        rel = str(fpath.relative_to(self.repo_path))
        module_path = self.file_to_module_path(fpath)

        for node in ast.walk(tree):
            name: Optional[str] = None
            etype = "function"
            sig = ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in target_names:
                    name = node.name
                    sig = self.extract_function_signature(node)
            elif isinstance(node, ast.ClassDef):
                if node.name in target_names:
                    name = node.name
                    etype = "class"

            if name is not None:
                record = self.make_record(
                    source_file=rel,
                    language="python",
                    entity_name=name,
                    entity_type=etype,
                    subcategory=subcategory,
                    module_path=module_path,
                    qualified_name=f"{module_path}.{name}",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature=sig,
                    docstring=ast.get_docstring(node) or "",
                    extraction_confidence=1.0,
                    compliance_tags=compliance_tags,
                    lifecycle_phase=lifecycle_phase,
                )
                self.write_record(record, str(self.output_path / OUTPUT_FILE))
                count += 1

        return count
