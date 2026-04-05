"""
Hookability extractor — discovers hook definitions, hook consumers,
dispatch keys, override protocols, profiler hooks, and module callbacks
across the PyTorch repository.

Passes
------
A. Hook definitions (FunctionDef nodes matching HOOK_METHODS)
B. Hook consumers  (Call nodes whose attribute matches HOOK_METHODS) — BUG-12 fix
C. Dispatch keys    (C++ enum class DispatchKey values)
D. Override protocols (__torch_function__, __torch_dispatch__, etc.)
E. Profiler hooks   (profile, record_function, etc.)
F. Module callbacks  (train, eval, apply, to, zero_grad, etc.)
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from .base import BaseExtractor, EntityRecord, compute_stable_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOOK_METHODS = {
    "register_forward_hook",
    "register_forward_pre_hook",
    "register_backward_hook",
    "register_full_backward_hook",
    "register_state_dict_pre_hook",
    "register_load_state_dict_post_hook",
    "register_module_forward_hook",
    "register_module_forward_pre_hook",
    "register_module_backward_hook",
    "register_module_full_backward_hook",
    "register_hook",
    "register_multi_grad_hook",
    "register_post_accumulate_grad_hook",
}

OVERRIDE_PROTOCOLS = {
    "__torch_function__",
    "__torch_dispatch__",
    "TorchFunctionMode",
    "TorchDispatchMode",
}

PROFILER_TARGETS = {
    "profile",
    "record_function",
    "emit_nvtx",
    "emit_itt",
    "ExecutionTraceObserver",
}

MODULE_CALLBACKS = {
    "train",
    "eval",
    "apply",
    "to",
    "zero_grad",
    "_apply",
    "_load_from_state_dict",
}

OUTPUT_FILE = "hookability.jsonl"


def _lifecycle_phase_for_hook(name: str) -> str:
    """Infer lifecycle phase from hook/callback name."""
    lower = name.lower()
    if "forward" in lower:
        return "forward_pass"
    if "backward" in lower or "grad" in lower:
        return "backward_pass"
    if "state_dict" in lower or "load" in lower:
        return "serialization"
    if "train" in lower or "eval" in lower:
        return "mode_switch"
    if "profile" in lower or "record" in lower or "trace" in lower:
        return "profiling"
    if "dispatch" in lower:
        return "dispatch"
    return "runtime"


class HookabilityExtractor(BaseExtractor):
    """Extract hook-related entities from the PyTorch repository."""

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__(
            name="hookability",
            repo_path=repo_path,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> int:
        """Run all six extraction passes and return total record count."""
        total = 0
        total += self._pass_a_hook_definitions()
        total += self._pass_b_hook_consumers()
        total += self._pass_c_dispatch_keys()
        total += self._pass_d_override_protocols()
        total += self._pass_e_profiler_hooks()
        total += self._pass_f_module_callbacks()

        out = str(self.output_path / OUTPUT_FILE)
        self.flush(out)
        self.report_stats()
        return total

    # ------------------------------------------------------------------
    # Pass A — Hook definitions
    # ------------------------------------------------------------------

    def _pass_a_hook_definitions(self) -> int:
        self.logger.info("Pass A — Hook definitions: starting")
        count = 0
        search_dirs = ["torch/nn", "torch/autograd", "torch/utils"]
        files = self._collect_python_files(search_dirs)

        for fpath in files:
            try:
                count += self._walk_hook_definitions(fpath)
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Unexpected error in Pass A for %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("Pass A — Hook definitions: %d records", count)
        return count

    def _walk_hook_definitions(self, fpath: Path) -> int:
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        tree = ast.parse(source, filename=str(fpath))
        self._files_processed += 1
        count = 0
        rel = str(fpath.relative_to(self.repo_path))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in HOOK_METHODS:
                    # Determine enclosing class
                    class_name = self._find_enclosing_class(tree, node)
                    module_path = self.file_to_module_path(fpath)
                    record = self.make_record(
                        source_file=rel,
                        language="python",
                        entity_name=node.name,
                        entity_type="method",
                        subcategory="hook_definition",
                        module_path=module_path,
                        qualified_name=self.compute_qualified_name(
                            module_path, class_name, node.name
                        ),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=self.extract_function_signature(node),
                        docstring=ast.get_docstring(node) or "",
                        raw_text=self.get_raw_text(
                            fpath, node.lineno,
                            getattr(node, "end_lineno", node.lineno),
                        ),
                        extraction_confidence=1.0,
                        compliance_tags=["eu_ai_act_art_61"],
                        lifecycle_phase=_lifecycle_phase_for_hook(node.name),
                    )
                    self.write_record(record, str(self.output_path / OUTPUT_FILE))
                    count += 1

        return count

    # ------------------------------------------------------------------
    # Pass B — Hook consumers (BUG-12 fix)
    # ------------------------------------------------------------------

    def _pass_b_hook_consumers(self) -> int:
        self.logger.info("Pass B — Hook consumers: starting")
        count = 0
        search_dirs = [
            "torch/nn", "torch/autograd", "torch/utils", "torch/nn/utils",
        ]
        files = self._collect_python_files(search_dirs)

        for fpath in files:
            try:
                count += self._walk_hook_consumers(fpath)
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Unexpected error in Pass B for %s: %s", fpath, exc)
                self._errors += 1

        self.logger.info("Pass B — Hook consumers: %d records", count)
        return count

    def _walk_hook_consumers(self, fpath: Path) -> int:
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        tree = ast.parse(source, filename=str(fpath))
        self._files_processed += 1
        count = 0
        rel = str(fpath.relative_to(self.repo_path))
        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in HOOK_METHODS:
                    lineno = node.lineno
                    # Build caller context (up to 200 chars around the call)
                    ctx_start = max(0, lineno - 3)
                    ctx_end = min(len(lines), lineno + 2)
                    caller_context = "\n".join(lines[ctx_start:ctx_end])[:200]

                    module_path = self.file_to_module_path(fpath)
                    record = self.make_record(
                        source_file=rel,
                        language="python",
                        entity_name=func.attr,
                        entity_type="method",
                        subcategory="hook_consumer",
                        module_path=module_path,
                        qualified_name=f"{module_path}.{func.attr}",
                        start_line=lineno,
                        end_line=getattr(node, "end_lineno", lineno),
                        raw_text=self.get_raw_text(
                            fpath, lineno,
                            getattr(node, "end_lineno", lineno),
                        ),
                        extraction_confidence=0.9,
                        compliance_tags=["eu_ai_act_art_61"],
                        lifecycle_phase=_lifecycle_phase_for_hook(func.attr),
                        metadata={"caller_context": caller_context},
                    )
                    self.write_record(record, str(self.output_path / OUTPUT_FILE))
                    count += 1

        return count

    # ------------------------------------------------------------------
    # Pass C — Dispatch keys
    # ------------------------------------------------------------------

    # [^{]* allows for `: uint16_t` or other type annotations between
    # the enum name and the opening brace (PyTorch uses `DispatchKey : uint16_t`)
    _DISPATCH_KEY_RE = re.compile(
        r"enum\s+class\s+DispatchKey[^{]*\{([^}]+)\}",
        re.DOTALL,
    )

    def _pass_c_dispatch_keys(self) -> int:
        self.logger.info("Pass C — Dispatch keys: starting")
        count = 0
        header_globs = [
            "c10/core/DispatchKey.h",
            "aten/src/ATen/core/dispatch/*.h",
        ]
        files: list[Path] = []
        for pattern in header_globs:
            found = self.find_files(pattern)
            files.extend(found)

        for fpath in files:
            try:
                count += self._extract_dispatch_keys(fpath)
            except Exception as exc:
                self.logger.error(
                    "Unexpected error in Pass C for %s: %s", fpath, exc
                )
                self._errors += 1

        self.logger.info("Pass C — Dispatch keys: %d records", count)
        return count

    def _extract_dispatch_keys(self, fpath: Path) -> int:
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        self._files_processed += 1
        count = 0
        rel = str(fpath.relative_to(self.repo_path))

        for match in self._DISPATCH_KEY_RE.finditer(source):
            body = match.group(1)
            # Find line number of the enum opening
            enum_start_offset = match.start()
            enum_start_line = source[:enum_start_offset].count("\n") + 1

            for i, raw_entry in enumerate(body.split(",")):
                entry = raw_entry.strip()
                if not entry or entry.startswith("//") or entry.startswith("/*"):
                    continue
                # Strip inline comments and assignments
                entry_name = entry.split("=")[0].split("//")[0].strip()
                if not entry_name or entry_name.startswith("#"):
                    continue

                record = self.make_record(
                    source_file=rel,
                    language="cpp",
                    entity_name=entry_name,
                    entity_type="enum",
                    subcategory="dispatch_key",
                    module_path="c10.core.DispatchKey",
                    qualified_name=f"DispatchKey::{entry_name}",
                    start_line=enum_start_line + i,
                    end_line=enum_start_line + i,
                    raw_text=raw_entry.strip()[:200],
                    extraction_confidence=0.8,
                    lifecycle_phase="dispatch",
                )
                self.write_record(record, str(self.output_path / OUTPUT_FILE))
                count += 1

        return count

    # ------------------------------------------------------------------
    # Pass D — Override protocols
    # ------------------------------------------------------------------

    def _pass_d_override_protocols(self) -> int:
        self.logger.info("Pass D — Override protocols: starting")
        count = 0
        files = self._collect_python_files(["torch"])

        for fpath in files:
            try:
                count += self._walk_override_protocols(fpath)
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error(
                    "Unexpected error in Pass D for %s: %s", fpath, exc
                )
                self._errors += 1

        self.logger.info("Pass D — Override protocols: %d records", count)
        return count

    def _walk_override_protocols(self, fpath: Path) -> int:
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        tree = ast.parse(source, filename=str(fpath))
        self._files_processed += 1
        count = 0
        rel = str(fpath.relative_to(self.repo_path))

        for node in ast.walk(tree):
            name: Optional[str] = None
            etype = "method"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in OVERRIDE_PROTOCOLS:
                    name = node.name
            elif isinstance(node, ast.ClassDef):
                if node.name in OVERRIDE_PROTOCOLS:
                    name = node.name
                    etype = "class"

            if name is not None:
                module_path = self.file_to_module_path(fpath)
                class_name = ""
                if etype == "method":
                    class_name = self._find_enclosing_class(tree, node)
                record = self.make_record(
                    source_file=rel,
                    language="python",
                    entity_name=name,
                    entity_type=etype,
                    subcategory="override_protocol",
                    module_path=module_path,
                    qualified_name=self.compute_qualified_name(
                        module_path, class_name, name
                    ),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature=(
                        self.extract_function_signature(node)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else ""
                    ),
                    docstring=ast.get_docstring(node) or "",
                    extraction_confidence=1.0,
                    lifecycle_phase="dispatch",
                )
                self.write_record(record, str(self.output_path / OUTPUT_FILE))
                count += 1

        return count

    # ------------------------------------------------------------------
    # Pass E — Profiler hooks
    # ------------------------------------------------------------------

    def _pass_e_profiler_hooks(self) -> int:
        self.logger.info("Pass E — Profiler hooks: starting")
        count = 0
        files = self._collect_python_files(["torch/profiler"])

        for fpath in files:
            try:
                count += self._walk_profiler_hooks(fpath)
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", fpath, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error(
                    "Unexpected error in Pass E for %s: %s", fpath, exc
                )
                self._errors += 1

        self.logger.info("Pass E — Profiler hooks: %d records", count)
        return count

    def _walk_profiler_hooks(self, fpath: Path) -> int:
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        tree = ast.parse(source, filename=str(fpath))
        self._files_processed += 1
        count = 0
        rel = str(fpath.relative_to(self.repo_path))

        for node in ast.walk(tree):
            name: Optional[str] = None
            etype = "function"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in PROFILER_TARGETS:
                    name = node.name
            elif isinstance(node, ast.ClassDef):
                if node.name in PROFILER_TARGETS:
                    name = node.name
                    etype = "class"

            if name is not None:
                module_path = self.file_to_module_path(fpath)
                record = self.make_record(
                    source_file=rel,
                    language="python",
                    entity_name=name,
                    entity_type=etype,
                    subcategory="profiler_hook",
                    module_path=module_path,
                    qualified_name=f"{module_path}.{name}",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature=(
                        self.extract_function_signature(node)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else ""
                    ),
                    docstring=ast.get_docstring(node) or "",
                    extraction_confidence=1.0,
                    lifecycle_phase="profiling",
                )
                self.write_record(record, str(self.output_path / OUTPUT_FILE))
                count += 1

        return count

    # ------------------------------------------------------------------
    # Pass F — Module callbacks
    # ------------------------------------------------------------------

    def _pass_f_module_callbacks(self) -> int:
        self.logger.info("Pass F — Module callbacks: starting")
        count = 0
        target = self.repo_path / "torch" / "nn" / "modules" / "module.py"
        if not target.exists():
            self.logger.warning(
                "Pass F — Module file not found: %s", target
            )
            self._warnings += 1
            return 0

        try:
            source = self.read_file_safe(target)
            if source is None:
                return 0
            tree = ast.parse(source, filename=str(target))
            self._files_processed += 1
            rel = str(target.relative_to(self.repo_path))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in MODULE_CALLBACKS:
                        class_name = self._find_enclosing_class(tree, node)
                        module_path = self.file_to_module_path(target)
                        record = self.make_record(
                            source_file=rel,
                            language="python",
                            entity_name=node.name,
                            entity_type="method",
                            subcategory="module_callback",
                            module_path=module_path,
                            qualified_name=self.compute_qualified_name(
                                module_path, class_name, node.name
                            ),
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            signature=self.extract_function_signature(node),
                            docstring=ast.get_docstring(node) or "",
                            extraction_confidence=1.0,
                            lifecycle_phase=_lifecycle_phase_for_hook(node.name),
                        )
                        self.write_record(
                            record, str(self.output_path / OUTPUT_FILE)
                        )
                        count += 1
        except SyntaxError as exc:
            self.logger.warning("Syntax error in %s: %s", target, exc)
            self._warnings += 1
        except Exception as exc:
            self.logger.error(
                "Unexpected error in Pass F for %s: %s", target, exc
            )
            self._errors += 1

        self.logger.info("Pass F — Module callbacks: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_python_files(self, subdirs: list[str]) -> list[Path]:
        """Collect *.py files from a list of repo-relative subdirectories."""
        files: list[Path] = []
        seen: set[Path] = set()
        for subdir in subdirs:
            root = self.repo_path / subdir
            if not root.exists():
                self.logger.warning("Directory not found: %s", root)
                self._warnings += 1
                continue
            for f in sorted(root.rglob("*.py")):
                if f not in seen:
                    seen.add(f)
                    files.append(f)
        return files

    @staticmethod
    def _find_enclosing_class(tree: ast.Module, target_node: ast.AST) -> str:
        """Return the name of the class that directly contains *target_node*, or ''."""
        for cls_node in ast.walk(tree):
            if isinstance(cls_node, ast.ClassDef):
                for child in ast.walk(cls_node):
                    if child is target_node:
                        return cls_node.name
        return ""
