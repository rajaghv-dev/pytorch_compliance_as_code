"""
Module Hierarchy Extractor — extracts nn.Module class hierarchy, method
signatures, inheritance relations, and functional API from PyTorch.

Compliance relevance:
    - All module classes map to EU AI Act Article 15 (accuracy, robustness, cybersecurity).
    - Training modules get lifecycle_phase "training_only"; inference-safe modules
      get "inference_safe".
    - Inheritance relations are tracked as structured ``relations`` entries.

The extractor walks ``torch/nn/modules/*.py``, ``torch/nn/parallel/*.py``,
``torch/nn/utils/*.py``, and ``torch/nn/functional.py``.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from .base import EntityRecord, BaseExtractor, compute_stable_id

logger = logging.getLogger("pct.extractors.module_hierarchy")

# Modules that are inference-safe (commonly used at inference time without
# modification).  Anything not in this set defaults to "training_only".
_INFERENCE_SAFE_MODULES: set[str] = {
    "Module", "Sequential", "ModuleList", "ModuleDict", "ParameterList",
    "ParameterDict", "Linear", "Conv1d", "Conv2d", "Conv3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
    "ReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax",
    "Dropout",  # commonly set to eval mode
    "Embedding", "EmbeddingBag",
    "Flatten", "Unflatten", "Identity",
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "TransformerEncoder", "TransformerDecoder", "Transformer",
    "MultiheadAttention",
}


class ModuleHierarchyExtractor(BaseExtractor):
    """
    Extract nn.Module class hierarchy, parameters, and forward logic.

    Walks module directories, parses ASTs, and produces EntityRecords for
    each class definition found, including inheritance relations, method
    metadata, and init-time parameter registrations.  Also extracts the
    public functional API from ``torch.nn.functional``.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the module hierarchy extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="module_hierarchy", repo_path=repo_path, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Run the module hierarchy extraction pass.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting module hierarchy extraction")
        output_file = str(self.output_path / "module_hierarchy.jsonl")

        # --- nn.Module classes from multiple directories ---
        module_patterns = [
            "torch/nn/modules/*.py",
            "torch/nn/parallel/*.py",
            "torch/nn/utils/*.py",
        ]
        for pattern in module_patterns:
            self._extract_module_classes(pattern, output_file)

        # --- Functional API (public functions from torch.nn.functional) ---
        self._extract_functional_api(output_file)

        # Flush remaining buffered records
        self.flush(output_file)

        self.logger.info(
            "Module hierarchy extraction complete — %d records produced",
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # Module class extraction
    # ------------------------------------------------------------------ #

    def _extract_module_classes(self, pattern: str, output_file: str) -> None:
        """
        Parse files matching *pattern* and emit records for every class definition.

        For each class we capture:
        - Base class names (using ``ast.unparse`` for correct rendering)
        - All methods with their arguments, decorators, docstring presence
        - Whether ``super().__init__()`` is called
        - Whether ``forward`` is abstract (contains ``raise NotImplementedError``)
        - Parameter/buffer registrations in ``__init__``
        - Inheritance relations as structured ``relations`` entries

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
                if not isinstance(node, ast.ClassDef):
                    continue

                # --- Extract base class names using ast.unparse ---
                bases: list[str] = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:
                        # Fallback for very unusual base expressions
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(getattr(base, "attr", "<unknown>"))
                        else:
                            bases.append("<unknown>")

                # --- Build structured relations for each inherited base ---
                relations: list[dict[str, str]] = [
                    {"type": "inherits", "target": base_name, "source": "ast"}
                    for base_name in bases
                ]

                # --- Extract methods and their metadata ---
                methods: dict[str, dict[str, Any]] = {}
                has_super_init = False
                forward_is_abstract = False

                for item in node.body:
                    if not isinstance(item, ast.FunctionDef):
                        continue

                    # Collect decorator strings
                    decorators = self.extract_decorators(item)

                    # Extract docstring presence
                    method_docstring = ast.get_docstring(item) or ""

                    methods[item.name] = {
                        "args": [a.arg for a in item.args.args],
                        "decorators": decorators,
                        "has_docstring": bool(method_docstring),
                        "line": item.lineno,
                    }

                    # Check if __init__ calls super().__init__()
                    if item.name == "__init__":
                        has_super_init = self._has_super_init_call(item)

                    # Check if forward is abstract (raises NotImplementedError)
                    if item.name == "forward":
                        forward_is_abstract = self._is_abstract_forward(item)

                # --- Extract __init__ parameter registrations ---
                init_registrations: list[str] = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                                if stmt.func.attr in {"register_buffer", "register_parameter"}:
                                    init_registrations.append(stmt.func.attr)

                # --- Determine lifecycle phase ---
                lifecycle_phase = (
                    "inference_safe" if node.name in _INFERENCE_SAFE_MODULES
                    else "training_only"
                )

                # --- Line range ---
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", node.lineno)

                # Qualified name for this class
                qualified_name = self.compute_qualified_name(module_path, "", node.name)

                # Docstring
                docstring = ast.get_docstring(node) or ""

                # Raw text — cap class preview at 60 lines for large classes
                preview_end = min(start_line + 60, end_line)
                raw_text = self.get_raw_text(filepath, start_line, preview_end)

                # Build metadata
                metadata: dict[str, Any] = {
                    "bases": bases,
                    "methods": methods,
                    "has_forward": "forward" in methods,
                    "has_extra_repr": "extra_repr" in methods,
                    "has_super_init": has_super_init,
                    "forward_is_abstract": forward_is_abstract,
                    "init_registrations": init_registrations,
                    "method_count": len(methods),
                }

                # All module classes get Article 15 compliance tag
                record = self.make_record(
                    source_file=rel_path,
                    language="python",
                    entity_name=node.name,
                    entity_type="module_class",
                    subcategory="nn_module",
                    module_path=module_path,
                    qualified_name=qualified_name,
                    start_line=start_line,
                    end_line=end_line,
                    raw_text=raw_text,
                    docstring=docstring,
                    relations=relations,
                    compliance_tags=["eu_ai_act_art_15"],
                    lifecycle_phase=lifecycle_phase,
                    extraction_confidence=1.0,
                    metadata=metadata,
                )

                self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # Functional API extraction
    # ------------------------------------------------------------------ #

    def _extract_functional_api(self, output_file: str) -> None:
        """
        Extract all public functions from ``torch.nn.functional``.

        Only functions whose names do not start with ``_`` are included.
        """
        for filepath in self.find_files("torch/nn/functional.py"):
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
                if not isinstance(node, ast.FunctionDef):
                    continue
                # Skip private functions
                if node.name.startswith("_"):
                    continue

                start_line = node.lineno
                end_line = getattr(node, "end_lineno", node.lineno)
                docstring = ast.get_docstring(node) or ""
                signature = self.extract_function_signature(node)
                decorators = self.extract_decorators(node)
                qualified_name = self.compute_qualified_name(module_path, "", node.name)

                metadata: dict[str, Any] = {
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": decorators,
                }

                record = self.make_record(
                    source_file=rel_path,
                    language="python",
                    entity_name=node.name,
                    entity_type="function",
                    subcategory="functional_api",
                    module_path=module_path,
                    qualified_name=qualified_name,
                    start_line=start_line,
                    end_line=end_line,
                    raw_text=self.get_raw_text(filepath, start_line, end_line),
                    docstring=docstring,
                    signature=signature,
                    compliance_tags=["eu_ai_act_art_15"],
                    lifecycle_phase="inference_safe",
                    extraction_confidence=1.0,
                    metadata=metadata,
                )

                self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # AST inspection helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _has_super_init_call(init_node: ast.FunctionDef) -> bool:
        """
        Return True if *init_node* contains a ``super().__init__(...)`` call.

        Walks every statement inside the ``__init__`` body looking for a Call
        node whose func is ``super().__init__``.
        """
        for stmt in ast.walk(init_node):
            if not isinstance(stmt, ast.Call):
                continue
            func = stmt.func
            # Pattern: super().__init__()  →  Call(func=Attribute(value=Call(func=Name(id='super')), attr='__init__'))
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "__init__"
                and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"
            ):
                return True
        return False

    @staticmethod
    def _is_abstract_forward(forward_node: ast.FunctionDef) -> bool:
        """
        Return True if *forward_node* contains ``raise NotImplementedError``.

        This is the standard pattern for abstract forward methods in
        ``torch.nn.Module``.
        """
        for stmt in ast.walk(forward_node):
            if isinstance(stmt, ast.Raise) and stmt.exc is not None:
                # Check for ``raise NotImplementedError`` or ``raise NotImplementedError(...)``
                exc = stmt.exc
                if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                    return True
                if (
                    isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id == "NotImplementedError"
                ):
                    return True
        return False
