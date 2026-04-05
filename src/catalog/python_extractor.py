"""
Python catalog extractor for the PyTorch Compliance Toolkit.

Walks all ``.py`` files under ``torch/``, ``functorch/``, ``torchgen/``, and
``test/`` in the PyTorch repository.  Uses ``ast.parse()`` to extract:

* **Classes** (including private) -- name, bases, methods, decorators, docstring,
  line range.
* **Functions / methods** (including private) -- name, args with type annotations,
  return annotation, decorators, docstring, line range.
* **Module-level assignments** -- name, simple literal value, line.
* **Imports** -- ``import X`` and ``from X import Y``.

Every entity is written as an ``EntityRecord`` to
``storage/raw/catalog_python.jsonl``.

Performance note: file reads go through the LRU-cached ``read_file_safe()``
method inherited from ``BaseExtractor``; we never call ``open()`` directly.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Optional

from ..extractors.base import BaseExtractor, EntityRecord, compute_stable_id

logger = logging.getLogger("pct.catalog.python")

# Directories under the PyTorch repo root to scan for Python files.
_SCAN_DIRS = ["torch", "functorch", "torchgen", "test"]

# Output file name (resolved against output_path by BaseExtractor).
_OUTPUT_FILE = "catalog_python.jsonl"


class PythonCatalogExtractor(BaseExtractor):
    """
    Catalog extractor that parses Python source files with the ``ast`` module.

    Produces EntityRecords for classes, functions/methods, module-level
    assignments, and import statements found in the PyTorch codebase.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the Python catalog extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="catalog_python", repo_path=repo_path, output_path=output_path)
        # Resolve the output file path relative to the output directory.
        self._output_file = str(self.output_path / _OUTPUT_FILE)

    # ------------------------------------------------------------------ #
    # Main extraction entry point
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Walk Python files, parse ASTs, and emit EntityRecords.

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

            self.logger.info("Starting Python catalog extraction for %s", target)
            py_files = self.find_files("*.py", root=target)

            for filepath in py_files:
                try:
                    records = self._process_file(filepath)
                    if records:
                        all_records.extend(records)
                        # Write in batches to the output file.
                        self.write_records(records, self._output_file)
                    self._files_processed += 1
                except Exception as exc:
                    self.logger.error(
                        "Unexpected error processing %s: %s", filepath, exc
                    )
                    self._errors += 1

        # Final flush to make sure all buffered records are written.
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
        Parse a single Python file and extract all entities.

        Parameters
        ----------
        filepath : Path
            Absolute path to the ``.py`` file.

        Returns
        -------
        list[EntityRecord]
            Records extracted from the file.
        """
        # Use the cached reader -- never open files directly (BUG-03).
        source = self.read_file_safe(filepath)
        if source is None:
            return []

        # Attempt AST parsing. Syntax errors are expected in some PyTorch
        # files (e.g. template files, stub files with deliberate errors).
        try:
            tree = ast.parse(source, filename=str(filepath))
        except SyntaxError as exc:
            self.logger.warning("AST parse failed for %s: %s", filepath, exc)
            self._warnings += 1
            return []

        # Compute the dotted module path from the file path.
        module_path = self.file_to_module_path(filepath)
        rel_path = str(filepath.relative_to(self.repo_path))

        records: list[EntityRecord] = []

        # Walk top-level AST nodes.  Classes are handled recursively to
        # capture methods nested inside them.
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                records.extend(
                    self._extract_class(node, filepath, rel_path, module_path)
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                records.append(
                    self._extract_function(
                        node, filepath, rel_path, module_path, class_name=""
                    )
                )
            elif isinstance(node, ast.Assign):
                rec = self._extract_assignment(
                    node, filepath, rel_path, module_path
                )
                if rec is not None:
                    records.append(rec)
            elif isinstance(node, ast.AnnAssign):
                rec = self._extract_annotated_assignment(
                    node, filepath, rel_path, module_path
                )
                if rec is not None:
                    records.append(rec)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                records.extend(
                    self._extract_import(node, filepath, rel_path, module_path)
                )

        return records

    # ------------------------------------------------------------------ #
    # Class extraction
    # ------------------------------------------------------------------ #

    def _extract_class(
        self,
        node: ast.ClassDef,
        filepath: Path,
        rel_path: str,
        module_path: str,
    ) -> list[EntityRecord]:
        """
        Extract a class definition and all its methods.

        Includes private classes (names starting with ``_``).

        Parameters
        ----------
        node : ast.ClassDef
            The AST class node.
        filepath : Path
            Source file path (absolute).
        rel_path : str
            Source file path relative to the repo root.
        module_path : str
            Dotted module path.

        Returns
        -------
        list[EntityRecord]
            One record for the class plus one per method.
        """
        records: list[EntityRecord] = []

        # Base class names as strings.
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append("<unknown>")

        # Method names (for the class-level metadata).
        method_names = [
            child.name
            for child in ast.iter_child_nodes(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        decorators = self.extract_decorators(node)
        docstring = ast.get_docstring(node) or ""

        # Determine line range. ast gives 1-based line numbers.
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

        qualified_name = self.compute_qualified_name(module_path, "", node.name)

        class_record = self.make_record(
            source_file=rel_path,
            language="python",
            entity_name=node.name,
            entity_type="class",
            subcategory="class_definition",
            module_path=module_path,
            qualified_name=qualified_name,
            start_line=start_line,
            end_line=end_line,
            raw_text=self.get_raw_text(filepath, start_line, end_line),
            docstring=docstring,
            signature=f"class {node.name}({', '.join(bases)}):",
            extraction_confidence=1.0,
            metadata={
                "bases": bases,
                "methods": method_names,
                "decorators": decorators,
            },
        )
        records.append(class_record)

        # Extract each method inside the class.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                records.append(
                    self._extract_function(
                        child, filepath, rel_path, module_path, class_name=node.name
                    )
                )
            # Handle nested classes recursively.
            elif isinstance(child, ast.ClassDef):
                records.extend(
                    self._extract_class(child, filepath, rel_path, module_path)
                )

        return records

    # ------------------------------------------------------------------ #
    # Function / method extraction
    # ------------------------------------------------------------------ #

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        filepath: Path,
        rel_path: str,
        module_path: str,
        class_name: str,
    ) -> EntityRecord:
        """
        Extract a function or method definition.

        Includes private functions/methods (names starting with ``_``).

        Parameters
        ----------
        node : ast.FunctionDef | ast.AsyncFunctionDef
            The AST function node.
        filepath : Path
            Absolute source file path.
        rel_path : str
            Repo-relative source file path.
        module_path : str
            Dotted module path.
        class_name : str
            Enclosing class name, or ``""`` for module-level functions.

        Returns
        -------
        EntityRecord
        """
        # Determine entity_type and subcategory based on whether this is a
        # method (inside a class) or a standalone function.
        if class_name:
            entity_type = "method"
            subcategory = "method_definition"
        else:
            entity_type = "function"
            subcategory = "function_definition"

        decorators = self.extract_decorators(node)
        docstring = ast.get_docstring(node) or ""
        signature = self.extract_function_signature(node)

        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

        qualified_name = self.compute_qualified_name(
            module_path, class_name, node.name
        )

        # Extract argument details: name + annotation pairs.
        args_info = self._extract_args_info(node.args)

        # Return annotation as a string.
        return_annotation = ""
        if node.returns is not None:
            try:
                return_annotation = ast.unparse(node.returns)
            except Exception:
                return_annotation = "<unknown>"

        is_async = isinstance(node, ast.AsyncFunctionDef)

        return self.make_record(
            source_file=rel_path,
            language="python",
            entity_name=node.name,
            entity_type=entity_type,
            subcategory=subcategory,
            module_path=module_path,
            qualified_name=qualified_name,
            start_line=start_line,
            end_line=end_line,
            raw_text=self.get_raw_text(filepath, start_line, end_line),
            docstring=docstring,
            signature=signature,
            extraction_confidence=1.0,
            metadata={
                "decorators": decorators,
                "args": args_info,
                "return_annotation": return_annotation,
                "is_async": is_async,
            },
        )

    def _extract_args_info(self, args_node: ast.arguments) -> list[dict]:
        """
        Extract argument names and type annotations from an ``ast.arguments`` node.

        Parameters
        ----------
        args_node : ast.arguments
            The arguments node of a function definition.

        Returns
        -------
        list[dict]
            Each dict has keys ``name`` and ``annotation``.
        """
        result: list[dict] = []
        for arg in args_node.args + args_node.posonlyargs + args_node.kwonlyargs:
            annotation = ""
            if arg.annotation is not None:
                try:
                    annotation = ast.unparse(arg.annotation)
                except Exception:
                    annotation = "<unknown>"
            result.append({"name": arg.arg, "annotation": annotation})

        # *args
        if args_node.vararg:
            annotation = ""
            if args_node.vararg.annotation:
                try:
                    annotation = ast.unparse(args_node.vararg.annotation)
                except Exception:
                    annotation = "<unknown>"
            result.append({"name": f"*{args_node.vararg.arg}", "annotation": annotation})

        # **kwargs
        if args_node.kwarg:
            annotation = ""
            if args_node.kwarg.annotation:
                try:
                    annotation = ast.unparse(args_node.kwarg.annotation)
                except Exception:
                    annotation = "<unknown>"
            result.append({"name": f"**{args_node.kwarg.arg}", "annotation": annotation})

        return result

    # ------------------------------------------------------------------ #
    # Module-level assignment extraction
    # ------------------------------------------------------------------ #

    def _extract_assignment(
        self,
        node: ast.Assign,
        filepath: Path,
        rel_path: str,
        module_path: str,
    ) -> Optional[EntityRecord]:
        """
        Extract a module-level assignment if the target is a simple name
        and the value is a simple literal (string, int, bool, or list of strings).

        Parameters
        ----------
        node : ast.Assign
            The AST assignment node.
        filepath : Path
            Absolute source file path.
        rel_path : str
            Repo-relative source file path.
        module_path : str
            Dotted module path.

        Returns
        -------
        EntityRecord | None
            A record if the assignment qualifies, otherwise ``None``.
        """
        # Only handle simple single-target assignments: ``NAME = value``.
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None

        name = target.id
        value = self._extract_simple_value(node.value)
        if value is None:
            # Value is not a simple literal; skip.
            return None

        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)
        qualified_name = self.compute_qualified_name(module_path, "", name)

        return self.make_record(
            source_file=rel_path,
            language="python",
            entity_name=name,
            entity_type="assignment",
            subcategory="module_assignment",
            module_path=module_path,
            qualified_name=qualified_name,
            start_line=start_line,
            end_line=end_line,
            raw_text=self.get_raw_text(filepath, start_line, end_line),
            extraction_confidence=1.0,
            metadata={"value": value},
        )

    def _extract_annotated_assignment(
        self,
        node: ast.AnnAssign,
        filepath: Path,
        rel_path: str,
        module_path: str,
    ) -> Optional[EntityRecord]:
        """
        Extract an annotated assignment (e.g. ``x: int = 5``) at module level.

        Parameters
        ----------
        node : ast.AnnAssign
            The AST annotated-assignment node.
        filepath, rel_path, module_path : str
            As in other extraction methods.

        Returns
        -------
        EntityRecord | None
        """
        if node.target is None or not isinstance(node.target, ast.Name):
            return None
        if node.value is None:
            return None

        name = node.target.id
        value = self._extract_simple_value(node.value)
        if value is None:
            return None

        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)
        qualified_name = self.compute_qualified_name(module_path, "", name)

        annotation_str = ""
        try:
            annotation_str = ast.unparse(node.annotation)
        except Exception:
            pass

        return self.make_record(
            source_file=rel_path,
            language="python",
            entity_name=name,
            entity_type="assignment",
            subcategory="module_assignment",
            module_path=module_path,
            qualified_name=qualified_name,
            start_line=start_line,
            end_line=end_line,
            raw_text=self.get_raw_text(filepath, start_line, end_line),
            extraction_confidence=1.0,
            metadata={"value": value, "annotation": annotation_str},
        )

    def _extract_simple_value(self, node: ast.expr) -> Any:
        """
        Return the Python value of an AST node if it is a simple literal.

        Supported: str, int, float, bool, None, list of str.
        Returns ``None`` (the Python sentinel) if the value is too complex.

        Parameters
        ----------
        node : ast.expr
            The value node from an assignment.

        Returns
        -------
        Any
            The literal value, or ``None`` if not extractable.
        """
        # ast.Constant covers str, int, float, bool, None in Python 3.8+.
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool)) or node.value is None:
                return node.value

        # List of strings: [\"a\", \"b\", ...]
        if isinstance(node, ast.List):
            values = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    values.append(elt.value)
                else:
                    return None  # Mixed or non-string list; skip.
            return values

        return None

    # ------------------------------------------------------------------ #
    # Import extraction
    # ------------------------------------------------------------------ #

    def _extract_import(
        self,
        node: ast.Import | ast.ImportFrom,
        filepath: Path,
        rel_path: str,
        module_path: str,
    ) -> list[EntityRecord]:
        """
        Extract import statements.  These feed the cross-reference graph.

        ``import X`` produces one record per alias.
        ``from X import Y, Z`` produces one record per imported name.

        Parameters
        ----------
        node : ast.Import | ast.ImportFrom
            The AST import node.
        filepath : Path
            Absolute source file path.
        rel_path : str
            Repo-relative source file path.
        module_path : str
            Dotted module path.

        Returns
        -------
        list[EntityRecord]
        """
        records: list[EntityRecord] = []
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

        if isinstance(node, ast.Import):
            # ``import X`` or ``import X as Y``
            for alias in node.names:
                imported_name = alias.name
                local_name = alias.asname or alias.name
                records.append(
                    self.make_record(
                        source_file=rel_path,
                        language="python",
                        entity_name=local_name,
                        entity_type="import",
                        subcategory="import_statement",
                        module_path=module_path,
                        qualified_name=f"{module_path}.import.{imported_name}",
                        start_line=start_line,
                        end_line=end_line,
                        raw_text=self.get_raw_text(filepath, start_line, end_line),
                        extraction_confidence=1.0,
                        metadata={
                            "imported_module": imported_name,
                            "alias": alias.asname,
                            "style": "import",
                        },
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            # ``from X import Y`` or ``from X import *``
            from_module = node.module or ""
            level = node.level  # Number of leading dots (relative imports).
            for alias in (node.names or []):
                imported_name = alias.name
                local_name = alias.asname or alias.name
                records.append(
                    self.make_record(
                        source_file=rel_path,
                        language="python",
                        entity_name=local_name,
                        entity_type="import",
                        subcategory="import_statement",
                        module_path=module_path,
                        qualified_name=f"{module_path}.import.{from_module}.{imported_name}",
                        start_line=start_line,
                        end_line=end_line,
                        raw_text=self.get_raw_text(filepath, start_line, end_line),
                        extraction_confidence=1.0,
                        metadata={
                            "from_module": from_module,
                            "imported_name": imported_name,
                            "alias": alias.asname,
                            "level": level,
                            "style": "from_import",
                        },
                    )
                )

        return records
