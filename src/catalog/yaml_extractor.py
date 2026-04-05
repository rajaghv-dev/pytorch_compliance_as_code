"""
YAML catalog extractor for the PyTorch Compliance Toolkit.

Parses ``aten/src/ATen/native/native_functions.yaml`` to extract ATen operator
definitions (~2000+ entries).  Each YAML entry describes an operator's signature,
dispatch table, and structural metadata.

Also scans ``torchgen/`` for additional YAML configuration files.

Output: ``storage/raw/catalog_yaml.jsonl``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ..extractors.base import BaseExtractor, EntityRecord, compute_stable_id

logger = logging.getLogger("pct.catalog.yaml")

_OUTPUT_FILE = "catalog_yaml.jsonl"

# The primary YAML file that defines ATen native operators.
_NATIVE_FUNCTIONS_YAML = "aten/src/ATen/native/native_functions.yaml"

# Regex to extract the operator name from the ``func`` field.
# The func field looks like: "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
# We want the part before the first ``(``.
_OP_NAME_RE = re.compile(r'^([^(]+)')


class YamlCatalogExtractor(BaseExtractor):
    """
    Catalog extractor for YAML-defined ATen operators and torchgen configs.

    Parses ``native_functions.yaml`` and any YAML files under ``torchgen/``.
    Each operator entry becomes an EntityRecord with the full YAML dict
    stored in ``metadata``.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__(name="catalog_yaml", repo_path=repo_path, output_path=output_path)
        self._output_file = str(self.output_path / _OUTPUT_FILE)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Parse YAML files and emit EntityRecords for each operator.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting YAML catalog extraction")

        # --- native_functions.yaml ---
        nf_path = self.repo_path / _NATIVE_FUNCTIONS_YAML
        if nf_path.is_file():
            self._parse_native_functions(nf_path)
        else:
            self.logger.warning(
                "native_functions.yaml not found at %s", nf_path
            )
            self._warnings += 1

        # --- torchgen/ YAML files ---
        torchgen_dir = self.repo_path / "torchgen"
        if torchgen_dir.is_dir():
            yaml_files = self.find_files("*.yaml", root=torchgen_dir)
            for yf in yaml_files:
                try:
                    self._parse_generic_yaml(yf)
                except Exception as exc:
                    self.logger.error(
                        "Unexpected error processing %s: %s", yf, exc
                    )
                    self._errors += 1
        else:
            self.logger.warning("torchgen/ directory not found")
            self._warnings += 1

        self.flush(self._output_file)
        self.logger.info(
            "Completed %d files, %d records",
            self._files_processed,
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # native_functions.yaml
    # ------------------------------------------------------------------ #

    def _parse_native_functions(self, filepath: Path) -> None:
        """
        Parse the main ``native_functions.yaml`` file.

        Each top-level list entry is expected to be a dict with at least a
        ``func`` key defining the operator signature.

        Parameters
        ----------
        filepath : Path
            Absolute path to ``native_functions.yaml``.
        """
        self.logger.info("Parsing native_functions.yaml at %s", filepath)

        # Use the cached file reader.
        content = self.read_file_safe(filepath)
        if content is None:
            return

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            self.logger.error("YAML parse error for %s: %s", filepath, exc)
            self._errors += 1
            return

        if not isinstance(data, list):
            self.logger.warning(
                "Expected a list in native_functions.yaml, got %s",
                type(data).__name__,
            )
            self._warnings += 1
            return

        try:
            rel_path = str(filepath.relative_to(self.repo_path))
        except ValueError:
            rel_path = str(filepath)

        records: list[EntityRecord] = []

        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue

            func_sig = entry.get("func", "")
            if not func_sig:
                continue

            # Extract operator name from the func signature.
            # e.g. "add.Tensor(Tensor self, ...) -> Tensor" => "add.Tensor"
            op_name_match = _OP_NAME_RE.match(func_sig)
            op_name = op_name_match.group(1).strip() if op_name_match else f"unknown_{idx}"

            # Extract specific fields of interest.
            dispatch = entry.get("dispatch", {})
            structured = entry.get("structured", False)
            structured_delegate = entry.get("structured_delegate", "")
            device_check = entry.get("device_check", "")
            device_guard = entry.get("device_guard", True)

            record = self.make_record(
                source_file=rel_path,
                language="yaml",
                entity_name=op_name,
                entity_type="operator",
                subcategory="aten_operator",
                module_path="aten.native",
                qualified_name=f"aten::{op_name}",
                start_line=0,  # YAML doesn't give us line numbers easily.
                end_line=0,
                raw_text=str(entry)[:5000],
                signature=func_sig,
                extraction_confidence=0.95,
                metadata={
                    "func": func_sig,
                    "dispatch": dispatch if isinstance(dispatch, dict) else {},
                    "structured": structured,
                    "structured_delegate": structured_delegate,
                    "device_check": device_check,
                    "device_guard": device_guard,
                    # Store the full entry for downstream consumers.
                    "full_yaml_entry": entry,
                },
            )
            records.append(record)

        if records:
            self.write_records(records, self._output_file)

        self._files_processed += 1
        self.logger.info(
            "Extracted %d operators from native_functions.yaml", len(records)
        )

    # ------------------------------------------------------------------ #
    # Generic YAML files under torchgen/
    # ------------------------------------------------------------------ #

    def _parse_generic_yaml(self, filepath: Path) -> None:
        """
        Parse a generic YAML file from the ``torchgen/`` directory.

        If the file contains a list of dicts with ``func`` keys, treat them
        as operator definitions (same as native_functions.yaml).  Otherwise,
        store top-level keys as config entries.

        Parameters
        ----------
        filepath : Path
            Absolute path to the YAML file.
        """
        content = self.read_file_safe(filepath)
        if content is None:
            return

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            self.logger.warning("YAML parse error for %s: %s", filepath, exc)
            self._warnings += 1
            return

        try:
            rel_path = str(filepath.relative_to(self.repo_path))
        except ValueError:
            rel_path = str(filepath)

        records: list[EntityRecord] = []

        if isinstance(data, list):
            # Could be another operator definition list.
            for idx, entry in enumerate(data):
                if not isinstance(entry, dict):
                    continue
                func_sig = entry.get("func", "")
                if func_sig:
                    op_name_match = _OP_NAME_RE.match(func_sig)
                    op_name = op_name_match.group(1).strip() if op_name_match else f"unknown_{idx}"

                    record = self.make_record(
                        source_file=rel_path,
                        language="yaml",
                        entity_name=op_name,
                        entity_type="operator",
                        subcategory="aten_operator",
                        module_path=self.file_to_module_path(filepath),
                        qualified_name=f"aten::{op_name}",
                        start_line=0,
                        end_line=0,
                        raw_text=str(entry)[:5000],
                        signature=func_sig,
                        extraction_confidence=0.95,
                        metadata={"func": func_sig, "full_yaml_entry": entry},
                    )
                    records.append(record)
                else:
                    # Non-operator list entry; store as a generic config entry.
                    name = entry.get("name", f"entry_{idx}")
                    record = self.make_record(
                        source_file=rel_path,
                        language="yaml",
                        entity_name=str(name),
                        entity_type="config_entry",
                        subcategory="yaml_config",
                        module_path=self.file_to_module_path(filepath),
                        qualified_name=f"{self.file_to_module_path(filepath)}.{name}",
                        start_line=0,
                        end_line=0,
                        raw_text=str(entry)[:5000],
                        extraction_confidence=0.95,
                        metadata={"full_yaml_entry": entry},
                    )
                    records.append(record)

        elif isinstance(data, dict):
            # Store each top-level key as a config entity.
            for key, value in data.items():
                record = self.make_record(
                    source_file=rel_path,
                    language="yaml",
                    entity_name=str(key),
                    entity_type="config_entry",
                    subcategory="yaml_config",
                    module_path=self.file_to_module_path(filepath),
                    qualified_name=f"{self.file_to_module_path(filepath)}.{key}",
                    start_line=0,
                    end_line=0,
                    raw_text=str(value)[:5000],
                    extraction_confidence=0.95,
                    metadata={"key": key, "value_type": type(value).__name__},
                )
                records.append(record)

        if records:
            self.write_records(records, self._output_file)

        self._files_processed += 1
        self.logger.info("Extracted %d records from %s", len(records), rel_path)
