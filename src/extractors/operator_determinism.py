"""
Operator determinism extractor — discovers determinism-related APIs, cuDNN
config knobs, RNG management functions, numerical stability patterns, and
determinism tests in the PyTorch repository.

Passes
------
1. Native functions YAML parse (device_check / device_guard flags)
2. Determinism API extraction (use_deterministic_algorithms, etc.)
3. cuDNN config (deterministic, benchmark, allow_tf32)
4. RNG management (manual_seed, set_rng_state, get_rng_state, fork_rng)
5. Numerical stability patterns (nan, overflow, eps, etc.) with false-positive filter
6. Determinism tests (test/test_deterministic.py)
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from .base import BaseExtractor, EntityRecord, compute_stable_id

OUTPUT_FILE = "operator_determinism.jsonl"

# ---------------------------------------------------------------------------
# Pass 5 — patterns & false-positive filter
# ---------------------------------------------------------------------------

_NUMERICAL_STABILITY_RE = re.compile(
    r"\b(nan|overflow|underflow|eps|epsilon|logsumexp|log_sum_exp)\b",
    re.IGNORECASE,
)

_FALSE_POSITIVE_PATTERNS = [
    re.compile(r"^\s*#"),                              # comment line
    re.compile(r"\bself\.assertEqual\s*\("),           # inside assertEqual
    re.compile(r"\blogging\.\w+\s*\("),                # inside logging calls
    re.compile(r"\bprint\s*\("),                       # inside print calls
    re.compile(r'==\s*["\']nan["\']', re.IGNORECASE),  # string comparison == "nan"
    re.compile(r'["\']nan["\']\s*==', re.IGNORECASE),  # "nan" == ...
]


def _is_false_positive(line: str, match_pos: int) -> bool:
    """
    Return True if the match at *match_pos* in *line* is likely a false
    positive (test assertion, log statement, comment, or string comparison).
    """
    for pat in _FALSE_POSITIVE_PATTERNS:
        if pat.search(line):
            return True
    return False


class OperatorDeterminismExtractor(BaseExtractor):
    """Extract determinism-related entities from the PyTorch repository."""

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__(
            name="operator_determinism",
            repo_path=repo_path,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> int:
        total = 0
        total += self._pass_1_native_functions_yaml()
        total += self._pass_2_determinism_api()
        total += self._pass_3_cudnn_config()
        total += self._pass_4_rng_management()
        total += self._pass_5_numerical_stability()
        total += self._pass_6_determinism_tests()

        out = str(self.output_path / OUTPUT_FILE)
        self.flush(out)
        self.report_stats()
        return total

    # ------------------------------------------------------------------
    # Pass 1 — Native functions YAML
    # ------------------------------------------------------------------

    def _pass_1_native_functions_yaml(self) -> int:
        self.logger.info("Pass 1 — Native functions YAML: starting")
        yaml_path = self.repo_path / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"
        if not yaml_path.exists():
            self.logger.warning("native_functions.yaml not found at %s", yaml_path)
            self._warnings += 1
            return 0

        count = 0
        try:
            # Use a lightweight YAML-like parser to avoid hard dependency on PyYAML.
            # We scan for operator entries with device_check: NoCheck or device_guard: False.
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            self.logger.warning(
                "PyYAML not installed; falling back to regex-based YAML parsing"
            )
            return self._parse_native_functions_regex(yaml_path)

        try:
            content = self.read_file_safe(yaml_path)
            if content is None:
                return 0
            self._files_processed += 1
            entries = yaml.safe_load(content)
            if not isinstance(entries, list):
                self.logger.warning("native_functions.yaml root is not a list")
                self._warnings += 1
                return 0

            rel = str(yaml_path.relative_to(self.repo_path))
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                func_name = entry.get("func", "")
                # Extract operator name from "func" field (format: "name.overload(args) -> ret")
                op_name = func_name.split("(")[0].split(".")[0].strip()
                if not op_name:
                    continue

                device_check = entry.get("device_check", "")
                device_guard = entry.get("device_guard", True)

                if str(device_check) == "NoCheck" or device_guard is False:
                    record = self.make_record(
                        source_file=rel,
                        language="yaml",
                        entity_name=op_name,
                        entity_type="operator",
                        subcategory="nondeterministic_candidate",
                        module_path="aten.native",
                        qualified_name=f"aten::{op_name}",
                        raw_text=str(entry)[:500],
                        extraction_confidence=0.7,
                        compliance_tags=["eu_ai_act_art_15"],
                        lifecycle_phase="runtime",
                        metadata={
                            "device_check": str(device_check),
                            "device_guard": str(device_guard),
                            "func_signature": func_name[:200],
                        },
                    )
                    self.write_record(record, str(self.output_path / OUTPUT_FILE))
                    count += 1

        except Exception as exc:
            self.logger.error("Error parsing native_functions.yaml: %s", exc)
            self._errors += 1

        self.logger.info("Pass 1 — Native functions YAML: %d records", count)
        return count

    def _parse_native_functions_regex(self, yaml_path: Path) -> int:
        """Fallback regex parser when PyYAML is not installed."""
        content = self.read_file_safe(yaml_path)
        if content is None:
            return 0
        self._files_processed += 1
        count = 0
        rel = str(yaml_path.relative_to(self.repo_path))

        # Match blocks that contain device_check: NoCheck or device_guard: False
        func_re = re.compile(r"^- func:\s*(.+)$", re.MULTILINE)
        for match in func_re.finditer(content):
            func_sig = match.group(1).strip()
            op_name = func_sig.split("(")[0].split(".")[0].strip()
            # Look ahead for device_check or device_guard within the next ~20 lines
            block_start = match.start()
            block_end = content.find("\n- ", block_start + 1)
            if block_end == -1:
                block_end = len(content)
            block = content[block_start:block_end]

            if "device_check: NoCheck" in block or "device_guard: False" in block:
                lineno = content[:block_start].count("\n") + 1
                record = self.make_record(
                    source_file=rel,
                    language="yaml",
                    entity_name=op_name,
                    entity_type="operator",
                    subcategory="nondeterministic_candidate",
                    module_path="aten.native",
                    qualified_name=f"aten::{op_name}",
                    start_line=lineno,
                    raw_text=block[:500],
                    extraction_confidence=0.6,
                    compliance_tags=["eu_ai_act_art_15"],
                    lifecycle_phase="runtime",
                    metadata={"func_signature": func_sig[:200]},
                )
                self.write_record(record, str(self.output_path / OUTPUT_FILE))
                count += 1

        self.logger.info("Pass 1 (regex fallback) — %d records", count)
        return count

    # ------------------------------------------------------------------
    # Pass 2 — Determinism API
    # ------------------------------------------------------------------

    _DETERMINISM_API_NAMES = {
        "use_deterministic_algorithms",
        "are_deterministic_algorithms_enabled",
    }

    def _pass_2_determinism_api(self) -> int:
        self.logger.info("Pass 2 — Determinism API: starting")
        count = 0
        target = self.repo_path / "torch" / "__init__.py"
        if not target.exists():
            self.logger.warning("torch/__init__.py not found")
            self._warnings += 1
            return 0

        try:
            count += self._extract_functions_by_name(
                target,
                self._DETERMINISM_API_NAMES,
                subcategory="determinism_api",
                compliance_tags=["eu_ai_act_art_15", "reproducibility"],
                lifecycle_phase="configuration",
            )
        except SyntaxError as exc:
            self.logger.warning("Syntax error in %s: %s", target, exc)
            self._warnings += 1
        except Exception as exc:
            self.logger.error("Error in Pass 2: %s", exc)
            self._errors += 1

        self.logger.info("Pass 2 — Determinism API: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Pass 3 — cuDNN config
    # ------------------------------------------------------------------

    _CUDNN_NAMES = {"deterministic", "benchmark", "allow_tf32"}

    def _pass_3_cudnn_config(self) -> int:
        self.logger.info("Pass 3 — cuDNN config: starting")
        count = 0
        target = self.repo_path / "torch" / "backends" / "cudnn" / "__init__.py"
        if not target.exists():
            self.logger.warning("torch/backends/cudnn/__init__.py not found")
            self._warnings += 1
            return 0

        try:
            source = self.read_file_safe(target)
            if source is None:
                return 0
            tree = ast.parse(source, filename=str(target))
            self._files_processed += 1
            rel = str(target.relative_to(self.repo_path))
            module_path = self.file_to_module_path(target)

            for node in ast.walk(tree):
                name = None
                etype = "config_entry"
                sig = ""
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in self._CUDNN_NAMES:
                        name = node.name
                        etype = "function"
                        sig = self.extract_function_signature(node)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id in self._CUDNN_NAMES:
                            name = t.id
                            break

                if name is not None:
                    record = self.make_record(
                        source_file=rel,
                        language="python",
                        entity_name=name,
                        entity_type=etype,
                        subcategory="cudnn_config",
                        module_path=module_path,
                        qualified_name=f"{module_path}.{name}",
                        start_line=getattr(node, "lineno", 0),
                        end_line=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                        signature=sig,
                        extraction_confidence=1.0,
                        compliance_tags=["eu_ai_act_art_15"],
                        lifecycle_phase="configuration",
                    )
                    self.write_record(record, str(self.output_path / OUTPUT_FILE))
                    count += 1

        except SyntaxError as exc:
            self.logger.warning("Syntax error in %s: %s", target, exc)
            self._warnings += 1
        except Exception as exc:
            self.logger.error("Error in Pass 3: %s", exc)
            self._errors += 1

        self.logger.info("Pass 3 — cuDNN config: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Pass 4 — RNG management
    # ------------------------------------------------------------------

    _RNG_NAMES = {"manual_seed", "set_rng_state", "get_rng_state", "fork_rng"}

    def _pass_4_rng_management(self) -> int:
        self.logger.info("Pass 4 — RNG management: starting")
        count = 0
        targets = [
            self.repo_path / "torch" / "__init__.py",
            self.repo_path / "torch" / "random.py",
        ]

        for target in targets:
            if not target.exists():
                self.logger.warning("RNG target not found: %s", target)
                self._warnings += 1
                continue
            try:
                count += self._extract_functions_by_name(
                    target,
                    self._RNG_NAMES,
                    subcategory="rng_management",
                    compliance_tags=["eu_ai_act_art_15", "reproducibility"],
                    lifecycle_phase="initialization",
                )
            except SyntaxError as exc:
                self.logger.warning("Syntax error in %s: %s", target, exc)
                self._warnings += 1
            except Exception as exc:
                self.logger.error("Error in Pass 4 for %s: %s", target, exc)
                self._errors += 1

        self.logger.info("Pass 4 — RNG management: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Pass 5 — Numerical stability patterns
    # ------------------------------------------------------------------

    def _pass_5_numerical_stability(self) -> int:
        self.logger.info("Pass 5 — Numerical stability patterns: starting")
        count = 0
        target = self.repo_path / "torch" / "nn" / "functional.py"
        if not target.exists():
            self.logger.warning("torch/nn/functional.py not found")
            self._warnings += 1
            return 0

        try:
            source = self.read_file_safe(target)
            if source is None:
                return 0
            self._files_processed += 1
            rel = str(target.relative_to(self.repo_path))
            module_path = self.file_to_module_path(target)
            lines = source.splitlines()

            seen_patterns: set[tuple[str, int]] = set()

            for lineno_0, line in enumerate(lines):
                for m in _NUMERICAL_STABILITY_RE.finditer(line):
                    matched_word = m.group(1).lower()
                    key = (matched_word, lineno_0)
                    if key in seen_patterns:
                        continue
                    if _is_false_positive(line, m.start()):
                        continue
                    seen_patterns.add(key)

                    record = self.make_record(
                        source_file=rel,
                        language="python",
                        entity_name=matched_word,
                        entity_type="config_entry",
                        subcategory="numerical_stability",
                        module_path=module_path,
                        qualified_name=f"{module_path}:{lineno_0 + 1}",
                        start_line=lineno_0 + 1,
                        end_line=lineno_0 + 1,
                        raw_text=line.strip()[:300],
                        extraction_confidence=0.7,
                        compliance_tags=["eu_ai_act_art_15"],
                        lifecycle_phase="runtime",
                        metadata={"pattern_matched": matched_word},
                    )
                    self.write_record(record, str(self.output_path / OUTPUT_FILE))
                    count += 1

        except Exception as exc:
            self.logger.error("Error in Pass 5: %s", exc)
            self._errors += 1

        self.logger.info("Pass 5 — Numerical stability patterns: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Pass 6 — Determinism tests
    # ------------------------------------------------------------------

    def _pass_6_determinism_tests(self) -> int:
        self.logger.info("Pass 6 — Determinism tests: starting")
        count = 0
        target = self.repo_path / "test" / "test_deterministic.py"
        if not target.exists():
            self.logger.warning("test/test_deterministic.py not found")
            self._warnings += 1
            return 0

        try:
            source = self.read_file_safe(target)
            if source is None:
                return 0
            tree = ast.parse(source, filename=str(target))
            self._files_processed += 1
            rel = str(target.relative_to(self.repo_path))
            module_path = self.file_to_module_path(target)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test"):
                        record = self.make_record(
                            source_file=rel,
                            language="python",
                            entity_name=node.name,
                            entity_type="test_case",
                            subcategory="determinism_test",
                            module_path=module_path,
                            qualified_name=f"{module_path}.{node.name}",
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            signature=self.extract_function_signature(node),
                            docstring=ast.get_docstring(node) or "",
                            extraction_confidence=1.0,
                            compliance_tags=["eu_ai_act_art_15", "reproducibility"],
                            lifecycle_phase="testing",
                        )
                        self.write_record(
                            record, str(self.output_path / OUTPUT_FILE)
                        )
                        count += 1

        except SyntaxError as exc:
            self.logger.warning("Syntax error in %s: %s", target, exc)
            self._warnings += 1
        except Exception as exc:
            self.logger.error("Error in Pass 6: %s", exc)
            self._errors += 1

        self.logger.info("Pass 6 — Determinism tests: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _extract_functions_by_name(
        self,
        fpath: Path,
        target_names: set[str],
        subcategory: str,
        compliance_tags: list[str],
        lifecycle_phase: str,
    ) -> int:
        """AST-walk *fpath* and extract functions/classes whose name is in *target_names*."""
        source = self.read_file_safe(fpath)
        if source is None:
            return 0
        tree = ast.parse(source, filename=str(fpath))
        self._files_processed += 1
        count = 0
        rel = str(fpath.relative_to(self.repo_path))
        module_path = self.file_to_module_path(fpath)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in target_names:
                    record = self.make_record(
                        source_file=rel,
                        language="python",
                        entity_name=node.name,
                        entity_type="function",
                        subcategory=subcategory,
                        module_path=module_path,
                        qualified_name=f"{module_path}.{node.name}",
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=self.extract_function_signature(node),
                        docstring=ast.get_docstring(node) or "",
                        extraction_confidence=1.0,
                        compliance_tags=compliance_tags,
                        lifecycle_phase=lifecycle_phase,
                    )
                    self.write_record(record, str(self.output_path / OUTPUT_FILE))
                    count += 1

        return count
