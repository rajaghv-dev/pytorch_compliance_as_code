"""
Base extractor infrastructure for the PyTorch Compliance Toolkit.

This module defines:

* **EntityRecord** — the canonical schema for every piece of evidence extracted
  from the PyTorch repository.
* **compute_stable_id()** — deterministic ID generation (BUG-01 fix).
* **BaseExtractor** — abstract base class that all concrete extractors inherit.
  Includes buffered batch writes (BUG-02), LRU-cached file reads (BUG-03),
  correct rglob usage (BUG-04), and ``ast.unparse`` decorator handling (BUG-05).
* **CheckpointManager** — saves and restores pipeline phase completion state.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import logging
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("pct.extractors.base")

# ---------------------------------------------------------------------------
# BUG-03 fix: module-level cached file reader
# ---------------------------------------------------------------------------
# ``functools.lru_cache`` cannot decorate a bound method directly because
# ``self`` would be part of the cache key and would prevent cache hits across
# instances.  Instead we cache on a free function keyed by (filepath, encoding).

_READ_CACHE_LOCK = threading.Lock()


@functools.lru_cache(maxsize=500)
def _cached_read_file(filepath_str: str, encoding: str = "utf-8") -> Optional[str]:
    """
    Read a file and return its contents as a string.

    Results are cached (LRU, 500 entries) so repeated reads of the same file
    across extractors do not hit the filesystem again.

    Parameters
    ----------
    filepath_str : str
        Absolute path to the file (string, not Path, because lru_cache needs
        hashable arguments).
    encoding : str
        Character encoding to use.

    Returns
    -------
    str | None
        File contents, or ``None`` if the file cannot be read.
    """
    try:
        with open(filepath_str, "r", encoding=encoding, errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        logger.warning("Could not read file %s: %s", filepath_str, exc)
        return None


# ---------------------------------------------------------------------------
# EntityRecord
# ---------------------------------------------------------------------------


@dataclass
class EntityRecord:
    """
    Canonical evidence record extracted from the PyTorch repository.

    Every extractor produces a list of ``EntityRecord`` instances.  Down-stream
    annotators enrich the ``compliance_tags``, ``lifecycle_phase``, and related
    fields; converters then serialise records to RDF, CSV, Markdown, etc.
    """

    # ---- Identity ----
    id: str = ""
    source_file: str = ""
    language: str = ""                       # "python"|"cpp"|"yaml"|"rst"|"markdown"|"config"
    entity_name: str = ""
    entity_type: str = ""                    # "function"|"method"|"class"|"operator"|"enum"|
                                             # "doc_directive"|"test_case"|"config_entry"|
                                             # "commit"|"dependency"|"license"
    subcategory: str = ""
    module_path: str = ""
    qualified_name: str = ""
    start_line: int = 0
    end_line: int = 0

    # ---- Content ----
    raw_text: str = ""
    docstring: str = ""
    signature: str = ""

    # ---- Relations ----
    relations: list = field(default_factory=list)

    # ---- Compliance Annotations ----
    compliance_tags: list = field(default_factory=list)
    lifecycle_phase: str = ""
    execution_level: str = ""
    distributed_safety: str = ""
    distributed_safety_notes: str = ""
    export_survival: dict = field(default_factory=dict)

    # ---- Confidence ----
    extraction_confidence: float = 1.0
    mapping_confidence: float = 0.0
    mapping_rationale: str = ""

    # ---- Metadata ----
    extractor: str = ""
    annotations: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ---- Serialisation helpers ----

    def to_dict(self) -> dict:
        """Convert the record to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EntityRecord":
        """
        Reconstruct an ``EntityRecord`` from a dictionary.

        Unknown keys are silently ignored so that records produced by a newer
        version of the toolkit can still be loaded by an older one (forward
        compatibility), and vice-versa (backward compatibility).

        Parameters
        ----------
        d : dict
            Dictionary with record field values.

        Returns
        -------
        EntityRecord
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        try:
            return cls(**filtered)
        except TypeError as exc:
            logger.error(
                "Failed to reconstruct EntityRecord from dict: %s — %s",
                exc,
                {k: type(v).__name__ for k, v in filtered.items()},
            )
            raise


# ---------------------------------------------------------------------------
# Stable-ID generation  (BUG-01 fix)
# ---------------------------------------------------------------------------
# The old algorithm only hashed (source_file, entity_name), which collided
# for overloaded methods, same-named helpers in different submodules, etc.
# The new key includes subcategory, start_line, module_path, and entity_type.


def compute_stable_id(record: EntityRecord) -> str:
    """
    Compute a deterministic 20-hex-char ID for an ``EntityRecord``.

    The hash input includes enough context to distinguish overloaded names,
    same-named functions at different line numbers, and different subcategories.

    Parameters
    ----------
    record : EntityRecord
        The record to identify.

    Returns
    -------
    str
        A 20-character hex digest (lowercase).
    """
    primary = (
        f"{record.source_file}:{record.module_path}:{record.entity_name}"
        f":{record.start_line}:{record.entity_type}:{record.subcategory}"
    )
    return hashlib.sha256(primary.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# BaseExtractor
# ---------------------------------------------------------------------------

# BUG-02: buffer size for batch writes.
_WRITE_BUFFER_FLUSH_THRESHOLD = 1000


class BaseExtractor(ABC):
    """
    Abstract base class for all evidence extractors.

    Subclasses must override :meth:`extract` to walk the relevant portion
    of the repository and yield ``EntityRecord`` instances.

    Key design decisions (bug-fix driven):

    * **BUG-01** — IDs are computed via ``compute_stable_id()``.
    * **BUG-02** — Records are buffered; disk writes happen every 100
      records or when the extractor finishes.
    * **BUG-03** — File reads go through ``read_file_safe()``, which
      delegates to a module-level LRU-cached function.
    * **BUG-04** — ``find_files()`` uses ``pathlib.Path.rglob()``.
    * **BUG-05** — ``extract_decorators()`` uses ``ast.unparse()``.
    """

    def __init__(self, name: str, repo_path: str | Path, output_path: str | Path):
        """
        Initialise the base extractor.

        Parameters
        ----------
        name : str
            Human-readable extractor name (used in logs and record metadata).
        repo_path : str | Path
            Root of the PyTorch repository checkout.
        output_path : str | Path
            Directory where output JSON files are written.
        """
        self.name = name
        self.repo_path = Path(repo_path)
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(f"pct.extractors.{name}")

        # Counters for the final stats report.
        self._files_processed: int = 0
        self._records_produced: int = 0
        self._errors: int = 0
        self._warnings: int = 0

        # BUG-02: write buffer keyed by output_file path.
        self._write_buffers: dict[str, list[dict]] = {}
        self._written_counts: dict[str, int] = {}

        self.logger.info("Initialised extractor '%s'", name)

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def extract(self) -> list[EntityRecord]:
        """
        Run the extractor and return all discovered records.

        Subclasses must implement this method.  They should call
        :meth:`write_record` / :meth:`write_records` to persist output,
        and call :meth:`flush` at the end if they want to guarantee all
        buffered data is written before returning.

        Returns
        -------
        list[EntityRecord]
            All records produced during this extraction pass.
        """
        ...

    # ------------------------------------------------------------------ #
    # File discovery  (BUG-04 fix)
    # ------------------------------------------------------------------ #

    def find_files(self, pattern: str, root: Path | None = None) -> list[Path]:
        """
        Recursively find files matching a glob *pattern* under *root*.

        Uses ``pathlib.Path.rglob()`` directly instead of manually splitting
        on ``**`` (BUG-04 fix).

        Parameters
        ----------
        pattern : str
            A glob pattern such as ``"*.py"`` or ``"torch/**/*.cpp"``.
        root : Path | None
            Starting directory.  Defaults to ``self.repo_path``.

        Returns
        -------
        list[Path]
            Sorted list of matching paths.
        """
        root = root or self.repo_path
        self.logger.info("Searching for '%s' under %s", pattern, root)
        try:
            matches = sorted(root.rglob(pattern))
            self.logger.info("  Found %d files matching '%s'", len(matches), pattern)
            return matches
        except OSError as exc:
            self.logger.error("Error during file search: %s", exc)
            self._errors += 1
            return []

    # ------------------------------------------------------------------ #
    # File reading  (BUG-03 fix)
    # ------------------------------------------------------------------ #

    def read_file_safe(self, filepath: Path, encoding: str = "utf-8") -> Optional[str]:
        """
        Read a file, returning ``None`` on failure.

        Delegates to the module-level ``_cached_read_file`` so that repeated
        reads of the same file are served from an LRU cache.

        Parameters
        ----------
        filepath : Path
            Absolute or relative path to the file.
        encoding : str
            Character encoding.

        Returns
        -------
        str | None
            File content, or ``None`` if unreadable.
        """
        content = _cached_read_file(str(filepath.resolve()), encoding)
        if content is None:
            self._warnings += 1
        return content

    # ------------------------------------------------------------------ #
    # Path / name helpers
    # ------------------------------------------------------------------ #

    def file_to_module_path(self, filepath: Path) -> str:
        """
        Convert a repository-relative file path to a dotted module path.

        Example: ``torch/nn/modules/module.py`` becomes
        ``torch.nn.modules.module``.

        Parameters
        ----------
        filepath : Path
            Path relative to the repository root.

        Returns
        -------
        str
            Dotted module path (without ``.py`` extension).
        """
        try:
            rel = filepath.relative_to(self.repo_path)
        except ValueError:
            # filepath is not under repo_path; use as-is.
            rel = filepath

        # Strip the .py extension and convert separators.
        parts = rel.with_suffix("").parts
        return ".".join(parts)

    def compute_qualified_name(
        self, module_path: str, class_name: str, method_name: str
    ) -> str:
        """
        Build a fully-qualified dotted name for an entity.

        Parameters
        ----------
        module_path : str
            Dotted module path (e.g. ``torch.nn.modules.module``).
        class_name : str
            Enclosing class name, or ``""`` if none.
        method_name : str
            Function / method name.

        Returns
        -------
        str
            Qualified name such as ``torch.nn.modules.module.Module.forward``.
        """
        parts = [module_path]
        if class_name:
            parts.append(class_name)
        if method_name:
            parts.append(method_name)
        return ".".join(parts)

    # ------------------------------------------------------------------ #
    # AST helpers
    # ------------------------------------------------------------------ #

    def extract_function_signature(self, node: ast.FunctionDef) -> str:
        """
        Return the full function signature as a string.

        Uses ``ast.unparse`` on the arguments node and reconstructs the
        ``def …(…) -> …:`` line.

        Parameters
        ----------
        node : ast.FunctionDef
            An AST function-definition node.

        Returns
        -------
        str
            Signature string like ``def foo(x: int, y: str = "bar") -> bool:``.
        """
        try:
            args_str = ast.unparse(node.args)
            ret = ""
            if node.returns is not None:
                ret = f" -> {ast.unparse(node.returns)}"
            return f"def {node.name}({args_str}){ret}:"
        except Exception as exc:
            self.logger.warning(
                "Failed to unparse signature for %s: %s", node.name, exc
            )
            self._warnings += 1
            return f"def {node.name}(...):"

    def extract_decorators(self, node: ast.FunctionDef | ast.ClassDef) -> list[str]:
        """
        Return human-readable strings for each decorator on *node*.

        Uses ``ast.unparse()`` (BUG-05 fix) so that complex decorator
        expressions such as ``@torch.no_grad()`` are rendered correctly
        instead of falling back to ``<decorator>``.

        Parameters
        ----------
        node : ast.FunctionDef | ast.ClassDef
            An AST node that may carry decorators.

        Returns
        -------
        list[str]
            Decorator strings, each prefixed with ``@``.
        """
        decorators: list[str] = []
        for dec in node.decorator_list:
            try:
                decorators.append(f"@{ast.unparse(dec)}")
            except Exception as exc:
                self.logger.warning(
                    "Could not unparse decorator on %s: %s",
                    getattr(node, "name", "<unknown>"),
                    exc,
                )
                self._warnings += 1
                decorators.append("@<unknown>")
        return decorators

    # ------------------------------------------------------------------ #
    # Raw-text extraction
    # ------------------------------------------------------------------ #

    def get_raw_text(self, filepath: Path, start: int, end: int) -> str:
        """
        Return the source lines ``[start, end]`` (1-based, inclusive).

        The result is capped at 5 000 characters to prevent oversized records
        from blowing up memory when serialised.

        Parameters
        ----------
        filepath : Path
            Source file.
        start : int
            First line number (1-based).
        end : int
            Last line number (1-based, inclusive).

        Returns
        -------
        str
            The extracted text, truncated to 5 000 chars if necessary.
        """
        content = self.read_file_safe(filepath)
        if content is None:
            return ""
        lines = content.splitlines()
        # Clamp to valid range.
        start = max(1, start)
        end = min(len(lines), end)
        snippet = "\n".join(lines[start - 1 : end])
        if len(snippet) > 5000:
            snippet = snippet[:5000] + "\n... [truncated]"
        return snippet

    # ------------------------------------------------------------------ #
    # Record creation helper
    # ------------------------------------------------------------------ #

    def make_record(self, **kwargs: Any) -> EntityRecord:
        """
        Build an ``EntityRecord`` with a stable ID and extractor metadata.

        Callers supply entity-specific fields as keyword arguments; this
        method fills in ``id``, ``extractor``, and ``timestamp`` automatically.

        Parameters
        ----------
        **kwargs
            Fields forwarded to the ``EntityRecord`` constructor.

        Returns
        -------
        EntityRecord
        """
        record = EntityRecord(extractor=self.name, **kwargs)
        record.id = compute_stable_id(record)
        return record

    # ------------------------------------------------------------------ #
    # Buffered writes  (BUG-02 fix)
    # ------------------------------------------------------------------ #

    def write_record(self, record: EntityRecord, output_file: str) -> None:
        """
        Add a single record to the write buffer for *output_file*.

        When the buffer reaches ``_WRITE_BUFFER_FLUSH_THRESHOLD`` (100),
        it is automatically flushed to disk.

        Parameters
        ----------
        record : EntityRecord
            The record to persist.
        output_file : str
            Path to the output JSON file.
        """
        buf = self._write_buffers.setdefault(output_file, [])
        buf.append(record.to_dict())
        self._records_produced += 1

        if len(buf) >= _WRITE_BUFFER_FLUSH_THRESHOLD:
            self._flush_writes(output_file)

    def write_records(self, records: list[EntityRecord], output_file: str) -> None:
        """
        Add a batch of records to the write buffer.

        Parameters
        ----------
        records : list[EntityRecord]
            Records to persist.
        output_file : str
            Path to the output JSON file.
        """
        buf = self._write_buffers.setdefault(output_file, [])
        for rec in records:
            buf.append(rec.to_dict())
            self._records_produced += 1

        # Flush if we've exceeded the threshold.
        if len(buf) >= _WRITE_BUFFER_FLUSH_THRESHOLD:
            self._flush_writes(output_file)

    def flush(self, output_file: str) -> None:
        """
        Force-flush the write buffer for a specific output file.

        Parameters
        ----------
        output_file : str
            Path to the output JSON file.
        """
        if output_file in self._write_buffers and self._write_buffers[output_file]:
            self._flush_writes(output_file)

    def flush_all(self) -> None:
        """Flush every pending write buffer to disk."""
        for output_file in list(self._write_buffers.keys()):
            self._flush_writes(output_file)

    def _flush_writes(self, output_file: str) -> None:
        """
        Persist buffered records to *output_file* by appending JSONL lines.

        Each record is written as a single JSON line (append mode).  A running
        total is tracked in ``self._written_counts`` so we never need to read
        the file back from disk.

        Parameters
        ----------
        output_file : str
            Destination JSONL file path.
        """
        buf = self._write_buffers.pop(output_file, [])
        if not buf:
            return

        out_path = Path(output_file)
        self.logger.debug("Flushing %d records to %s", len(buf), out_path.name)

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as fh:
                for rec in buf:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            prev = self._written_counts.get(output_file, 0)
            total = prev + len(buf)
            self._written_counts[output_file] = total

            # Log a milestone every 5000 records
            if total // 5000 > prev // 5000:
                self.logger.info(
                    "%s: %d records written …", out_path.name, total
                )
        except Exception as exc:
            self.logger.error(
                "Failed to flush records to %s: %s", output_file, exc
            )
            self._errors += 1
            # Put the records back so they aren't lost.
            existing_buf = self._write_buffers.get(output_file, [])
            self._write_buffers[output_file] = buf + existing_buf

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def report_stats(self) -> dict[str, int]:
        """
        Log a summary of extraction statistics and return them as a dict.

        Returns
        -------
        dict[str, int]
            Keys: ``files_processed``, ``records_produced``, ``errors``,
            ``warnings``.
        """
        stats = {
            "files_processed": self._files_processed,
            "records_produced": self._records_produced,
            "errors": self._errors,
            "warnings": self._warnings,
        }
        self.logger.info(
            "Extractor '%s' stats — files: %d | records: %d | errors: %d | warnings: %d",
            self.name,
            stats["files_processed"],
            stats["records_produced"],
            stats["errors"],
            stats["warnings"],
        )
        return stats


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """
    Saves and restores pipeline phase completion state.

    State is persisted to ``<storage_path>/session/checkpoint.json`` so that
    a pipeline run can be resumed after interruption without re-executing
    already-completed phases.
    """

    def __init__(self, storage_path: Path):
        """
        Initialise the checkpoint manager.

        Parameters
        ----------
        storage_path : Path
            Root storage directory (typically ``./storage``).
        """
        self.path = storage_path / "session" / "checkpoint.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("pct.checkpoint")
        self._state = self._load_or_init()
        self.logger.info("CheckpointManager initialised at %s", self.path)

    # ---- internal persistence ----

    def _load_or_init(self) -> dict:
        """Load existing checkpoint or create a fresh state dict."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    state = json.load(fh)
                self.logger.info("Loaded existing checkpoint with %d completed phases",
                                 len(state.get("completed_phases", {})))
                return state
            except (json.JSONDecodeError, OSError) as exc:
                self.logger.warning(
                    "Corrupt checkpoint file %s: %s — starting fresh", self.path, exc
                )
        return {
            "completed_phases": {},
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "config_hash": "",
        }

    def _persist(self) -> None:
        """Write current state to disk atomically."""
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.path.parent),
                prefix=".checkpoint_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self._state, fh, indent=2, ensure_ascii=False)
                os.replace(tmp_path, str(self.path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            self.logger.error("Failed to persist checkpoint: %s", exc)

    # ---- public API ----

    def mark_done(self, phase: str, metadata: Optional[dict] = None) -> None:
        """
        Record that *phase* completed successfully.

        Parameters
        ----------
        phase : str
            Phase name (e.g. ``"catalog"``, ``"extract"``).
        metadata : dict | None
            Optional metadata to store alongside the completion timestamp
            (e.g. ``{"records": 12345}``).
        """
        entry = {
            "done_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            entry.update(metadata)
        self._state.setdefault("completed_phases", {})[phase] = entry
        self._persist()
        self.logger.info("Marked phase '%s' as done", phase)

    def is_done(self, phase: str) -> bool:
        """
        Check whether *phase* has already been completed.

        Parameters
        ----------
        phase : str
            Phase name.

        Returns
        -------
        bool
        """
        done = phase in self._state.get("completed_phases", {})
        self.logger.info("Phase '%s' done? %s", phase, done)
        return done

    def get_metadata(self, phase: str) -> dict:
        """
        Retrieve metadata stored when *phase* was marked done.

        Parameters
        ----------
        phase : str
            Phase name.

        Returns
        -------
        dict
            Metadata dict, or empty dict if the phase is not recorded.
        """
        return dict(self._state.get("completed_phases", {}).get(phase, {}))

    def reset(self, phase: Optional[str] = None) -> None:
        """
        Reset checkpoint state.

        Parameters
        ----------
        phase : str | None
            If given, reset only that phase.  If ``None``, reset everything.
        """
        if phase is None:
            self.logger.info("Resetting ALL checkpoint state")
            self._state["completed_phases"] = {}
        else:
            self.logger.info("Resetting checkpoint for phase '%s'", phase)
            self._state.get("completed_phases", {}).pop(phase, None)
        self._persist()

    def save_state(self, state: dict) -> None:
        """
        Merge arbitrary key/value pairs into the checkpoint.

        Parameters
        ----------
        state : dict
            Arbitrary state to persist (must be JSON-serialisable).
        """
        self._state.update(state)
        self._persist()
        self.logger.info("Saved additional state keys: %s", list(state.keys()))

    def load_state(self) -> dict:
        """
        Return the full checkpoint state dictionary.

        Returns
        -------
        dict
            Complete checkpoint state.
        """
        return dict(self._state)
