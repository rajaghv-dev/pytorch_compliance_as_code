"""
Data Provenance Extractor — extracts data-loading, autograd, and checkpointing
entities from the PyTorch repository.

Compliance relevance:
    - Data loading classes/functions map to EU AI Act Article 10 (data governance).
    - Autograd provenance entities map to Article 12 (record-keeping).
    - Checkpointing utilities map to Article 12 (record-keeping).

The extractor walks targeted directories, parses Python AST, and emits
EntityRecord instances for each matching class or function definition.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from .base import EntityRecord, BaseExtractor, compute_stable_id

logger = logging.getLogger("pct.extractors.data_provenance")

# ---------------------------------------------------------------------------
# Target entity names grouped by subcategory
# ---------------------------------------------------------------------------

# Data loading pipeline entities — Article 10 compliance
DATA_LOADING_TARGETS: set[str] = {
    "Dataset", "IterableDataset", "DataLoader", "Sampler",
    "BatchSampler", "DistributedSampler", "RandomSampler",
    "SequentialSampler", "ConcatDataset", "Subset",
    "random_split", "get_worker_info", "default_collate",
    "TensorDataset", "ChainDataset", "StackDataset",
}

# Autograd provenance entities — Article 12 compliance
AUTOGRAD_TARGETS: set[str] = {
    "backward", "grad", "retain_grad", "no_grad",
    "enable_grad", "set_grad_enabled", "inference_mode",
    "GradMode", "AccumulateGrad",
}

# Checkpointing entities — Article 12 compliance
CHECKPOINT_TARGETS: set[str] = {
    "checkpoint", "checkpoint_sequential",
    "set_checkpoint_early_stop", "CheckpointFunction",
}

# ---------------------------------------------------------------------------
# Compliance tag and lifecycle-phase mappings per subcategory
# ---------------------------------------------------------------------------

_SUBCATEGORY_COMPLIANCE: dict[str, list[str]] = {
    "data_loading":   ["eu_ai_act_art_10"],
    "autograd_provenance": ["eu_ai_act_art_12"],
    "checkpointing":  ["eu_ai_act_art_12"],
}

_SUBCATEGORY_LIFECYCLE: dict[str, str] = {
    "data_loading":   "data_preparation",
    "autograd_provenance": "training_only",
    "checkpointing":  "training_only",
}


class DataProvenanceExtractor(BaseExtractor):
    """
    Extract data-loading, autograd, and checkpointing entities from PyTorch.

    Walks specific directories under the repo, parses Python ASTs, and matches
    class/function definitions against known target names.  Each match becomes
    an EntityRecord tagged with the appropriate compliance articles and lifecycle
    phase.
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the data provenance extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="data_provenance", repo_path=repo_path, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Run the data provenance extraction pass.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting data provenance extraction")
        output_file = str(self.output_path / "data_provenance.jsonl")

        # --- Data loading entities (torch/utils/data/*.py) ---
        self._extract_entities(
            pattern="torch/utils/data/*.py",
            subcategory="data_loading",
            targets=DATA_LOADING_TARGETS,
            output_file=output_file,
        )

        # --- Autograd provenance entities (torch/autograd/*.py) ---
        self._extract_entities(
            pattern="torch/autograd/*.py",
            subcategory="autograd_provenance",
            targets=AUTOGRAD_TARGETS,
            output_file=output_file,
        )

        # --- Checkpointing entities (torch/utils/checkpoint.py) ---
        self._extract_entities(
            pattern="torch/utils/checkpoint.py",
            subcategory="checkpointing",
            targets=CHECKPOINT_TARGETS,
            output_file=output_file,
        )

        # Flush any remaining buffered records to disk
        self.flush(output_file)

        self.logger.info(
            "Data provenance extraction complete — %d records produced",
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_entities(
        self,
        pattern: str,
        subcategory: str,
        targets: set[str],
        output_file: str,
    ) -> None:
        """
        Walk files matching *pattern*, parse AST, and emit records for
        class/function definitions whose name appears in *targets*.

        Parameters
        ----------
        pattern : str
            Glob pattern relative to the repo root.
        subcategory : str
            Subcategory label (e.g. ``"data_loading"``).
        targets : set[str]
            Set of entity names to match.
        output_file : str
            Destination JSONL path.
        """
        # Look up compliance tags and lifecycle phase for this subcategory
        compliance_tags = _SUBCATEGORY_COMPLIANCE.get(subcategory, [])
        lifecycle_phase = _SUBCATEGORY_LIFECYCLE.get(subcategory, "")

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

            # Compute the dotted module path and repo-relative source path
            module_path = self.file_to_module_path(filepath)
            rel_path = str(filepath.relative_to(self.repo_path))

            # Walk the entire AST looking for matching definitions
            for node in ast.walk(tree):
                name = getattr(node, "name", None)
                if name is None or name not in targets:
                    continue

                # Only process class and function definitions
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                # Determine entity type
                is_class = isinstance(node, ast.ClassDef)
                entity_type = "class" if is_class else "function"

                # Line range
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", node.lineno)

                # Extract docstring if present
                docstring = ast.get_docstring(node) or ""

                # For classes, collect method names as metadata
                methods: list[str] = []
                if is_class:
                    methods = [
                        item.name for item in ast.walk(node)
                        if isinstance(item, ast.FunctionDef)
                    ]

                # Build the qualified name (module.ClassName or module.function_name)
                qualified_name = self.compute_qualified_name(module_path, "", name)

                # Raw source text for the entity (capped internally at 5000 chars)
                raw_text = self.get_raw_text(filepath, start_line, end_line)

                # AST-found entities get confidence 1.0
                extraction_confidence = 1.0

                # Build metadata dict
                metadata: dict[str, Any] = {}
                if methods:
                    metadata["methods"] = methods

                # Create the record using the new EntityRecord schema
                record = self.make_record(
                    source_file=rel_path,
                    language="python",
                    entity_name=name,
                    entity_type=entity_type,
                    subcategory=subcategory,
                    module_path=module_path,
                    qualified_name=qualified_name,
                    start_line=start_line,
                    end_line=end_line,
                    raw_text=raw_text,
                    docstring=docstring,
                    compliance_tags=list(compliance_tags),  # copy to avoid sharing
                    lifecycle_phase=lifecycle_phase,
                    extraction_confidence=extraction_confidence,
                    metadata=metadata,
                )

                self.write_record(record, output_file)
