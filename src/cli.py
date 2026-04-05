"""
Command-line interface (CLI) for the PyTorch Compliance Toolkit.

This is the main entry point registered in pyproject.toml:
    [project.scripts]
    pct = "src.cli:main"

USAGE EXAMPLES
--------------
    pct                                       # full pipeline, built-in defaults
    pct --config configs/talk_demo.yaml       # override config file
    pct --repo /path/to/pytorch              # override repo path
    pct --phase catalog,extract              # run only the first two phases
    pct --resume                             # skip phases already completed
    pct --reset                              # wipe checkpoint, run from scratch
    pct --workers 4                          # parallel file processing
    pct --loglevel DEBUG                     # verbose output for debugging
    pct --logdir storage/logs               # also write logs to a file

PIPELINE PHASES (in order)
---------------------------
    1. catalog   — scan the repo for all .py / .cpp / .yaml / .rst files
    2. extract   — run specialized extractors (hookability, determinism, …)
    3. annotate  — enrich records with compliance tags (in memory)
    4. organize  — dedup, cross-reference, index, validate
    5. convert   — write final output (RDF, CSV, Markdown, notebooks)

SESSION CHECKPOINTING
---------------------
After every phase completes, its name is written to:
    storage/session/checkpoint.json

Re-running with --resume will skip phases already in the checkpoint.
This lets you restart a long pipeline run without repeating earlier work.

AUTO-SAVE
---------
During the extract phase (which can take hours on the full PyTorch repo),
the session state is automatically saved to disk every 2 minutes so that
if the process is killed, progress is not completely lost.

GRACEFUL SHUTDOWN
-----------------
Press Ctrl-C (or send SIGTERM) once to request a graceful stop.  The
pipeline will finish its current phase and then exit cleanly with a
non-zero return code so that CI systems can detect the interruption.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from .config import Config, load_config
from .extractors.base import CheckpointManager
from .gpu_monitor import gpu_monitor
from .logging_setup import setup_logging, phase_logger
from .security import (
    SecurityError,
    validate_ollama_url,
    validate_phase_list,
)

logger = logging.getLogger("pct.cli")

# How often to auto-save session state during long phases (seconds).
_AUTO_SAVE_INTERVAL = 120  # 2 minutes

# ---------------------------------------------------------------------------
# Graceful shutdown via Ctrl-C / SIGTERM
# ---------------------------------------------------------------------------
# We use a module-level flag instead of raising an exception in the signal
# handler, because exceptions in signal handlers can cause unpredictable
# behaviour in multi-threaded code.

_shutdown_requested: bool = False


def _on_shutdown(signum: int, frame) -> None:
    """Set the shutdown flag so the pipeline exits after the current phase."""
    global _shutdown_requested
    _shutdown_requested = True
    # Print directly to stderr so the message appears even if logging is
    # not yet configured.
    print(
        "\n[pct] Shutdown requested — finishing current phase, then exiting…",
        file=sys.stderr, flush=True,
    )


signal.signal(signal.SIGINT,  _on_shutdown)
signal.signal(signal.SIGTERM, _on_shutdown)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argparse parser for the `pct` command.

    Adding a new CLI flag:
        1. Add parser.add_argument(...) here.
        2. Read args.<flag> in main() and apply it to the config.
    """
    parser = argparse.ArgumentParser(
        prog="pct",
        description=(
            "PyTorch Compliance Toolkit — extract EU AI Act / GDPR evidence "
            "from the PyTorch source tree."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pct                                   full pipeline with built-in defaults
  pct --config configs/talk_demo.yaml   offline demo (3 extractors, no LLM)
  pct --phase catalog,extract           run only first two phases
  pct --resume                          skip already-completed phases
  pct --reset                           wipe checkpoint and start fresh
  pct --loglevel DEBUG                  verbose output for debugging
""",
    )

    # ---- Config source ----
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to a YAML config file.  Uses built-in defaults if omitted.",
    )

    # ---- Path overrides (override the corresponding YAML fields) ----
    parser.add_argument(
        "--repo", metavar="DIR",
        help="Override repo_path in config (path to PyTorch clone).",
    )
    parser.add_argument(
        "--legal", metavar="DIR",
        help="Override legal_path in config (path to legal reference docs).",
    )
    parser.add_argument(
        "--out", metavar="DIR",
        help="Override output_path in config (where output files are written).",
    )

    # ---- Phase selection ----
    parser.add_argument(
        "--phase", metavar="PHASES",
        help=(
            "Comma-separated list of phases to run, e.g. 'catalog,extract'. "
            "Overrides the phases list in the config file."
        ),
    )

    # ---- Checkpoint controls ----
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip phases that are already recorded as complete in the checkpoint.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help=(
            "Delete the existing checkpoint file before starting, "
            "so all phases run from scratch."
        ),
    )

    # ---- Parallelism ----
    parser.add_argument(
        "--workers", type=int, metavar="N",
        help=(
            "Number of parallel worker processes for file-intensive phases. "
            "Defaults to the value in the config (usually 1)."
        ),
    )

    # ---- Logging ----
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--logdir", metavar="DIR",
        help=(
            "Directory where a rotating daily log file is written. "
            "If omitted, output goes to the console only."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Phase runner functions
# ---------------------------------------------------------------------------
# Each phase is a standalone function that takes (config, storage_path,
# checkpoint) as arguments.  Phases that produce in-memory results also
# accept / return a shared data dict so that consecutive phases (annotate
# → organize → convert) don't need to re-read from disk.


def run_catalog_phase(
    config: Config,
    storage_path: Path,
    checkpoint: CheckpointManager,
) -> None:
    """
    Phase 1 — Catalog: enumerate all source files in the repo.

    Runs four extractors in sequence:
      - Python (.py files under torch/, functorch/, test/)
      - C++    (.cpp, .h, .cu, .cuh files)
      - YAML   (native_functions.yaml operator catalog)
      - RST    (documentation directives and compliance notes)

    Output: storage/raw/catalog_*.json
    """
    from .catalog.python_extractor import PythonCatalogExtractor
    from .catalog.cpp_extractor import CppCatalogExtractor
    from .catalog.yaml_extractor import YamlCatalogExtractor
    from .catalog.rst_extractor import RstCatalogExtractor

    raw_dir = storage_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # List of (label, extractor_instance) pairs.
    # Add new catalog extractors here if you add more language support.
    catalog_extractors = [
        ("python", PythonCatalogExtractor(config.repo_path, str(raw_dir))),
        ("cpp",    CppCatalogExtractor(config.repo_path, str(raw_dir))),
        ("yaml",   YamlCatalogExtractor(config.repo_path, str(raw_dir))),
        ("rst",    RstCatalogExtractor(config.repo_path, str(raw_dir))),
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    def _run_one(label_extractor):
        label, extractor = label_extractor
        n = extractor.extract()   # extract() returns int (count), not a list
        extractor.flush_all()
        return label, n if isinstance(n, int) else len(n)

    t0 = _time.perf_counter()
    total = 0
    max_workers = min(4, len(catalog_extractors))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, le): le[0] for le in catalog_extractors}
        for future in as_completed(futures):
            if _shutdown_requested:
                break
            label, n = future.result()
            total += n
            logger.info("Catalog: %s → %d records", label, n)

    elapsed = _time.perf_counter() - t0
    checkpoint.mark_done("catalog", {"records": total})
    logger.info("Catalog phase complete — %d total records — elapsed %.1fs", total, elapsed)


def run_extract_phase(
    config: Config,
    storage_path: Path,
    checkpoint: CheckpointManager,
) -> None:
    """
    Phase 2 — Extract: run specialised evidence extractors.

    Each extractor in config.extractors.enabled is looked up in
    EXTRACTOR_REGISTRY, instantiated, and run.  Failures in individual
    extractors are logged but do not stop the pipeline.

    Auto-saves the checkpoint every 2 minutes during this phase, because
    it can take a long time on the full PyTorch repository (~50 k files).

    Output: storage/raw/<extractor_name>.json  (one file per extractor)
    """
    from .extractors import EXTRACTOR_REGISTRY

    raw_dir = storage_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    enabled_names = config.extractors.enabled
    logger.info("Extract: running %d extractor(s): %s", len(enabled_names), enabled_names)

    t0 = time.perf_counter()
    total = 0
    last_save = time.time()

    for name in enabled_names:
        if _shutdown_requested:
            logger.warning("Shutdown requested — stopping extract phase early")
            break

        # Auto-save checkpoint periodically so partial progress is preserved.
        if time.time() - last_save >= _AUTO_SAVE_INTERVAL:
            checkpoint.save_state({"extract_autosave_at": time.time()})
            last_save = time.time()
            logger.info("Auto-saved checkpoint during extract phase")

        ExtractorClass = EXTRACTOR_REGISTRY.get(name)
        if ExtractorClass is None:
            logger.warning("Extract: unknown extractor '%s' — skipping", name)
            continue

        logger.info("Extract: starting '%s' …", name)
        try:
            extractor = ExtractorClass(config.repo_path, str(raw_dir))
            n = extractor.extract()   # returns int count
            extractor.flush_all()
            stats = extractor.report_stats()
            total += n if isinstance(n, int) else len(n)
            logger.info(
                "Extract: '%s' → %d records  (files=%d, errors=%d, warnings=%d)",
                name, n if isinstance(n, int) else len(n),
                stats.get("files_processed", 0),
                stats.get("errors", 0),
                stats.get("warnings", 0),
            )
        except Exception:
            logger.error(
                "Extract: extractor '%s' failed — continuing with others",
                name, exc_info=True,
            )

    elapsed = time.perf_counter() - t0
    checkpoint.mark_done("extract", {"records": total})
    logger.info("Extract phase complete — %d total records — elapsed %.1fs", total, elapsed)


def run_annotate_phase(
    config: Config,
    storage_path: Path,
    checkpoint: CheckpointManager,
) -> dict:
    """
    Phase 3 — Annotate: enrich records with compliance metadata.

    Loads all raw records from storage/raw/, then runs each annotator in
    the canonical order (see src/annotators/__init__.py for the order).

    Returns a dict with the annotated records list so that the organize
    phase can use them without re-reading from disk.

    Returns
    -------
    dict
        {"records": list[EntityRecord]}
    """
    from .organizer.dedup import Deduplicator
    from .annotators import run_all_annotators

    raw_dir = storage_path / "raw"

    # Load all records produced by the catalog and extract phases.
    logger.info("Annotate: loading records from %s …", raw_dir)
    loader = Deduplicator()
    records = loader.load_all_records(raw_dir)
    logger.info("Annotate: loaded %d records", len(records))

    # Run all annotators.  Each one enriches the records in memory.
    records = run_all_annotators(records)
    logger.info("Annotate: %d records annotated", len(records))

    checkpoint.mark_done("annotate", {"records": len(records)})
    return {"records": records}


def run_organize_phase(
    config: Config,
    storage_path: Path,
    checkpoint: CheckpointManager,
    records: list | None = None,
) -> dict:
    """
    Phase 4 — Organize: deduplicate, index, validate, and compute statistics.

    Accepts in-memory records from the annotate phase or reloads them from
    disk if this phase is run in isolation (--phase organize).

    Writes four files to storage/organized/:
      - all_records.jsonl       — deduplicated records
      - cross_references.json   — call / co-occurrence graph
      - entity_index.json       — multi-dimensional lookup indexes
      - statistics.json         — aggregate counts and distributions
      - validation_report.json  — data-quality issues

    Returns
    -------
    dict
        {"records": list[EntityRecord], "stats": dict}
    """
    from .organizer.cross_references import CrossReferenceBuilder
    from .organizer.dedup import Deduplicator
    from .organizer.entity_index import EntityIndexer
    from .organizer.statistics import StatisticsComputer
    from .organizer.validation import Validator

    organized_dir = storage_path / "organized"
    organized_dir.mkdir(parents=True, exist_ok=True)

    # If records weren't passed through the pipeline, reload from disk.
    if records is None:
        logger.info("Organize: reloading records from %s …", storage_path / "raw")
        loader = Deduplicator()
        records = loader.load_all_records(storage_path / "raw")

    # Step 4a: Deduplication.
    logger.info("Organize: deduplicating %d records …", len(records))
    dedup = Deduplicator()
    unique = dedup.deduplicate(records)
    dedup.write_results(unique, organized_dir / "all_records.jsonl")
    logger.info("Organize: %d unique records after dedup", len(unique))

    # Step 4b: Cross-references.
    logger.info("Organize: building cross-reference graph …")
    xref = CrossReferenceBuilder()
    graph = xref.build(unique)
    xref.write_results(graph, organized_dir / "cross_references.json")

    # Step 4c: Entity index.
    logger.info("Organize: building entity index …")
    indexer = EntityIndexer()
    indexes = indexer.build_indexes(unique)
    id_lookup = indexer.build_id_lookup(unique)
    indexer.write_results(indexes, id_lookup, organized_dir)

    # Step 4d: Statistics.
    logger.info("Organize: computing statistics …")
    stats_computer = StatisticsComputer()
    stats = stats_computer.compute(unique)
    stats_computer.write_results(stats, organized_dir / "statistics.json")

    # Step 4e: Validation.
    logger.info("Organize: validating records …")
    validator = Validator()
    report = validator.validate(unique)
    validator.write_results(report, organized_dir / "validation_report.json")

    if report["errors"] > 0:
        logger.warning(
            "Organize: %d validation errors found — see %s for details",
            report["errors"], organized_dir / "validation_report.json",
        )
    else:
        logger.info("Organize: validation passed with no errors")

    checkpoint.mark_done(
        "organize",
        {"records": len(unique), "errors": report["errors"]},
    )
    logger.info(
        "Organize phase complete — %d unique records, %d validation errors",
        len(unique), report["errors"],
    )
    return {"records": unique, "stats": stats}


def run_convert_phase(
    config: Config,
    storage_path: Path,
    checkpoint: CheckpointManager,
    records: list | None = None,
) -> None:
    """
    Phase 5 — Convert: write final output in all configured formats.

    Supported formats (controlled by config.outputs.*):
      - CSV      → storage/output/compliance_evidence.csv
      - Markdown → storage/output/compliance_report.md
      - RDF      → storage/output/compliance_graph.ttl
      - Notebook → (placeholder — not yet implemented)
      - Talk assets → (placeholder — not yet implemented)

    Parameters
    ----------
    records : list | None
        Records from the organize phase.  Reloaded from disk if None.
    """
    from .converters.csv_converter import CsvConverter
    from .converters.markdown_converter import MarkdownConverter
    from .converters.rdf_converter import RdfConverter

    output_dir = storage_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reload records if not passed through the in-memory pipeline.
    if records is None:
        logger.info("Convert: loading records from %s …", storage_path / "organized")
        from .organizer.dedup import Deduplicator
        loader = Deduplicator()
        records = loader.load_all_records(storage_path / "organized")

    if not records:
        logger.warning("Convert: no records available — skipping all converters")
        checkpoint.mark_done("convert", {"records": 0})
        return

    outputs = config.outputs

    if outputs.csv:
        logger.info("Convert: writing CSV …")
        CsvConverter().convert(records, output_dir / "compliance_evidence.csv")

    if outputs.markdown:
        logger.info("Convert: writing Markdown …")
        MarkdownConverter().convert(records, output_dir / "compliance_report.md")

    if outputs.rdf:
        logger.info("Convert: writing RDF/Turtle …")
        RdfConverter().convert(records, output_dir / "compliance_graph.ttl")

    if outputs.notebook:
        logger.info("Convert: writing SPARQL query notebook …")
        from .converters.sparql_notebook import SparqlNotebookConverter
        ttl_path = output_dir / "compliance_graph.ttl"
        notebook_path = output_dir / "compliance_queries.ipynb"
        SparqlNotebookConverter().convert(
            output_path=notebook_path,
            ttl_path=str(ttl_path),
        )
        logger.info("Convert: SPARQL notebook written to %s", notebook_path)

    if outputs.talk_assets:
        logger.info("Convert: generating talk assets …")
        assets_dir = storage_path / "talk_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        try:
            from .assets.heatmap import ComplianceHeatmapAsset
            heatmap_path = ComplianceHeatmapAsset().render(records, assets_dir)
            logger.info("Convert: heatmap → %s", heatmap_path)
        except Exception:
            logger.warning("Convert: heatmap failed (matplotlib/squarify missing?)",
                           exc_info=True)
        try:
            from .assets.survival_matrix import SurvivalMatrixAsset
            matrix_path = SurvivalMatrixAsset().render(records, assets_dir)
            logger.info("Convert: survival matrix → %s", matrix_path)
        except Exception:
            logger.warning("Convert: survival matrix failed", exc_info=True)
        try:
            from .assets.handout import HandoutAsset
            handout_path = HandoutAsset().render(records, assets_dir)
            logger.info("Convert: handout → %s", handout_path)
        except ImportError:
            logger.warning(
                "Convert: handout skipped — reportlab not installed "
                "(pip install reportlab)"
            )
        except Exception:
            logger.warning("Convert: handout failed", exc_info=True)

    checkpoint.mark_done("convert", {"records": len(records)})
    logger.info("Convert phase complete")


def run_llm_phase(
    config: Config,
    storage_path: Path,
    checkpoint: CheckpointManager,
) -> None:
    """
    Phase 6 — LLM-Enrich: validate compliance mappings and parse legal text.

    Two steps (both require Ollama to be running locally):

    Step 6a — Mapping validation (phi4):
        Loads the organised records, runs phi4 to rate each entity-article
        mapping, updates mapping_confidence and removes false-positive tags,
        and writes an updated all_records.jsonl.

    Step 6b — Legal parsing (mistral-nemo / qwen2.5:14b):
        Reads legal reference files from data/legal/, extracts structured
        obligation objects, and writes storage/organized/legal_obligations.json.

    The phase is skipped entirely if llm.enabled = false in config.

    Output
    ------
    - storage/organized/all_records.jsonl  (updated confidence / rationale)
    - storage/organized/legal_obligations.json
    """
    if not config.llm.enabled:
        logger.info("LLM phase skipped — llm.enabled = false in config")
        checkpoint.mark_done("llm", {"skipped": True})
        return

    # Wait for the GPU to reach a safe starting temperature before kicking
    # off sustained LLM inference.  A laptop GPU sitting at 76°C after the
    # pipeline run will hit 90-95°C within minutes under phi4 load — thermal
    # throttling then halves inference speed.  Waiting for 53°C gives ~40°C
    # of headroom before throttling starts (typical limit: ~90-95°C).
    gpu_monitor.wait_until_cool()

    from .llm.ollama_client import OllamaClient
    from .llm.mapping_validator import MappingValidator
    from .llm.legal_parser import LegalParser
    from .organizer.dedup import Deduplicator

    organized_dir = storage_path / "organized"
    organized_dir.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(base_url=config.llm.ollama_url)

    # --- Step 6a: mapping validation ---
    logger.info("LLM: loading records for mapping validation …")
    loader = Deduplicator()
    records = loader.load_all_records(organized_dir)
    logger.info("LLM: loaded %d records", len(records))

    # Candidates are records where mapping_confidence < 0.7 and have tags.
    # phi4 rates each entity-article pair as direct / indirect / none and
    # removes tags rated "none" (false-positive elimination).
    validator = MappingValidator(client)
    low_conf_count = sum(
        1 for r in records if r.mapping_confidence < 0.7 and r.compliance_tags
    )
    logger.info(
        "LLM: sending %d low-confidence records to phi4 for mapping validation …",
        low_conf_count,
    )
    records = validator.validate_records(records)

    # Persist updated records back to all_records.jsonl so the convert phase
    # (if re-run with --resume --phase convert) picks up the improved data.
    dedup = Deduplicator()
    dedup.write_results(records, organized_dir / "all_records.jsonl")
    logger.info("LLM: mapping validation complete — wrote %d records", len(records))

    # --- Step 6b: legal text parsing ---
    logger.info("LLM: parsing legal reference files …")
    try:
        legal_data_dir = Path(config.legal_path)
        legal_parser = LegalParser(client, storage_path)
        result = legal_parser.parse_articles(legal_data_dir)
        total_obligations = sum(
            len(obs)
            for article_map in result.values()
            for obs in article_map.values()
        )
        logger.info(
            "LLM: legal parsing complete — %d obligation objects extracted",
            total_obligations,
        )
    except Exception:
        logger.warning("LLM: legal parsing failed — continuing", exc_info=True)

    checkpoint.mark_done("llm", {"records": len(records)})
    logger.info("LLM phase complete")


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------
# Maps phase name → runner function.
# To add a new pipeline phase:
#   1. Write a run_<phase>_phase() function above.
#   2. Add it to this dict.
# Nothing else needs to change.

_PHASE_RUNNERS: dict[str, object] = {
    "catalog":  run_catalog_phase,
    "extract":  run_extract_phase,
    "annotate": run_annotate_phase,
    "organize": run_organize_phase,
    "convert":  run_convert_phase,
    "llm":      run_llm_phase,
}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(config: Config, args: argparse.Namespace) -> int:
    """
    Execute the pipeline phases in order.

    Handles:
      - Checkpoint reset (--reset)
      - Phase skipping (--resume)
      - Passing annotated records between annotate → organize → convert
        to avoid redundant disk reads
      - Shutdown signal checking between phases
      - Final summary table

    Parameters
    ----------
    config : Config
        Validated configuration object.
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    storage_path = Path(config.output_path)
    checkpoint = CheckpointManager(storage_path)

    if args.reset:
        logger.info("--reset: clearing all checkpoint state")
        checkpoint.reset()

    phases = config.phases
    logger.info("Pipeline will run phases: %s", phases)
    logger.info("Resume mode: %s", args.resume)

    # Shared records dict passed between annotate → organize → convert.
    # Avoids re-reading from disk at each phase boundary.
    _shared: dict = {"records": None, "stats": None}

    wall_start = time.time()

    for phase in phases:
        # Check for graceful shutdown between phases (never mid-phase).
        if _shutdown_requested:
            logger.info("Shutdown flag set — exiting pipeline after current position")
            break

        if args.resume and checkpoint.is_done(phase):
            logger.info("SKIP (already done): %s", phase)
            continue

        runner = _PHASE_RUNNERS.get(phase)
        if runner is None:
            logger.warning("No runner registered for phase '%s' — skipping", phase)
            continue

        with phase_logger(phase):
            try:
                # Phases that share in-memory data need special call patterns.
                if phase == "annotate":
                    result = runner(config, storage_path, checkpoint)
                    _shared["records"] = result.get("records")

                elif phase == "organize":
                    result = runner(
                        config, storage_path, checkpoint, _shared["records"]
                    )
                    _shared["records"] = result.get("records")
                    _shared["stats"]   = result.get("stats")

                elif phase == "convert":
                    runner(config, storage_path, checkpoint, _shared["records"])

                else:
                    # catalog and extract don't return in-memory data.
                    runner(config, storage_path, checkpoint)

            except Exception:
                logger.error(
                    "Phase '%s' raised an unhandled exception — aborting pipeline",
                    phase, exc_info=True,
                )
                return 1  # Signal failure to the shell / CI system.

    wall_elapsed = time.time() - wall_start
    _print_summary(checkpoint, phases, wall_elapsed)
    return 0


def _print_summary(
    checkpoint: CheckpointManager,
    phases: list[str],
    elapsed: float,
) -> None:
    """Log a brief table showing which phases completed and their record counts."""
    state = checkpoint.load_state()
    completed = state.get("completed_phases", {})

    logger.info("-" * 56)
    logger.info("PIPELINE SUMMARY")
    logger.info("-" * 56)
    for phase in phases:
        if phase in completed:
            meta = completed[phase]
            records = meta.get("records", "?")
            done_at = meta.get("done_at", "")[:19]   # YYYY-MM-DDTHH:MM:SS
            logger.info(
                "  %-10s  ✓  records=%-8s  completed_at=%s",
                phase, records, done_at,
            )
        else:
            logger.info("  %-10s  —  (skipped or not reached)", phase)
    logger.info("Total wall time: %.1f s", elapsed)
    logger.info("-" * 56)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Entry point for the `pct` command.

    Steps:
        1. Parse CLI arguments.
        2. Set up logging (console + optional file).
        3. Load and validate the YAML config.
        4. Apply CLI overrides to the config.
        5. Run security checks (Ollama URL, phase names).
        6. Execute the pipeline.
        7. Exit with 0 (success) or 1 (failure).
    """
    parser = build_parser()
    args = parser.parse_args()

    # Step 2 — Logging must be set up before any other output.
    log_dir = Path(args.logdir) if args.logdir else Path("storage/logs")
    setup_logging(level=args.loglevel, log_dir=log_dir)
    logger.info("PyTorch Compliance Toolkit — starting up")

    # GPU detection: log device info once at startup so it's visible in logs.
    gpu_monitor.log_device_info()

    # Step 3 — Load config.
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load configuration: %s", exc)
        sys.exit(1)

    # Step 4 — Apply CLI overrides.
    if args.repo:
        config.repo_path = args.repo
        logger.info("CLI override: repo_path = %s", args.repo)
    if args.legal:
        config.legal_path = args.legal
        logger.info("CLI override: legal_path = %s", args.legal)
    if args.out:
        config.output_path = args.out
        logger.info("CLI override: output_path = %s", args.out)
    if args.workers:
        config.workers = args.workers
        logger.info("CLI override: workers = %d", args.workers)
    if args.phase:
        requested = [p.strip() for p in args.phase.split(",") if p.strip()]
        safe_phases = validate_phase_list(requested)
        if not safe_phases:
            logger.error(
                "No valid phases in --phase '%s'.  "
                "Valid options: %s",
                args.phase,
                ", ".join(["catalog", "extract", "annotate", "organize", "convert", "llm"]),
            )
            sys.exit(1)
        config.phases = safe_phases
        logger.info("CLI override: phases = %s", safe_phases)

    logger.info(config.summary())

    # Step 5 — Security checks.
    try:
        if config.llm.enabled:
            validate_ollama_url(config.llm.ollama_url)
    except SecurityError as exc:
        logger.error("Security check failed: %s", exc)
        sys.exit(1)

    # Step 6 — Run the pipeline.
    exit_code = run_pipeline(config, args)
    sys.exit(exit_code)
