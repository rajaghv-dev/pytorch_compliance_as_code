"""
Logging setup for the PyTorch Compliance Toolkit.

Call setup_logging() ONCE at the very start of your program (e.g. in cli.py).
After that, every logger under the "pct.*" hierarchy will automatically write
to both the terminal and a rotating daily log file.

Why a dedicated module?
    All 40+ source files use logging.getLogger("pct.something").  By
    configuring the parent "pct" logger here once, every child logger inherits
    the same handlers and level without any extra setup.

Usage:
    from src.logging_setup import setup_logging, phase_logger

    # At startup (once):
    setup_logging(level="INFO", log_dir=Path("storage/logs"))

    # Around each pipeline phase:
    with phase_logger("catalog"):
        run_catalog(...)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The root logger name for the entire toolkit.
# All pct.* loggers are children of this one.
PCT_ROOT_LOGGER = "pct"

# Log format for the terminal: short and easy to scan.
CONSOLE_FORMAT = "%(asctime)s [%(levelname)-8s] %(message)s"

# Log format for the file: includes the logger name for traceability.
FILE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"

# Short time format for the console (HH:MM:SS).
DATE_FMT_CONSOLE = "%H:%M:%S"

# Full date+time for log files (ISO-like).
DATE_FMT_FILE = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# setup_logging()
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_filename: str = "pct.log",
) -> logging.Logger:
    """
    Configure logging for the entire PyTorch Compliance Toolkit.

    This function is idempotent — calling it more than once is safe and
    will not add duplicate handlers.

    Parameters
    ----------
    level : str
        Logging level for both console and file output.
        One of "DEBUG", "INFO", "WARNING", "ERROR".
    log_dir : Path | None
        Directory where the rotating log file is written.
        If None, file logging is disabled; output goes to stdout only.
    log_filename : str
        Base name of the log file inside *log_dir*.
        The file rotates at midnight, keeping up to 7 days.

    Returns
    -------
    logging.Logger
        The root pct logger (all child loggers inherit from it).
    """
    root = logging.getLogger(PCT_ROOT_LOGGER)

    # Idempotent guard: do nothing if we already set up handlers.
    if root.handlers:
        return root

    numeric_level = _parse_level(level)
    root.setLevel(numeric_level)

    # ------------------------------------------------------------------ #
    # Console handler — writes to stdout
    # ------------------------------------------------------------------ #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FMT_CONSOLE)
    )
    root.addHandler(console_handler)

    # ------------------------------------------------------------------ #
    # File handler — optional, rotating daily log
    # ------------------------------------------------------------------ #
    if log_dir is not None:
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / log_filename

            # TimedRotatingFileHandler rotates at midnight.
            # backupCount=7 keeps the last week of logs.
            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=str(log_path),
                when="midnight",
                backupCount=7,
                encoding="utf-8",
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(
                logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT_FILE)
            )
            root.addHandler(file_handler)
            root.info(
                "File logging enabled: %s  (rotate=daily, keep=7 days)",
                log_path,
            )
        except OSError as exc:
            # Don't crash if we can't create the log directory.
            root.warning(
                "Could not enable file logging at '%s': %s", log_dir, exc
            )

    root.info(
        "Logging initialised — level=%s, handlers=%d", level, len(root.handlers)
    )
    return root


# ---------------------------------------------------------------------------
# phase_logger()  — context manager for pipeline phase timing
# ---------------------------------------------------------------------------


@contextmanager
def phase_logger(
    phase_name: str,
    logger: logging.Logger | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that logs the start, end, and duration of a pipeline phase.

    Use this to wrap each phase so it is easy to find phase boundaries in
    the log output and diagnose which phase is slow.

    Parameters
    ----------
    phase_name : str
        Human-readable phase name, e.g. "catalog" or "extract".
    logger : Logger | None
        Logger to use.  Defaults to the root pct logger.

    Example
    -------
        with phase_logger("catalog"):
            run_catalog_phase(...)

        # Output:
        # 12:00:01 [INFO    ] ============================================================
        # 12:00:01 [INFO    ] PHASE START: CATALOG
        # 12:00:01 [INFO    ] ============================================================
        # ... (catalog logs) ...
        # 12:00:45 [INFO    ] PHASE DONE:  CATALOG  (44.1s)
    """
    log = logger or logging.getLogger(PCT_ROOT_LOGGER)
    separator = "=" * 60
    log.info(separator)
    log.info("PHASE START: %s", phase_name.upper())
    log.info(separator)

    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        log.info(separator)
        log.info("PHASE DONE:  %s  (%.1fs)", phase_name.upper(), elapsed)
        log.info(separator)
    except Exception:
        elapsed = time.perf_counter() - start
        log.error(
            "PHASE FAILED: %s after %.1fs",
            phase_name.upper(), elapsed,
            exc_info=True,
        )
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_level(level: str) -> int:
    """
    Convert a level string to a numeric logging level.

    Accepts "DEBUG", "INFO", "WARNING", "ERROR" (case-insensitive).
    Falls back to INFO for unknown values.
    """
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        logging.getLogger(PCT_ROOT_LOGGER).warning(
            "Unknown log level '%s' — defaulting to INFO", level
        )
        return logging.INFO
    return numeric
