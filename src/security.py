"""
Security utilities for the PyTorch Compliance Toolkit.

WHY THIS FILE EXISTS
--------------------
The toolkit reads from user-supplied paths (PyTorch repo, legal docs) and
writes output under a user-specified output directory.  Without explicit
guards, a crafted configuration file could cause:

  - Path traversal: writing files outside the intended output directory
    (e.g. output_path="../../../etc/cron.d/evil").
  - SSRF: Ollama URL pointing to an internal service that should not receive
    the prompts we send (which can contain proprietary code excerpts).
  - Memory exhaustion: loading a multi-GB JSON "record" file that was
    created or modified by an attacker.
  - Log injection: entity names containing newline characters that make
    fake log entries appear in the audit trail.

WHEN TO CALL THESE FUNCTIONS
-----------------------------
  safe_path()            — before writing any file whose path came from config
  sanitize_name()        — before using entity_name in a filename or log line
  safe_json_load()       — before loading external JSON files
  validate_ollama_url()  — once at startup, before the first Ollama request
  validate_phase_list()  — when accepting phase names from CLI / config

These checks run at system boundaries only, not inside tight loops.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


logger = logging.getLogger("pct.security")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SecurityError(Exception):
    """Raised when a security check fails.

    Callers should catch this and log an error before exiting.  Never
    silently swallow it — a security violation must be surfaced to the user.
    """


# ---------------------------------------------------------------------------
# 1.  Path traversal protection
# ---------------------------------------------------------------------------

# Never allow writes to these system roots, regardless of allowed_root.
# This is a belt-and-suspenders check on top of the relative_to() test.
_BLOCKED_SYSTEM_PREFIXES: tuple[str, ...] = (
    "/etc", "/usr", "/bin", "/sbin", "/boot",
    "/proc", "/sys", "/dev", "/lib", "/lib64",
    "/var/log", "/root",
)


def safe_path(target: str | Path, allowed_root: str | Path) -> Path:
    """
    Resolve *target* and confirm it lives inside *allowed_root*.

    This is the primary defence against path traversal.  Call it whenever
    you are about to create or write a file whose path came from external
    input (config file, CLI argument, entity metadata).

    Parameters
    ----------
    target : str | Path
        The path to validate.  May be relative.
    allowed_root : str | Path
        The directory that *target* must be inside.

    Returns
    -------
    Path
        The fully resolved, validated absolute path.

    Raises
    ------
    SecurityError
        If *target* is outside *allowed_root* or points to a blocked system root.
    """
    target_resolved = Path(target).resolve()
    root_resolved = Path(allowed_root).resolve()

    # Ensure target is inside allowed_root.
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError:
        raise SecurityError(
            f"Path traversal blocked: '{target_resolved}' escapes "
            f"the allowed root '{root_resolved}'. "
            "Check your config for unexpected '..' components."
        )

    # Belt-and-suspenders: reject known system roots.
    target_str = str(target_resolved)
    for blocked in _BLOCKED_SYSTEM_PREFIXES:
        if target_str == blocked or target_str.startswith(blocked + os.sep):
            raise SecurityError(
                f"Write to system directory blocked: '{target_resolved}'. "
                "The output_path must point to a user-writable directory."
            )

    logger.debug("safe_path OK: %s", target_resolved)
    return target_resolved


# ---------------------------------------------------------------------------
# 2.  Name sanitisation
# ---------------------------------------------------------------------------

# Allow only characters that are safe in filenames on Linux and Windows.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^a-zA-Z0-9._\-]")

# Characters that could corrupt log output (newlines, carriage returns, etc.)
_LOG_INJECTION_CHARS = re.compile(r"[\r\n\t\x00-\x1f\x7f]")


def sanitize_name(name: str, max_length: int = 128) -> str:
    """
    Strip dangerous characters from a name used in filenames or log output.

    Replaces any character outside ``[a-zA-Z0-9._-]`` with an underscore,
    strips control characters (including newlines), and truncates to
    *max_length* characters.

    Parameters
    ----------
    name : str
        Raw name, e.g. an entity_name or extractor name from a config.
    max_length : int
        Maximum allowed length for the result.

    Returns
    -------
    str
        A filename-safe, log-safe string.
    """
    if not name:
        return "_empty_"

    # Normalise Unicode to NFC, then strip all control characters.
    name = unicodedata.normalize("NFC", name)
    name = _LOG_INJECTION_CHARS.sub("", name)

    # Replace any remaining unsafe characters with underscores.
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)

    # Truncate to the maximum length.
    name = name[:max_length]

    return name or "_empty_"


def sanitize_log_value(value: str) -> str:
    """
    Strip newlines and control characters from a value before logging it.

    Use this when logging untrusted strings (entity names, docstrings, paths
    from external files) to prevent log injection attacks.

    Parameters
    ----------
    value : str
        Raw string from external input.

    Returns
    -------
    str
        The same string with all control characters removed.
    """
    return _LOG_INJECTION_CHARS.sub("", value)


# ---------------------------------------------------------------------------
# 3.  Safe JSON loading with size limit
# ---------------------------------------------------------------------------

# Default maximum JSON file size: 200 MB.
# Records larger than this are almost certainly corrupted or malicious.
DEFAULT_MAX_JSON_BYTES: int = 200 * 1024 * 1024  # 200 MB


def safe_json_load(
    filepath: str | Path,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> Any:
    """
    Load a JSON file, rejecting files that exceed *max_bytes*.

    Prevents an attacker from causing an OOM crash by placing a huge
    "record" file in the storage directory.

    Parameters
    ----------
    filepath : str | Path
        Path to the JSON or JSONL file.
    max_bytes : int
        Maximum allowed file size in bytes before loading.

    Returns
    -------
    Any
        Parsed JSON content.

    Raises
    ------
    SecurityError
        If the file is larger than *max_bytes*.
    json.JSONDecodeError
        If the file contains invalid JSON.
    OSError
        If the file cannot be opened.
    """
    path = Path(filepath)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0  # The actual open below will raise if the file does not exist.

    if size > max_bytes:
        raise SecurityError(
            f"JSON file is too large to load safely: '{path}' "
            f"({size:,} bytes > {max_bytes:,} byte limit). "
            "This may indicate data corruption or a malicious file."
        )

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 4.  Ollama URL validation (SSRF guard)
# ---------------------------------------------------------------------------

# Ollama should always run on localhost.
# Allowing remote hosts would let a crafted config file exfiltrate prompt
# data (which can contain proprietary code snippets and legal text).
_ALLOWED_OLLAMA_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
)


def validate_ollama_url(url: str) -> None:
    """
    Verify that the Ollama base URL points to a local server.

    Parameters
    ----------
    url : str
        The Ollama URL from config, e.g. "http://localhost:11434".

    Raises
    ------
    SecurityError
        If the URL points to a non-local host.
    ValueError
        If the URL cannot be parsed.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Cannot parse Ollama URL '{url}': {exc}") from exc

    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_OLLAMA_HOSTS:
        raise SecurityError(
            f"Ollama URL '{url}' points to a non-local host '{host}'. "
            "Only localhost / 127.0.0.1 are permitted because prompts may "
            "contain proprietary code or legal document text. "
            "Set 'ollama_url: http://localhost:11434' in your config."
        )

    logger.debug("Ollama URL validated: %s (host=%s)", url, host)


# ---------------------------------------------------------------------------
# 5.  Phase list validation
# ---------------------------------------------------------------------------

_KNOWN_PHASES: frozenset[str] = frozenset(
    {"catalog", "extract", "annotate", "organize", "convert", "llm"}
)


def validate_phase_list(phases: list[str]) -> list[str]:
    """
    Filter a list of phase names, keeping only the known and safe ones.

    Strips whitespace, rejects non-identifier strings (which could be
    shell injection attempts), and warns about unknown phase names.

    Parameters
    ----------
    phases : list[str]
        Phase names from config or the --phase CLI argument.

    Returns
    -------
    list[str]
        Only the valid, recognised phase names in the original order.
        Unknown or malformed phases are removed with a warning.
    """
    safe: list[str] = []
    for raw in phases:
        clean = raw.strip()
        # Reject empty strings and strings with shell-special characters.
        if not clean:
            continue
        if not clean.replace("_", "").isalnum():
            logger.warning(
                "Security: ignoring malformed phase name %r "
                "(only alphanumeric + underscore allowed)",
                clean,
            )
            continue
        if clean not in _KNOWN_PHASES:
            logger.warning(
                "Ignoring unknown phase '%s'. "
                "Known phases: %s",
                clean, ", ".join(sorted(_KNOWN_PHASES)),
            )
            continue
        safe.append(clean)
    return safe
