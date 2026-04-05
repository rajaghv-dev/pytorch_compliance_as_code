"""
Tests for src/security.py — path traversal, name sanitisation,
JSON size limit, and Ollama URL validation.

WHAT IS TESTED
--------------
- safe_path() blocks paths that escape the allowed root.
- safe_path() blocks writes to known system directories.
- safe_path() accepts legitimate subdirectory paths.
- sanitize_name() removes control characters and unsafe chars.
- sanitize_name() truncates long names.
- safe_json_load() raises SecurityError for oversized files.
- safe_json_load() parses normal JSON correctly.
- validate_ollama_url() accepts localhost variants.
- validate_ollama_url() rejects non-local hosts.
- validate_phase_list() filters unknown and malformed phase names.

HOW TO RUN
----------
    pytest tests/test_security.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.security import (
    SecurityError,
    sanitize_name,
    safe_json_load,
    safe_path,
    validate_ollama_url,
    validate_phase_list,
)


# ---------------------------------------------------------------------------
# safe_path()
# ---------------------------------------------------------------------------


class TestSafePath:

    def test_accepts_child_path(self, tmp_path):
        """A path that is a child of allowed_root should be accepted."""
        child = tmp_path / "output" / "file.json"
        result = safe_path(child, tmp_path)
        assert result == child.resolve()

    def test_blocks_path_traversal_dotdot(self, tmp_path):
        """A path using '../' to escape the root must be rejected."""
        escaped = tmp_path / "output" / ".." / ".." / "etc" / "passwd"
        with pytest.raises(SecurityError, match="traversal"):
            safe_path(escaped, tmp_path)

    def test_blocks_path_outside_root(self, tmp_path):
        """An absolute path outside allowed_root must be rejected."""
        # Use a path that is definitely outside tmp_path.
        outside = Path("/tmp/some_other_dir/file.txt")
        with pytest.raises(SecurityError):
            safe_path(outside, tmp_path)

    def test_blocks_etc_directory(self, tmp_path):
        """Writing to /etc must always be blocked regardless of allowed_root."""
        # This tests the belt-and-suspenders check.
        # We mock allowed_root to be "/" so the relative_to() check passes,
        # but the system-directory check should still catch it.
        with pytest.raises(SecurityError):
            safe_path("/etc/cron.d/evil", "/")

    def test_resolves_relative_paths(self, tmp_path):
        """safe_path() should resolve relative paths against the current dir."""
        # A relative path under tmp_path should be accepted.
        child = tmp_path / "subdir"
        child.mkdir()
        result = safe_path(str(child), str(tmp_path))
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# sanitize_name()
# ---------------------------------------------------------------------------


class TestSanitizeName:

    def test_alphanumeric_name_unchanged(self):
        """A name with only safe characters should not be modified."""
        assert sanitize_name("register_forward_hook") == "register_forward_hook"

    def test_removes_control_characters(self):
        """Newlines and other control chars must be stripped."""
        result = sanitize_name("evil\nname\r")
        assert "\n" not in result
        assert "\r" not in result

    def test_replaces_spaces_with_underscores(self):
        """Spaces should be replaced by underscores."""
        result = sanitize_name("my entity name")
        assert " " not in result
        assert "_" in result

    def test_replaces_path_separators(self):
        """Path separators must be replaced by underscores."""
        result = sanitize_name("torch/nn/module")
        assert "/" not in result

    def test_truncates_long_names(self):
        """Names longer than max_length must be truncated."""
        long_name = "a" * 200
        result = sanitize_name(long_name, max_length=128)
        assert len(result) <= 128

    def test_empty_string_returns_placeholder(self):
        """An empty name should return the safe placeholder string."""
        assert sanitize_name("") == "_empty_"

    def test_unicode_normalisation(self):
        """Unicode characters outside the safe set are replaced."""
        # The 'é' character is not in [a-zA-Z0-9._-].
        result = sanitize_name("résumé")
        # Should not raise; result should only contain safe chars.
        import re
        assert re.match(r"^[a-zA-Z0-9._\-]+$", result), (
            f"Unexpected chars in: {result!r}"
        )


# ---------------------------------------------------------------------------
# safe_json_load()
# ---------------------------------------------------------------------------


class TestSafeJsonLoad:

    def test_loads_valid_json(self, tmp_path):
        """A normal-sized JSON file should be loaded correctly."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = safe_json_load(path)
        assert loaded == data

    def test_rejects_oversized_file(self, tmp_path):
        """A file exceeding max_bytes must raise SecurityError."""
        big = tmp_path / "big.json"
        # Write 11 bytes; use a 10-byte limit.
        big.write_text(json.dumps({"x": "y" * 5}), encoding="utf-8")
        size = big.stat().st_size

        with pytest.raises(SecurityError, match="large"):
            safe_json_load(big, max_bytes=size - 1)

    def test_raises_on_invalid_json(self, tmp_path):
        """Malformed JSON must raise json.JSONDecodeError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            safe_json_load(bad)

    def test_raises_on_missing_file(self, tmp_path):
        """A missing file must raise OSError (or FileNotFoundError)."""
        with pytest.raises(OSError):
            safe_json_load(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# validate_ollama_url()
# ---------------------------------------------------------------------------


class TestValidateOllamaUrl:

    @pytest.mark.parametrize("url", [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://localhost:11434/api",
    ])
    def test_accepts_local_urls(self, url):
        """Local Ollama URLs should not raise."""
        validate_ollama_url(url)  # Should not raise.

    @pytest.mark.parametrize("url", [
        "http://ollama.example.com:11434",
        "http://10.0.0.1:11434",
        "http://192.168.1.100:11434",
        "https://remote-server.io/api",
    ])
    def test_rejects_remote_urls(self, url):
        """Remote Ollama URLs must raise SecurityError."""
        with pytest.raises(SecurityError):
            validate_ollama_url(url)


# ---------------------------------------------------------------------------
# validate_phase_list()
# ---------------------------------------------------------------------------


class TestValidatePhaseList:

    def test_keeps_valid_phases(self):
        """All known phase names should be kept."""
        phases = ["catalog", "extract", "annotate", "organize", "convert"]
        result = validate_phase_list(phases)
        assert result == phases

    def test_removes_unknown_phases(self):
        """Unknown phase names should be silently removed."""
        result = validate_phase_list(["catalog", "unknown_phase", "extract"])
        assert result == ["catalog", "extract"]

    def test_removes_malformed_names(self):
        """Names with shell-special characters must be removed."""
        result = validate_phase_list(["catalog", "$(evil)", "extract"])
        assert result == ["catalog", "extract"]

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped before validation."""
        result = validate_phase_list(["  catalog  ", " extract "])
        assert result == ["catalog", "extract"]

    def test_empty_list_returns_empty(self):
        """An empty input list should produce an empty output list."""
        assert validate_phase_list([]) == []
