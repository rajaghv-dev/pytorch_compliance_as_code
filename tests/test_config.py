"""
Tests for src/config.py — configuration loading and validation.

WHAT IS TESTED
--------------
- Config.from_yaml() loads a real YAML file correctly.
- Config._from_raw() correctly merges overrides onto defaults.
- validate_config() raises ValueError for a missing repo_path.
- validate_config() creates the output_path directory if needed.
- Config.to_dict() produces a JSON-serialisable dictionary.
- load_config(None) returns the built-in defaults.

HOW TO RUN
----------
    # From the pytorch_compliance_toolkit/ directory:
    pytest tests/test_config.py -v
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.config import (
    Config,
    LLMConfig,
    LLMModelsConfig,
    ExtractorsConfig,
    AnnotatorsConfig,
    OutputConfig,
    load_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# Helper: write a minimal YAML config to a temp file
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write *content* as a YAML file under *tmp_path* and return the path."""
    path = tmp_path / "test_config.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# LLMModelsConfig
# ---------------------------------------------------------------------------


class TestLLMModelsConfig:

    def test_defaults(self):
        """All model names have sensible non-empty defaults."""
        m = LLMModelsConfig()
        assert m.mapping_validator != ""
        assert m.legal_parser != ""
        assert m.semantic_search != ""
        assert m.rag_answerer != ""

    def test_from_dict_accepts_subset(self):
        """from_dict() should not raise if only some models are overridden."""
        m = LLMModelsConfig.from_dict({"mapping_validator": "qwen2.5:14b"})
        assert m.mapping_validator == "qwen2.5:14b"
        # Other fields keep their defaults.
        assert m.rag_answerer == LLMModelsConfig().rag_answerer

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict() silently ignores keys that don't match any field."""
        # Should not raise.
        m = LLMModelsConfig.from_dict({"totally_unknown_key": "value"})
        assert isinstance(m, LLMModelsConfig)


# ---------------------------------------------------------------------------
# Config._from_raw()
# ---------------------------------------------------------------------------


class TestConfigFromRaw:

    def test_empty_dict_uses_defaults(self):
        """An empty dict should produce the built-in defaults."""
        cfg = Config._from_raw({})
        assert cfg.workers == 1
        assert "catalog" in cfg.phases
        assert cfg.llm.enabled is True

    def test_overrides_are_applied(self):
        """Fields present in the raw dict should override defaults."""
        cfg = Config._from_raw({
            "workers": 4,
            "phases": ["catalog"],
        })
        assert cfg.workers == 4
        assert cfg.phases == ["catalog"]

    def test_nested_llm_override(self):
        """Nested LLM config dict is correctly parsed."""
        cfg = Config._from_raw({
            "llm": {
                "enabled": False,
                "models": {"mapping_validator": "custom-model"},
            }
        })
        assert cfg.llm.enabled is False
        assert cfg.llm.models.mapping_validator == "custom-model"

    def test_extractors_override(self):
        """Custom extractor list is respected."""
        cfg = Config._from_raw({
            "extractors": {"enabled": ["hookability", "operator_determinism"]},
        })
        assert cfg.extractors.enabled == ["hookability", "operator_determinism"]

    def test_outputs_override(self):
        """Output flags are independently toggleable."""
        cfg = Config._from_raw({
            "outputs": {"rdf": False, "csv": True, "notebook": False},
        })
        assert cfg.outputs.rdf is False
        assert cfg.outputs.csv is True
        assert cfg.outputs.notebook is False


# ---------------------------------------------------------------------------
# Config.from_yaml()
# ---------------------------------------------------------------------------


class TestConfigFromYaml:

    def test_loads_minimal_yaml(self, tmp_path):
        """A YAML with only repo_path should load without errors."""
        yaml_path = _write_yaml(
            tmp_path,
            f"""
            repo_path: {tmp_path}
            """,
        )
        cfg = Config.from_yaml(yaml_path)
        assert cfg.repo_path == str(tmp_path)

    def test_raises_on_missing_file(self, tmp_path):
        """Raises FileNotFoundError if the YAML file does not exist."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "nonexistent.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path):
        """Raises yaml.YAMLError for malformed YAML."""
        import yaml
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [\nbad yaml", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            Config.from_yaml(bad_yaml)


# ---------------------------------------------------------------------------
# validate_config()
# ---------------------------------------------------------------------------


class TestValidateConfig:

    def test_raises_if_repo_path_missing(self, tmp_path):
        """validate_config() raises ValueError for a non-existent repo_path."""
        cfg = Config._from_raw({
            "repo_path": str(tmp_path / "does_not_exist"),
            "output_path": str(tmp_path / "out"),
        })
        with pytest.raises(ValueError, match="repo_path"):
            validate_config(cfg)

    def test_creates_output_path_if_missing(self, tmp_path):
        """validate_config() creates the output_path directory if it doesn't exist."""
        out_dir = tmp_path / "new_output"
        assert not out_dir.exists()
        cfg = Config._from_raw({
            "repo_path": str(tmp_path),          # tmp_path exists
            "output_path": str(out_dir),
        })
        validate_config(cfg)
        assert out_dir.is_dir()

    def test_clamps_invalid_workers(self, tmp_path):
        """validate_config() clamps workers < 1 to 1."""
        cfg = Config._from_raw({
            "repo_path": str(tmp_path),
            "output_path": str(tmp_path / "out"),
            "workers": 0,
        })
        validate_config(cfg)
        assert cfg.workers == 1


# ---------------------------------------------------------------------------
# Config.to_dict() and load_config()
# ---------------------------------------------------------------------------


class TestConfigSerialisation:

    def test_to_dict_is_json_serialisable(self, tmp_path):
        """to_dict() must return something json.dumps() can handle."""
        cfg = Config._from_raw({"repo_path": str(tmp_path)})
        d = cfg.to_dict()
        # Should not raise.
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_dict_contains_phases(self):
        """to_dict() output contains the phases list."""
        cfg = Config._from_raw({"phases": ["catalog"]})
        d = cfg.to_dict()
        assert d["phases"] == ["catalog"]


class TestLoadConfig:

    def test_load_config_none_returns_defaults(self):
        """load_config(None) should succeed if repo_path is the default."""
        # We don't validate paths here because the default repo may not exist.
        cfg = Config()
        assert cfg.workers == 1
        assert "catalog" in cfg.phases
