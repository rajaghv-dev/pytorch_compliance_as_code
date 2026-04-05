"""
Configuration loader for the PyTorch Compliance Toolkit.

Loads YAML configuration, merges over sensible defaults, and validates that
required paths and model references are plausible before the pipeline starts.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("pct.config")

# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMModelsConfig:
    """Model identifiers used by the various LLM-powered stages."""

    mapping_validator: str = "phi4"
    legal_parser: str = "mistral-nemo"
    semantic_search: str = "nomic-embed-text"
    rag_answerer: str = "llama3.1"
    commit_classifier: str = "qwen2.5-coder:7b"
    cpp_translator_stage1: str = "qwen2.5-coder:7b"
    cpp_translator_stage2: str = "qwen2.5:14b"

    @classmethod
    def from_dict(cls, d: dict) -> "LLMModelsConfig":
        """Create an LLMModelsConfig from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class LLMConfig:
    """Top-level LLM configuration block."""

    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    models: LLMModelsConfig = field(default_factory=LLMModelsConfig)
    rebel_large: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        """Create an LLMConfig from a dictionary, handling nested models."""
        models_raw = d.get("models", {})
        models = LLMModelsConfig.from_dict(models_raw) if isinstance(models_raw, dict) else LLMModelsConfig()
        return cls(
            enabled=d.get("enabled", True),
            ollama_url=d.get("ollama_url", "http://localhost:11434"),
            models=models,
            rebel_large=d.get("rebel_large", True),
        )


@dataclass
class ExtractorsConfig:
    """Which extractors to run."""

    enabled: list = field(default_factory=lambda: [
        "hookability", "operator_determinism", "export_boundary",
        "data_provenance", "module_hierarchy", "supply_chain",
        "api_documentation", "sphinx_notes", "test_suite",
        "compliance_tools", "commit_history",
    ])

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractorsConfig":
        """Create an ExtractorsConfig from a dictionary."""
        return cls(enabled=d.get("enabled", cls.__dataclass_fields__["enabled"].default_factory()))


@dataclass
class AnnotatorsConfig:
    """Which annotators to run."""

    enabled: list = field(default_factory=lambda: [
        "compliance_tagger", "dispatcher_level", "distributed_safety",
        "lifecycle", "export_survival", "confidence", "hook_consumers",
    ])

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotatorsConfig":
        """Create an AnnotatorsConfig from a dictionary."""
        return cls(enabled=d.get("enabled", cls.__dataclass_fields__["enabled"].default_factory()))


@dataclass
class OutputConfig:
    """Which output formats to produce."""

    rdf: bool = True
    markdown: bool = True
    csv: bool = True
    notebook: bool = True
    talk_assets: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "OutputConfig":
        """Create an OutputConfig from a dictionary."""
        return cls(
            rdf=d.get("rdf", True),
            markdown=d.get("markdown", True),
            csv=d.get("csv", True),
            notebook=d.get("notebook", True),
            talk_assets=d.get("talk_assets", True),
        )


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

# Default paths — used when no YAML is provided or a field is missing.
_DEFAULT_REPO_PATH = "/home/raja/gemini-torch/pytorch"
_DEFAULT_LEGAL_PATH = "/home/raja/compliance-as-code/pytorch_compliance_toolkit/data/legal"
_DEFAULT_OUTPUT_PATH = "./storage"

# Pipeline phases in canonical order.
_DEFAULT_PHASES = ["catalog", "extract", "annotate", "organize", "convert"]


@dataclass
class Config:
    """
    Root configuration object for the PyTorch Compliance Toolkit.

    Prefer ``Config.from_yaml(path)`` to build from a YAML file, which
    merges user overrides on top of built-in defaults.
    """

    repo_path: str = _DEFAULT_REPO_PATH
    legal_path: str = _DEFAULT_LEGAL_PATH
    output_path: str = _DEFAULT_OUTPUT_PATH
    phases: list = field(default_factory=lambda: list(_DEFAULT_PHASES))
    workers: int = 1
    llm: LLMConfig = field(default_factory=LLMConfig)
    extractors: ExtractorsConfig = field(default_factory=ExtractorsConfig)
    annotators: AnnotatorsConfig = field(default_factory=AnnotatorsConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    # ---- construction helpers ----

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Load a YAML config file and merge it over the built-in defaults.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        Config
            Fully populated configuration object.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        yaml.YAMLError
            If the file contains invalid YAML.
        """
        path = Path(path)
        logger.info("Loading configuration from %s", path)

        if not path.exists():
            logger.error("Configuration file not found: %s", path)
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw: dict = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse YAML configuration: %s", exc)
            raise

        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict) -> "Config":
        """Build a Config from a raw dictionary (already parsed YAML)."""
        llm_raw = raw.get("llm", {})
        llm = LLMConfig.from_dict(llm_raw) if isinstance(llm_raw, dict) else LLMConfig()

        ext_raw = raw.get("extractors", {})
        extractors = ExtractorsConfig.from_dict(ext_raw) if isinstance(ext_raw, dict) else ExtractorsConfig()

        ann_raw = raw.get("annotators", {})
        annotators = AnnotatorsConfig.from_dict(ann_raw) if isinstance(ann_raw, dict) else AnnotatorsConfig()

        out_raw = raw.get("outputs", {})
        outputs = OutputConfig.from_dict(out_raw) if isinstance(out_raw, dict) else OutputConfig()

        return cls(
            repo_path=raw.get("repo_path", _DEFAULT_REPO_PATH),
            legal_path=raw.get("legal_path", _DEFAULT_LEGAL_PATH),
            output_path=raw.get("output_path", _DEFAULT_OUTPUT_PATH),
            phases=raw.get("phases", list(_DEFAULT_PHASES)),
            workers=raw.get("workers", 1),
            llm=llm,
            extractors=extractors,
            annotators=annotators,
            outputs=outputs,
        )

    # ---- serialisation ----

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary (for checkpoint storage).

        Returns
        -------
        dict
            JSON-safe dictionary representation.
        """
        return asdict(self)

    # ---- pretty printing ----

    def summary(self) -> str:
        """Return a human-readable summary of the key settings."""
        lines = [
            "=== PyTorch Compliance Toolkit — Config Summary ===",
            f"  repo_path   : {self.repo_path}",
            f"  legal_path  : {self.legal_path}",
            f"  output_path : {self.output_path}",
            f"  phases      : {', '.join(self.phases)}",
            f"  workers     : {self.workers}",
            f"  llm.enabled : {self.llm.enabled}",
            f"  extractors  : {len(self.extractors.enabled)} enabled",
            f"  annotators  : {len(self.annotators.enabled)} enabled",
            f"  outputs     : rdf={self.outputs.rdf} md={self.outputs.markdown} "
            f"csv={self.outputs.csv} nb={self.outputs.notebook} talk={self.outputs.talk_assets}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> Config:
    """
    Convenience function: load a YAML config or return defaults.

    Parameters
    ----------
    path : str | Path | None
        Optional path to a YAML file.  If ``None``, built-in defaults are used.

    Returns
    -------
    Config
        Validated configuration object.
    """
    if path is None:
        logger.info("No config file specified — using built-in defaults")
        cfg = Config()
    else:
        cfg = Config.from_yaml(path)

    validate_config(cfg)
    return cfg


def validate_config(config: Config) -> None:
    """
    Validate the loaded configuration.

    Checks that required filesystem paths exist, that phase names are
    recognised, and that extractor / annotator names look reasonable.
    Logs INFO for each successful check and WARN / ERR for problems.

    Parameters
    ----------
    config : Config
        The configuration object to validate.

    Raises
    ------
    ValueError
        If a critical validation check fails (e.g. repo_path missing).
    """
    logger.info("Validating configuration ...")

    # -- repo_path --
    repo = Path(config.repo_path)
    if repo.is_dir():
        logger.info("  repo_path exists: %s", repo)
    else:
        logger.error("  repo_path does NOT exist: %s", repo)
        raise ValueError(
            f"repo_path does not exist: {repo}. "
            "Please clone the PyTorch repo or update the config."
        )

    # -- legal_path --
    legal = Path(config.legal_path)
    if legal.is_dir():
        logger.info("  legal_path exists: %s", legal)
    else:
        logger.warning(
            "  legal_path does not exist: %s — legal extractors will be skipped",
            legal,
        )

    # -- output_path (create if missing) --
    out = Path(config.output_path)
    if not out.exists():
        logger.info("  Creating output_path: %s", out)
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("  Failed to create output_path %s: %s", out, exc)
            raise ValueError(f"Cannot create output_path: {out}") from exc
    else:
        logger.info("  output_path exists: %s", out)

    # -- phases --
    known_phases = {"catalog", "extract", "annotate", "organize", "convert", "llm"}
    for phase in config.phases:
        if phase not in known_phases:
            logger.warning("  Unknown phase '%s' — it will be ignored by the runner", phase)
        else:
            logger.info("  Phase '%s' is recognised", phase)

    # -- workers --
    if config.workers < 1:
        logger.warning("  workers=%d is invalid, clamping to 1", config.workers)
        config.workers = 1
    else:
        logger.info("  workers=%d", config.workers)

    # -- LLM models (light check: non-empty strings) --
    if config.llm.enabled:
        models = config.llm.models
        for fname in LLMModelsConfig.__dataclass_fields__:
            val = getattr(models, fname)
            if not val or not isinstance(val, str):
                logger.warning("  LLM model '%s' is empty or invalid: %r", fname, val)
            else:
                # DEBUG: model names are low-value noise at INFO level
                logger.debug("  LLM model %-25s = %s", fname, val)
    else:
        logger.info("  LLM disabled — skipping model validation")

    # -- extractors --
    known_extractors = {
        "hookability", "operator_determinism", "export_boundary",
        "data_provenance", "module_hierarchy", "supply_chain",
        "api_documentation", "sphinx_notes", "test_suite",
        "compliance_tools", "commit_history",
    }
    for ext in config.extractors.enabled:
        if ext not in known_extractors:
            logger.warning("  Unknown extractor '%s'", ext)
        else:
            logger.info("  Extractor '%s' enabled", ext)

    # -- annotators --
    known_annotators = {
        "compliance_tagger", "dispatcher_level", "distributed_safety",
        "lifecycle", "export_survival", "confidence", "hook_consumers",
    }
    for ann in config.annotators.enabled:
        if ann not in known_annotators:
            logger.warning("  Unknown annotator '%s'", ann)
        else:
            logger.info("  Annotator '%s' enabled", ann)

    logger.info("Configuration validation complete.")
