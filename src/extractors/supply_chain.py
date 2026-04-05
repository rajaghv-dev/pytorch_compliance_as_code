"""
Supply Chain Extractor — extracts dependencies, licenses, build configs,
and CI workflows for software bill-of-materials (SBOM) compliance.

Compliance relevance:
    - All supply chain entries map to EU AI Act Article 17 (quality management).

Sources scanned:
    - ``setup.py``, ``pyproject.toml``, ``requirements*.txt`` — Python deps
    - ``CMakeLists.txt``, ``cmake/*.cmake`` — C++ / CMake deps
    - ``third_party/*/LICENSE*`` — third-party library licenses
    - ``.github/workflows/*.yml`` — CI/CD workflow configs
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any

from .base import EntityRecord, BaseExtractor, compute_stable_id

logger = logging.getLogger("pct.extractors.supply_chain")


class SupplyChainExtractor(BaseExtractor):
    """
    Extract dependencies, licenses, and build/CI configs from the PyTorch repo.

    Produces EntityRecords with ``language = "config"`` for YAML, TOML, and
    CMake sources.  All records are tagged with ``eu_ai_act_art_17`` (quality
    management system).
    """

    def __init__(self, repo_path: Path, output_path: Path) -> None:
        """
        Initialise the supply chain extractor.

        Parameters
        ----------
        repo_path : Path
            Root of the PyTorch repository checkout.
        output_path : Path
            Directory where output JSONL files are written.
        """
        super().__init__(name="supply_chain", repo_path=repo_path, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self) -> int:
        """
        Run the full supply chain extraction pass.

        Returns
        -------
        int
            Total number of records produced.
        """
        self.logger.info("Starting supply chain extraction")
        output_file = str(self.output_path / "supply_chain.jsonl")

        # Extract from each source category
        self._extract_python_deps(output_file)
        self._extract_cmake_deps(output_file)
        self._extract_third_party(output_file)
        self._extract_ci_configs(output_file)

        # Flush remaining buffered records
        self.flush(output_file)

        self.logger.info(
            "Supply chain extraction complete — %d records produced",
            self._records_produced,
        )
        self.report_stats()
        return self._records_produced

    # ------------------------------------------------------------------ #
    # Python dependencies
    # ------------------------------------------------------------------ #

    def _extract_python_deps(self, output_file: str) -> None:
        """
        Extract Python dependencies from ``setup.py`` and ``requirements*.txt``.

        Parses ``install_requires`` lists in setup.py via regex, and reads
        each line from requirements files.  Each dependency becomes an
        EntityRecord with ``entity_type = "dependency"``.
        """
        # --- setup.py ---
        for filepath in self.find_files("setup.py"):
            content = self.read_file_safe(filepath)
            if content is None:
                self.logger.warning("Skipping unreadable file: %s", filepath)
                continue

            self._files_processed += 1
            rel_path = str(filepath.relative_to(self.repo_path))

            # Regex to capture install_requires list content
            for match in re.finditer(
                r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL
            ):
                deps = re.findall(r'["\']([^"\']+)["\']', match.group(1))
                for dep in deps:
                    # Strip version specifier to get bare package name
                    dep_name = re.split(r'[>=<!\[]', dep)[0].strip()
                    if not dep_name:
                        continue

                    record = self.make_record(
                        source_file=rel_path,
                        language="config",
                        entity_name=dep_name,
                        entity_type="dependency",
                        subcategory="python_dependency",
                        module_path="setup.py",
                        qualified_name=f"dep.python.{dep_name}",
                        start_line=0,
                        end_line=0,
                        raw_text=dep,
                        compliance_tags=["eu_ai_act_art_17"],
                        extraction_confidence=1.0,
                        metadata={"source": "setup.py", "spec": dep},
                    )
                    self.write_record(record, output_file)

        # --- requirements*.txt files ---
        for filepath in self.find_files("requirements*.txt"):
            content = self.read_file_safe(filepath)
            if content is None:
                self.logger.warning("Skipping unreadable file: %s", filepath)
                continue

            self._files_processed += 1
            rel_path = str(filepath.relative_to(self.repo_path))

            for line_num, line in enumerate(content.splitlines(), start=1):
                line = line.strip()
                # Skip empty lines, comments, and option flags
                if not line or line.startswith("#") or line.startswith("-"):
                    continue

                dep_name = re.split(r'[>=<!\[]', line)[0].strip()
                if not dep_name:
                    continue

                record = self.make_record(
                    source_file=rel_path,
                    language="config",
                    entity_name=dep_name,
                    entity_type="dependency",
                    subcategory="python_dependency",
                    module_path=filepath.name,
                    qualified_name=f"dep.python.{dep_name}",
                    start_line=line_num,
                    end_line=line_num,
                    raw_text=line,
                    compliance_tags=["eu_ai_act_art_17"],
                    extraction_confidence=1.0,
                    metadata={"source": filepath.name, "spec": line},
                )
                self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # CMake / C++ dependencies
    # ------------------------------------------------------------------ #

    def _extract_cmake_deps(self, output_file: str) -> None:
        """
        Extract C++ dependencies from ``CMakeLists.txt`` and ``cmake/*.cmake``.

        Looks for ``find_package(...)`` and ``option(...)`` directives.
        """
        # Compiled regex patterns for CMake directives
        find_pkg_re = re.compile(r'find_package\s*\(\s*(\w+)', re.MULTILINE)
        option_re = re.compile(r'option\s*\(\s*(\w+)\s+"([^"]*)"', re.MULTILINE)

        # Search both CMakeLists.txt and cmake/*.cmake files
        cmake_patterns = ["CMakeLists.txt", "cmake/*.cmake"]

        for pattern in cmake_patterns:
            for filepath in self.find_files(pattern):
                content = self.read_file_safe(filepath)
                if content is None:
                    self.logger.warning("Skipping unreadable file: %s", filepath)
                    continue

                self._files_processed += 1
                rel_path = str(filepath.relative_to(self.repo_path))

                # --- find_package directives ---
                for match in find_pkg_re.finditer(content):
                    pkg_name = match.group(1)
                    line_num = content[:match.start()].count("\n") + 1

                    record = self.make_record(
                        source_file=rel_path,
                        language="config",
                        entity_name=pkg_name,
                        entity_type="dependency",
                        subcategory="cmake_dependency",
                        module_path=rel_path,
                        qualified_name=f"dep.cmake.{pkg_name}",
                        start_line=line_num,
                        end_line=line_num,
                        raw_text=match.group(0),
                        compliance_tags=["eu_ai_act_art_17"],
                        extraction_confidence=1.0,
                        metadata={"directive": "find_package"},
                    )
                    self.write_record(record, output_file)

                # --- option directives ---
                for match in option_re.finditer(content):
                    opt_name = match.group(1)
                    opt_desc = match.group(2)
                    line_num = content[:match.start()].count("\n") + 1

                    record = self.make_record(
                        source_file=rel_path,
                        language="config",
                        entity_name=opt_name,
                        entity_type="config_entry",
                        subcategory="cmake_option",
                        module_path=rel_path,
                        qualified_name=f"cmake.option.{opt_name}",
                        start_line=line_num,
                        end_line=line_num,
                        raw_text=match.group(0),
                        compliance_tags=["eu_ai_act_art_17"],
                        extraction_confidence=1.0,
                        metadata={"directive": "option", "description": opt_desc},
                    )
                    self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # Third-party licenses
    # ------------------------------------------------------------------ #

    def _extract_third_party(self, output_file: str) -> None:
        """
        Extract third-party library license metadata from ``third_party/``.

        Scans each subdirectory for ``LICENSE*`` files and attempts to identify
        the license type by checking the first 500 characters.
        """
        third_party = self.repo_path / "third_party"
        if not third_party.exists():
            self.logger.info("No third_party directory found — skipping")
            return

        for subdir in sorted(third_party.iterdir()):
            if not subdir.is_dir():
                continue

            for license_file in subdir.glob("LICENSE*"):
                content = self.read_file_safe(license_file)
                if content is None:
                    self.logger.warning("Skipping unreadable license: %s", license_file)
                    continue

                self._files_processed += 1

                # Heuristic license type detection from the header
                header = content[:500]
                license_type = "unknown"
                if "MIT" in header:
                    license_type = "MIT"
                elif "Apache" in header:
                    license_type = "Apache-2.0"
                elif "BSD" in header:
                    license_type = "BSD"
                elif "GPL" in header:
                    license_type = "GPL"

                rel_path = str(license_file.relative_to(self.repo_path))

                record = self.make_record(
                    source_file=rel_path,
                    language="config",
                    entity_name=subdir.name,
                    entity_type="license",
                    subcategory="third_party_license",
                    module_path=f"third_party.{subdir.name}",
                    qualified_name=f"license.{subdir.name}",
                    start_line=1,
                    end_line=content.count("\n"),
                    raw_text=content[:2000],
                    compliance_tags=["eu_ai_act_art_17"],
                    extraction_confidence=1.0,
                    metadata={
                        "library": subdir.name,
                        "license_type": license_type,
                    },
                )
                self.write_record(record, output_file)

    # ------------------------------------------------------------------ #
    # CI/CD workflow configs
    # ------------------------------------------------------------------ #

    def _extract_ci_configs(self, output_file: str) -> None:
        """
        Extract CI/CD workflow configurations from ``.github/workflows/*.yml``.

        Each workflow YAML file is emitted as a single ``config_entry`` record.
        """
        for filepath in self.find_files(".github/workflows/*.yml"):
            content = self.read_file_safe(filepath)
            if content is None:
                self.logger.warning("Skipping unreadable CI config: %s", filepath)
                continue

            self._files_processed += 1
            rel_path = str(filepath.relative_to(self.repo_path))

            record = self.make_record(
                source_file=rel_path,
                language="config",
                entity_name=filepath.stem,
                entity_type="config_entry",
                subcategory="ci_config",
                module_path=".github.workflows",
                qualified_name=f"ci.workflow.{filepath.stem}",
                start_line=1,
                end_line=content.count("\n"),
                raw_text=content[:5000],
                compliance_tags=["eu_ai_act_art_17"],
                extraction_confidence=1.0,
                metadata={"filename": filepath.name},
            )
            self.write_record(record, output_file)
