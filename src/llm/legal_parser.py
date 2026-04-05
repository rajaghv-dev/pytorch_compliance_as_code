"""
Parse EU AI Act and GDPR article texts into structured obligations using qwen2.5:14b.

Reads legal article files from data/legal/, chunks them, sends each chunk to
qwen2.5:14b for obligation extraction, and stores the structured output in
storage/organized/legal_obligations.json.

EU AI Act and GDPR are distinct legal frameworks with different obligations,
scopes, and enforcement bodies. They are parsed separately and stored under
separate top-level keys ("eu_ai_act" and "gdpr") in the output JSON.

Model choice: qwen2.5:14b outperforms mistral-nemo on structured extraction
from dense legal text — higher parameter count gives better coverage of nested
obligation lists and cross-article references.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .ollama_client import OllamaClient, OllamaError


logger = logging.getLogger("legal_parser")

# qwen3.5:27b: strong structured extraction on dense legal text with improved
# reasoning for nested obligation structures and cross-article dependencies.
# EU AI Act and GDPR are separate legal frameworks; both parsed by the same model
# but stored under distinct keys so downstream consumers never conflate them.
MODEL = "qwen3.5:27b"

# Maximum characters per chunk sent to the LLM
CHUNK_SIZE = 2000


class LegalParser:
    """Extracts structured obligations from legal text using mistral-nemo."""

    # All available EU AI Act chapter files, ordered by chapter number.
    # Chapters 04-06 are the core high-risk system obligations; 01-03 and 07
    # provide definitional and innovation-support context.
    EU_AI_ACT_ARTICLES: list[str] = [
        "eu_ai_act/chapters/01_preamble.md",
        "eu_ai_act/chapters/02_general_provisions.md",
        "eu_ai_act/chapters/03_prohibited_practices.md",
        "eu_ai_act/chapters/04_high_risk_systems.md",
        "eu_ai_act/chapters/05_obligations.md",
        "eu_ai_act/chapters/06_transparency.md",
        "eu_ai_act/chapters/07_innovation.md",
    ]

    # All available GDPR chapter files.
    # Chapters 02-03 cover core data-processing obligations; 01, 04, 05 cover
    # scope, controller/processor duties, and cross-border transfers.
    GDPR_ARTICLES: list[str] = [
        "gdpr/chapters/01_general_provisions.md",
        "gdpr/chapters/02_principles.md",
        "gdpr/chapters/03_data_subject_rights.md",
        "gdpr/chapters/04_controller_processor.md",
        "gdpr/chapters/05_transfers.md",
    ]

    def __init__(self, client: OllamaClient, storage_dir: Path):
        """
        Initialise the legal parser.

        Parameters
        ----------
        client : OllamaClient
            Configured Ollama client instance.
        storage_dir : Path
            Root storage directory for output files.
        """
        self.client = client
        self.storage_dir = storage_dir
        self.output_path = storage_dir / "organized" / "legal_obligations.json"
        self.logger = logging.getLogger("legal_parser")

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
        """Split text into chunks of approximately chunk_size characters.

        Splits on paragraph boundaries where possible to preserve context.

        Parameters
        ----------
        text : str
            Legal text to chunk.
        chunk_size : int
            Target chunk size in characters.

        Returns
        -------
        list[str]
            List of text chunks.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _parse_chunk(self, chunk: str) -> list[dict[str, Any]]:
        """Send a single text chunk to mistral-nemo for obligation extraction.

        Parameters
        ----------
        chunk : str
            Legal text chunk to parse.

        Returns
        -------
        list[dict]
            List of obligation dicts, or empty list on failure.
        """
        system_prompt = (
            "/no_think\n"
            "You are a legal analyst specialising in EU technology regulation. "
            "Extract all specific legal obligations from the provided article text as a JSON array. "
            "EU AI Act and GDPR are distinct legal frameworks with different scopes — "
            "do not conflate their obligations. "
            "For each obligation output: "
            '{"id": "Art15-3a", "framework": "EU AI Act"|"GDPR", "text": "...", '
            '"applies_to": "...", "requires": "...", "evidence_type": "..."}\n'
            "Respond with ONLY the JSON array, no explanation. No thinking or reasoning."
        )

        try:
            start_time = time.time()
            result = self.client.generate(
                model=MODEL,
                prompt=chunk,
                system=system_prompt,
                temperature=0.1,
                format="json",
                max_retries=3,
            )
            elapsed = time.time() - start_time
            self.logger.info(
                "Parsed chunk (%d chars) in %.2fs", len(chunk), elapsed,
            )

            # Result may be a list (desired) or a dict wrapping a list
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # Some models wrap the array in an object
                for key in ("obligations", "items", "results", "data"):
                    if key in result and isinstance(result[key], list):
                        return result[key]
                # If the dict itself looks like a single obligation, wrap it
                if "id" in result and "text" in result:
                    return [result]
                self.logger.warning(
                    "WARN: Unexpected dict structure from LLM: %s",
                    list(result.keys()),
                )
                return []
            else:
                self.logger.warning("WARN: Unexpected response type: %s", type(result))
                return []

        except OllamaError as e:
            self.logger.warning("WARN: Failed to parse chunk: %s", e)
            return []
        except Exception as e:
            self.logger.warning("WARN: Unexpected error parsing chunk: %s", e)
            return []

    def _extract_article_id(self, obligation_id: str) -> str:
        """Extract the article number from an obligation ID like 'Art15-3a'.

        Parameters
        ----------
        obligation_id : str
            Obligation ID string.

        Returns
        -------
        str
            Article key like "Art15".
        """
        # Try to extract "ArtNN" prefix
        import re
        match = re.match(r"(Art\d+)", obligation_id)
        if match:
            return match.group(1)
        return "Unknown"

    def parse_articles(self, data_dir: Path) -> dict[str, dict[str, list[dict]]]:
        """Parse all configured legal article files into structured obligations.

        Parameters
        ----------
        data_dir : Path
            Root data directory containing legal/ subdirectory.

        Returns
        -------
        dict
            Nested structure: {"eu_ai_act": {"Art15": [...], ...}, "gdpr": {...}}
        """
        # Check model availability
        if not self.client.check_model_available(MODEL):
            self.logger.warning("WARN: Model %s not available, skipping legal parsing", MODEL)
            return {}

        # Check for existing output (resume support)
        if self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if existing:
                    self.logger.info("Loaded existing legal obligations from %s", self.output_path)
                    return existing
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning("WARN: Could not load existing obligations: %s", e)

        result: dict[str, dict[str, list[dict]]] = {
            "eu_ai_act": {},
            "gdpr": {},
        }

        # data_dir is already the legal root (e.g. "data/legal/") — the
        # chapter paths in EU_AI_ACT_ARTICLES / GDPR_ARTICLES are relative
        # to it, so no further "legal" segment is needed.
        legal_dir = data_dir

        # Parse EU AI Act articles
        self.logger.info("Parsing EU AI Act articles...")
        for article_path_str in self.EU_AI_ACT_ARTICLES:
            full_path = legal_dir / article_path_str
            self._parse_file(full_path, result["eu_ai_act"])

        # Parse GDPR articles
        self.logger.info("Parsing GDPR articles...")
        for article_path_str in self.GDPR_ARTICLES:
            full_path = legal_dir / article_path_str
            self._parse_file(full_path, result["gdpr"])

        # Save output
        self._save_output(result)

        return result

    def _parse_file(
        self,
        file_path: Path,
        target: dict[str, list[dict]],
    ) -> None:
        """Parse a single legal article file and merge obligations into target.

        Parameters
        ----------
        file_path : Path
            Path to the markdown file.
        target : dict
            Dict to merge results into, keyed by article ID.
        """
        if not file_path.exists():
            self.logger.warning("WARN: Legal file not found: %s", file_path)
            return

        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as e:
            self.logger.error("ERR: Could not read %s: %s", file_path, e)
            return

        self.logger.info("Parsing %s (%d chars)", file_path.name, len(text))

        chunks = self._chunk_text(text)
        self.logger.info("Split into %d chunks", len(chunks))

        for i, chunk in enumerate(chunks):
            obligations = self._parse_chunk(chunk)
            self.logger.info(
                "Chunk %d/%d: extracted %d obligations",
                i + 1, len(chunks), len(obligations),
            )

            for obligation in obligations:
                ob_id = obligation.get("id", "Unknown")
                article_key = self._extract_article_id(ob_id)
                if article_key not in target:
                    target[article_key] = []
                target[article_key].append(obligation)

    def _save_output(self, result: dict) -> None:
        """Save parsed obligations to storage.

        Parameters
        ----------
        result : dict
            The obligations structure to persist.
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self.logger.info("Saved legal obligations to %s", self.output_path)
        except OSError as e:
            self.logger.error("ERR: Failed to save obligations: %s", e)
