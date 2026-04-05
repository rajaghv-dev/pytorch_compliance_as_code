"""
Semantic relation extraction via HuggingFace REBEL-large.

WHAT IT DOES
------------
For every entity with a non-empty docstring, this module runs the docstring
through REBEL-large (Babelscape/rebel-large) to extract semantic triples of
the form (subject, relation, object).  The triples are appended to the
entity's `relations` list with source="rebel-large".

WHY REBEL-LARGE (not Ollama)
-----------------------------
REBEL-large is a dedicated seq2seq information-extraction model fine-tuned
on Wikipedia relation triples.  It produces structured (S, R, O) output
that maps naturally to our RDF knowledge graph.  Ollama's instruction-tuned
models would require extensive prompt engineering to match this reliability.

REBEL OUTPUT FORMAT
-------------------
REBEL outputs a tagged string:
  <triplet> subject <subj> object <obj> relation <triplet> ...

Multiple triplets can appear in one output, separated by <triplet>.

GPU USAGE
---------
Uses gpu_monitor.get_torch_device() so inference runs on GPU when available
and temperature is safe, falling back to CPU automatically.

USAGE
-----
    from src.llm.relation_extractor import RelationExtractor

    extractor = RelationExtractor()
    records = extractor.extract_all(records)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.llm.relation_extractor")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

# HuggingFace model for relation extraction.
_MODEL_NAME = "Babelscape/rebel-large"

# Maximum input length for REBEL (tokens). Docstrings are truncated to this.
_MAX_INPUT_CHARS = 512

# Regex to parse REBEL's tagged output into (subject, relation, object) triples.
# Actual REBEL format: <triplet> SUBJ <subj> OBJ <obj> REL
# (the <subj>/<obj> tags separate subject→object→relation, not subject→relation→object)
_REBEL_RE = re.compile(
    r"<triplet>\s*(?P<subj>[^<]+?)\s*<subj>\s*(?P<obj>[^<]+?)\s*<obj>\s*(?P<rel>[^<]+?)(?=\s*<triplet>|\s*</s>|\s*$)",
    re.DOTALL,
)


# ----------------------------------------------------------------------- #
# RelationExtractor
# ----------------------------------------------------------------------- #

class RelationExtractor:
    """
    Extracts semantic triples from entity docstrings using REBEL-large.

    The model is loaded lazily on the first call to extract_all() or
    extract_one() to avoid importing HuggingFace at import time (which
    would slow down CLI startup).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (default: Babelscape/rebel-large).
    batch_size : int
        Number of docstrings to process in one pipeline() call.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._tokenizer: Any = None   # Lazy-loaded HuggingFace tokenizer.
        self._model: Any = None       # Lazy-loaded seq2seq model.
        self._device: Any = None      # torch.device selected by gpu_monitor.

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract_all(
        self, records: list["EntityRecord"]
    ) -> list["EntityRecord"]:
        """
        Run relation extraction on all records with a non-empty docstring.

        Parameters
        ----------
        records : list[EntityRecord]
            All entity records.

        Returns
        -------
        list[EntityRecord]
            Same list with `relations` lists extended in-place.
        """
        eligible = [r for r in records if r.docstring]
        total = len(eligible)
        logger.info(
            "RelationExtractor: extracting relations from %d docstrings "
            "using %s …",
            total,
            self.model_name,
        )

        # Lazy-load the model.
        self._load_pipeline()
        if self._tokenizer is None or self._model is None:
            logger.warning(
                "RelationExtractor: model not loaded — skipping extraction"
            )
            return records

        updated = 0
        # Process in batches to control memory usage.
        for batch_start in range(0, total, self.batch_size):
            batch = eligible[batch_start: batch_start + self.batch_size]
            texts = [r.docstring[:_MAX_INPUT_CHARS] for r in batch]

            try:
                import torch
                inputs = self._tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self._device)
                with torch.no_grad():
                    ids = self._model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=3,
                    )
                decoded = self._tokenizer.batch_decode(
                    ids, skip_special_tokens=False
                )
            except Exception as exc:
                logger.warning(
                    "RelationExtractor: inference error on batch %d-%d: %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                continue

            for rec, raw_text in zip(batch, decoded):
                triples = self._parse_rebel_output(raw_text)
                if triples:
                    rec.relations = list(rec.relations or [])
                    rec.relations.extend(triples)
                    updated += 1

            logger.info(
                "RelationExtractor: processed %d / %d docstrings …",
                min(batch_start + self.batch_size, total),
                total,
            )

        logger.info(
            "RelationExtractor: added relations to %d / %d records",
            updated,
            total,
        )
        return records

    def extract_one(self, rec: "EntityRecord") -> list[dict]:
        """
        Extract triples from a single record's docstring.

        Returns a list of {"source": "rebel-large", "subject": ...,
        "relation": ..., "object": ...} dicts, or [] if none found.
        """
        if not rec.docstring:
            return []

        self._load_pipeline()
        if self._tokenizer is None or self._model is None:
            return []

        try:
            import torch
            inputs = self._tokenizer(
                rec.docstring[:_MAX_INPUT_CHARS],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self._device)
            with torch.no_grad():
                ids = self._model.generate(**inputs, max_length=256, num_beams=3)
            raw_text = self._tokenizer.decode(ids[0], skip_special_tokens=False)
            return self._parse_rebel_output(raw_text)
        except Exception as exc:
            logger.debug(
                "RelationExtractor.extract_one failed for %r: %s",
                rec.entity_name,
                exc,
            )
            return []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_pipeline(self) -> None:
        """
        Lazily import transformers and load REBEL tokenizer + model directly.

        Selects GPU or CPU based on the gpu_monitor singleton's current
        temperature reading.  Falls back to CPU if CUDA is unavailable or
        too hot.

        Sets self._tokenizer = self._model = None on failure (caller checks).
        """
        if self._tokenizer is not None:
            return  # Already loaded.

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from ..gpu_monitor import gpu_monitor

            device_str = gpu_monitor.get_torch_device()
            self._device = torch.device(device_str)

            logger.info(
                "RelationExtractor: loading %s on device=%s …",
                self.model_name,
                device_str,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model = self._model.to(self._device)
            self._model.eval()

            logger.info("RelationExtractor: model loaded successfully")

        except ImportError:
            logger.warning(
                "RelationExtractor: `transformers` not installed — "
                "run `pip install transformers` to enable relation extraction"
            )
        except Exception as exc:
            logger.warning(
                "RelationExtractor: failed to load model %s: %s",
                self.model_name,
                exc,
            )

    def _parse_rebel_output(self, text: str) -> list[dict]:
        """
        Parse REBEL's tagged output into a list of triple dicts.

        REBEL format example:
          <triplet> register_hook <subj> Module <obj> method of

        Each triple becomes:
          {
            "source":   "rebel-large",
            "subject":  "register_hook",
            "relation": "method of",
            "object":   "Module",
          }

        Returns an empty list if the text contains no parseable triples.
        """
        triples: list[dict] = []

        for match in _REBEL_RE.finditer(text):
            subj = match.group("subj").strip()
            obj = match.group("obj").strip()
            rel = match.group("rel").strip()

            if subj and rel and obj:
                triples.append({
                    "source":   "rebel-large",
                    "subject":  subj,
                    "relation": rel,
                    "object":   obj,
                })

        if triples:
            logger.debug(
                "RelationExtractor: parsed %d triples from output",
                len(triples),
            )

        return triples
