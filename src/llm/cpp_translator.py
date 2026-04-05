"""
Two-stage C++ → compliance description translator.

WHY THIS EXISTS
---------------
PyTorch's core is written in C++.  The operator_determinism, export_boundary,
and hookability extractors pull raw C++ function text into EntityRecords, but
the compliance tagger only understands English.  This module bridges the gap
with a two-stage LLM pipeline:

  Stage 1 (qwen2.5-coder:7b):
      "What does this C++ function do in a PyTorch ML pipeline?"
      → one-sentence English description stored in entity.docstring

  Stage 2 (qwen2.5:14b):
      Feed the English description through a mapping-validator style pass
      to produce a structured compliance mapping and a refined docstring.

TARGET FILES
------------
  aten/src/ATen/native/*.cpp
  c10/core/Dispatch*.h
  torch/csrc/autograd/*.cpp

USAGE
-----
    from src.llm.cpp_translator import CppTranslator
    from src.llm.ollama_client import OllamaClient

    client = OllamaClient()
    translator = CppTranslator(client)
    records = translator.translate_all(records)
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extractors.base import EntityRecord
    from .ollama_client import OllamaClient

logger = logging.getLogger("pct.llm.cpp_translator")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

# Stage 1: code-focused model for initial English description.
_STAGE1_MODEL = "qwen3-coder:30b"

# Stage 2: reasoning-capable model for compliance mapping.
_STAGE2_MODEL = "qwen3.5:35b"

# Source file patterns that target files must match (partial path match).
_TARGET_PATTERNS = (
    "aten/src/ATen/native/",
    "c10/core/Dispatch",
    "torch/csrc/autograd/",
)

_STAGE1_SYSTEM = (
    "You are a senior ML engineer explaining C++ PyTorch internals to "
    "compliance analysts. Describe what this C++ function does in the "
    "context of a PyTorch ML pipeline. Respond in ONE sentence, in plain "
    "English, without C++ jargon. Focus on the compliance-relevant "
    "behaviour (determinism, logging, export, hooks, data handling)."
)

_STAGE2_SYSTEM = (
    "You are a compliance analyst. Given a description of a PyTorch C++ "
    "function, identify which EU AI Act article it most directly supports.\n"
    "Respond with ONLY a JSON object:\n"
    '{"article": "Art.9"|"Art.10"|"Art.11"|"Art.12"|"Art.13"|'
    '"Art.14"|"Art.15"|"none",\n'
    ' "confidence": 0.0-1.0,\n'
    ' "refined_description": "one sentence"}'
)

# Regex to recognise valid EU AI Act article values from stage-2 output.
_ARTICLE_RE = re.compile(r"^Art\.\d+$")


# ----------------------------------------------------------------------- #
# CppTranslator
# ----------------------------------------------------------------------- #

class CppTranslator:
    """
    Two-stage LLM pipeline for C++ entity compliance translation.

    Only processes records whose source_file matches one of the target
    path patterns.  Other records are returned unchanged.

    Parameters
    ----------
    client : OllamaClient
        Configured Ollama HTTP client.
    stage1_model : str
        Model for Stage 1 (code → English description).
    stage2_model : str
        Model for Stage 2 (English → compliance mapping).
    """

    def __init__(
        self,
        client: "OllamaClient",
        stage1_model: str = _STAGE1_MODEL,
        stage2_model: str = _STAGE2_MODEL,
    ) -> None:
        self.client = client
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def translate_all(
        self, records: list["EntityRecord"]
    ) -> list["EntityRecord"]:
        """
        Run the two-stage translation on all matching C++ records.

        Parameters
        ----------
        records : list[EntityRecord]
            All entity records (mix of Python and C++).

        Returns
        -------
        list[EntityRecord]
            Same list with C++ target records updated in-place.
        """
        targets = [r for r in records if self._is_target(r)]
        total = len(targets)
        logger.info(
            "CppTranslator: translating %d C++ records (%s → %s) …",
            total,
            self.stage1_model,
            self.stage2_model,
        )

        updated = 0
        for i, rec in enumerate(targets):
            success = self._translate_one(rec)
            if success:
                updated += 1
            if (i + 1) % 25 == 0:
                logger.info(
                    "CppTranslator: processed %d / %d records …",
                    i + 1,
                    total,
                )

        logger.info(
            "CppTranslator: translated %d / %d C++ records successfully",
            updated,
            total,
        )
        return records

    def translate_one(self, rec: "EntityRecord") -> bool:
        """
        Translate a single record.  Returns True on success.
        """
        if not self._is_target(rec):
            return False
        return self._translate_one(rec)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_target(self, rec: "EntityRecord") -> bool:
        """Return True if the record's source file is a C++ translation target."""
        src = rec.source_file or ""
        return any(pat in src for pat in _TARGET_PATTERNS)

    def _translate_one(self, rec: "EntityRecord") -> bool:
        """
        Run stage-1 and stage-2 on a single record, updating it in-place.

        Stage 1: C++ text → one-sentence English description.
        Stage 2: English description → structured compliance mapping.
        """
        # Stage 1 --------------------------------------------------------
        raw_cpp = (rec.raw_text or "")[:1000]
        stage1_prompt = (
            f"```cpp\n{raw_cpp}\n```"
        )

        try:
            s1_result = self.client.generate(
                model=self.stage1_model,
                prompt=stage1_prompt,
                system=_STAGE1_SYSTEM,
                temperature=0.1,
                format="",      # plain text — not JSON
            )
            description = s1_result.get("response", "").strip()
        except Exception as exc:
            logger.debug(
                "CppTranslator stage1 failed for %r: %s",
                rec.entity_name,
                exc,
            )
            return False

        if not description:
            logger.debug(
                "CppTranslator: empty stage-1 response for %r",
                rec.entity_name,
            )
            return False

        # Update docstring with English description (keep original if present).
        if not rec.docstring:
            rec.docstring = description

        # Stage 2 --------------------------------------------------------
        stage2_prompt = (
            f"Function: {rec.entity_name}\n"
            f"Description: {description}"
        )

        try:
            s2_result = self.client.generate(
                model=self.stage2_model,
                prompt=stage2_prompt,
                system=_STAGE2_SYSTEM,
                temperature=0.1,
                format="json",
            )
        except Exception as exc:
            logger.debug(
                "CppTranslator stage2 failed for %r: %s",
                rec.entity_name,
                exc,
            )
            # Stage 1 succeeded — partial success is OK.
            return True

        self._apply_stage2(rec, s2_result)
        return True

    def _apply_stage2(self, rec: "EntityRecord", result: dict) -> None:
        """
        Apply stage-2 compliance mapping to the record.

        Updates compliance_tags, mapping_confidence, mapping_rationale,
        and optionally refines the docstring.
        """
        article = result.get("article", "none")
        confidence = result.get("confidence", 0.0)
        refined = result.get("refined_description", "")

        # Validate article.
        if not _ARTICLE_RE.match(str(article)):
            article = "none"

        # Validate confidence.
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.3

        # Map "Art.N" to tag format.
        tag = ""
        if article and article != "none":
            num = article.replace("Art.", "")
            tag = f"eu_ai_act_art_{num}"

        # Only update if the LLM found a relevant article.
        if tag:
            existing = set(rec.compliance_tags or [])
            existing.add(tag)
            rec.compliance_tags = sorted(existing)
            rec.mapping_confidence = max(
                float(rec.mapping_confidence or 0.0), confidence
            )
            rec.mapping_rationale = (
                f"CppTranslator ({self.stage1_model}→{self.stage2_model}): "
                f"article={article}, confidence={confidence:.2f}"
            )

        # Use refined description if better than current docstring.
        if refined and len(refined) > len(rec.docstring or ""):
            rec.docstring = refined

        logger.debug(
            "CppTranslator stage2: %r → article=%s  confidence=%.2f",
            rec.entity_name,
            article,
            confidence,
        )
