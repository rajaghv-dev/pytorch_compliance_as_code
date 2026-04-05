"""
Commit compliance classifier using qwen2.5-coder:7b via Ollama.

WHAT IT DOES
------------
For every commit record produced by the commit_history extractor, this
module re-classifies the commit's compliance relevance using an LLM,
replacing the keyword-based classification with a more nuanced LLM
judgment.

CLASSIFICATION TYPES
--------------------
  security_fix       — patches a vulnerability (→ Art.15)
  determinism_change — adds/removes deterministic behaviour (→ Art.9/15)
  deprecation        — removes or marks a public API deprecated (→ Art.11)
  breaking_change    — backward-incompatible API or behaviour change
  hook_change        — modifies hook registration/dispatch (→ Art.14)
  data_handling      — changes DataLoader, dataset, preprocessing (→ Art.10)
  not_relevant       — no compliance relevance

USAGE
-----
    from src.llm.commit_classifier import CommitClassifier
    from src.llm.ollama_client import OllamaClient

    client = OllamaClient()
    classifier = CommitClassifier(client)
    classified = classifier.classify_all(commit_records)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extractors.base import EntityRecord
    from .ollama_client import OllamaClient

logger = logging.getLogger("pct.llm.commit_classifier")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

# LLM model for commit classification — qwen3-coder:30b for deeper code understanding.
_MODEL = "qwen3-coder:30b"

# Valid classification types that the LLM must output.
_VALID_TYPES = frozenset({
    "security_fix",
    "determinism_change",
    "deprecation",
    "breaking_change",
    "hook_change",
    "data_handling",
    "not_relevant",
})

# Article tags mapped from classification types.
_TYPE_TO_ARTICLE: dict[str, str] = {
    "security_fix":        "eu_ai_act_art_15",
    "determinism_change":  "eu_ai_act_art_9",
    "deprecation":         "eu_ai_act_art_11",
    "breaking_change":     "eu_ai_act_art_11",
    "hook_change":         "eu_ai_act_art_14",
    "data_handling":       "eu_ai_act_art_10",
    "not_relevant":        "",
}

# System prompt sent to qwen2.5-coder for each commit.
_SYSTEM_PROMPT = (
    "You are a compliance analyst reviewing PyTorch git commits.\n"
    "Classify the commit's compliance relevance for the EU AI Act.\n"
    "Respond with ONLY a JSON object — no extra text:\n"
    '{"type": "security_fix"|"determinism_change"|"deprecation"|'
    '"breaking_change"|"hook_change"|"data_handling"|"not_relevant",\n'
    ' "confidence": 0.0-1.0,\n'
    ' "article": "Art.9"|"Art.10"|"Art.11"|"Art.14"|"Art.15"|"none"}'
)


# ----------------------------------------------------------------------- #
# CommitClassifier
# ----------------------------------------------------------------------- #

class CommitClassifier:
    """
    Uses qwen2.5-coder:7b to classify commit records by compliance type.

    Only processes records whose entity_type == "commit".
    Non-commit records are passed through unchanged.

    Parameters
    ----------
    client : OllamaClient
        Configured Ollama HTTP client.
    model : str
        Override the default model name.
    """

    def __init__(
        self,
        client: "OllamaClient",
        model: str = _MODEL,
    ) -> None:
        self.client = client
        self.model = model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify_all(
        self, records: list["EntityRecord"]
    ) -> list["EntityRecord"]:
        """
        Classify all commit records and return the full record list.

        Commit records are updated in-place with the LLM's classification.
        Non-commit records are returned unchanged.

        Parameters
        ----------
        records : list[EntityRecord]
            All entity records (mix of commits and non-commits is fine).

        Returns
        -------
        list[EntityRecord]
            Same records with commit records' compliance_tags and
            mapping_rationale updated by the LLM.
        """
        commit_records = [r for r in records if r.entity_type == "commit"]
        total = len(commit_records)
        logger.info(
            "CommitClassifier: classifying %d commit records with %s …",
            total,
            self.model,
        )

        updated = 0
        for i, rec in enumerate(commit_records):
            success = self._classify_one(rec)
            if success:
                updated += 1
            if (i + 1) % 50 == 0:
                logger.info(
                    "CommitClassifier: processed %d / %d commits …",
                    i + 1,
                    total,
                )

        logger.info(
            "CommitClassifier: classified %d / %d commits successfully",
            updated,
            total,
        )
        return records

    def classify_one(self, rec: "EntityRecord") -> bool:
        """
        Classify a single commit record.

        Returns True if the LLM responded with a valid classification.
        """
        return self._classify_one(rec)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _classify_one(self, rec: "EntityRecord") -> bool:
        """
        Send one commit to the LLM and update the record in-place.

        Returns True on success, False if the call fails or returns garbage.
        """
        # Build prompt from commit subject + first 500 chars of body.
        subject = rec.entity_name or "(no subject)"
        body = (rec.raw_text or "")[:500]

        prompt = f"Commit: {subject}\nDiff: {body}"

        try:
            result = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.1,
                format="json",
            )
        except Exception as exc:
            logger.debug(
                "CommitClassifier: Ollama call failed for commit %r: %s",
                subject[:60],
                exc,
            )
            return False

        return self._apply_classification(rec, result)

    def _apply_classification(
        self, rec: "EntityRecord", result: dict
    ) -> bool:
        """
        Validate the LLM response and update the EntityRecord.

        Parameters
        ----------
        rec : EntityRecord
            The commit record to update.
        result : dict
            Parsed JSON from the LLM.

        Returns
        -------
        bool
            True if the classification was valid and applied.
        """
        commit_type = result.get("type", "")
        confidence = result.get("confidence", 0.0)
        article_str = result.get("article", "none")

        # Validate type field.
        if commit_type not in _VALID_TYPES:
            logger.debug(
                "CommitClassifier: invalid type %r — ignoring commit %r",
                commit_type,
                rec.entity_name[:60],
            )
            return False

        # Validate confidence is a float in [0, 1].
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        # Override the keyword-based classification.
        tag = _TYPE_TO_ARTICLE.get(commit_type, "")
        if tag:
            rec.compliance_tags = [tag]
        else:
            # not_relevant: clear any tags the keyword pass may have set.
            rec.compliance_tags = []

        rec.mapping_confidence = confidence
        rec.mapping_rationale = (
            f"LLM ({self.model}): type={commit_type}, article={article_str}"
        )
        rec.subcategory = commit_type   # overwrite keyword-based subcategory

        logger.debug(
            "CommitClassifier: commit=%r  type=%s  confidence=%.2f  article=%s",
            rec.entity_name[:60],
            commit_type,
            confidence,
            article_str,
        )
        return True
