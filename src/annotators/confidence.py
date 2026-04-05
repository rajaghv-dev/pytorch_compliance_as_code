"""
Confidence annotator for PyTorch EntityRecords.

Applies dual confidence scales and boosts confidence for cross-referenced
entities:
  - extraction_confidence: how certain we are the entity was correctly extracted
  - mapping_confidence: how certain we are the compliance mapping is correct

Special handling:
  - Entities appearing in deterministic.md get boosted to 0.95 on both scales
  - Test cases with compliance tags get extraction_confidence = 0.95
  - Tiered defaults for mapping_confidence based on evidence strength
"""

from __future__ import annotations

import logging
from typing import Optional

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.annotators.confidence")


class ConfidenceAnnotator:
    """Applies dual confidence scales and boosts confidence for cross-referenced entities."""

    # Import the compliance API map for checking direct API matches
    # We reference it lazily to avoid circular imports.
    _COMPLIANCE_API_MAP: Optional[dict] = None

    def __init__(self, determinism_operators: Optional[set[str]] = None):
        """
        Initialise the confidence annotator.

        Parameters
        ----------
        determinism_operators : set[str] | None
            Set of operator names extracted from Sphinx deterministic.md notes.
            Entities matching these names get boosted confidence on both scales.
        """
        self.determinism_operators: set[str] = determinism_operators or set()

    @classmethod
    def _get_api_map(cls) -> dict[str, list[str]]:
        """Lazily load the compliance API map to check for direct matches."""
        if cls._COMPLIANCE_API_MAP is None:
            from .compliance_tagger import ComplianceTagger
            cls._COMPLIANCE_API_MAP = ComplianceTagger.COMPLIANCE_API_MAP
        return cls._COMPLIANCE_API_MAP

    def annotate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Adjust confidence scores for all records.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records to adjust.

        Returns
        -------
        list[EntityRecord]
            The same list, with confidence scores adjusted.
        """
        logger.info("Annotating %d records with ConfidenceAnnotator", len(records))
        boosted_count = 0
        defaulted_count = 0

        api_map = self._get_api_map()

        for rec in records:
            changed = False

            # ---- Boost: determinism operators from Sphinx notes ----
            # Entities that appear in deterministic.md are strong evidence
            # of compliance relevance for Art 15 (accuracy/robustness).
            if rec.entity_name in self.determinism_operators:
                rec.extraction_confidence = max(rec.extraction_confidence, 0.95)
                rec.mapping_confidence = max(rec.mapping_confidence, 0.95)
                changed = True

            # ---- Boost: test cases with compliance tags ----
            # Tests are strong evidence of extraction correctness because
            # they explicitly exercise the compliance-relevant behaviour.
            if rec.entity_type == "test_case" and rec.compliance_tags:
                rec.extraction_confidence = max(rec.extraction_confidence, 0.95)
                changed = True

            # ---- Default mapping_confidence for tagged but unscored ----
            # Apply tiered defaults if the entity has tags but the confidence
            # hasn't been set by the compliance tagger.
            if rec.compliance_tags and rec.mapping_confidence == 0.0:
                confidence = self._default_confidence(rec, api_map)
                rec.mapping_confidence = confidence
                defaulted_count += 1
                changed = True

            if changed:
                boosted_count += 1

        logger.info(
            "Tagged %d records (%d confidence-defaulted)", boosted_count, defaulted_count
        )
        return records

    def _default_confidence(
        self, rec: EntityRecord, api_map: dict[str, list[str]]
    ) -> float:
        """
        Assign a default mapping_confidence based on evidence strength.

        Tiers:
        - 0.8+: Direct API-to-article match (entity in COMPLIANCE_API_MAP)
        - 0.6: Capability-to-article (e.g. DataLoader -> Art 10)
        - 0.4: Keyword / phrase match (docstring-based)
        - 0.2: Tagged but no clear rationale

        Parameters
        ----------
        rec : EntityRecord
            The record to score.
        api_map : dict
            The compliance API map for checking direct matches.

        Returns
        -------
        float
            Default mapping confidence score.
        """
        # Direct API match
        if rec.entity_name in api_map:
            return 0.8

        # Has a mapping rationale suggesting structured evidence
        if rec.mapping_rationale:
            # Tier-1 or Tier-2 evidence mentioned in rationale
            if "Tier-1" in rec.mapping_rationale or "Tier-2" in rec.mapping_rationale:
                return 0.6
            # Tier-3 (semantic) evidence
            if "Tier-3" in rec.mapping_rationale:
                return 0.4

        # Tagged but no clear rationale — weakest signal
        return 0.2


def annotate_confidence(
    records: list[EntityRecord],
    determinism_operators: Optional[set[str]] = None,
) -> list[EntityRecord]:
    """Module-level convenience function for confidence annotation."""
    return ConfidenceAnnotator(
        determinism_operators=determinism_operators
    ).annotate(records)
