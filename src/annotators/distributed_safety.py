"""
Distributed safety annotator for PyTorch EntityRecords.

Tags entities with a distributed safety level:
  - safe: no known distributed synchronisation issues
  - unsafe: contains collective operations that create sync points
  - conditional: may be unsafe depending on FSDP shard visibility
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.annotators.distributed_safety")


class DistributedSafetyAnnotator:
    """Tags entities with distributed safety level: safe | unsafe | conditional."""

    # Patterns indicating collective operations that create sync points.
    # These are unsafe because they block across all ranks.
    UNSAFE_PATTERNS = [
        re.compile(r"\btorch\.distributed\.all_reduce\b"),
        re.compile(r"\btorch\.distributed\.all_gather\b"),
        re.compile(r"\btorch\.distributed\.broadcast\b"),
        re.compile(r"\btorch\.distributed\.barrier\b"),
        re.compile(r"\ball_reduce\s*\("),
        re.compile(r"\ball_gather\s*\("),
    ]

    # Patterns indicating parameter access that may be conditional on FSDP
    # shard visibility — safe in non-distributed settings but potentially
    # unsafe under FSDP where only a shard of parameters is visible.
    CONDITIONAL_PATTERNS = [
        re.compile(r"module\.parameters\s*\("),
        re.compile(r"module\.named_parameters\s*\("),
        re.compile(r"\.parameters\s*\(\s*\)"),
    ]

    def annotate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Assign distributed_safety to each record that has compliance_tags.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records to classify.

        Returns
        -------
        list[EntityRecord]
            The same list, with distributed_safety / distributed_safety_notes
            populated for compliance-relevant entities.
        """
        logger.info(
            "Annotating %d records with DistributedSafetyAnnotator", len(records)
        )
        tagged_count = 0
        new_count = 0

        for rec in records:
            # Only classify entities that have compliance tags
            if not rec.compliance_tags:
                continue

            had_safety = bool(rec.distributed_safety)
            safety, notes = self._classify(rec)
            rec.distributed_safety = safety
            rec.distributed_safety_notes = notes
            tagged_count += 1
            if not had_safety:
                new_count += 1

        logger.info(
            "Tagged %d records (%d new tags)", tagged_count, new_count
        )
        return records

    def _classify(self, rec: EntityRecord) -> tuple[str, str]:
        """
        Determine the distributed safety level for a single record.

        Returns
        -------
        tuple[str, str]
            (safety_level, notes) — level is one of "safe", "unsafe",
            "conditional".
        """
        raw = rec.raw_text or ""

        # ---- Check for unsafe collective operations ----
        for pattern in self.UNSAFE_PATTERNS:
            match = pattern.search(raw)
            if match:
                return (
                    "unsafe",
                    f"Contains distributed sync point: '{match.group()}'. "
                    "This creates a blocking collective that must execute on "
                    "all ranks simultaneously.",
                )

        # ---- Check for conditional patterns (FSDP shard visibility) ----
        # Exclude _summon_full_params which is a safe FSDP primitive that
        # explicitly handles shard gathering before parameter access.
        if "_summon_full_params" not in raw:
            for pattern in self.CONDITIONAL_PATTERNS:
                match = pattern.search(raw)
                if match:
                    return (
                        "conditional",
                        f"Accesses parameters via '{match.group()}'. "
                        "Under FSDP, only a shard of parameters may be "
                        "visible — use _summon_full_params or ensure "
                        "full parameter visibility before access.",
                    )

        # ---- Default: safe ----
        return ("safe", "")


def annotate_distributed_safety(records: list[EntityRecord]) -> list[EntityRecord]:
    """Module-level convenience function for distributed safety annotation."""
    return DistributedSafetyAnnotator().annotate(records)
