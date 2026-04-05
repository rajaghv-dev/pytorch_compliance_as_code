"""
Hook consumers annotator for PyTorch EntityRecords.

Finds where hooks are CALLED (consumed), not just defined.  This is the
BUG-12 fix implemented as an annotation pass: if the hookability extractor's
Pass B missed consumers, this annotator catches them.

Hook consumers are compliance-relevant under EU AI Act Art 61 (post-market
monitoring) because they represent code that exercises hook points.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.annotators.hook_consumers")


class HookConsumersAnnotator:
    """Finds where hooks are CALLED (consumed), not just defined -- BUG-12 fix as annotation pass."""

    # All known hook registration method names.  If raw_text contains one
    # of these as a call (not a definition), the entity is a hook consumer.
    HOOK_METHODS: set[str] = {
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_backward_hook",
        "register_full_backward_hook",
        "register_state_dict_pre_hook",
        "register_load_state_dict_post_hook",
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_backward_hook",
        "register_module_full_backward_hook",
        "register_hook",
        "register_multi_grad_hook",
        "register_post_accumulate_grad_hook",
    }

    # Pre-compiled pattern matching any hook method as a call site:
    #   .register_forward_hook(   or   register_forward_hook(
    # We look for the method name followed by '(' to indicate a call.
    _HOOK_CALL_RE = re.compile(
        r"(?:^|[.\s])"
        r"("
        + "|".join(re.escape(m) for m in sorted(HOOK_METHODS))
        + r")"
        r"\s*\(",
        re.MULTILINE,
    )

    # Pattern that indicates a hook DEFINITION rather than a call:
    #   def register_forward_hook(
    _HOOK_DEF_RE = re.compile(
        r"def\s+("
        + "|".join(re.escape(m) for m in sorted(HOOK_METHODS))
        + r")\s*\(",
        re.MULTILINE,
    )

    def annotate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Ensure all hook consumers are tagged with eu_ai_act_art_61.

        Two passes:
        1. Entities already marked as hook_consumer but missing the tag.
        2. Entities whose raw_text contains hook calls but are not yet
           classified as hook_consumer.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records to scan.

        Returns
        -------
        list[EntityRecord]
            The same list, with hook consumers properly tagged.
        """
        logger.info(
            "Annotating %d records with HookConsumersAnnotator", len(records)
        )
        tagged_count = 0
        new_count = 0

        for rec in records:
            changed = False

            # ---- Pass 1: existing hook_consumer missing Art 61 tag ----
            if rec.subcategory == "hook_consumer":
                if "eu_ai_act_art_61" not in rec.compliance_tags:
                    rec.compliance_tags.append("eu_ai_act_art_61")
                    changed = True
                rec.mapping_confidence = max(rec.mapping_confidence, 0.85)
                if not rec.mapping_rationale:
                    rec.mapping_rationale = (
                        "Hook consumer: this code invokes a compliance hook point"
                    )

            # ---- Pass 2: detect hook calls in raw_text ----
            elif rec.raw_text and rec.subcategory != "hook_definition":
                if self._has_hook_call(rec.raw_text):
                    # This entity calls a hook method but isn't marked as a consumer
                    rec.subcategory = "hook_consumer"
                    if "eu_ai_act_art_61" not in rec.compliance_tags:
                        rec.compliance_tags.append("eu_ai_act_art_61")
                    rec.mapping_confidence = max(rec.mapping_confidence, 0.85)
                    rec.mapping_rationale = (
                        "Hook consumer: this code invokes a compliance hook point"
                    )
                    changed = True

            if changed:
                new_count += 1
            if rec.subcategory == "hook_consumer":
                tagged_count += 1

        logger.info(
            "Tagged %d records (%d new tags)", tagged_count, new_count
        )
        return records

    def _has_hook_call(self, raw_text: str) -> bool:
        """
        Check whether raw_text contains a hook CALL (not a definition).

        We look for hook method names followed by '(' but exclude lines
        where the method is being defined (``def register_*_hook(``).

        Parameters
        ----------
        raw_text : str
            Source code text of the entity.

        Returns
        -------
        bool
            True if the text contains at least one hook call.
        """
        # Find all hook method references that look like calls
        calls = self._HOOK_CALL_RE.findall(raw_text)
        if not calls:
            return False

        # Exclude definitions: if ALL occurrences are defs, it's not a consumer
        defs = set(self._HOOK_DEF_RE.findall(raw_text))
        call_methods = set(calls)

        # If there are call-site matches beyond what's defined, it's a consumer
        return bool(call_methods - defs)


def annotate_hook_consumers(records: list[EntityRecord]) -> list[EntityRecord]:
    """Module-level convenience function for hook consumer annotation."""
    return HookConsumersAnnotator().annotate(records)
