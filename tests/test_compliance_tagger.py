"""
Tests for src/annotators/compliance_tagger.py — 3-tier compliance tagging.

WHAT IS TESTED
--------------
Tier 1 (name match):
  - register_forward_hook → eu_ai_act_art_61
  - use_deterministic_algorithms → eu_ai_act_art_15
  - DataLoader → eu_ai_act_art_10
  - save → eu_ai_act_art_12
  - clip_grad_norm_ → eu_ai_act_art_9 AND eu_ai_act_art_15

Tier 2 (structural):
  - subcategory=autograd_extension → eu_ai_act_art_15
  - subcategory=hook_consumer → eu_ai_act_art_61
  - inherits from autograd.Function → eu_ai_act_art_15

Tier 3 (docstring):
  - "non-deterministic" in docstring → eu_ai_act_art_15
  - "privacy" in docstring → eu_ai_act_art_10 + gdpr_art_5
  - "audit" in docstring → eu_ai_act_art_12 + eu_ai_act_art_61

Other behaviour:
  - Unknown entity name with empty docstring → no tags added.
  - Tags from multiple tiers are merged (union).
  - Confidence is set correctly per tier.

HOW TO RUN
----------
    pytest tests/test_compliance_tagger.py -v
"""

from __future__ import annotations

import pytest

from src.annotators.compliance_tagger import ComplianceTagger, annotate_compliance
from src.extractors.base import EntityRecord


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _tag_record(**kwargs) -> EntityRecord:
    """Build and tag a single EntityRecord."""
    rec = EntityRecord(**kwargs)
    tagger = ComplianceTagger()
    return tagger.annotate([rec])[0]


# ---------------------------------------------------------------------------
# Tier 1: direct name match
# ---------------------------------------------------------------------------


class TestTier1NameMatch:

    def test_register_forward_hook_gets_art61(self):
        """register_forward_hook must be tagged with Art. 61."""
        rec = _tag_record(
            entity_name="register_forward_hook",
            entity_type="method",
        )
        assert "eu_ai_act_art_61" in rec.compliance_tags

    def test_use_deterministic_algorithms_gets_art15(self):
        """use_deterministic_algorithms must be tagged with Art. 15."""
        rec = _tag_record(
            entity_name="use_deterministic_algorithms",
            entity_type="function",
        )
        assert "eu_ai_act_art_15" in rec.compliance_tags

    def test_dataloader_gets_art10(self):
        """DataLoader must be tagged with Art. 10 (data governance)."""
        rec = _tag_record(entity_name="DataLoader", entity_type="class")
        assert "eu_ai_act_art_10" in rec.compliance_tags

    def test_save_gets_art12(self):
        """save() must be tagged with Art. 12 (record-keeping)."""
        rec = _tag_record(entity_name="save", entity_type="function")
        assert "eu_ai_act_art_12" in rec.compliance_tags

    def test_clip_grad_norm_gets_both_art9_and_art15(self):
        """clip_grad_norm_ is mapped to both Art. 9 and Art. 15."""
        rec = _tag_record(entity_name="clip_grad_norm_", entity_type="function")
        assert "eu_ai_act_art_9" in rec.compliance_tags
        assert "eu_ai_act_art_15" in rec.compliance_tags

    def test_tier1_confidence_is_at_least_08(self):
        """Tier-1 name matches should set mapping_confidence >= 0.8."""
        rec = _tag_record(
            entity_name="register_forward_hook",
            entity_type="method",
        )
        assert rec.mapping_confidence >= 0.8

    def test_rationale_mentions_tier1(self):
        """The mapping_rationale should mention the name match."""
        rec = _tag_record(
            entity_name="register_forward_hook",
            entity_type="method",
        )
        assert "Tier-1" in rec.mapping_rationale or "name match" in rec.mapping_rationale.lower()


# ---------------------------------------------------------------------------
# Tier 2: structural patterns
# ---------------------------------------------------------------------------


class TestTier2Structural:

    def test_autograd_extension_subcategory_gets_art15(self):
        """Entities with subcategory=autograd_extension → Art. 15."""
        rec = _tag_record(
            entity_name="MyCustomBackward",
            entity_type="class",
            subcategory="autograd_extension",
        )
        assert "eu_ai_act_art_15" in rec.compliance_tags

    def test_hook_consumer_subcategory_gets_art61(self):
        """Entities with subcategory=hook_consumer → Art. 61."""
        rec = _tag_record(
            entity_name="_call_forward_hooks",
            entity_type="method",
            subcategory="hook_consumer",
        )
        assert "eu_ai_act_art_61" in rec.compliance_tags

    def test_inherits_from_autograd_function_gets_art15(self):
        """Entities that inherit from autograd.Function → Art. 15."""
        rec = _tag_record(
            entity_name="MyOp",
            entity_type="class",
            relations=[{"type": "inherits", "target": "torch.autograd.Function"}],
        )
        assert "eu_ai_act_art_15" in rec.compliance_tags

    def test_tier2_confidence_is_at_least_085(self):
        """Tier-2 structural matches should set mapping_confidence >= 0.85."""
        rec = _tag_record(
            entity_name="MyOp",
            entity_type="class",
            subcategory="autograd_extension",
        )
        assert rec.mapping_confidence >= 0.85


# ---------------------------------------------------------------------------
# Tier 3: docstring semantic
# ---------------------------------------------------------------------------


class TestTier3Semantic:

    def test_non_deterministic_phrase_gets_art15(self):
        """'non-deterministic' in docstring → Art. 15."""
        rec = _tag_record(
            entity_name="scatter_add_",
            entity_type="method",
            docstring="This operation is non-deterministic on CUDA devices.",
        )
        assert "eu_ai_act_art_15" in rec.compliance_tags

    def test_privacy_phrase_gets_art10_and_gdpr(self):
        """'privacy' in docstring → Art. 10 (data governance) + GDPR Art. 5."""
        rec = _tag_record(
            entity_name="load_user_data",
            entity_type="function",
            docstring="Loads data with privacy protections applied.",
        )
        assert "eu_ai_act_art_10" in rec.compliance_tags
        assert "gdpr_art_5" in rec.compliance_tags

    def test_audit_phrase_gets_art12_and_art61(self):
        """'audit' in docstring → Art. 12 + Art. 61."""
        rec = _tag_record(
            entity_name="log_event",
            entity_type="function",
            docstring="Write an audit log entry for this operation.",
        )
        assert "eu_ai_act_art_12" in rec.compliance_tags
        assert "eu_ai_act_art_61" in rec.compliance_tags

    def test_tier3_confidence_is_at_most_05(self):
        """Tier-3 (docstring) confidence should be ≤ 0.5 for pure semantic matches."""
        # A record with ONLY a tier-3 match (unknown entity_name, no subcategory).
        rec = _tag_record(
            entity_name="totally_unknown_function_xyz",
            entity_type="function",
            subcategory="",
            docstring="This operation may produce non-deterministic results.",
        )
        # Tier-3 sets confidence to 0.5; it should not exceed that.
        assert rec.mapping_confidence <= 0.5

    def test_empty_docstring_gets_no_tier3_tags(self):
        """An entity with an empty docstring should get no tier-3 tags."""
        rec = _tag_record(
            entity_name="truly_unknown_xyz",
            entity_type="function",
            subcategory="",
            docstring="",
        )
        # Only check that no tags come purely from tier-3 (empty docstring).
        # Tags from tier-1 or tier-2 are fine.
        assert not any(
            tag for tag in rec.compliance_tags
            if tag.startswith(("eu_ai_act", "gdpr"))
            # A record with unknown name and no subcategory should not gain tags
            # from tier-3 if the docstring is empty.
        ), (
            "Expected no compliance tags for unknown entity with empty docstring"
        )


# ---------------------------------------------------------------------------
# Tag merging and idempotency
# ---------------------------------------------------------------------------


class TestTagMerging:

    def test_tags_are_merged_not_duplicated(self):
        """Running the tagger twice must not duplicate tags."""
        rec = EntityRecord(
            entity_name="register_forward_hook",
            entity_type="method",
            docstring="Registers a monitoring hook.",
        )
        tagger = ComplianceTagger()
        tagger.annotate([rec])
        tags_after_first = list(rec.compliance_tags)
        tagger.annotate([rec])  # Run again.
        # Each tag should appear exactly once.
        for tag in set(rec.compliance_tags):
            assert rec.compliance_tags.count(tag) == 1

    def test_multiple_tier_matches_produce_union(self):
        """Tags from different tiers should be combined (union)."""
        # register_forward_hook matches tier-1 (art_61)
        # "non-deterministic" in docstring matches tier-3 (art_15)
        rec = _tag_record(
            entity_name="register_forward_hook",
            entity_type="method",
            docstring="Warning: non-deterministic if called during distributed training.",
        )
        assert "eu_ai_act_art_61" in rec.compliance_tags
        assert "eu_ai_act_art_15" in rec.compliance_tags


# ---------------------------------------------------------------------------
# module-level convenience function
# ---------------------------------------------------------------------------


class TestAnnotateCompliance:

    def test_annotate_compliance_returns_same_length(self, sample_records):
        """annotate_compliance() should return the same number of records."""
        result = annotate_compliance(sample_records)
        assert len(result) == len(sample_records)

    def test_annotate_compliance_tags_known_entities(self, sample_records):
        """Known entities in sample_records should receive compliance tags."""
        result = annotate_compliance(sample_records)
        tagged = [r for r in result if r.compliance_tags]
        # At least the hook and determinism records should be tagged.
        assert len(tagged) >= 2
