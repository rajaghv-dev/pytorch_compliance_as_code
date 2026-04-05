"""
3-tier compliance tagger for PyTorch EntityRecords.

Tier 1: Direct name matching against a known compliance API map.
Tier 2: AST structural patterns (subcategory, inheritance relations).
Tier 3: Docstring semantic phrase matching.

Each tier assigns compliance_tags and a mapping_confidence reflecting the
strength of the evidence.  Tags from multiple tiers are merged (union)
and the highest confidence wins.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.annotators.compliance_tagger")


class ComplianceTagger:
    """3-tier compliance tagger: regex -> AST structure -> docstring semantic."""

    # ------------------------------------------------------------------
    # Tier 1: Known compliance API names (entity_name -> articles)
    # ------------------------------------------------------------------
    COMPLIANCE_API_MAP: dict[str, list[str]] = {
        # Art 61 — Post-market monitoring / hooks (+ Art.14 human oversight, Art.12 record-keeping)
        "register_forward_hook": ["eu_ai_act_art_61", "eu_ai_act_art_14", "eu_ai_act_art_12"],
        "register_forward_pre_hook": ["eu_ai_act_art_61"],
        "register_backward_hook": ["eu_ai_act_art_61", "eu_ai_act_art_14", "eu_ai_act_art_12"],
        "register_full_backward_hook": ["eu_ai_act_art_61", "eu_ai_act_art_14"],
        "register_state_dict_pre_hook": ["eu_ai_act_art_61"],
        "register_load_state_dict_post_hook": ["eu_ai_act_art_61"],
        "register_module_forward_hook": ["eu_ai_act_art_61", "eu_ai_act_art_14"],
        "register_module_forward_pre_hook": ["eu_ai_act_art_61"],
        "register_module_backward_hook": ["eu_ai_act_art_61", "eu_ai_act_art_14"],
        "register_module_full_backward_hook": ["eu_ai_act_art_61"],
        "register_hook": ["eu_ai_act_art_61", "eu_ai_act_art_14", "eu_ai_act_art_12"],
        "register_multi_grad_hook": ["eu_ai_act_art_61"],
        "register_post_accumulate_grad_hook": ["eu_ai_act_art_61"],
        # Art 15 — Accuracy, robustness, cybersecurity / determinism
        "use_deterministic_algorithms": ["eu_ai_act_art_15"],
        "are_deterministic_algorithms_enabled": ["eu_ai_act_art_15"],
        "manual_seed": ["eu_ai_act_art_15", "eu_ai_act_art_14"],
        "set_rng_state": ["eu_ai_act_art_15"],
        "get_rng_state": ["eu_ai_act_art_15"],
        "fork_rng": ["eu_ai_act_art_15"],
        "backward": ["eu_ai_act_art_15"],
        "autograd": ["eu_ai_act_art_15"],
        "grad": ["eu_ai_act_art_15"],
        "no_grad": ["eu_ai_act_art_15", "eu_ai_act_art_14"],
        "enable_grad": ["eu_ai_act_art_15", "eu_ai_act_art_14"],
        "set_grad_enabled": ["eu_ai_act_art_15", "eu_ai_act_art_14"],
        "inference_mode": ["eu_ai_act_art_15"],
        "anomaly_mode": ["eu_ai_act_art_15"],
        "detect_anomaly": ["eu_ai_act_art_15"],
        "set_detect_anomaly": ["eu_ai_act_art_15"],
        "GradScaler": ["eu_ai_act_art_15"],
        # Art 10 — Data and data governance (+ GDPR Art.5 data principles)
        "DataLoader": ["eu_ai_act_art_10", "gdpr_art_5"],
        "Dataset": ["eu_ai_act_art_10", "gdpr_art_5"],
        "IterableDataset": ["eu_ai_act_art_10", "gdpr_art_5"],
        "Sampler": ["eu_ai_act_art_10"],
        "BatchSampler": ["eu_ai_act_art_10"],
        "RandomSampler": ["eu_ai_act_art_10"],
        "SequentialSampler": ["eu_ai_act_art_10"],
        "DistributedSampler": ["eu_ai_act_art_10"],
        "WeightedRandomSampler": ["eu_ai_act_art_10"],
        "SubsetRandomSampler": ["eu_ai_act_art_10"],
        "ConcatDataset": ["eu_ai_act_art_10"],
        "TensorDataset": ["eu_ai_act_art_10", "gdpr_art_5"],
        # Art 12 — Record-keeping
        "save": ["eu_ai_act_art_12"],
        "load": ["eu_ai_act_art_12"],
        "checkpoint": ["eu_ai_act_art_12", "gdpr_art_5"],
        "save_state_dict": ["eu_ai_act_art_12"],
        "load_state_dict": ["eu_ai_act_art_12"],
        "state_dict": ["eu_ai_act_art_12"],
        # Art 11 — Technical documentation
        "ExportedProgram": ["eu_ai_act_art_11"],
        "export": ["eu_ai_act_art_11"],
        # Art 13 — Transparency
        "SummaryWriter": ["eu_ai_act_art_13"],
        # Art 9 — Risk management
        "clip_grad_norm_": ["eu_ai_act_art_9", "eu_ai_act_art_15"],
        "clip_grad_value_": ["eu_ai_act_art_9", "eu_ai_act_art_15"],
        # GDPR Art.5 — data principles (lawfulness, fairness, transparency, purpose limitation, storage limitation, accuracy)
        "serialize": ["gdpr_art_5"],
        "deserialize": ["gdpr_art_5"],
        "save_state": ["gdpr_art_5", "eu_ai_act_art_12"],
        "audit_log": ["gdpr_art_5", "eu_ai_act_art_12"],
        "data_loader": ["gdpr_art_5", "eu_ai_act_art_10"],
        "Subset": ["gdpr_art_5", "eu_ai_act_art_10"],
        "random_split": ["gdpr_art_5", "eu_ai_act_art_10"],
        "data_provenance": ["gdpr_art_5", "eu_ai_act_art_10"],
        # GDPR Art.17 — right to erasure
        "delete_model": ["gdpr_art_17"],
        "forget": ["gdpr_art_17"],
        "unlearn": ["gdpr_art_17", "eu_ai_act_art_10"],
        "machine_unlearning": ["gdpr_art_17"],
        "gradient_deletion": ["gdpr_art_17"],
        # GDPR Art.22 — automated decision-making
        "predict": ["gdpr_art_22", "eu_ai_act_art_13"],
        "inference": ["gdpr_art_22", "eu_ai_act_art_13"],
        "decision": ["gdpr_art_22", "eu_ai_act_art_14"],
        "score": ["gdpr_art_22"],
        "classify": ["gdpr_art_22", "eu_ai_act_art_13"],
        "AutoModelForSequenceClassification": ["gdpr_art_22", "eu_ai_act_art_13"],
        # GDPR Art.25 — privacy by design
        "differential_privacy": ["gdpr_art_25", "eu_ai_act_art_10"],
        "noise_multiplier": ["gdpr_art_25"],
        "privacy_engine": ["gdpr_art_25"],
        "PrivacyEngine": ["gdpr_art_25"],
        "anonymize": ["gdpr_art_25", "eu_ai_act_art_10"],
        "pseudonymize": ["gdpr_art_25"],
        "make_private": ["gdpr_art_25"],
        "GradSampleModule": ["gdpr_art_25"],
        "DPOptimizer": ["gdpr_art_25"],
        "GaussianMechanism": ["gdpr_art_25"],
        "RDPAccountant": ["gdpr_art_25"],
        "privacy_accountant": ["gdpr_art_25"],
        "PrivacyAccountant": ["gdpr_art_25"],
        "get_privacy_spent": ["gdpr_art_25"],
        "clip_per_sample_gradients": ["gdpr_art_25", "eu_ai_act_art_15"],
        "noise_scale": ["gdpr_art_25"],
        "max_grad_norm": ["gdpr_art_25", "eu_ai_act_art_15"],
        "per_sample_gradient_clip": ["gdpr_art_25"],
        # GDPR Art.32 — security
        "encrypt": ["gdpr_art_32"],
        "decrypt": ["gdpr_art_32"],
        "secure_aggregation": ["gdpr_art_32"],
        "ssl_context": ["gdpr_art_32"],
        "tls": ["gdpr_art_32"],
        "create_ssl_context": ["gdpr_art_32"],
        "ssl_socket": ["gdpr_art_32"],
        "tls_config": ["gdpr_art_32"],
        "secure_channel": ["gdpr_art_32"],
        "compute_hash": ["gdpr_art_32"],
        "verify_keys": ["gdpr_art_32"],
        "hmac": ["gdpr_art_32"],
        "SecureAggregator": ["gdpr_art_32"],
        "encrypt_model": ["gdpr_art_32"],
        "hash_tensor": ["gdpr_art_32"],
        # EU AI Act Art.14 — human oversight (hooks already merged above with Art.61)
        "register_comm_hook": ["eu_ai_act_art_14"],
        "save_checkpoint": ["eu_ai_act_art_14", "eu_ai_act_art_12"],
        "load_checkpoint": ["eu_ai_act_art_14", "eu_ai_act_art_12"],
        "DistributedDataParallel": ["eu_ai_act_art_14", "eu_ai_act_art_15"],
        "FullyShardedDataParallel": ["eu_ai_act_art_14", "eu_ai_act_art_15"],
        "ProcessGroup": ["eu_ai_act_art_14"],
        "barrier": ["eu_ai_act_art_14"],
        "callback": ["eu_ai_act_art_14", "eu_ai_act_art_12"],
        "Callback": ["eu_ai_act_art_14", "eu_ai_act_art_12"],
        "Trainer": ["eu_ai_act_art_14", "eu_ai_act_art_17"],
        "EarlyStopping": ["eu_ai_act_art_14", "eu_ai_act_art_9"],
    }

    # ------------------------------------------------------------------
    # Tier 3: Docstring semantic phrases (phrase -> articles)
    # ------------------------------------------------------------------
    COMPLIANCE_PHRASES: dict[str, list[str]] = {
        "non-deterministic": ["eu_ai_act_art_15"],
        "different results": ["eu_ai_act_art_15"],
        "not guaranteed to produce identical": ["eu_ai_act_art_15"],
        "hardware dependent": ["eu_ai_act_art_15"],
        "cuda atomics": ["eu_ai_act_art_15"],
        "numerical stability": ["eu_ai_act_art_15"],
        "privacy": ["eu_ai_act_art_10", "gdpr_art_5"],
        "tamper": ["eu_ai_act_art_12"],
        "audit": ["eu_ai_act_art_12", "eu_ai_act_art_61"],
        "reproducib": ["eu_ai_act_art_15"],
        "fairness": ["eu_ai_act_art_10"],
        "bias": ["eu_ai_act_art_10"],
        "transparency": ["eu_ai_act_art_13"],
        "human oversight": ["eu_ai_act_art_14"],
        "monitoring": ["eu_ai_act_art_61", "eu_ai_act_art_14", "eu_ai_act_art_12"],
        "risk": ["eu_ai_act_art_9"],
        "data governance": ["eu_ai_act_art_10"],
        "documentation": ["eu_ai_act_art_11"],
        "record": ["eu_ai_act_art_12"],
        "accuracy": ["eu_ai_act_art_15"],
        "robustness": ["eu_ai_act_art_15"],
        "cybersecurity": ["eu_ai_act_art_15"],
        # GDPR Art.5 — data principles
        "storage limitation": ["gdpr_art_5"],
        "purpose limitation": ["gdpr_art_5"],
        "data minimization": ["gdpr_art_5", "eu_ai_act_art_10"],
        "lawful basis": ["gdpr_art_5", "eu_ai_act_art_10"],
        "data subject": ["gdpr_art_5", "gdpr_art_22"],
        "personal data": ["gdpr_art_5", "eu_ai_act_art_10"],
        "retention": ["gdpr_art_5", "eu_ai_act_art_12"],
        "audit trail": ["gdpr_art_5", "eu_ai_act_art_12"],
        # GDPR Art.17 — erasure
        "right to erasure": ["gdpr_art_17"],
        "right to be forgotten": ["gdpr_art_17"],
        "machine unlearning": ["gdpr_art_17"],
        "unlearn": ["gdpr_art_17"],
        "forget": ["gdpr_art_17"],
        # GDPR Art.22 — automated decisions
        "automated decision": ["gdpr_art_22", "eu_ai_act_art_14"],
        "profiling": ["gdpr_art_22", "eu_ai_act_art_13"],
        "automated processing": ["gdpr_art_22"],
        # GDPR Art.25 — privacy by design
        "privacy by design": ["gdpr_art_25", "eu_ai_act_art_10"],
        "differential privacy": ["gdpr_art_25", "eu_ai_act_art_10"],
        "anonymisation": ["gdpr_art_25"],
        "anonymization": ["gdpr_art_25"],
        "pseudonymisation": ["gdpr_art_25"],
        "noise multiplier": ["gdpr_art_25"],
        "per-sample gradient": ["gdpr_art_25"],
        "per sample gradient": ["gdpr_art_25"],
        "privacy budget": ["gdpr_art_25"],
        "dp-sgd": ["gdpr_art_25"],
        "private learning": ["gdpr_art_25"],
        "gradient privatization": ["gdpr_art_25"],
        "opacus": ["gdpr_art_25"],
        "max grad norm": ["gdpr_art_25", "eu_ai_act_art_15"],
        "clip per-sample": ["gdpr_art_25"],
        "privacy accountant": ["gdpr_art_25"],
        "privacy engine": ["gdpr_art_25"],
        "sensitive data": ["gdpr_art_25", "gdpr_art_5"],
        # GDPR Art.32 — security
        "encryption": ["gdpr_art_32", "eu_ai_act_art_15"],
        "secure channel": ["gdpr_art_32"],
        "authenticated": ["gdpr_art_32"],
        "integrity check": ["gdpr_art_32"],
        "checksum": ["gdpr_art_32"],
        "digital signature": ["gdpr_art_32"],
        "hashing": ["gdpr_art_32"],
        "access control": ["gdpr_art_32"],
        "authorization": ["gdpr_art_32"],
        "credential": ["gdpr_art_32"],
        "certificate": ["gdpr_art_32"],
        "tls handshake": ["gdpr_art_32"],
        "ssl certificate": ["gdpr_art_32"],
        "secure aggregation": ["gdpr_art_32"],
        "federated learning": ["gdpr_art_32", "gdpr_art_25"],
        # Art.14 — human oversight (note: "monitoring" merged with Art.61 entry above)
        "intervention": ["eu_ai_act_art_14"],
        "manual override": ["eu_ai_act_art_14"],
        "human review": ["eu_ai_act_art_14"],
        "human-in-the-loop": ["eu_ai_act_art_14"],
        "supervision": ["eu_ai_act_art_14"],
        "stop training": ["eu_ai_act_art_14", "eu_ai_act_art_9"],
        "early stopping": ["eu_ai_act_art_14", "eu_ai_act_art_9"],
        "gradient clipping": ["eu_ai_act_art_14", "eu_ai_act_art_9", "eu_ai_act_art_15"],
    }

    def annotate(self, records: list[EntityRecord]) -> list[EntityRecord]:
        """
        Apply 3-tier compliance tagging to every record in *records*.

        Records are modified in-place.  The list is returned for chaining.

        Parameters
        ----------
        records : list[EntityRecord]
            Entity records to tag.

        Returns
        -------
        list[EntityRecord]
            The same list, with compliance_tags / mapping_confidence /
            mapping_rationale populated where applicable.
        """
        logger.info("Annotating %d records with ComplianceTagger", len(records))
        tagged_count = 0
        new_tag_count = 0

        for rec in records:
            had_tags_before = bool(rec.compliance_tags)
            tags_before = set(rec.compliance_tags)
            rationale_parts: list[str] = []

            # ---- Tier 1: direct name match ----
            if rec.entity_name in self.COMPLIANCE_API_MAP:
                articles = self.COMPLIANCE_API_MAP[rec.entity_name]
                for art in articles:
                    if art not in rec.compliance_tags:
                        rec.compliance_tags.append(art)
                rec.mapping_confidence = max(rec.mapping_confidence, 0.8)
                rationale_parts.append(
                    f"Tier-1 name match: '{rec.entity_name}' -> {articles}"
                )

            # ---- Tier 2: AST structural patterns ----
            tier2_tags = self._tier2_structural(rec)
            if tier2_tags:
                for art in tier2_tags:
                    if art not in rec.compliance_tags:
                        rec.compliance_tags.append(art)
                rec.mapping_confidence = max(rec.mapping_confidence, 0.85)
                rationale_parts.append(
                    f"Tier-2 structural: subcategory='{rec.subcategory}' -> {tier2_tags}"
                )

            # ---- Tier 3: docstring semantic phrases ----
            tier3_tags = self._tier3_semantic(rec)
            if tier3_tags:
                for art, phrase in tier3_tags:
                    if art not in rec.compliance_tags:
                        rec.compliance_tags.append(art)
                # Semantic matches are weaker evidence
                rec.mapping_confidence = max(rec.mapping_confidence, 0.5)
                matched_phrases = list({p for _, p in tier3_tags})
                rationale_parts.append(
                    f"Tier-3 semantic: phrases={matched_phrases}"
                )

            # ---- Set rationale if any tags were added ----
            if set(rec.compliance_tags) != tags_before:
                new_tag_count += 1
                if rationale_parts:
                    rec.mapping_rationale = "; ".join(rationale_parts)

            if rec.compliance_tags:
                tagged_count += 1

            # ---- Warn on conflicting signals ----
            self._check_conflicts(rec)

        logger.info(
            "Tagged %d records (%d new tags added)", tagged_count, new_tag_count
        )
        return records

    def _tier2_structural(self, rec: EntityRecord) -> list[str]:
        """
        Tier 2: Infer compliance tags from AST-level structural information.

        - autograd_extension subcategory -> Art 15 (accuracy/robustness)
        - hook_consumer subcategory -> Art 61 (post-market monitoring)
        - Inherits from torch.autograd.Function -> Art 15
        """
        tags: list[str] = []

        # Subcategory: autograd extension entities relate to Art 15
        if rec.subcategory == "autograd_extension":
            tags.append("eu_ai_act_art_15")

        # Subcategory: hook consumer entities relate to Art 61
        if rec.subcategory == "hook_consumer":
            tags.append("eu_ai_act_art_61")

        # Check inheritance: if entity inherits from torch.autograd.Function
        for rel in rec.relations:
            target = rel.get("target", "") if isinstance(rel, dict) else ""
            if "autograd.Function" in target or "autograd_function" in target.lower():
                if "eu_ai_act_art_15" not in tags:
                    tags.append("eu_ai_act_art_15")
                break

        # Art.14 — hook_definition subcategory (all hooks are human oversight touchpoints)
        if rec.subcategory == "hook_definition":
            for art in ["eu_ai_act_art_14", "eu_ai_act_art_12"]:
                if art not in tags:
                    tags.append(art)

        # Art.14 — override_protocol subcategory (__torch_function__ / __torch_dispatch__)
        if rec.subcategory == "override_protocol":
            if "eu_ai_act_art_14" not in tags:
                tags.append("eu_ai_act_art_14")

        # GDPR Art.5 / AI Act Art.10 — data_loading subcategory
        if rec.subcategory == "data_loading":
            for art in ["gdpr_art_5", "eu_ai_act_art_10"]:
                if art not in tags:
                    tags.append(art)

        # GDPR Art.22 — function/method whose name IS or starts/ends with an
        # ML-decision keyword.  We require whole-word boundaries to avoid matching
        # shape/type-inference utilities (infer_size, inferShapeType, …).
        _name_lower = (rec.entity_name or "").lower()
        _decision_kws = ("predict", "classify", "score", "decision")
        if rec.entity_type in ("function", "method") and any(
            _name_lower == kw
            or _name_lower.startswith(kw + "_")
            or _name_lower.endswith("_" + kw)
            for kw in _decision_kws
        ):
            for art in ["gdpr_art_22", "eu_ai_act_art_13"]:
                if art not in tags:
                    tags.append(art)

        # GDPR Art.25 — privacy by design: entity name contains well-known DP/
        # privacy-specific substrings.  Note: we avoid bare "private" to prevent
        # false positives from Python's _private_helper naming convention.
        # "privacy" is safe (no collision with Python conventions).
        # "_private" requires an underscore prefix, matching make_private but
        # not standalone __private_state internal attributes.
        _dp_kws = ("privacy", "_private", "anonymi", "pseudonym", "dp_optimizer",
                   "grad_sample", "noise_multi", "per_sample")
        if any(kw in _name_lower for kw in _dp_kws):
            if "gdpr_art_25" not in tags:
                tags.append("gdpr_art_25")

        # GDPR Art.32 — security: entity name contains security-related stems.
        # We require structural prefixes/suffixes (ssl_, _ssl, etc.) to avoid
        # matching unrelated words like "hash_map" (data structure) vs "hash_"
        # (cryptographic hash utility).  "checksum" and "hmac" are unambiguous.
        _sec_kws = ("encrypt", "decrypt", "ssl_", "_ssl", "tls_", "_tls",
                    "secure_", "_secure", "hmac", "hash_data", "hash_tensor",
                    "checksum", "verify_key", "certif", "crypto")
        if any(kw in _name_lower for kw in _sec_kws):
            if "gdpr_art_32" not in tags:
                tags.append("gdpr_art_32")

        return tags

    def _tier3_semantic(self, rec: EntityRecord) -> list[tuple[str, str]]:
        """
        Tier 3: Search the docstring for compliance-relevant phrases.

        Returns a list of (article, phrase) tuples found.
        """
        if not rec.docstring:
            return []

        doc_lower = rec.docstring.lower()
        results: list[tuple[str, str]] = []

        for phrase, articles in self.COMPLIANCE_PHRASES.items():
            if phrase in doc_lower:
                for art in articles:
                    results.append((art, phrase))

        return results

    def _check_conflicts(self, rec: EntityRecord) -> None:
        """Warn when an entity carries tags from potentially conflicting domains."""
        tags = set(rec.compliance_tags)
        # Training-focused (art 15) + deployment-focused (art 11) can be a signal
        # worth reviewing manually.
        if "eu_ai_act_art_15" in tags and "eu_ai_act_art_11" in tags:
            logger.warning(
                "Entity '%s' (%s) has both Art 15 (robustness) and Art 11 "
                "(documentation) tags — review for conflicting lifecycle signals",
                rec.entity_name,
                rec.id,
            )


def annotate_compliance(records: list[EntityRecord]) -> list[EntityRecord]:
    """Module-level convenience function for compliance tagging."""
    return ComplianceTagger().annotate(records)
