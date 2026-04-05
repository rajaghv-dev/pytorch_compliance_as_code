"""
Validates entity-to-article compliance mappings using phi4 via Ollama.

For each entity where mapping_confidence < 0.7 and compliance_tags is non-empty,
the validator asks phi4 to rate how directly the entity supports the mapped
legal obligation, then updates confidence and rationale accordingly.

EU AI Act and GDPR are separate legal frameworks. They are kept in distinct
description dicts so the system prompt correctly names the applicable framework
for each tag — asking phi4 about a "GDPR obligation" vs an "EU AI Act obligation"
produces meaningfully different and more accurate relevance ratings.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.extractors.base import EntityRecord

from .ollama_client import OllamaClient, OllamaError


logger = logging.getLogger("mapping_validator")

# Model used for mapping validation — qwen3.5:35b provides stronger
# reasoning for cross-article dependency chains and confidence calibration.
MODEL = "qwen3.5:35b"


class MappingValidator:
    """Validates entity->article compliance mappings using a reasoning LLM.

    EU AI Act and GDPR are distinct legal frameworks; each has its own
    description dict so the phi4 prompt correctly names the framework.
    """

    # EU AI Act — Regulation (EU) 2024/1689 on Artificial Intelligence
    EU_AI_ACT_DESCRIPTIONS: dict[str, str] = {
        "eu_ai_act_art_9":  "Risk management system: identify, analyse, evaluate and mitigate risks throughout the AI system lifecycle",
        "eu_ai_act_art_10": "Data governance: training, validation and test data quality, relevance, and representativeness",
        "eu_ai_act_art_11": "Technical documentation: comprehensive documentation before market placement or service provision",
        "eu_ai_act_art_12": "Record-keeping: automatic logging of events to enable post-deployment audit trail",
        "eu_ai_act_art_13": "Transparency: clear instructions for use, capability disclosures, limitations",
        "eu_ai_act_art_14": "Human oversight: measures enabling humans to effectively monitor and intervene in AI system operation",
        "eu_ai_act_art_15": "Accuracy, robustness and cybersecurity: performance levels, resilience to errors and adversarial inputs",
        "eu_ai_act_art_17": "Quality management system: documented policies covering design, development, testing, monitoring",
        "eu_ai_act_art_43": "Conformity assessment: procedures to verify compliance before deployment",
        "eu_ai_act_art_61": "Post-market monitoring: proactive collection and review of data throughout the system lifetime",
        "eu_ai_act_art_72": "Serious incident reporting: notification obligations to market surveillance authorities",
    }

    # GDPR — Regulation (EU) 2016/679 on data protection (General Data Protection Regulation)
    # Distinct from the EU AI Act: different legal basis, enforcement body (DPAs vs. national AI authorities),
    # and scope (personal data processing vs. AI system risk classification).
    GDPR_DESCRIPTIONS: dict[str, str] = {
        "gdpr_art_5":  "Principles of data processing: lawfulness, fairness, transparency, purpose limitation, data minimisation, accuracy, storage limitation, integrity and confidentiality",
        "gdpr_art_6":  "Lawfulness of processing: legal bases including consent, contract, legal obligation, vital interests, public task, legitimate interests",
        "gdpr_art_17": "Right to erasure ('right to be forgotten'): obligation to erase personal data on request under specified conditions",
        "gdpr_art_22": "Automated individual decision-making: rights and safeguards where decisions are made solely by automated processing with significant legal or similarly significant effects",
        "gdpr_art_25": "Data protection by design and by default: technical and organisational measures to implement data protection principles from the outset",
        "gdpr_art_32": "Security of processing: appropriate technical and organisational measures to ensure a level of security appropriate to the risk (encryption, pseudonymisation, resilience)",
    }

    # Combined lookup for convenience
    @property
    def _all_descriptions(self) -> dict[str, tuple[str, str]]:
        """Returns {tag: (framework_label, description)} for all known tags."""
        result = {tag: ("EU AI Act", desc) for tag, desc in self.EU_AI_ACT_DESCRIPTIONS.items()}
        result.update({tag: ("GDPR", desc) for tag, desc in self.GDPR_DESCRIPTIONS.items()})
        return result

    def __init__(self, client: OllamaClient):
        """
        Initialise the mapping validator.

        Parameters
        ----------
        client : OllamaClient
            Configured Ollama client instance.
        """
        self.client = client
        self.logger = logging.getLogger("mapping_validator")

    def validate_single(
        self,
        entity_name: str,
        docstring: str,
        article_tag: str,
    ) -> dict:
        """Validate a single entity-article mapping via phi4.

        Parameters
        ----------
        entity_name : str
            Name of the PyTorch entity.
        docstring : str
            Entity docstring (truncated to 500 chars).
        article_tag : str
            Compliance tag key (e.g. "eu_ai_act_art_9").

        Returns
        -------
        dict
            {"relevance": "direct"|"indirect"|"none", "confidence": float, "reason": str}
            Returns empty dict on failure.
        """
        all_desc = self._all_descriptions
        if article_tag not in all_desc:
            return {}

        framework_label, article_description = all_desc[article_tag]
        # Human-readable article label, e.g. "Art 15" or "Art 22"
        article_label = article_tag.split("_art_")[-1].replace("_", ".")
        full_label = f"{framework_label} Art.{article_label}"

        system_prompt = (
            f"You are a compliance analyst specialising in EU technology regulation. "
            f"EU AI Act and GDPR are distinct legal frameworks. "
            f"Rate how directly a PyTorch entity supports a specific {framework_label} obligation. "
            'Respond with ONLY a JSON object: {"relevance": "direct"|"indirect"|"none", '
            '"confidence": 0.0-1.0, "reason": "one sentence"}'
        )

        user_prompt = (
            f"PyTorch entity: {entity_name}\n"
            f"Docstring: {docstring[:500]}\n"
            f"Legal obligation ({framework_label}): {full_label} — {article_description}"
        )

        try:
            start_time = time.time()
            result = self.client.generate(
                model=MODEL,
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.1,
                format="json",
                max_retries=3,
            )
            elapsed = time.time() - start_time
            self.logger.info(
                "Validated %s <-> %s in %.2fs: relevance=%s confidence=%.2f",
                entity_name, article_tag, elapsed,
                result.get("relevance", "?"), result.get("confidence", 0),
            )

            # Validate response structure
            relevance = result.get("relevance", "")
            if relevance not in ("direct", "indirect", "none"):
                self.logger.warning(
                    "WARN: Unexpected relevance value '%s' for %s, treating as 'none'",
                    relevance, entity_name,
                )
                result["relevance"] = "none"

            confidence = result.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                self.logger.warning(
                    "WARN: Invalid confidence %.2f for %s, clamping",
                    confidence, entity_name,
                )
                result["confidence"] = max(0.0, min(1.0, float(confidence)))

            return result

        except OllamaError as e:
            self.logger.error("ERR: Validation failed for %s <-> %s: %s", entity_name, article_tag, e)
            return {}
        except Exception as e:
            self.logger.error("ERR: Unexpected error validating %s: %s", entity_name, e)
            return {}

    def validate_records(self, records: list["EntityRecord"]) -> list["EntityRecord"]:
        """Validate all records with low-confidence mappings in batch.

        For each entity where mapping_confidence < 0.7 and compliance_tags is
        non-empty, validates each tag against the entity and updates confidence,
        rationale, and tag list accordingly.

        Parameters
        ----------
        records : list[EntityRecord]
            All entity records to process.

        Returns
        -------
        list[EntityRecord]
            The same records list, mutated in place with updated fields.
        """
        # Check model availability first
        if not self.client.check_model_available(MODEL):
            self.logger.warning("WARN: Model %s not available, skipping mapping validation", MODEL)
            return records

        # Filter to records needing validation
        candidates = [
            r for r in records
            if r.mapping_confidence < 0.7 and r.compliance_tags
        ]
        self.logger.info(
            "Mapping validation: %d candidates out of %d total records",
            len(candidates), len(records),
        )

        processed = 0
        for record in candidates:
            # Skip if already validated by LLM (resume support)
            if record.metadata.get("llm_mapping_validated"):
                continue

            tags_to_remove: list[str] = []

            all_desc = self._all_descriptions
            for tag in list(record.compliance_tags):
                if tag not in all_desc:
                    continue

                result = self.validate_single(
                    entity_name=record.entity_name,
                    docstring=record.docstring,
                    article_tag=tag,
                )

                if not result:
                    # LLM call failed; keep existing mapping unchanged
                    continue

                relevance = result.get("relevance", "none")
                confidence = result.get("confidence", 0.0)
                reason = result.get("reason", "")

                if relevance == "direct":
                    record.mapping_confidence = max(record.mapping_confidence, confidence)
                    record.mapping_rationale = reason
                elif relevance == "indirect":
                    record.mapping_confidence = max(record.mapping_confidence, confidence * 0.7)
                    # Keep the tag but note it's indirect
                elif relevance == "none":
                    tags_to_remove.append(tag)

            # Remove tags that the LLM determined are irrelevant
            for tag in tags_to_remove:
                if tag in record.compliance_tags:
                    record.compliance_tags.remove(tag)
                    self.logger.info(
                        "Removed tag %s from %s (LLM: not relevant)",
                        tag, record.entity_name,
                    )

            # Mark as validated for resume support
            record.metadata["llm_mapping_validated"] = True

            processed += 1
            if processed % 10 == 0:
                self.logger.info("Mapping validation progress: %d/%d", processed, len(candidates))

        self.logger.info("Mapping validation complete: %d entities processed", processed)
        return records
