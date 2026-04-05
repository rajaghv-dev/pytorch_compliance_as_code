# torchcomply — Regulatory Mapping

This document maps every torchcomply component to the specific regulation articles it satisfies.
Links point to the official consolidated texts published in the EU Official Journal and GDPR-info.eu.

**Regulation texts:**
- EU AI Act (2024/1689): https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
- GDPR (2016/679): https://gdpr-info.eu/

---

## Component → Article mapping

| torchcomply Component | Class / Function | Regulation | Article | Obligation |
|---|---|---|---|---|
| Audit Trail (Mechanism 1) | `AuditChain`, `register_compliance_hooks` | EU AI Act | [Art. 12](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_12) | Record-keeping; immutable hash-chained logs of all high-risk AI operations |
| Dispatcher Trace (Mechanism 2) | `ComplianceTensor.__torch_function__` | EU AI Act | [Art. 12](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_12) | Operator-level traceability; every ATen call logged with shapes and timestamps |
| Gradient Provenance (Mechanism 3) | `ProvenanceLinear` (custom Autograd) | GDPR | [Art. 17](https://gdpr-info.eu/art-17-gdpr/) | Right to Erasure; per-subject gradient attribution enabling machine unlearning |
| Gradient Provenance (Mechanism 3) | `ProvenanceLinear` | EU AI Act | [Art. 12](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_12) | Fine-grained record-keeping: which training samples influenced which weight updates |
| Fairness Gate | `FairnessGate`, `compute_demographic_parity` | EU AI Act | [Art. 10(2)(f)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_10) | Examination of training data for possible biases; demographic parity enforcement |
| Compliant Dataset | `CompliantDataset`, `ConsentRegistry` | GDPR | [Art. 6](https://gdpr-info.eu/art-6-gdpr/), [Art. 7](https://gdpr-info.eu/art-7-gdpr/) | Lawfulness of processing; explicit consent per data subject and per purpose |
| Compliant Dataset | `DatasetProfile` warnings | EU AI Act | [Art. 10(2)(f)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_10) | Class imbalance detection; data representativeness assessment |
| Consent Access Log | `ConsentRegistry.access_log` | GDPR | [Art. 5(2)](https://gdpr-info.eu/art-5-gdpr/) | Accountability principle; auditable record of all consent decisions and denials |
| Privacy Engine (DP-SGD) | `CompliancePrivacyEngine` (Opacus) | GDPR | [Art. 25](https://gdpr-info.eu/art-25-gdpr/) | Privacy by Design; DP-SGD Gaussian noise injection during training |
| Privacy Engine (DP-SGD) | `CompliancePrivacyEngine` (Opacus) | GDPR | [Art. 32](https://gdpr-info.eu/art-32-gdpr/) | Security of processing; technical measures to protect personal data during ML training |
| Secure Inference (MPC) | `ComplianceSecureInference` (CrypTen) | GDPR | [Art. 25](https://gdpr-info.eu/art-25-gdpr/) | Privacy by Default; encrypted inference ensures server never sees plaintext inputs |
| Explainability | `ComplianceExplainer` (Captum) | EU AI Act | [Art. 13](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_13) | Transparency; human-interpretable token/pixel attributions via Integrated Gradients |
| Explainability | `ComplianceExplainer` (Captum) | EU AI Act | [Art. 14(3)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_14) | Human oversight; enables operators to understand and verify system output |
| Explainability | `ComplianceExplainer` (Captum) | GDPR | [Art. 22(3)](https://gdpr-info.eu/art-22-gdpr/) | Right to explanation for automated individual decisions |
| Compliance Engine | `ComplianceEngine.generate_report` | EU AI Act | [Art. 11](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_11) | Technical documentation requirement for conformity assessment |
| Annex IV PDF | `AnnexIVReport` | EU AI Act | [Art. 11](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_11) | Technical documentation drawn up before market placement; covers §1–§4 of Annex IV |
| Annex IV PDF | `AnnexIVReport` | EU AI Act | [Art. 43](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_43) | Conformity assessment; PDF provides evidence package for notified body review |
| Annex IV PDF | `AnnexIVReport` | EU AI Act | [Annex IV](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#anx_IV) | All six Annex IV sections: general description, design specs, training methodology, validation |
| MLflow Logger | `ComplianceMLflowLogger` | EU AI Act | [Art. 9](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_9) | Risk management system; compliance evidence versioned alongside model artifacts |
| MLflow Logger | `ComplianceMLflowLogger` | EU AI Act | [Art. 12](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_12) | Record-keeping; logs can be used post-hoc to verify system behaviour |
| MLflow Logger | `ComplianceMLflowLogger` | EU AI Act | [Art. 17](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_17) | Quality management system; versioned metrics support ongoing QMS evidence |
| OTel Logger | `OtelComplianceLogger` | EU AI Act | [Art. 14](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_14) | Human oversight; real-time observability of AI system decisions in production |
| OTel Logger | `OtelComplianceLogger` | EU AI Act | [Art. 61](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_61) | Post-market monitoring; spans provide live observability during deployment |
| Chain Integrity | `AuditChain.verify()` | EU AI Act | [Art. 12](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_12) | Tamper detection; cryptographic (SHA-256) proof of log integrity |
| Summary Box | `ComplianceEngine.summary()` | EU AI Act | [Art. 17](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_17) | Quality management; operator-visible compliance status at a glance |
| Post-Market Monitor | `09_deployment_monitor` example | EU AI Act | [Art. 61](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_61) | Post-market monitoring; bias drift detection during live deployment |
| Compliance Diff | `ComplianceDiff`, `ComplianceSnapshot` | EU AI Act | [Art. 9](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_9) | Ongoing risk management; run-to-run regression detection in CI |

---

## Three PyTorch Compliance Mechanisms

torchcomply uses all three PyTorch-native extension points as compliance substrates:

| # | Mechanism | PyTorch API | Granularity | Primary Use |
|---|-----------|-------------|-------------|-------------|
| 1 | Forward hooks | `register_forward_hook` | Module-level (per layer) | Audit trail: what ran, when, with what shapes |
| 2 | Dispatcher | `__torch_function__` on Tensor subclass | Operator-level (per ATen op) | Full operator trace without model code changes |
| 3 | Custom Autograd | `torch.autograd.Function` | Gradient-level (per backward pass) | Data provenance: which subjects influenced which weight updates |

PyTorch extension point documentation:
- Forward hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- `__torch_function__`: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api
- Custom Autograd: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd

---

## GDPR Article 22 — Automated Individual Decision-Making

Article 22 gives data subjects the right not to be subject to solely automated decisions
that produce legal or significant effects. Where Article 22 applies:

- The `ComplianceExplainer` (Captum) satisfies Art. 22(3): *"meaningful information about the
  logic involved"* — per-token attributions make model reasoning interpretable.
- The `FairnessGate` provides evidence that the automated system does not systematically
  disadvantage protected groups.
- The `AnnexIVReport` documents the system's decision-making logic for regulatory review.

Reference: https://gdpr-info.eu/art-22-gdpr/
