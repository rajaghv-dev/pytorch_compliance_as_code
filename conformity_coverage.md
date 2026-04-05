# EU AI Act — High-Risk AI System Conformity Coverage

> **Coverage: 18 of 37 requirements fully covered (49%); 34 of 37 addressed (92%)**
>
> *"Addressed" = fully covered (✅) + partially covered (⚠️). The 3 out-of-scope items
> are administrative/legal obligations that no Python library can fulfill.*

This document maps each requirement to the torchcomply component that satisfies it.
Article references link to the official consolidated EU AI Act text in the EU Official Journal:
https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689

Legend: ✅ Fully covered | ⚠️ Partial (framework support only, or deployment-layer required) | ❌ Out of scope (legal/administrative)

---

## Chapter III, Section 2 — Obligations for providers of high-risk AI systems

| # | Article | Requirement | Coverage | torchcomply Component |
|---|---------|-------------|----------|-----------------------|
| 1 | Art. 9(1) | Establish risk management system | ✅ | `ComplianceEngine`, `FairnessGate` |
| 2 | Art. 9(2) | Identify and analyse known risks | ⚠️ | `FairnessGate` (fairness risks only) |
| 3 | Art. 9(3) | Adopt risk management measures | ⚠️ | `CompliancePrivacyEngine` (DP), `FairnessGate` |
| 4 | Art. 9(4) | Test to identify appropriate measures | ⚠️ | Test suite (`pytest torchcomply/tests/` — 37 tests) |
| 5 | Art. 10(1) | Data governance practices | ✅ | `CompliantDataset`, `ConsentRegistry` |
| 6 | Art. 10(2) | Training/validation/test data relevance | ⚠️ | `DatasetProfile` imbalance warnings |
| 7 | Art. 10(3) | Data practices suitable for intended purpose | ⚠️ | `DatasetProfile`, `ConsentRegistry` |
| 8 | Art. 10(4) | Examine data for biases | ✅ | `FairnessGate`, `compute_demographic_parity` |
| 9 | Art. 10(5) | Data protection measures | ✅ | `CompliancePrivacyEngine` (DP-SGD), `ComplianceSecureInference` (CrypTen) |
| 10 | Art. 11(1) | Technical documentation drawn up | ✅ | `AnnexIVReport.save_pdf()` |
| 11 | Art. 11(2) | Keep documentation updated | ⚠️ | Manual; `generated_at` timestamp in report |
| 12 | Art. 12(1) | Logging capabilities | ✅ | `AuditChain`, `register_compliance_hooks` |
| 13 | Art. 12(2) | Automatically log events | ✅ | `AuditChain` (forward hook — every operation) |
| 14 | Art. 12(3) | Log events for a minimum period | ⚠️ | `AuditChain.to_json()` persists; retention policy is a deployment concern |
| 15 | Art. 13(1) | Ensure sufficient transparency | ✅ | `ComplianceExplainer` (Captum), `engine.summary()` |
| 16 | Art. 13(2) | Instructions for use | ⚠️ | `regulatory_mapping.md`, `SCREENSHOT_GUIDE.md` |
| 17 | Art. 13(3) | Content of instructions for use | ⚠️ | `AnnexIVReport` Section 1 (General Description) |
| 18 | Art. 14(1) | Human oversight measures | ✅ | `FairnessGate` blocks training; `AuditChain` enables review |
| 19 | Art. 14(3) | Ability to understand system output | ✅ | `ComplianceExplainer` (Captum IntegratedGradients) |
| 20 | Art. 14(4) | Ability to decide not to use output | ✅ | `FairnessGate.on_epoch_end()` raises `ComplianceViolation` |
| 21 | Art. 14(5) | Oversight by natural persons | ⚠️ | Framework for human review; actual human process is external |
| 22 | Art. 15(1) | Achieve appropriate accuracy | ⚠️ | Test suite; no automated accuracy threshold enforcement yet |
| 23 | Art. 15(2) | Accuracy metrics stated | ⚠️ | `AnnexIVReport` training config section |
| 24 | Art. 15(3) | Robustness to errors | ⚠️ | `AuditChain.verify()` detects tampering |
| 25 | Art. 15(4) | Cybersecurity measures | ⚠️ | `ComplianceSecureInference` (CrypTen MPC) |
| 26 | Art. 17(1) | Quality management system | ✅ | `ComplianceEngine` + full test suite (37 torchcomply tests) |
| 27 | Art. 17(2) | QMS covers design and development | ✅ | `AnnexIVReport` Sections 2–4 |
| 28 | Art. 17(3) | QMS covers post-market monitoring | ✅ | `09_deployment_monitor` example |

## Chapter VI — Post-market monitoring

| # | Article | Requirement | Coverage | torchcomply Component |
|---|---------|-------------|----------|-----------------------|
| 29 | Art. 61(1) | Post-market monitoring system | ✅ | `09_deployment_monitor`, `AuditChain` |
| 30 | Art. 61(3) | Serious incident reporting | ❌ | Out of scope — deployment/ops layer |
| 31 | Art. 61(4) | Document monitoring plan | ⚠️ | Root-level docs (`regulatory_mapping.md`, `static_analysis.md`); no automated plan generation |

## Annex IV — Technical Documentation

| # | Section | Requirement | Coverage | torchcomply Component |
|---|---------|-------------|----------|-----------------------|
| 32 | §1 | General description | ✅ | `AnnexIVReport` title page |
| 33 | §2 | Design specifications | ✅ | `ModelIntrospector` + Section 2 |
| 34 | §3 | Training methodology | ✅ | `AnnexIVReport` Section 4 |
| 35 | §4 | Validation/testing | ⚠️ | `AnnexIVReport` fairness section |
| 36 | §5 | Changes and versions | ❌ | Manual versioning; not automated |
| 37 | §6 | EU declaration of conformity | ❌ | Legal/administrative; out of scope |

---

## Summary

| Category | Count | % of 37 |
|----------|-------|---------|
| ✅ Fully covered | **18** | 49% |
| ⚠️ Partial coverage | **16** | 43% |
| ❌ Out of scope | **3** | 8% |
| **Total tracked** | **37** | 100% |

**Addressed (✅ + ⚠️): 34 of 37 = 92%**

The 3 out-of-scope requirements are administrative/legal obligations:
- **Serious incident reporting** (Art. 61(3)) — requires organisational processes and notified body contact
- **Change and version history** (Annex IV §5) — requires manual versioning and CI/CD changelog discipline
- **EU Declaration of Conformity** (Annex IV §6) — legal document issued by the provider's responsible person after notified body assessment

No Python library can generate these. torchcomply generates the *technical evidence* that supports them.

### How this compares to existing tools

| Tool | Approach | Coverage |
|------|----------|----------|
| torchcomply | Runtime enforcement + automated Annex IV PDF | 92% addressed |
| IBM AI Fairness 360 | Bias metrics only | ~Art. 10 only |
| Opacus alone | DP-SGD only | ~Art. 25, 32 only |
| Captum alone | Attribution only | ~Art. 13 only |
| Manual compliance | Document-only checklists | 0% automated |

References:
- IBM AI Fairness 360: https://aif360.mybluemix.net/
- Microsoft Fairlearn: https://fairlearn.org/
- EU AI Act official text: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
