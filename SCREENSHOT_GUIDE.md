# torchcomply — Screenshot Guide for Conference Slides

Each screenshot maps directly to one or more abstract claims for the talk
*"Building Trust for Users and Regulators Alike: A Cost-Efficient PyTorch Path to Compliance-as-Code"*
at PyTorch Conference Europe 2026, Station F, Paris — April 8.

## Terminal setup before screenshotting

- Font: ≥ 14pt monospace (JetBrains Mono or Fira Code recommended)
- Theme: dark terminal (Monokai / Solarized Dark / Nord)
- Width: ≥ 120 columns so table borders render cleanly
- Run each example fresh (no cached output) immediately before screenshotting

---

## Screenshots to capture

### Example 01 — Audit Trail (`examples/01_audit_trail/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 1 | Terminal output | Audit chain entries with SHA-256 hashes (10 DistilBERT reviews) | Claim 3, 7 |
| 2 | Terminal output | Tamper detection — `INTEGRITY VIOLATION` message after manual hash modification | Claim 7 |
| 3 | `sample_output/audit_waterfall.png` | Waterfall chart of 670 hook entries across all leaf modules | Claim 3 |

---

### Example 02 — Fairness Gate (`examples/02_fairness_gate/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 4 | Terminal output | Epoch table: all 8 epochs BLOCKED, parity 0.137–0.180 > threshold 0.10 | Claim 3, 5 |
| 5 | `sample_output/fairness_trajectory.png` | Dual-axis chart: training loss + demographic parity per epoch | Claim 3 |

---

### Example 03 — Captum Explainability (`examples/03_captum_explain/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 6 | Terminal output | Per-token attribution scores for 5 sentiment reviews | Claim 2, 5 |
| 7 | `sample_output/captum_attribution.png` | Attribution bar chart — top/bottom tokens highlighted | Claim 2 |
| 8 | `sample_output/attribution_heatmap.png` | Heatmap of attribution scores across all 5 samples | Claim 2 |

---

### Example 04 — Opacus DP-SGD (`examples/04_opacus_dp/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 9  | Terminal output | `EpsilonBudgetExceeded` fired at ε=26.0 > max=8.0; final ε=50.0 | Claim 2 |
| 10 | `sample_output/dp_budget_gauge.png` | Gauge chart showing ε progress toward budget limit | Claim 2 |
| 11 | `sample_output/dp_accuracy_tradeoff.png` | Privacy-accuracy tradeoff curve | Claim 2 |

---

### Example 05 — Compliant Dataset (`examples/05_compliant_dataset/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 12 | Terminal output | Consent denial log: 15 subjects denied, reason `consent_withdrawn` | Claim 3 |
| 13 | `sample_output/consent_scatter.png` | Grouped bar: granted vs denied samples across 10 subject groups | Claim 3 |
| 14 | `sample_output/class_distribution.png` | Class distribution with 10:1 imbalance warning | Claim 3 |

---

### Example 06 — Before vs After (`examples/06_before_after/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 15 | Terminal output | Standard checkpoint (46.8 MB) vs compliant checkpoint: 180-entry audit chain, root hash, parity=0.040 | Claim 4, 7 |
| 16 | `sample_output/before_after_comparison.png` | Side-by-side comparison chart | Claim 4 |

---

### Example 07 — CrypTen Secure Inference (`examples/07_crypten_secure/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 17 | Terminal output | MPC encrypted inference timing; all layers covered; error < 0.001 | Claim 2 |
| 18 | `sample_output/crypten_comparison.png` | Standard vs encrypted inference comparison | Claim 2 |

---

### Example 08 — Three Mechanisms (`examples/08_three_mechanisms/`)

**This is the most important technical slide — shows all three PyTorch extension points.**

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 19 | Terminal output | Three-section output: hooks (15 entries) + dispatcher (3 ops) + provenance (1 gradient) | Claim 6 |
| 20 | `sample_output/three_mechanisms.png` | Architecture diagram with all three mechanisms annotated | Claim 6 |

---

### Example 09 — Deployment Monitor (`examples/09_deployment_monitor/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 21 | Terminal output | Concept drift detected at batch 15 (request 700+); 620 routed to human review | Claim 4, 5 |
| 22 | `sample_output/bias_drift.png` | Entropy over batches with drift detection threshold line | Claim 4 |
| 23 | `sample_output/human_interventions.png` | Human review routing count per batch | Claim 4 |

---

### Example 10 — Connected Pipeline (`examples/10_connected_pipeline/`)

**The "everything together" demo — most impactful for the talk summary slide.**

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 24 | Terminal output | All 10 stage summary with pass/fail per stage | All claims |
| 25 | Terminal output | `ComplianceEngine.summary()` box — the compliance certificate | Claim 1, 4, 7 |
| 26 | `sample_output/pipeline_timeline.png` | Timeline of all stages with duration | All claims |
| 27 | `sample_output/coverage_radar.png` | Radar chart of regulatory coverage | All claims |
| 28 | `sample_output/compliance_report.pdf` (page 1) | Annex IV title page | Claim 4, 7 |
| 29 | Terminal output | MLflow Stage 9: experiment logged, run ID shown | Claim 4 |
| 30 | Terminal output | OTel Stage 10: 4 finished spans shown | Claim 5 |

---

### Example 11 — LLM Fine-Tuning (`examples/11_llm_finetune/`)

| # | Source | Description | Abstract claim |
|---|--------|-------------|----------------|
| 31 | Terminal output | LoRA applied: 405,504 trainable / 82M total (0.49%) | Claim 5 |
| 32 | Terminal output | Training table: 3 epochs, loss values, audit entry counts (5,311 total) | Claim 3 |
| 33 | Terminal output | Captum top-5 attributed tokens on generated text | Claim 2, 6 |
| 34 | `sample_output/lora_compliance_card.png` | LoRA architecture card with compliance metadata overlay | Claim 6 |

---

## Abstract claim coverage

| Abstract Claim | Key Screenshots |
|---|---|
| Claim 1: Traditional compliance is document-only | 25 (automated summary box IS the alternative) |
| Claim 2: Opacus + Captum + CrypTen used in isolation today | 7, 9, 17 (each tool separately; 24 shows them connected) |
| Claim 3: Compliance-as-code embeds controls in execution | 1, 2, 4, 12 (live hooks inside transformers) |
| Claim 4: Continuous compliance, reduced audit cost | 21, 22, 28, 29 (post-market monitoring + auto PDF) |
| Claim 5: PyTorch dynamic execution enables runtime checks | 4, 9, 30 (hooks on real attention layers; OTel traces) |
| Claim 6: Dispatcher, autograd, hooks as compliance substrate | 19, 20, 33, 34 (three mechanisms together) |
| Claim 7: Automated documentation and tamper-proof audit trails | 2, 25, 28 (tamper detection + PDF + summary box) |

