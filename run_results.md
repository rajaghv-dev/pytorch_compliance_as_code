# torchcomply — Example Run Results

> **Status: 11/11 examples passing** | Last run: 2026-04-05 | Environment: CPU-only, PyTorch 2.11.0

---

## Summary Table

| # | Example | Status | Time | Key Result |
|---|---------|--------|------|------------|
| 01 | `audit_trail` | ✅ PASS | 5.3s | 670-entry SHA-256 chain; tamper detection demonstrated; chain integrity verified |
| 02 | `fairness_gate` | ✅ PASS | 2.7s | Parity 0.137 > threshold 0.10 → all 8 epochs BLOCKED |
| 03 | `captum_explain` | ✅ PASS | 4.5s | 5 DistilBERT reviews attributed via LayerIntegratedGradients; persisted to JSONL |
| 04 | `opacus_dp` | ✅ PASS | 15.6s | ε=50.0; `EpsilonBudgetExceeded` fired at ε=26.0 > max=8.0 (epoch 1); DP accuracy 39.9% vs baseline 54.8% |
| 05 | `compliant_dataset` | ✅ PASS | 0.5s | 485 granted, 15 denied; 10:1 class imbalance warning; purpose-scope demo (analytics: 213 denied) |
| 06 | `before_after` | ✅ PASS | 4.6s | Standard checkpoint vs compliant: 180-entry audit chain, parity=0.040, root hash |
| 07 | `crypten_secure` | ✅ PASS | 2.4s | MPC encrypted inference ~3× overhead (17ms vs 6ms); max error 0.000878; all layers covered |
| 08 | `three_mechanisms` | ✅ PASS | 1.4s | Hooks (15 entries) + Dispatcher (3 ops) + Autograd (1 gradient provenance record) |
| 09 | `deployment_monitor` | ✅ PASS | 11.5s | Concept drift detected at batch 15 (request 700+); 620 routed to human review |
| 10 | `connected_pipeline` | ✅ PASS | 5.5s | All 10 stages pass; 108 audit entries; parity=0.061; Annex IV PDF; MLflow + OTel |
| 11 | `llm_finetune` | ✅ PASS | 22.4s | DistilGPT-2 + LoRA (0.49% trainable); 5,311 audit entries; 30 granted / 10 denied |

**Total wall time: ~76s for all 11 examples**

---

## Test Suite

```
37 passed, 6 warnings in 9.92s
```

All 37 unit tests pass on CPU. The 6 warnings are cosmetic deprecation notices from upstream libraries, not from torchcomply itself.

---

## Generated Artifacts

Each example writes to its own `sample_output/` directory.

| Example | PNG files | Other |
|---------|-----------|-------|
| 01 | `audit_waterfall.png` | `audit_chain.jsonl` (1.4 MB, 670 entries) |
| 02 | `fairness_trajectory.png` | — |
| 03 | `captum_attribution.png`, `attribution_heatmap.png` | `attribution_log.jsonl` (5 records) |
| 04 | `dp_budget_gauge.png`, `dp_accuracy_tradeoff.png` | — |
| 05 | `consent_scatter.png`, `class_distribution.png` | — |
| 06 | `before_after_comparison.png` | `audit_chain.jsonl` (354 KB, 180 entries) |
| 07 | `crypten_comparison.png` | — |
| 08 | `three_mechanisms.png` | — |
| 09 | `bias_drift.png`, `human_interventions.png` | MLflow run logged |
| 10 | `pipeline_timeline.png`, `coverage_radar.png` | `compliance_report.pdf`, `audit_chain.jsonl` |
| 11 | `lora_compliance_card.png` | `compliance_report.pdf`, `audit_chain.jsonl` |

---

## Known Limitations

| Item | Detail |
|------|--------|
| CrypTen (Ex 07, 10) | Requires `pip install crypten --no-build-isolation`; examples handle missing install gracefully |
| Example 04 first run | CIFAR-10 download (~170 MB); subsequent runs use local cache |
| Emoji rendering | `❌` / `✅` glyphs missing from DejaVu Sans in matplotlib — cosmetic only, does not affect output files |
