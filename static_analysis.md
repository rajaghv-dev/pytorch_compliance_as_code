# Static Analysis Pipeline

The static analysis pipeline answers a deceptively simple question: *which parts of PyTorch already do compliance work, and for which obligations?* Before you can enforce anything at runtime, you need to know what the codebase can actually support. This pipeline turns that question into machine-readable evidence.

---

## The Core Insight

PyTorch is not compliance-unaware. `register_forward_hook` is an observability primitive that maps directly to EU AI Act Art. 12 (record-keeping) and Art. 61 (post-market monitoring). `use_deterministic_algorithms` is a reproducibility contract that addresses Art. 15 (robustness). `DataLoader` with consent-gated access is GDPR Art. 6/7 in code form.

None of this is labelled. No one wrote "this function is for Article 12." The compliance meaning is latent — embedded in naming conventions, docstrings, behavioural patterns, C++ dispatch keys, and test coverage. The pipeline's job is to surface it systematically, at scale, across a codebase with millions of lines.

---

## Pipeline Architecture

```
PyTorch source tree
      │
      ▼
Phase 1: Catalog       — find all .py / .cpp / .rst / .yaml files
      │
      ▼
Phase 2: Extract       — 11 specialized extractors → EntityRecord list
      │
      ▼
Phase 3: Annotate      — 3-tier compliance tagger enriches every record
      │
      ▼
Phase 4: Organize      — dedup, cross-reference, index, validate
      │
      ▼
Phase 5: Convert       — RDF/Turtle, CSV, Markdown chapters, SPARQL notebooks
      │
      ▼
Phase 6: LLM Enrich    — Ollama-based legal parsing, relation extraction, validation
```

Each phase writes its completion to `storage/session/checkpoint.json`. Re-running with `--resume` skips already-completed phases. During Phase 2 (which can run for hours on the full PyTorch repo), the session auto-saves every 2 minutes so an interrupted run does not lose progress.

Run the pipeline with:

```bash
pct --config configs/default.yaml          # full pipeline, all 11 extractors
pct --config configs/talk_demo.yaml        # fast offline demo, 3 extractors, no LLM
pct --resume                               # skip phases already completed
```

---

## Phase 1 — Catalog

Four language-aware crawlers walk the repository tree:

| Crawler | Target | What it finds |
|---------|--------|---------------|
| `python_extractor` | `*.py` | Functions, methods, classes via AST |
| `cpp_extractor` | `*.cpp`, `*.h` | Operators, dispatch keys, C++ hooks |
| `rst_extractor` | `*.rst` | Sphinx directives, `.. deprecated::`, `.. warning::` |
| `yaml_extractor` | `native_functions.yaml` | All ~2,000 native operator declarations |

The catalog does not interpret anything yet. It builds a map of what exists, with source locations, before any compliance reasoning happens. This keeps later phases honest — they cannot invent entities that were not actually found.

---

## Phase 2 — Extraction

### The EntityRecord

Every piece of evidence is a typed `EntityRecord` — a 27-field dataclass with a stable SHA-256 identity:

```python
id = SHA256(source_file + module_path + entity_name +
            start_line + entity_type + subcategory)[:20]
```

The six-field key prevents collision for overloaded methods, same-named helpers across submodules, and entities at different line numbers. The record carries identity, content, relations, compliance annotations, and dual confidence scores (`extraction_confidence`, `mapping_confidence`) so downstream steps always know how much to trust a signal.

### The 11 Extractors

Each extractor targets one axis of compliance evidence. Together they cover the full regulatory surface.

**Hookability**

Runs 6 passes over both Python AST and C++ source:

1. Hook *definitions* — functions with names in `HOOK_METHODS`
2. Hook *consumers* — call sites that invoke those hooks (knowing where hooks are *used*, not just *defined*)
3. C++ `DispatchKey` enum values — the operator dispatch layer
4. Override protocols — `__torch_function__`, `__torch_dispatch__`, `TorchFunctionMode`
5. Profiler hooks — `profile`, `record_function`, `ExecutionTraceObserver`
6. Module lifecycle callbacks — `train`, `eval`, `apply`, `zero_grad`

Hookability is the technical prerequisite for every runtime compliance control. `register_forward_hook` is what makes `AuditChain` (Art. 12) and `OtelComplianceLogger` (Art. 14) possible. The extractor maps the *attachment surface* available for compliance instrumentation.

**Operator Determinism**

Six independent passes:

1. `native_functions.yaml` — `device_check` flags on ~2,000 operators
2. `use_deterministic_algorithms` / `are_deterministic_algorithms_enabled` call sites
3. cuDNN benchmark/deterministic configuration (`torch.backends.cudnn.*`)
4. RNG management — seeds, `fork_rng`, `get_rng_state`, `set_rng_state`
5. Numerical stability patterns (with false-positive filter: excludes matches in comments, assertions, log strings)
6. Determinism test cases

EU AI Act Art. 15 requires that accuracy and robustness claims be substantiated. This extractor produces the evidence trail: which operators are flagged non-deterministic, which are gated by a determinism mode, and whether tests exist that verify the behaviour.

**API Documentation**

Extracts Sphinx docstrings, deprecation notices, type annotations, and cross-references. When the Annex IV report references a function, the documentation extractor has already collected what the API itself says about its behaviour, limitations, and usage contracts.

**Commit History**

Walks `git log` for commits that touch compliance-relevant files, classifying messages by pattern: determinism fixes, privacy additions, security patches, data governance changes. This is evidence of ongoing Art. 9 risk management — the regulator can see that the maintainers tracked and addressed these concerns over time.

**Compliance Tools**

Detects Opacus (`PrivacyEngine`, `GradSampleModule`, `DPOptimizer`, `RDPAccountant`), Captum (`LayerIntegratedGradients`, `IntegratedGradients`), and CrypTen (`crypten.mpc`, `SecretShare`) usage patterns. Bridges between what PyTorch natively offers and what the compliance-specific libraries add on top.

**Data Provenance**

Tracks `DataLoader`, `Dataset`, `Subset`, `random_split`, `DistributedSampler`, and consent-adjacent patterns. Data governance under Art. 10 requires knowing not just *what* data was used, but *how* it was selected, split, and weighted.

**Export Boundary**

Maps which APIs survive `torch.export()` and which do not — hooks, custom functions, dynamic control flow. The export boundary is a compliance boundary: a model deployed after export behaves differently from one deployed inline. Knowing which compliance mechanisms survive export prevents the gap where a well-instrumented training run produces a stripped inference artifact.

**Module Hierarchy**

Builds the inheritance graph of `nn.Module` subclasses. Compliance controls that rely on `register_forward_hook` only work if the hook attachment point is reachable. Understanding which modules expose hookable surfaces — and which are wrapped or frozen — is prerequisite to knowing where `AuditChain` and `ComplianceTensor` can actually be attached.

**Sphinx Notes**

Extracts `.. deprecated::`, `.. versionadded::`, `.. warning::`, `.. note::` directives from RST documentation. Deprecation notices are an under-appreciated compliance signal: an API marked deprecated but still present is a liability if it carries compliance semantics.

**Supply Chain**

Scans `requirements*.txt`, `setup.py`, `pyproject.toml`, `CMakeLists.txt` for dependencies and their compliance relevance. Supply chain transparency (Art. 11, SBOM) means knowing that Opacus provides differential privacy, that CrypTen provides MPC, that OpenTelemetry provides spans — declared, versioned, and auditable.

**Test Suite**

Discovers `test_determinism_*`, `test_privacy_*`, `test_fairness_*`, and `test_audit_*` test functions. Tests are evidence. Art. 9 risk management requires ongoing verification; the presence of compliance-specific tests demonstrates that properties are continuously validated, not just claimed once.

---

## Phase 3 — Annotation

The compliance tagger applies three layers of reasoning to every extracted record. Results are merged by union (any tier can contribute tags; the highest confidence wins).

### Tier 1 — Direct API Name Match (confidence 0.80)

A curated map of 116 API names to regulation articles. Examples:

| API | Articles |
|-----|---------|
| `register_forward_hook` | Art. 61, Art. 14, Art. 12 |
| `use_deterministic_algorithms` | Art. 15 |
| `DataLoader` | Art. 10, GDPR Art. 5 |
| `save` / `load` / `state_dict` | Art. 12 |
| `PrivacyEngine` | GDPR Art. 25 |
| `predict` / `inference` / `classify` | GDPR Art. 22, Art. 13 |

The map was built from the regulatory text: each article was read, its operative obligations identified, and the PyTorch APIs that mechanically implement those obligations were enumerated. A function is only in the map if there is a direct, articulable connection.

### Tier 2 — AST Structural Patterns (confidence 0.85)

Structural evidence is stronger than name evidence because it is harder to spoof. Tier 2 inspects:

- `subcategory` field — set by extractors based on AST node type, C++ enum membership, or YAML flags
- Inheritance relations — a class inheriting from `BasePrivacyEngine` implies GDPR Art. 25 regardless of what it is named
- Decorator patterns — `@torch.no_grad()`, `@torch.inference_mode()` imply Art. 15 semantics

Tier 2 catches renamed APIs, internal helpers, and extension points that Tier 1 misses because they do not appear in the canonical API surface.

### Tier 3 — Docstring Semantic Phrase Matching (confidence 0.50)

The lowest-confidence tier, essential for catching the long tail. Phrases like:

- `"non-deterministic"`, `"hardware dependent"`, `"cuda atomics"` → Art. 15
- `"audit trail"`, `"tamper"`, `"record"` → Art. 12
- `"differential privacy"`, `"per-sample gradient"`, `"noise multiplier"` → GDPR Art. 25
- `"right to erasure"`, `"machine unlearning"` → GDPR Art. 17
- `"human-in-the-loop"`, `"manual override"`, `"stop training"` → Art. 14

These phrases appear in docstrings of internal implementation functions that are never exposed as public APIs. Without Tier 3, the compliance signal from these functions would be invisible.

A Tier-3 tag is a lead, not a verdict. The LLM enrichment phase (Phase 6) reviews low-confidence annotations and either confirms or rejects them.

---

## Phase 4 — Organize

Four operations clean and connect the annotated records:

**Deduplication.** The stable SHA-256 ID prevents the same entity from appearing twice if multiple extractors find it. Without deduplication, a function like `register_forward_hook` — discovered by the hookability extractor, referenced by the API documentation extractor, and found in a test by the test suite extractor — would produce three records that look independent.

**Cross-referencing.** Links are drawn between hook definitions ↔ hook consumers, operators ↔ their determinism flags, compliance tool APIs ↔ their test coverage, and export-surviving APIs ↔ their documentation entries. These links enable multi-hop SPARQL queries.

**Entity indexing.** A lookup structure by entity type, module path, compliance tag, and confidence band.

**Validation.** Checks that every record has a stable ID, a source file that still exists, at least one compliance tag if its `mapping_confidence` exceeds 0.5, and that confidence values are in range. Records that fail validation are flagged but not silently dropped.

---

## Phase 5 — Conversion

The same evidence is serialised in four forms:

**RDF/Turtle knowledge graph** (`compliance_graph.ttl`)

```turtle
pct:entity_3a7f9b2c41 a pct:ComplianceEntity ;
    pct:entityName "register_forward_hook" ;
    pct:sourceFile "torch/nn/modules/module.py" ;
    pct:hasComplianceTag eu:art_61, eu:art_14, eu:art_12 ;
    pct:mappingConfidence 0.80 ;
    pct:lifecyclePhase "forward_pass" .
```

RDF enables SPARQL queries that no CSV can answer:

```sparql
SELECT ?name ?article WHERE {
    ?e pct:hasComplianceTag ?article ;
       pct:lifecyclePhase "training_only" ;
       pct:exportSurvival "false" .
    BIND(pct:entityName(?e) AS ?name)
}
```

This query finds all training-phase compliance mechanisms that do not survive `torch.export()` — a gap analysis in one query. The knowledge graph makes compliance coverage a queryable property of the codebase, not a manually maintained spreadsheet.

**CSV** — flat record export for analysis in pandas, Excel, or any compliance tool that accepts tabular input.

**Markdown chapters** — human-readable sections grouped by regulation article, suitable for inclusion in technical documentation or Annex IV supplements.

**SPARQL notebooks** — Jupyter notebooks with pre-written queries covering the most common compliance questions. Pre-built queries are in `queries/`.

---

## Phase 6 — LLM Enrichment

Static analysis reaches a ceiling: it cannot resolve ambiguous names, interpret complex docstrings, or assess whether a given implementation actually satisfies a legal obligation. Four LLM agents extend the pipeline beyond that ceiling.

**Legal parser** — takes low-confidence (Tier-3) annotations and asks: *does this docstring actually describe a mechanism that would satisfy this article?*

**Relation extractor** — reads cross-references and identifies which ones imply legal relationships beyond co-occurrence.

**Mapping validator** — reviews the Tier-1 name map against the current API surface and flags names that have been removed, renamed, or whose semantics have shifted.

**Semantic search** — embeds compliance obligations from the regulation text and retrieves the most semantically similar EntityRecords, surfacing APIs that satisfy an obligation without appearing in any keyword list.

All LLM calls go through a local Ollama instance (`configs/default.yaml` specifies the models). No data leaves the environment. The GB10 validation run processed 5,778 records in 7.8 hours using `qwen3.5:35b`.

---

## Connection to the Runtime Library

The static analysis pipeline and the `torchcomply` runtime library are two views of the same compliance surface.

The pipeline discovers that `register_forward_hook` maps to Art. 12 — the runtime library implements `AuditChain` using exactly that hook. The pipeline discovers that `PrivacyEngine` maps to GDPR Art. 25 — the runtime library wraps it in `CompliancePrivacyEngine`. The pipeline discovers that export boundaries strip hooks — Example 06 (`before_after`) demonstrates this concretely by comparing standard vs. compliant checkpoints.

The RDF graph is not just documentation. When `ComplianceDiff` detects a run-to-run regression, the knowledge graph can be queried to identify which APIs changed and which obligations are at risk.

The pipeline answers: *what can be compliant, and how do we know?*  
The runtime library answers: *is this run actually compliant, right now?*

Together they close the loop: static evidence of capability, dynamic evidence of execution.

---

## Design Decisions Worth Explaining

**Why SHA-256 for entity IDs, not sequential integers?**  
A sequential integer is only stable for the lifetime of one pipeline run. SHA-256 over six fields is stable across runs, machines, and time — unless the underlying code actually changes. Two runs of the pipeline can be diffed by ID, enabling Art. 9 ongoing monitoring without a separate versioning system.

**Why three confidence tiers instead of one model?**  
A single model would require training data that does not exist at the scale of a framework codebase. Three tiers make the evidence hierarchy explicit: name matches are the most defensible in an audit (the API literally *is called* `use_deterministic_algorithms`), structural matches are mechanically derivable, phrase matches are interpretive and labelled as such.

**Why LRU-cached file reads across extractors?**  
The 11 extractors all need to read the same files. Without a shared cache, a file like `torch/nn/modules/module.py` would be read 11 times. With a 500-entry LRU cache on a free function (not a bound method — which would make cache hits impossible across instances), each file is read once per pipeline run.

**Why buffered writes at 1,000 records?**  
On a full PyTorch run, a single extractor can produce tens of thousands of records. Writing each record immediately creates excessive inode pressure. Buffering at 1,000 records keeps I/O predictable without holding more than a few hundred KB in memory.

**Why graceful shutdown on SIGINT instead of immediate termination?**  
A killed pipeline mid-phase leaves partial output that the checkpoint manager cannot distinguish from completed output. Finishing the current phase cleanly means the checkpoint is always consistent: either a phase completed fully or it did not start. `--resume` is safe to use without manual inspection.
