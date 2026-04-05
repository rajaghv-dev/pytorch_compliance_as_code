# torchcomply — References

Every entry here is cited in at least one source file, example, or documentation page
in this repository. The [Component Index](#component-index) maps each source file to
the references it uses.

---

## Table of Contents

1. [EU AI Act & GDPR](#1-eu-ai-act--gdpr)
2. [PyTorch Extension Points](#2-pytorch-extension-points)
3. [PyTorch Conference](#3-pytorch-conference)
4. [Differential Privacy & Opacus](#4-differential-privacy--opacus)
5. [Explainability & Captum](#5-explainability--captum)
6. [Fairness](#6-fairness)
7. [Secure MPC & CrypTen](#7-secure-mpc--crypten)
8. [Machine Unlearning](#8-machine-unlearning)
9. [LLM Fine-Tuning & LoRA](#9-llm-fine-tuning--lora)
10. [Observability — MLflow & OpenTelemetry](#10-observability--mlflow--opentelemetry)
11. [Static Analysis Pipeline](#11-static-analysis-pipeline)
12. [Component Index](#component-index)
13. [Bibliography](#bibliography)

---

## 1. EU AI Act & GDPR

### EU AI Act (Regulation 2024/1689)

The primary legal instrument driving this project. High-risk AI systems must comply
with Articles 9–17 and produce Annex IV Technical Documentation before market placement.

| Article | Topic | URL |
|---------|-------|-----|
| Full text | EUR-Lex consolidated | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689 |
| Art. 9 | Risk management system | (section IV.2 of above) |
| Art. 10 | Data governance, bias examination | (section IV.3) |
| Art. 11 | Technical documentation | (section IV.4) |
| Art. 12 | Record-keeping and traceability | (section IV.5) |
| Art. 13 | Transparency | (section IV.6) |
| Art. 14 | Human oversight | (section IV.7) |
| Art. 17 | Quality management system | (section IV.10) |
| Art. 43 | Conformity assessment | (section V.2) |
| Art. 61 | Post-market monitoring | (section VII.2) |
| Annex IV | Technical documentation content | (end of consolidated text) |
| EU Charter Art. 21 | Non-discrimination | https://www.europarl.europa.eu/charter/pdf/text_en.pdf |

### GDPR (Regulation 2016/679)

| Article | Topic | URL |
|---------|-------|-----|
| Art. 6 | Lawfulness of processing | https://gdpr-info.eu/art-6-gdpr/ |
| Art. 7 | Conditions for consent | https://gdpr-info.eu/art-7-gdpr/ |
| Art. 17 | Right to erasure | https://gdpr-info.eu/art-17-gdpr/ |
| Art. 22 | Automated individual decision-making | https://gdpr-info.eu/art-22-gdpr/ |
| Art. 25 | Privacy by design and by default | https://gdpr-info.eu/art-25-gdpr/ |
| Art. 32 | Security of processing | https://gdpr-info.eu/art-32-gdpr/ |

### Supervisory Authority Guidance

| Authority | Resource | URL |
|-----------|----------|-----|
| ICO (UK) | GDPR consent guidance | https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/consent/ |
| EDPB (EU) | Recommendations 01/2020 — transfers to third countries | https://edpb.europa.eu/our-work-tools/our-documents/recommendations/recommendations-012020-measures-supplements-transfer_en |
| NIST | SP 800-92 — Guide to Computer Security Log Management | https://csrc.nist.gov/publications/detail/sp/800-92/final |

---

## 2. PyTorch Extension Points

torchcomply is built on three PyTorch-native extension points. These are the canonical
documentation pages for each mechanism.

### Mechanism 1 — Forward Hooks

Used by: `torchcomply/core/audit.py`

| Resource | URL |
|----------|-----|
| `register_forward_hook` API reference | https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook |

### Mechanism 2 — `__torch_function__` Dispatcher

Used by: `torchcomply/core/dispatcher_hooks.py`

| Resource | URL |
|----------|-----|
| Extending PyTorch — `__torch_function__` protocol | https://pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api |
| Subclassing `torch.Tensor` | https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor |
| "Let's talk about the PyTorch dispatcher" — Edward Yang | https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/ |
| Python `threading.local` (reentrancy guard) | https://docs.python.org/3/library/threading.html#thread-local-data |

### Mechanism 3 — Custom Autograd Function

Used by: `torchcomply/core/autograd_provenance.py`

| Resource | URL |
|----------|-----|
| Extending PyTorch Autograd | https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd |

---

## 3. PyTorch Conference

The talk this repo accompanies, and the broader conference context.

| Resource | URL |
|----------|-----|
| PyTorch Conference Europe 2026 | Station F, Paris — April 8 |
| PyTorch Conference 2024 — all talks (YouTube playlist) | https://www.youtube.com/playlist?list=PL_lsbAsL_o2B_znuvm-pDtV_cRhpqZb8l |
| PyTorch 2 paper — Ansel et al., ASPLOS 2024 | https://dl.acm.org/doi/10.1145/3620665.3640366 |

---

## 4. Differential Privacy & Opacus

Used by: `torchcomply/integrations/opacus_bridge.py`, `examples/04_opacus_dp/`, `examples/10_connected_pipeline/`

**The Algorithmic Foundations of Differential Privacy** *(free textbook — essential background)*
- Cynthia Dwork, Aaron Roth — Foundations and Trends in TCS, 2014
- https://www.cis.upenn.edu/~aaroth/privacybook.html

**Deep Learning with Differential Privacy** *(the DP-SGD algorithm)*
- Martin Abadi et al. (Google) — ACM CCS, 2016
- https://arxiv.org/abs/1607.00133

**Rényi Differential Privacy of the Gaussian Mechanism** *(privacy accountant used in Opacus)*
- Ilya Mironov (Google Brain) — IEEE CSF, 2017
- https://arxiv.org/abs/1702.07476

**Opacus: User-Friendly Differential Privacy Library in PyTorch**
- Ashkan Yousefpour et al. (Meta) — 2021
- https://arxiv.org/abs/2109.12298
- Website: https://opacus.ai/ | GitHub: https://github.com/pytorch/opacus

---

## 5. Explainability & Captum

Used by: `torchcomply/integrations/captum_explain.py`, `examples/03_captum_explain/`, `examples/10_connected_pipeline/`, `examples/11_llm_finetune/`

**Axiomatic Attribution for Deep Networks** *(Integrated Gradients — the algorithm in Captum)*
- Mukund Sundararajan, Ankur Taly, Qiqi Yan (Google) — ICML, 2017
- https://arxiv.org/abs/1703.01365
- ICML PDF: https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf

**Captum: A Unified and Generic Model Interpretability Library for PyTorch**
- Narine Kokhlikyan et al. (Facebook AI) — 2020
- https://arxiv.org/abs/2009.07896
- Website: https://captum.ai/
- `LayerIntegratedGradients` API: https://captum.ai/api/layer.html#captum.attr.LayerIntegratedGradients

---

## 6. Fairness

Used by: `torchcomply/core/fairness.py`, `examples/02_fairness_gate/`

**Fairness and Machine Learning: Limitations and Opportunities** *(free textbook)*
- Solon Barocas, Moritz Hardt, Arvind Narayanan — MIT Press, 2023
- https://fairmlbook.org/

**Equality of Opportunity in Supervised Learning** *(equalized odds / demographic parity)*
- Moritz Hardt, Eric Price, Nathan Srebro — NeurIPS, 2016
- https://arxiv.org/abs/1610.02413

**Learning from Imbalanced Data** *(class imbalance threshold rationale)*
- Haibo He, Edwardo A. Garcia — IEEE TKDE, 2009
- https://doi.org/10.1109/TKDE.2008.239

**Fairlearn** *(practical fairness toolkit for ML practitioners)*
- Microsoft — https://fairlearn.org/

**Google What-If Tool** *(visual fairness exploration)*
- Google PAIR — https://pair-code.github.io/what-if-tool/

---

## 7. Secure MPC & CrypTen

Used by: `torchcomply/integrations/crypten_bridge.py`, `examples/07_crypten_secure/`, `examples/10_connected_pipeline/`

**CrypTen: Secure Multi-Party Computation Meets Machine Learning**
- Brian Knott et al. (Meta) — NeurIPS, 2021
- https://arxiv.org/abs/2109.00984
- Website: https://crypten.ai/ | GitHub: https://github.com/facebookresearch/CrypTen

**SecureML: A System for Scalable Privacy-Preserving Machine Learning** *(secret sharing foundations)*
- Payman Mohassel, Yupeng Zhang — IEEE S&P, 2017
- https://eprint.iacr.org/2017/396

---

## 8. Machine Unlearning

Used by: `torchcomply/core/autograd_provenance.py`, `examples/08_three_mechanisms/`

`ProvenanceLinear` tracks which subject IDs appear in each training batch and the
resulting gradient norm. This enables the *identify* step of any unlearning workflow —
once affected training steps are known, SISA retraining or influence function
perturbations can efficiently remove a subject's contribution.

**Understanding Black-box Predictions via Influence Functions** *(identifies which data influenced a prediction)*
- Pang Wei Koh, Percy Liang — ICML, 2017
- https://arxiv.org/abs/1703.04730

**Towards Making Systems Forget with Machine Unlearning** *(first formal treatment)*
- Yinzhi Cao, Junfeng Yang — IEEE S&P, 2015
- https://arxiv.org/abs/1906.00707

---

## 9. LLM Fine-Tuning & LoRA

Used by: `examples/11_llm_finetune/`

Run result: DistilGPT-2 (82M params) + LoRA rank=8, alpha=16 → 405,504 trainable / 82,318,080 total (0.49%). 5,311 audit entries across 3 training epochs on 30 samples.

**LoRA: Low-Rank Adaptation of Large Language Models**
- Edward J. Hu et al. (Microsoft) — ICLR, 2022
- https://arxiv.org/abs/2106.09685

**PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning** *(the library used in example 11)*
- Hugging Face
- Docs: https://huggingface.co/docs/peft/en/index | GitHub: https://github.com/huggingface/peft

---

## 10. Observability — MLflow & OpenTelemetry

### MLflow

Used by: `torchcomply/integrations/mlflow_logger.py`, `examples/10_connected_pipeline/`

**Accelerating the Machine Learning Lifecycle with MLflow**
- Matei Zaharia et al. (Databricks / UC Berkeley) — IEEE Data Eng. Bulletin, 2018
- https://people.eecs.berkeley.edu/~matei/papers/2018/ieee_mlflow.pdf
- Docs: https://mlflow.org/docs/latest/tracking.html
- `mlflow.log_metric` API: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric

### OpenTelemetry

Used by: `torchcomply/integrations/otel.py`, `examples/10_connected_pipeline/`

| Resource | URL |
|----------|-----|
| Python SDK docs | https://opentelemetry-python.readthedocs.io/ |
| OTel specification | https://opentelemetry.io/docs/reference/specification/ |
| OTLP protocol | https://opentelemetry.io/docs/reference/specification/protocol/otlp/ |
| Semantic conventions | https://opentelemetry.io/docs/reference/specification/trace/semantic_conventions/ |
| Jaeger (trace backend) | https://www.jaegertracing.io/ |
| Grafana Tempo (trace backend) | https://grafana.com/oss/tempo/ |
| Zipkin (trace backend) | https://zipkin.io/ |

### ReportLab

Used by: `torchcomply/reports/annex_iv.py`

| Resource | URL |
|----------|-----|
| ReportLab user guide | https://docs.reportlab.com/reportlab/userguide/ch1_intro/ |
| Platypus layout engine | https://docs.reportlab.com/reportlab/userguide/ch6_intro/ |

---

## 11. Static Analysis Pipeline

Used by: `src/` — the full pipeline (`pct` CLI)

The static analysis pipeline scans a PyTorch source tree and produces a compliance knowledge graph (RDF/OWL), CSV, Markdown, and SPARQL notebooks. See `static_analysis.md` for the full design rationale.

| Resource | URL |
|----------|-----|
| rdflib (Python RDF library) | https://rdflib.readthedocs.io/ |
| SPARQL 1.1 specification | https://www.w3.org/TR/sparql11-query/ |
| Ollama (local LLM inference) | https://ollama.com/ |
| PyTorch native_functions.yaml | `aten/src/ATen/native/native_functions.yaml` in PyTorch source |

---

## Component Index

| Source File | References Used |
|------------|----------------|
| `torchcomply/core/audit.py` | EU AI Act Art. 12; `register_forward_hook` API; NIST SP 800-92 |
| `torchcomply/core/dispatcher_hooks.py` | EU AI Act Art. 12; `__torch_function__` docs; Edward Yang blog; `threading.local` |
| `torchcomply/core/autograd_provenance.py` | GDPR Art. 17; EU AI Act Art. 12; Koh & Liang 2017; Cao & Yang 2015 |
| `torchcomply/core/fairness.py` | EU AI Act Art. 10; EU Charter Art. 21; Barocas et al.; Hardt et al. 2016; Fairlearn; What-If Tool |
| `torchcomply/core/dataset.py` | GDPR Art. 6, 7; EU AI Act Art. 10; ICO consent guide; He & Garcia 2009 |
| `torchcomply/core/engine.py` | EU AI Act Art. 9–17; `register_forward_hook` API |
| `torchcomply/core/diff.py` | EU AI Act Art. 9; ComplianceSnapshot / ComplianceDiff |
| `torchcomply/integrations/captum_explain.py` | EU AI Act Art. 13, 14; GDPR Art. 22; Sundararajan et al. 2017; Kokhlikyan et al. 2020 |
| `torchcomply/integrations/opacus_bridge.py` | GDPR Art. 25, 32; Abadi et al. 2016; Mironov 2017; Opacus |
| `torchcomply/integrations/crypten_bridge.py` | GDPR Art. 25; EDPB Recommendations; Knott et al. 2021; Mohassel & Zhang 2017 |
| `torchcomply/integrations/mlflow_logger.py` | EU AI Act Art. 9, 12, 17; MLflow docs |
| `torchcomply/integrations/otel.py` | EU AI Act Art. 14, 61; OTel spec; Jaeger; Grafana Tempo; Zipkin |
| `torchcomply/reports/annex_iv.py` | EU AI Act Art. 11, 43, Annex IV; ReportLab docs |
| `examples/11_llm_finetune/` | LoRA (Hu et al. 2022); PEFT library |
| `src/` (static analysis pipeline) | rdflib; SPARQL 1.1; Ollama |

---

## Bibliography

Numbered citations in order of first appearance in the codebase:

1. **EU AI Act.** Regulation (EU) 2024/1689. *Official Journal of the EU*, 2024. https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689

2. **GDPR.** Regulation (EU) 2016/679. *Official Journal of the EU*, 2016. https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng

3. **NIST.** SP 800-92 — Guide to Computer Security Log Management, 2006. https://csrc.nist.gov/publications/detail/sp/800-92/final

4. **Yang, E.** Let's Talk About the PyTorch Dispatcher. Blog, 2020. https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/

5. **Dwork, C., Roth, A.** The Algorithmic Foundations of Differential Privacy. *Found. Trends TCS*, 2014. https://www.cis.upenn.edu/~aaroth/privacybook.html

6. **Abadi, M. et al.** Deep Learning with Differential Privacy. *ACM CCS*, 2016. https://arxiv.org/abs/1607.00133

7. **Mironov, I.** Rényi Differential Privacy of the Gaussian Mechanism. *IEEE CSF*, 2017. https://arxiv.org/abs/1702.07476

8. **Yousefpour, A. et al.** Opacus: User-Friendly Differential Privacy Library in PyTorch. 2021. https://arxiv.org/abs/2109.12298

9. **Sundararajan, M., Taly, A., Yan, Q.** Axiomatic Attribution for Deep Networks. *ICML*, 2017. https://arxiv.org/abs/1703.01365

10. **Kokhlikyan, N. et al.** Captum: A Unified and Generic Model Interpretability Library for PyTorch. 2020. https://arxiv.org/abs/2009.07896

11. **Barocas, S., Hardt, M., Narayanan, A.** *Fairness and Machine Learning*. MIT Press, 2023. https://fairmlbook.org/

12. **Hardt, M., Price, E., Srebro, N.** Equality of Opportunity in Supervised Learning. *NeurIPS*, 2016. https://arxiv.org/abs/1610.02413

13. **He, H., Garcia, E. A.** Learning from Imbalanced Data. *IEEE TKDE*, 2009. https://doi.org/10.1109/TKDE.2008.239

14. **Knott, B. et al.** CrypTen: Secure Multi-Party Computation Meets Machine Learning. *NeurIPS*, 2021. https://arxiv.org/abs/2109.00984

15. **Mohassel, P., Zhang, Y.** SecureML: A System for Scalable Privacy-Preserving Machine Learning. *IEEE S&P*, 2017. https://eprint.iacr.org/2017/396

16. **Koh, P. W., Liang, P.** Understanding Black-box Predictions via Influence Functions. *ICML*, 2017. https://arxiv.org/abs/1703.04730

17. **Cao, Y., Yang, J.** Towards Making Systems Forget with Machine Unlearning. *IEEE S&P*, 2015. https://arxiv.org/abs/1906.00707

18. **Hu, E. J. et al.** LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*, 2022. https://arxiv.org/abs/2106.09685

19. **Zaharia, M. et al.** Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Eng. Bulletin*, 2018. https://people.eecs.berkeley.edu/~matei/papers/2018/ieee_mlflow.pdf

20. **Ansel, J. et al.** PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. *ASPLOS*, 2024. https://dl.acm.org/doi/10.1145/3620665.3640366
