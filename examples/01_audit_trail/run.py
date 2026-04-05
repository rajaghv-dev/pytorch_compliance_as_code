"""
Example 1: Immutable Audit Trail — EU AI Act Article 12
Demonstrates hash-chained forward hooks on DistilBERT (66M params).
"""

import time
from collections import Counter

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from torchcomply.core.audit import IntegrityViolation
from torchcomply.core.engine import ComplianceEngine

import pathlib

SEPARATOR = "=" * 70

REVIEWS = [
    "This film is an absolute masterpiece. Every scene is breathtaking.",
    "A complete waste of time. The plot made no sense whatsoever.",
    "Brilliant performances from the entire cast. Highly recommended.",
    "The worst movie I have seen in years. Terrible script and direction.",
    "An emotional rollercoaster that left me speechless at the end.",
    "Mediocre at best. Nothing new or interesting to offer the audience.",
    "A stunning visual experience with a deeply moving story.",
    "Painfully boring and predictable. I fell asleep halfway through.",
    "One of the finest films ever made. A true cinematic achievement.",
    "Disappointing sequel that fails to capture the original's magic.",
]

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def main():
    t0 = time.time()

    print(SEPARATOR)
    print("EXAMPLE 1: Immutable Audit Trail — EU AI Act Article 12")
    print("Model: DistilBERT (66M params)")
    print(SEPARATOR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    print("\nLoading DistilBERT sentiment model…")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model = model.to(device).eval()
    print("Model loaded.")

    # Context manager API: hooks removed automatically on exit — even on exception.
    # This is the preferred production pattern vs manual engine.attach() / engine.detach().
    engine = ComplianceEngine(regulations=["eu_ai_act", "gdpr"])
    model = engine.attach(model)
    print(f"  (using `with engine` context manager — hooks removed automatically on exit)")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\nRunning inference on 10 reviews with audit hooks active…\n")
    for review in REVIEWS:
        enc = tokenizer(
            review,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        pred = int(logits.argmax(dim=-1).item())
        conf = float(torch.softmax(logits, dim=-1)[0, pred].item())
        print(f"  '{review[:50]}…' → {LABELS[pred]} ({conf:.1%})")

    # ------------------------------------------------------------------
    # Audit chain stats
    # ------------------------------------------------------------------
    chain = engine.audit_chain
    op_counts = Counter(e.operator_type for e in chain.entries)
    unique_ops = len(op_counts)

    n_reviews = len(REVIEWS)
    n_modules = len(chain) // n_reviews if n_reviews else len(chain)
    print(f"\nAudit chain: {len(chain)} entries")
    print(f"  = {n_reviews} forward passes × {n_modules} DistilBERT modules each")
    print(f"  Unique operator types: {unique_ops}")

    print(f"\n{'Operator Type':<30} | {'Count':>5}")
    print("-" * 38)
    for op, cnt in op_counts.most_common(10):
        print(f"  {op:<28} | {cnt:>5}")

    # ------------------------------------------------------------------
    # Sample entries
    # ------------------------------------------------------------------
    sample_indices = [0, 1, 2, len(chain) // 4, len(chain) // 2]
    print(f"\n{'Entry':<6} | {'Module':<40} | {'Type':<14} | {'Out Shape':<16} | {'OutHash':<10} | ChainHash")
    print("-" * 110)
    for i in sample_indices:
        if i >= len(chain):
            continue
        e = chain.entries[i]
        shape = str(e.output_shape)[:14]
        out_h = e.output_hash[:8] + "…" if e.output_hash else "(n/a)"
        print(
            f"  #{i:<4} | {e.module_name[:40]:<40} | {e.operator_type:<14} | {shape:<16} | {out_h:<10} | {e.hash[:8]}…"
        )
    print(f"\n  output_hash = SHA-256[:16] of the tensor values at that layer")
    print(f"  Modifying output post-hoc would change output_hash → break chain.verify()")

    # ------------------------------------------------------------------
    # Chain integrity + root hash
    # ------------------------------------------------------------------
    print("\nVerifying chain integrity…")
    try:
        chain.verify()
        root = chain.root_hash()
        print(f"✅ CHAIN VALID — all {len(chain)} hashes verified")
        print(f"   Root hash (chain fingerprint): {root}")
        print(f"   → Include this in regulatory filings; any tampering changes it.")
    except IntegrityViolation as exc:
        print(f"❌ INTEGRITY VIOLATION at entry #{exc.index}")

    # ------------------------------------------------------------------
    # Tamper detection demo — output values, not just metadata
    # ------------------------------------------------------------------
    print("\nSimulating tampering — modifying entry #5 output_hash (as if output values changed)…")
    original_oh = chain.entries[5].output_hash
    chain.entries[5].output_hash = "deadbeef00000000"
    try:
        chain.verify()
    except IntegrityViolation as exc:
        print(f"❌ INTEGRITY VIOLATION at entry #{exc.index} — hash mismatch!")
        print(f"   The chain bound the output tensor values, not just metadata.")
    chain.entries[5].output_hash = original_oh  # restore

    # ------------------------------------------------------------------
    # Persist to disk — durable audit log
    # ------------------------------------------------------------------
    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / "audit_chain.jsonl"
    chain.flush_jsonl(jsonl_path)
    print(f"\nAudit chain persisted → {jsonl_path}  ({jsonl_path.stat().st_size:,} bytes)")
    print(f"  Each line = one AuditEntry | append-only | survives process exit")

    # ------------------------------------------------------------------
    # Visualisation — audit waterfall
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    op_types = sorted({e.operator_type for e in chain.entries})
    color_map = {op: cm.tab20(i / max(len(op_types), 1)) for i, op in enumerate(op_types)}
    n = len(chain)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2]})

    # Left panel — operator type distribution (summary bar chart)
    op_counts = Counter(e.operator_type for e in chain.entries)
    sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
    op_names = [o for o, _ in sorted_ops]
    op_vals = [c for _, c in sorted_ops]
    bar_colors = [color_map[o] for o in op_names]
    axes[0].barh(op_names, op_vals, color=bar_colors, alpha=0.85, edgecolor="white")
    for i, v in enumerate(op_vals):
        axes[0].text(v + 1, i, str(v), va="center", fontsize=9)
    axes[0].set_xlabel("Entry count")
    axes[0].set_title(f"Operator Types\n({n} total entries)", fontsize=10, fontweight="bold")
    axes[0].invert_yaxis()

    # Right panel — sampled 30 entries showing hash chain
    step = max(1, n // 30)
    sample_indices = list(range(0, n, step))[:30]
    sample_entries = [chain.entries[i] for i in sample_indices]
    t_base = chain.entries[0].timestamp
    x_offsets = [(e.timestamp - t_base) / 1e6 for e in sample_entries]
    s_colors = [color_map[e.operator_type] for e in sample_entries]
    y_pos = np.arange(len(sample_entries))
    axes[1].barh(y_pos, [max(x, 0.1) for x in x_offsets], color=s_colors, alpha=0.85, height=0.7)
    labels = [f"#{sample_indices[i]:>3}  {e.module_name.split('.')[-1][:20]:<20}  {e.hash[:8]}…"
              for i, e in enumerate(sample_entries)]
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=7.5, fontfamily="monospace")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Time offset from first entry (ms)")
    axes[1].set_title(f"30 Sampled Entries (every {step}th)\nHash chain links each to previous",
                      fontsize=10, fontweight="bold")
    axes[1].text(0.99, 0.01, f"Root hash: {chain.root_hash()[:16]}…",
                 transform=axes[1].transAxes, ha="right", fontsize=7.5,
                 color="grey", style="italic")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[op], label=op) for op in op_types]
    fig.legend(handles=handles, loc="lower center", fontsize=8, ncol=len(op_types),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Compliance Audit Trail — {n} entries | DistilBERT 66M | EU AI Act Article 12",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "audit_waterfall.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved → {out_path}")

    engine.detach()
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
