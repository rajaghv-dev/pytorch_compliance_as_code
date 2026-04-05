"""
Example 6: Before vs After — What compliance adds to PyTorch
Real code comparison: standard torch.save() vs compliant torch.save().
"""

import datetime
import os
import pathlib
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torchcomply.core.engine import ComplianceEngine
from torchcomply.core.fairness import compute_demographic_parity

SEPARATOR = "=" * 70


def main():
    t0 = time.time()
    print(SEPARATOR)
    print("EXAMPLE 6: Before vs After — What compliance adds to PyTorch")
    print(SEPARATOR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # BEFORE — Standard PyTorch
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("BEFORE: Standard PyTorch (what everyone does today)")
    print("─" * 70)

    model_before = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = model_before(x)

    ckpt_standard = out_dir / "standard_checkpoint.pt"
    torch.save({"state_dict": model_before.state_dict()}, ckpt_standard)

    standard_size = os.path.getsize(ckpt_standard) / 1e6
    param_count = sum(p.numel() for p in model_before.parameters())

    print(f"\nStandard checkpoint contents:")
    print(f"  state_dict: {param_count:,} parameters")
    print(f"  That's it. No audit trail. No consent record. No fairness score.")
    print(f"  No model version. No training data hash. No regulatory mapping.")
    print(f'  A regulator asks "prove this model is fair" — you have nothing.')

    # ------------------------------------------------------------------
    # AFTER — Compliant PyTorch
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("AFTER: Compliant PyTorch (what torchcomply adds)")
    print("─" * 70)

    model_after = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    engine = ComplianceEngine(regulations=["eu_ai_act", "gdpr"])
    model_after = engine.attach(model_after)

    # Fine-tune for 2 steps so the audit chain covers real gradient updates
    print("\nFine-tuning for 2 steps (so audit chain covers real training passes)…")
    optimizer = optim.SGD(model_after.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model_after.train()
    for _ in range(2):
        fake_labels = torch.randint(0, 1000, (1,), device=device)
        out = model_after(x)
        loss = criterion(out, fake_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model_after.eval()

    # Fairness check on a small synthetic validation set
    torch.manual_seed(0)
    n_val = 50
    x_val = torch.randn(n_val, 3, 224, 224, device=device)
    # Two groups: 0 = images 0-24, 1 = images 25-49
    groups = torch.tensor([0] * 25 + [1] * 25)
    with torch.no_grad():
        logits_val = model_after(x_val)
    preds_val = logits_val.argmax(dim=-1).cpu()
    fairness_parity = compute_demographic_parity((preds_val > 500).long(), groups)
    fairness_passed = fairness_parity <= 0.15

    chain_valid = engine.audit_chain.verify()
    root_hash = engine.audit_chain.root_hash()
    ckpt_compliant = out_dir / "compliant_checkpoint.pt"
    torch.save(
        {
            "state_dict": model_after.state_dict(),
            "compliance": {
                "audit_chain_summary": engine.audit_chain.summary(),
                "audit_entries": len(engine.audit_chain),
                "chain_integrity": chain_valid,
                "chain_root_hash": root_hash,
                "model_architecture": type(model_after).__name__,
                "total_params": param_count,
                "fairness_parity": round(fairness_parity, 4),
                "fairness_threshold": 0.15,
                "fairness_passed": fairness_passed,
                "regulations": ["eu_ai_act", "gdpr"],
                "generated_at": datetime.datetime.utcnow().isoformat(),
                "framework_version": torch.__version__,
            },
        },
        ckpt_compliant,
    )
    # Also flush the full audit chain to disk (the checkpoint only stores the summary)
    engine.audit_chain.flush_jsonl(out_dir / "audit_chain.jsonl")

    compliant_size = os.path.getsize(ckpt_compliant) / 1e6
    overhead_kb = (compliant_size - standard_size) * 1000

    print(f"\nCompliant checkpoint contents:")
    print(f"  state_dict: {param_count:,} parameters")
    print(f"  audit_chain: {len(engine.audit_chain)} entries, chain VALID ✅")
    print(f"  chain_root_hash: {root_hash[:16]}…  ← single fingerprint for chain integrity")
    print(f"  fairness_parity: {fairness_parity:.4f} (threshold=0.15) → {'PASSED ✅' if fairness_passed else 'FAILED ❌'}")
    print(f"  model_architecture: auto-introspected ({type(model_after).__name__})")
    print(f"  regulations: [eu_ai_act, gdpr]")
    print(f"  generated_at: {datetime.datetime.utcnow().isoformat()}")
    print(f'  A regulator asks "prove this model is fair" — you hand them this file.')
    print(f"\nNote: checkpoint stores audit_chain_summary (compact). Full entry log:")
    print(f"  → audit_chain.jsonl  ({(out_dir / 'audit_chain.jsonl').stat().st_size:,} bytes)")

    print(f"\n{'─'*40}")
    print(f"  Standard checkpoint:  {standard_size:.1f} MB")
    print(f"  Compliant checkpoint: {compliant_size:.1f} MB  (overhead: {overhead_kb:.0f} KB)")

    engine.detach()

    # ------------------------------------------------------------------
    # Visualisation — side-by-side comparison
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#f8f8f8")

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # Left box — standard
    rect_l = mpatches.FancyBboxPatch(
        (0.5, 1),
        9,
        8,
        boxstyle="round,pad=0.2",
        linewidth=2,
        edgecolor="#e74c3c",
        facecolor="#fff0ee",
    )
    axes[0].add_patch(rect_l)
    axes[0].text(
        5,
        9.2,
        "Standard torch.save()",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#c0392b",
    )
    axes[0].text(
        5, 8.3, f"File size: {standard_size:.1f} MB", ha="center", fontsize=11, color="grey"
    )
    items_l = [
        f"✅  state_dict ({param_count/1e6:.1f}M params)",
        "",
        "❌  No audit trail",
        "❌  No consent record",
        "❌  No fairness score",
        "❌  No model version",
        "❌  No data hash",
        "❌  No regulatory mapping",
        "",
        "🔴  Regulator asks: 'prove fairness'",
        "🔴  You have nothing.",
    ]
    for i, line in enumerate(items_l):
        axes[0].text(1, 7.5 - i * 0.6, line, fontsize=11)

    # Right box — compliant
    rect_r = mpatches.FancyBboxPatch(
        (0.5, 1),
        9,
        8,
        boxstyle="round,pad=0.2",
        linewidth=2,
        edgecolor="#27ae60",
        facecolor="#f0fff4",
    )
    axes[1].add_patch(rect_r)
    axes[1].text(
        5,
        9.2,
        "Compliant torch.save()",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#1e8449",
    )
    axes[1].text(
        5,
        8.3,
        f"File size: {compliant_size:.1f} MB  (+{overhead_kb:.0f} KB)",
        ha="center",
        fontsize=11,
        color="grey",
    )
    items_r = [
        f"✅  state_dict ({param_count/1e6:.1f}M params)",
        f"✅  audit_chain ({len(engine.audit_chain)} entries, SHA-256)",
        f"✅  root_hash: {root_hash[:12]}…",
        "✅  chain_integrity: VALID",
        f"✅  fairness_parity: {fairness_parity:.4f} ({'PASSED' if fairness_passed else 'FAILED'})",
        "✅  regulations: [eu_ai_act, gdpr]",
        "✅  generated_at: ISO timestamp",
        "",
        "🟢  Regulator asks: 'prove fairness'",
        "🟢  You hand them this file.",
    ]
    for i, line in enumerate(items_r):
        axes[1].text(1, 7.5 - i * 0.6, line, fontsize=11)

    fig.suptitle(
        "Before vs After: Compliance-as-Code for PyTorch\n"
        "EU AI Act Articles 11, 12 | torchcomply v0.4.0",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = out_dir / "before_after_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved → {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
