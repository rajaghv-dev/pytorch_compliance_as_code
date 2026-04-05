"""
Example 8: All Three PyTorch Mechanisms for Compliance
Abstract claim: "Dispatcher, custom Autograd functions, and the hook system"
This is Slide 9 — the most critical example for PyTorch engineers.
"""

import hashlib
import pathlib
import time
from collections import Counter

import torch
import torch.nn as nn

from torchcomply.core.audit import AuditChain, register_compliance_hooks
from torchcomply.core.autograd_provenance import ProvenanceLinear
from torchcomply.core.dispatcher_hooks import ComplianceTensor


def _subject_hash(subject_idx: int) -> str:
    """Return a short hashed subject ID — realistic GDPR Art.17 identifier."""
    raw = f"subject-salt-{subject_idx}".encode()
    return "sub_" + hashlib.sha256(raw).hexdigest()[:12]


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 8: Three PyTorch Mechanisms for Compliance")
    print('Abstract claim: "Dispatcher, custom Autograd functions, and the hook system"')
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)

    X = torch.randn(5, 10, device=device)
    # Hashed subject IDs — realistic GDPR Art.17 identifiers, not raw integers
    subject_hashes = [_subject_hash(i) for i in range(5)]
    print(f"\nSubject IDs (hashed for GDPR Art.17): {subject_hashes}")
    # ProvenanceLinear uses integer indices internally; we track the hash mapping
    subject_ids = torch.arange(5, dtype=torch.long)  # indices into subject_hashes

    # ------------------------------------------------------------------
    # Mechanism 1 — Forward Hooks
    # ------------------------------------------------------------------
    print("\n═══ MECHANISM 1: Forward Hooks (register_forward_hook) ═══")
    print("Use case: Audit logging — EU AI Act Article 12")

    chain = AuditChain()
    handles = register_compliance_hooks(model, chain)
    with torch.no_grad():
        for _ in range(5):
            _ = model(X)
    for h in handles:
        h.remove()

    print(f"Hook audit trail: {len(chain)} entries recorded")
    for i in [0, len(chain) // 2, len(chain) - 1]:
        e = chain.entries[i]
        print(
            f"  #{i:<4} {e.module_name:<40} {e.operator_type:<12} {str(e.output_shape):<16} {e.hash[:8]}…"
        )
    chain.verify()
    print("✅ Chain integrity verified")

    # ------------------------------------------------------------------
    # Mechanism 2 — Dispatcher (__torch_function__)
    # ------------------------------------------------------------------
    print("\n═══ MECHANISM 2: Dispatcher (__torch_function__) ═══")
    print("Use case: Operator-level monitoring — invisible to model code")

    ComplianceTensor.clear_log()
    # Run on CPU to keep tensor type consistent
    model_cpu = model.cpu()
    x_ct = ComplianceTensor(torch.randn(5, 10))
    with torch.no_grad():
        _ = model_cpu(x_ct)

    disp_log = ComplianceTensor.get_log()
    op_counts = Counter(e["operator"] for e in disp_log)
    print(f"Dispatcher log: {len(disp_log)} tensor operations intercepted")
    for op, cnt in op_counts.most_common(5):
        print(f"  {op}: {cnt} call{'s' if cnt > 1 else ''}")
    print("This logging required ZERO changes to model code")

    # ------------------------------------------------------------------
    # Mechanism 3 — Custom Autograd Function
    # ------------------------------------------------------------------
    print("\n═══ MECHANISM 3: Custom Autograd Function (ProvenanceLinear) ═══")
    print("Use case: Data provenance through backpropagation — GDPR Article 17")

    ProvenanceLinear.clear_log()
    in_features, out_features = 10, 32
    weight = model_cpu[0].weight.detach().clone().requires_grad_(True)
    bias = model_cpu[0].bias.detach().clone().requires_grad_(True)
    x_plain = torch.randn(5, in_features)

    out = ProvenanceLinear.apply(x_plain, weight, bias, subject_ids)
    loss = out.sum()
    loss.backward()

    prov_log = ProvenanceLinear.get_provenance_log()
    entry = prov_log[0]
    raw_ids = entry['subject_ids']
    hashed_ids = [subject_hashes[i] for i in (raw_ids.tolist() if hasattr(raw_ids, 'tolist') else raw_ids)]
    print(f"Provenance log: {len(prov_log)} gradient update tracked")
    print(f"  Gradient influenced by subjects (hashed IDs): {hashed_ids}")
    print(f"  Gradient norm: {entry['grad_norm']:.4f}")
    print(f"\nIf '{hashed_ids[0]}' requests erasure (GDPR Art. 17), we know exactly")
    print("which gradients their data influenced — enabling targeted unlearning.")
    print("\nNote: 'Experimental' means ProvenanceLinear only covers layers *explicitly*")
    print("  wrapped with it. Upstream layers use standard autograd — their provenance")
    print("  is NOT tracked. Full graph coverage would require wrapping every layer,")
    print("  which is currently impractical for deep models.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nThree mechanisms demonstrated:")
    print(f"  Hooks:      Module-level audit logging  ({len(chain)} entries — production-ready)")
    print(f"  Dispatcher: Operator-level monitoring   ({len(disp_log)} ops — zero model changes)")
    print(f"  Autograd:   Gradient provenance tracking ({len(prov_log)} records — experimental)")
    print("All three use NATIVE PyTorch APIs. No framework modifications needed.")
    print("\nAbstract claim: \"Dispatcher, custom Autograd functions, and the hook system\"")
    print(f"  ✅ Hooks:      {len(chain)} entries, root_hash={chain.root_hash()[:12]}…")
    print(f"  ✅ Dispatcher: {len(disp_log)} ops intercepted, 0 model changes")
    print(f"  ✅ Autograd:   {len(prov_log)} gradient provenance records")

    # ------------------------------------------------------------------
    # Visualisation — three-column comparison
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#f8f8f8")

    specs = [
        {
            "title": "Forward Hooks\nregister_forward_hook",
            "color": "#2980b9",
            "level": "Module level",
            "count": f"{len(chain)} entries",
            "status": "Production",
            "regulation": "EU AI Act Art. 12",
            "desc": "Intercepts at module\nboundaries. Clean API.\nUsed in AuditChain.",
        },
        {
            "title": "__torch_function__\nDispatcher",
            "color": "#8e44ad",
            "level": "Operator level",
            "count": f"{len(disp_log)} ops (3-layer demo)",
            "status": "Production",
            "regulation": "EU AI Act Art. 12",
            "desc": "Intercepts EVERY tensor\nop. Zero model changes.\nReal models: 100s of ops.",
        },
        {
            "title": "ProvenanceLinear\nCustom Autograd",
            "color": "#27ae60",
            "level": "Gradient level",
            "count": f"{len(prov_log)} provenance",
            "status": "Experimental",
            "regulation": "GDPR Art. 17",
            "desc": "Tracks which subjects\ninfluenced each gradient.\nEnables unlearning.",
        },
    ]

    for ax, spec in zip(axes, specs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        rect = mpatches.FancyBboxPatch(
            (0.3, 0.5),
            9.4,
            9,
            boxstyle="round,pad=0.3",
            linewidth=2,
            edgecolor=spec["color"],
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(
            5,
            9.0,
            spec["title"],
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=spec["color"],
            va="center",
        )
        ax.axhline(8.2, xmin=0.05, xmax=0.95, color=spec["color"], linewidth=0.8, alpha=0.4)
        rows = [
            ("Level", spec["level"]),
            ("Logged", spec["count"]),
            ("Readiness", spec["status"]),
            ("Maps to", spec["regulation"]),
            ("", ""),
            ("", spec["desc"]),
        ]
        for i, (k, v) in enumerate(rows):
            if k:
                ax.text(1, 7.5 - i * 1.1, f"{k}:", fontsize=9, color="grey", va="top")
                ax.text(
                    3.8,
                    7.5 - i * 1.1,
                    v,
                    fontsize=9,
                    color="#222",
                    va="top",
                    fontweight="bold" if k == "Readiness" else "normal",
                )
            else:
                ax.text(1, 7.5 - i * 1.1, v, fontsize=8.5, color="#444", va="top", style="italic")

    fig.suptitle(
        "Three PyTorch Mechanisms for Compliance\n"
        "Claim 6: 'Dispatcher, custom Autograd functions, and the hook system'",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "three_mechanisms.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved → {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
