"""
Example 11: LLM Fine-Tuning with Compliance
DistilGPT-2 + LoRA (PEFT) + ComplianceEngine + Captum + Annex IV PDF.
Proves compliance-as-code works for LLM fine-tuning, not just image models.
"""

import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchcomply.core.dataset import CompliantDataset, ConsentRegistry, ConsentViolation
from torchcomply.core.engine import ComplianceEngine

SAMPLE_TEXTS = [
    # EU AI Act / regulatory principles
    "The most important principle of AI safety is transparency in decision-making.",
    "AI systems must be designed with human oversight mechanisms built in.",
    "Data minimisation is a core principle of privacy-respecting AI development.",
    "Robust testing reduces the risk of harmful outputs from AI systems.",
    "Fairness requires that AI systems treat all demographic groups equitably.",
    "Audit trails provide accountability and support post-hoc review of AI decisions.",
    "High-risk AI systems require comprehensive technical documentation.",
    "Explainability allows users to understand and contest automated decisions.",
    "AI governance frameworks must align with existing human rights law.",
    "Privacy by design integrates data protection from the earliest stages.",
    # ML engineering / PyTorch context
    "Gradient checkpointing reduces memory usage during training of large models.",
    "Differential privacy adds calibrated noise to gradients to protect training data.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Model quantisation reduces inference latency with minimal accuracy loss.",
    "Federated learning enables training on decentralised data without raw sharing.",
    "Hook-based monitoring intercepts forward passes without modifying model code.",
    "Custom autograd functions enable non-standard differentiable operations.",
    "The dispatcher pattern intercepts tensor operations for operator-level logging.",
    "LoRA adapters reduce trainable parameters while preserving base model knowledge.",
    "Secure multi-party computation allows inference on encrypted inputs.",
    # Ethics and governance
    "Algorithmic accountability requires that decisions can be explained to affected people.",
    "Bias in training data propagates into model predictions and affects downstream outcomes.",
    "Human oversight of automated systems is required for high-stakes decisions.",
    "Regulatory compliance is not a constraint on innovation but a prerequisite for trust.",
    "Open-source tooling enables reproducible compliance evidence for AI systems.",
    "Consent management ensures that personal data is only used for agreed purposes.",
    "Risk assessment for AI must consider both technical and social dimensions.",
    "Continuous monitoring detects distribution shifts that degrade model reliability.",
    "Incident reporting mechanisms allow harms to be identified and remediated quickly.",
    "International AI standards require alignment between technical controls and legal obligations.",
    # Diverse sentence structures for language model training
    "What constitutes meaningful consent under data protection law is context-dependent.",
    "The tension between model performance and privacy requires careful engineering trade-offs.",
    "Regulatory sandboxes provide a space to test AI systems before full deployment.",
    "Stakeholder engagement in AI development improves the legitimacy of automated decisions.",
    "Post-market monitoring is as important as pre-deployment testing for AI assurance.",
    "Supply chain transparency in AI requires disclosure of data sources and model provenance.",
    "The right to explanation requires that AI systems surface the reasoning behind decisions.",
    "Adversarial robustness testing should be part of every high-risk AI assessment.",
    "Model cards and datasheets make AI systems more interpretable to non-technical stakeholders.",
    "Proportionality in data use means collecting only what is necessary for the stated purpose.",
]  # 40 unique sentences — no repetition


class _TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = []
        self.sids = []
        for i, t in enumerate(texts):
            enc = tokenizer(
                t, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
            )
            self.encodings.append(enc)
            self.sids.append(f"author_{i % 20:03d}")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        ids = self.encodings[i]["input_ids"].squeeze(0)
        mask = self.encodings[i]["attention_mask"].squeeze(0)
        return ids, ids.clone(), mask, self.sids[i]


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 11: LLM Fine-Tuning with Compliance")
    print("Model: DistilGPT-2 (82M params) + LoRA (PEFT)")

    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "distilgpt2"
    print(f"\nLoading {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        # Target modules c_attn / c_proj are GPT-2 specific attention projections.
        # For other architectures: LLaMA → q_proj/v_proj, BERT → query/value,
        # Falcon → query_key_value. Adjust based on model.named_modules() output.
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        peft_ok = True
        print(
            f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)"
        )
    except Exception as ex:
        print(f"PEFT unavailable ({ex}) — full fine-tune")
        peft_ok = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

    model = model.to(device)

    # ------------------------------------------------------------------
    # Compliance engine + dataset
    # ------------------------------------------------------------------
    engine = ComplianceEngine(regulations=["eu_ai_act", "gdpr"])
    model = engine.attach(model)

    import random as _random
    _rng = _random.Random(99)  # fixed seed → reproducible but not deterministic by index
    all_author_ids = [f"author_{i:03d}" for i in range(20)]
    denied_ids = set(_rng.sample(all_author_ids, 5))  # 5 random authors opt out
    records = {
        f"author_{i:03d}": {
            "consent": f"author_{i:03d}" not in denied_ids,
            "purposes": ["training"],
        }
        for i in range(20)
    }
    print(f"\nConsent: denied authors (randomly selected, seed=99): {sorted(denied_ids)}")
    registry = ConsentRegistry(records)
    base_ds = _TextDataset(SAMPLE_TEXTS, tokenizer)

    # Filter out denied samples before training (collect allowed indices)
    allowed_ids = []
    allowed_labels = []
    allowed_masks = []
    denied_count = 0
    for i in range(len(base_ds)):
        ids, labels, mask, sid = base_ds[i]
        if registry.has_consent(sid, "training"):
            allowed_ids.append(ids)
            allowed_labels.append(labels)
            allowed_masks.append(mask)
        else:
            denied_count += 1

    from torch.utils.data import TensorDataset

    allowed_ds = TensorDataset(
        torch.stack(allowed_ids), torch.stack(allowed_labels), torch.stack(allowed_masks)
    )
    train_loader = DataLoader(allowed_ds, batch_size=8, shuffle=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    EPOCHS = 3

    print(
        f"\nTraining for {EPOCHS} epochs on {len(allowed_ids)} samples ({denied_count} denied)…\n"
    )
    print(f"{'Epoch':<6} | {'Loss':>8} | {'Audit entries':>14} | {'Consent denied':>15}")
    print("─" * 52)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for input_ids, labels, attention_mask in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        avg = total_loss / len(train_loader)
        print(f"  {epoch:<4}  | {avg:>8.3f} | {len(engine.audit_chain):>14} | {denied_count:>15}")

    # ------------------------------------------------------------------
    # Generate text
    # ------------------------------------------------------------------
    model.eval()
    prompt = "The most important principle of AI safety is"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            enc.input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n  {generated}")

    # ------------------------------------------------------------------
    # Captum attribution on prompt
    # ------------------------------------------------------------------
    try:
        from captum.attr import LayerIntegratedGradients

        base_model = model.base_model if peft_ok else model
        embed_layer = base_model.transformer.wte if hasattr(base_model, "transformer") else None
        if embed_layer is not None:

            def fwd(input_ids):
                return model(input_ids=input_ids).logits[:, -1, :]

            lig = LayerIntegratedGradients(fwd, embed_layer)
            inp = enc.input_ids
            baseline = torch.zeros_like(inp)
            # target = the token the model generated (first new token)
            target_token = int(out_ids[0, inp.shape[1]].item())
            attrs, _ = lig.attribute(
                inp,
                baselines=baseline,
                n_steps=20,
                target=target_token,
                return_convergence_delta=True,
            )
            scores = attrs.sum(dim=-1).squeeze(0)
            raw_tokens = tokenizer.convert_ids_to_tokens(inp[0].tolist())
            # Strip BPE prefix (Ġ = space marker in GPT-2 tokenizer)
            tokens = [t.replace("Ġ", "").replace("Ċ", "\\n") if t else t for t in raw_tokens]
            top5 = sorted(zip(scores.tolist(), tokens), key=lambda x: abs(x[0]), reverse=True)[:5]
            print(f"\nCaptum top-5 attributed prompt tokens:")
            for s, t in top5:
                print(f"  '{t}' ({s:+.3f})")
        captum_ok = True
    except Exception as ex:
        print(f"\nCaptum unavailable: {ex}")
        captum_ok = False

    # ------------------------------------------------------------------
    # Annex IV PDF
    # ------------------------------------------------------------------
    from torchcomply.reports.annex_iv import AnnexIVReport, ModelIntrospector

    mi = ModelIntrospector(model)
    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / "compliance_report.pdf"

    report = AnnexIVReport(
        model_introspection=mi,
        audit_chain=engine.audit_chain,
        fairness_log=[],
        training_config={
            "base_model": model_name,
            "lora_rank": 8 if peft_ok else "N/A",
            "lora_alpha": 16 if peft_ok else "N/A",
            "trainable_params": f"{trainable:,}",
            "total_params": f"{total:,}",
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "generated_sample": generated[:120],
        },
        dataset_info={
            "system_name": "LLM Fine-Tune Demo",
            "dataset_name": "AI Safety Texts",
            "size": len(allowed_ids),
            "denied": denied_count,
        },
        regulations=["eu_ai_act", "gdpr"],
    )
    report.save_pdf(str(pdf_path))

    consent_summary = registry.access_log_summary()
    pct_trainable = 100 * trainable / max(total, 1)

    print(f"\nLLM Compliance Summary:")
    print(f"  Base model:      {model_name} ({total:,} params)")
    print(f"  Adapter:         {'LoRA (rank=8, alpha=16)' if peft_ok else 'Full fine-tune'}")
    print(f"  Trainable:       {pct_trainable:.2f}% of params")
    print(f"  Training:        {EPOCHS} epochs on {len(allowed_ids)} samples")
    print(f"  Audit trail:     {len(engine.audit_chain)} entries")
    print(
        f"  Consent:         {consent_summary['granted']} granted, {consent_summary['denied']} denied"
    )
    print(f"  Report:          {pdf_path}")

    engine.detach()

    # ------------------------------------------------------------------
    # Visualisation — LoRA compliance card (with attribution if available)
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#f5f5f5")

    # Base model block
    base_rect = mpatches.FancyBboxPatch(
        (0.5, 1), 6, 6, boxstyle="round,pad=0.2", linewidth=2, edgecolor="grey", facecolor="#dde3f0"
    )
    ax.add_patch(base_rect)
    ax.text(
        3.5,
        7.3,
        f"Base Model: {model_name}",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#2c3e50",
    )
    ax.text(3.5, 6.6, f"{total:,} parameters", ha="center", fontsize=9, color="grey")
    ax.text(3.5, 5.8, "🔒 Frozen weights", ha="center", fontsize=9, color="#7f8c8d")
    for i, layer in enumerate(["wte (embed)", "h.0…h.5 (attn)", "wpe (pos)", "ln_f (norm)"]):
        rect = mpatches.FancyBboxPatch(
            (0.8, 4.2 - i * 0.7),
            5.4,
            0.55,
            boxstyle="round,pad=0.05",
            linewidth=1,
            edgecolor="#bdc3c7",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(3.5, 4.52 - i * 0.7, layer, ha="center", fontsize=8.5, color="#555")

    # LoRA adapters
    lora_y_positions = [5.2, 4.5]
    for j, (yp, nm) in enumerate(zip(lora_y_positions, ["c_attn (layer 0)", "c_attn (layer 3)"])):
        lora_rect = mpatches.FancyBboxPatch(
            (7.2, yp),
            2.5,
            0.5,
            boxstyle="round,pad=0.05",
            linewidth=2,
            edgecolor="#e67e22",
            facecolor="#fdebd0",
        )
        ax.add_patch(lora_rect)
        ax.text(
            8.45,
            yp + 0.28,
            f"LoRA: {nm}",
            ha="center",
            fontsize=8,
            color="#e67e22",
            fontweight="bold",
        )
        ax.annotate(
            "",
            xy=(7.2, yp + 0.25),
            xytext=(6.1, 4.52 - j * 0.7),
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5),
        )

    # Compliance metadata
    ax.text(10.2, 7.2, "Compliance metadata", fontsize=10, fontweight="bold", color="#27ae60")
    meta = [
        f"✅  Audit trail: {len(engine.audit_chain)} entries",
        f"✅  Consent: {consent_summary['granted']} granted",
        f"✅  Trainable: {pct_trainable:.2f}%",
        f"✅  Regulations: EU AI Act, GDPR",
        f"✅  Report: compliance_report.pdf",
        f"✅  Captum: {'attributed' if captum_ok else 'unavailable'}",
    ]
    for i, line in enumerate(meta):
        ax.text(10.2, 6.6 - i * 0.7, line, fontsize=9, color="#2c3e50")

    # Attribution panel — top-5 positive and negative tokens
    if captum_ok and "top5" in dir():
        ax.text(10.2, 3.8, "Captum attributions (last gen.)", fontsize=9,
                fontweight="bold", color="#8e44ad")
        ax.text(10.2, 3.3, "Top positive tokens:", fontsize=8, color="#27ae60")
        pos_tokens = [(s, t) for s, t in top5 if s > 0][:3]
        neg_tokens = [(s, t) for s, t in top5 if s < 0][:2]
        for i, (s, t) in enumerate(pos_tokens):
            clean_t = t.replace("Ġ", "").replace("Ċ", "\\n")
            ax.text(10.2, 2.9 - i * 0.45, f"  '{clean_t}' (+{s:.3f})", fontsize=8, color="#27ae60")
        if neg_tokens:
            ax.text(10.2, 1.6, "Top negative tokens:", fontsize=8, color="#c0392b")
            for i, (s, t) in enumerate(neg_tokens):
                clean_t = t.replace("Ġ", "").replace("Ċ", "\\n")
                ax.text(10.2, 1.2 - i * 0.45, f"  '{clean_t}' ({s:.3f})", fontsize=8, color="#c0392b")

    ax.set_title(
        "LoRA + Compliance — LLM Fine-Tuning Card\n"
        "EU AI Act Articles 11, 12, 13 | GDPR Articles 6, 7",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "lora_compliance_card.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved → {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
