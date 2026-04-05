"""
Example 3: Model Explainability — EU AI Act Article 13 (Transparency)
Uses Captum LayerIntegratedGradients on DistilBERT to attribute predictions.
"""

import hashlib
import json
import pathlib
import time

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from torchcomply.integrations.captum_explain import ComplianceExplainer

REVIEWS = [
    "This film is absolutely stunning. A true masterpiece of cinema.",
    "Terrible movie, boring plot, awful acting. Complete waste of time.",
    "An emotional and uplifting story. I laughed and cried throughout.",
    "Dull and predictable. The director showed no creative vision.",
    "Magnificent cinematography and a gripping narrative. Must watch.",
]

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 3: Model Explainability — EU AI Act Article 13 (Transparency)")
    print("Model: DistilBERT (66M params) | Captum IntegratedGradients")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device).eval()

    explainer = ComplianceExplainer(model, tokenizer)

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    attribution_log_path = out_dir / "attribution_log.jsonl"

    print(f"\nNote: Attribution baseline = zero embedding vector (padding-token equivalent).")
    print(f"      Scores are relative to this reference, not an absolute importance measure.")
    print("\n" + "-" * 70)

    SPECIAL_TOKENS = {"[SEP]", "[CLS]", "[PAD]"}

    results = []
    for review in REVIEWS:
        # Primary attribution: for the predicted class
        result = explainer.explain_text(review, target_class=None)
        results.append({"review": review, **result})

        tokens = result["tokens"]
        scores = result["attribution_scores"]
        paired = sorted(zip(scores, tokens), key=lambda x: abs(x[0]), reverse=True)
        # Filter special tokens from top-5 display
        top5 = [(s, t) for s, t in paired if t not in SPECIAL_TOKENS][:5]

        print(f"\nReview: '{review[:60]}…'")
        print(f"  Prediction: {LABELS[result['predicted_class']]} ({result['confidence']:.1%})")
        print(f"  Attribution (for predicted class={result['predicted_class']}):")
        print("    " + "  ".join(f"'{t}' ({s:+.3f})" for s, t in top5))

        # Contrastive: also show attribution for the *other* class when confidence < 80%
        if result["confidence"] < 0.80:
            other_class = 1 - result["predicted_class"]
            result_contra = explainer.explain_text(review, target_class=other_class)
            paired_c = [
                (s, t) for s, t in sorted(
                    zip(result_contra["attribution_scores"], result_contra["tokens"]),
                    key=lambda x: abs(x[0]), reverse=True
                ) if t not in SPECIAL_TOKENS
            ][:5]
            print(f"  Contrastive (for class={other_class}, confidence was only {result['confidence']:.1%}):")
            print("    " + "  ".join(f"'{t}' ({s:+.3f})" for s, t in paired_c))

        # Persist to JSONL — one record per review, durable attribution log
        import datetime as _dt
        input_hash = hashlib.sha256(review.encode()).hexdigest()[:16]
        log_entry = {
            "timestamp": int(time.time_ns()),
            "iso_timestamp": _dt.datetime.fromtimestamp(time.time(), tz=_dt.timezone.utc).isoformat(),
            "input_hash": input_hash,
            "review_snippet": review[:80],
            "predicted_class": result["predicted_class"],
            "confidence": round(result["confidence"], 4),
            "top5_tokens": [t for _, t in top5],  # special tokens already filtered
            "top5_scores": [round(s, 4) for s, _ in top5],
            "baseline": "zero_embedding_vector",
            "regulation": "EU AI Act Art.13",
        }
        with open(attribution_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print("-" * 70)
    print(f"\nAttributions persisted → {attribution_log_path}  ({len(results)} records)")
    print(f"  Each record: input_hash, predicted class, top-5 tokens + scores, timestamp")

    # ------------------------------------------------------------------
    # Visualisation 1 — Side-by-side: POSITIVE review[0] and NEGATIVE review[1]
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, r_idx in enumerate([0, 1]):
        focus = results[r_idx]
        tokens = [t.replace("##", "") for t in focus["tokens"][:20]]
        scores = focus["attribution_scores"][:20]
        colors = ["seagreen" if s >= 0 else "crimson" for s in scores]
        ax = axes[col]
        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, scores, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Attribution score")
        sentiment_label = LABELS[focus["predicted_class"]]
        ax.set_title(
            f"{sentiment_label} ({focus['confidence']:.1%})\n"
            f"\"{focus['review'][:50]}…\"",
            fontsize=9.5
        )

    fig.suptitle(
        "Token Attributions — Captum IntegratedGradients\n"
        "EU AI Act Article 13 | Left: POSITIVE review | Right: NEGATIVE review",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    out1 = out_dir / "captum_attribution.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization 1 saved → {out1}")

    # ------------------------------------------------------------------
    # Visualisation 2 — Heatmap across all 5 reviews (20 token positions)
    # ------------------------------------------------------------------
    MAX_TOKENS = 20
    matrix = []
    y_labels = []
    # Use the first result's tokens as representative X-axis labels (padded with position indices)
    x_token_labels = [t.replace("##", "") for t in results[0]["tokens"][:MAX_TOKENS]]
    x_token_labels += [str(i) for i in range(len(x_token_labels), MAX_TOKENS)]
    for r in results:
        row = r["attribution_scores"][:MAX_TOKENS]
        row += [0.0] * (MAX_TOKENS - len(row))
        matrix.append(row)
        y_labels.append(
            f"{LABELS[r['predicted_class']]} | {r['review'][:28]}…"
        )

    mat = np.array(matrix)
    vmax = np.abs(mat).max() or 1.0

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xticks(range(MAX_TOKENS))
    ax.set_xticklabels(x_token_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Token")
    ax.set_title("Attribution patterns across 5 reviews — EU AI Act Article 13")
    plt.colorbar(im, ax=ax, label="Attribution score")
    plt.tight_layout()

    out2 = out_dir / "attribution_heatmap.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Visualization 2 saved → {out2}")
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
