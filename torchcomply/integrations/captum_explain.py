"""
captum_explain.py — Captum-based explainability for EU AI Act Article 13 (Transparency).

ComplianceExplainer wraps Captum's LayerIntegratedGradients / IntegratedGradients
and structures attribution results for inclusion in Annex IV compliance reports.

Integrated Gradients (Sundararajan et al., 2017)
-------------------------------------------------
Integrated Gradients satisfies two desirable axioms for attribution methods:

  Sensitivity: if changing a feature changes the output, that feature gets
  non-zero attribution (rules out vanilla gradients which can be zero at flat regions).

  Implementation Invariance: two functionally identical networks always produce
  the same attributions (rules out methods that depend on network internals).

The method interpolates the input from a baseline b (typically zeros or a
neutral reference) to the actual input x, integrates the gradient along that
path, and multiplies by (x - b):

    IG_i(x) = (x_i - b_i) × ∫₀¹ ∂F(b + α(x-b)) / ∂x_i dα

For text models, ``LayerIntegratedGradients`` applies IG at the embedding layer
and sums over the embedding dimension to get a scalar score per token.

Baseline selection guidance:
  - Text models: zero-embedding vector (pad token ID = 0)
  - Image models: black image (all-zeros) or blurred baseline (Fong & Vedaldi, 2017)
  - Tabular: mean of training data, or zero for normalised features

Regulatory references:
  EU AI Act Art. 13 — Transparency and provision of information to users
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 13)
  EU AI Act Art. 14(3) — Ability to understand system output
  GDPR Art. 22(3) — Right to explanation for automated decisions
    https://gdpr-info.eu/art-22-gdpr/

Key papers:
  Sundararajan, Taly & Yan (2017) — Axiomatic Attribution for Deep Networks
    https://arxiv.org/abs/1703.01365
  Kokhlikyan et al. (2020) — Captum: A Unified and Generic Model Interpretability
  Library for PyTorch
    https://arxiv.org/abs/2009.07896

Captum documentation:
  https://captum.ai/
  LayerIntegratedGradients: https://captum.ai/api/layer.html#captum.attr.LayerIntegratedGradients
"""

from __future__ import annotations

from typing import List, Optional

import torch


class ComplianceExplainer:
    """
    Provides Captum attribution explanations with compliance logging.

    Args:
        model: A PyTorch model (DistilBERT / BERT for text, CNN for images).
        tokenizer: HuggingFace tokenizer (required for explain_text).
    """

    def __init__(self, model: torch.nn.Module, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Text explanation (LayerIntegratedGradients on embedding layer)
    # ------------------------------------------------------------------

    def explain_text(self, text: str, target_class: Optional[int] = None) -> dict:
        """
        Attribute a text prediction to individual tokens.

        Returns dict with keys: tokens, attribution_scores, predicted_class, confidence.
        """
        from captum.attr import LayerIntegratedGradients

        if self.tokenizer is None:
            raise ValueError(
                "tokenizer is required for explain_text(). "
                "Pass tokenizer= to ComplianceExplainer.__init__()."
            )
        self.model.eval()
        device = next(self.model.parameters()).device

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        predicted_class = int(logits.argmax(dim=-1).item())
        confidence = float(torch.softmax(logits, dim=-1)[0, predicted_class].item())
        if target_class is None:
            target_class = predicted_class

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        attribution_scores: list = []

        embedding_layer = self._get_embedding_layer()
        if embedding_layer is None:
            import warnings

            warnings.warn(
                "No supported embedding layer found (tried .distilbert, .bert, .roberta). "
                "attribution_scores will be empty. "
                "Override _get_embedding_layer() or pass the layer explicitly.",
                UserWarning,
                stacklevel=2,
            )
        if embedding_layer is not None:
            lig = LayerIntegratedGradients(self._text_forward, embedding_layer)
            baseline_ids = torch.zeros_like(input_ids)
            attrs, _ = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=target_class,
                n_steps=30,
                return_convergence_delta=True,
            )
            attribution_scores = attrs.sum(dim=-1).squeeze(0).tolist()

        return {
            "tokens": tokens,
            "attribution_scores": attribution_scores,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

    def _text_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def _get_embedding_layer(self):
        """Return the token embedding layer for LayerIntegratedGradients."""
        if hasattr(self.model, "distilbert"):
            return self.model.distilbert.embeddings
        if hasattr(self.model, "bert"):
            return self.model.bert.embeddings
        if hasattr(self.model, "roberta"):
            return self.model.roberta.embeddings
        return None

    # ------------------------------------------------------------------
    # Image explanation (IntegratedGradients)
    # ------------------------------------------------------------------

    def explain_image(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> dict:
        """
        Attribute an image prediction to pixel regions.

        Returns dict with keys: attribution_map, predicted_class, confidence.
        """
        from captum.attr import IntegratedGradients

        self.model.eval()
        device = next(self.model.parameters()).device
        x = (image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor).to(device)
        x = x.float()

        with torch.no_grad():
            logits = self.model(x)

        predicted_class = int(logits.argmax(dim=-1).item())
        confidence = float(torch.softmax(logits, dim=-1)[0, predicted_class].item())
        if target_class is None:
            target_class = predicted_class

        ig = IntegratedGradients(self.model)
        baseline = torch.zeros_like(x)
        attrs, _ = ig.attribute(
            x,
            baselines=baseline,
            target=target_class,
            n_steps=30,
            return_convergence_delta=True,
        )

        return {
            "attribution_map": attrs.squeeze(0),
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_explanation_report(self, explanations: List[dict]) -> str:
        """Format a list of explanation dicts as a human-readable compliance section."""
        lines = [
            "=== Explainability Report (Captum Integrated Gradients) ===",
            "Regulation: EU AI Act Article 13 — Transparency and information provision",
            "",
        ]
        for i, exp in enumerate(explanations, start=1):
            lines.append(f"Explanation {i}:")
            lines.append(f"  Predicted class : {exp.get('predicted_class', 'N/A')}")
            lines.append(f"  Confidence      : {exp.get('confidence', 0):.1%}")
            tokens = exp.get("tokens", [])
            scores = exp.get("attribution_scores", [])
            if tokens and scores:
                paired = sorted(zip(scores, tokens), key=lambda x: abs(x[0]), reverse=True)
                top5 = paired[:5]
                lines.append(
                    "  Top 5 tokens    : " + ", ".join(f"'{t}' ({s:+.3f})" for s, t in top5)
                )
            lines.append("")
        return "\n".join(lines)
