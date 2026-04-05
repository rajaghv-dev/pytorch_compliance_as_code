"""Tests for ComplianceExplainer (Captum) — EU AI Act Article 13."""

import pytest

pytest.importorskip("captum", reason="captum not installed")
pytest.importorskip("transformers", reason="transformers not installed")


def test_text_explanation():
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

    from torchcomply.integrations.captum_explain import ComplianceExplainer

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model.eval()

    explainer = ComplianceExplainer(model, tokenizer)
    result = explainer.explain_text("This film is absolutely wonderful!")

    assert "tokens" in result
    assert "attribution_scores" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert len(result["tokens"]) > 0
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_explanation_report():
    from torchcomply.integrations.captum_explain import ComplianceExplainer

    import torch

    dummy_model = torch.nn.Linear(10, 2)  # not used for this test
    explainer = ComplianceExplainer(dummy_model)

    explanations = [
        {
            "tokens": ["[CLS]", "great", "film", "[SEP]"],
            "attribution_scores": [0.01, 0.85, 0.42, 0.02],
            "predicted_class": 1,
            "confidence": 0.94,
        },
        {
            "tokens": ["[CLS]", "terrible", "movie", "[SEP]"],
            "attribution_scores": [0.0, -0.90, -0.35, 0.01],
            "predicted_class": 0,
            "confidence": 0.87,
        },
    ]

    report = explainer.generate_explanation_report(explanations)
    assert "Explanation 1" in report
    assert "Explanation 2" in report
    assert "94.0%" in report or "0.94" in report or "94" in report
    assert "great" in report
