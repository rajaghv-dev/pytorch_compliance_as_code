"""
annex_iv.py — EU AI Act Annex IV Technical Documentation PDF generator.

AnnexIVReport builds a professional A4 PDF covering all Annex IV sections:
architecture, training data, methodology, privacy, explainability, fairness,
audit trail, and regulatory mapping.

EU AI Act Annex IV Requirements
---------------------------------
Annex IV specifies the minimum content for the Technical Documentation that
high-risk AI system providers must draw up before placing a system on the market
(Article 11). The six sections required by Annex IV are:

  §1 General description of the AI system
  §2 Detailed description of the elements of the AI system and process for its development
  §3 Detailed information about the monitoring, functioning and control of the AI system
  §4 Description of the appropriateness of the performance metrics
  §5 Description of any change made to the system through its lifecycle
  §6 EU declaration of conformity (by reference)

This report covers §1–§4 programmatically. §5 (change history) requires manual
versioning and CI/CD integration. §6 (declaration of conformity) is an
administrative/legal document issued by the provider's legal function.

PDF generation uses ReportLab, a production-grade Python PDF library used by
organisations including NASA and the US SEC for regulatory filings.

Regulatory references:
  EU AI Act Annex IV — Technical Documentation
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Annex IV)
  EU AI Act Art. 11 — Technical documentation
  EU AI Act Art. 43 — Conformity assessment procedure

ReportLab documentation:
  https://docs.reportlab.com/reportlab/userguide/ch1_intro/
  Platypus (Page Layout and Typography Using Scripts):
    https://docs.reportlab.com/reportlab/userguide/ch6_intro/
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn

# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


@dataclass
class _LayerInfo:
    name: str
    layer_type: str
    param_count: int


class ModelIntrospector:
    """Extracts architecture metadata from a PyTorch model."""

    def __init__(self, model: nn.Module) -> None:
        self.architecture: str = type(model).__name__
        self.layers: List[dict] = []
        self.total_params: int = 0
        self.trainable_params: int = 0
        self._inspect(model)

    def _inspect(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if not list(module.children()):  # leaf only
                params = sum(p.numel() for p in module.parameters())
                self.layers.append(
                    {"name": name, "type": type(module).__name__, "param_count": params}
                )
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


class AnnexIVReport:
    """
    EU AI Act Annex IV Technical Documentation PDF.

    Args:
        model_introspection: ``ModelIntrospector`` result.
        audit_chain: ``AuditChain`` instance from torchcomply.core.audit.
        fairness_log: List of per-epoch fairness dicts from FairnessGate.
        training_config: Dict of training hyper-parameters.
        dataset_info: Dict with dataset name, size, class distribution, etc.
        dp_info: Optional DP metadata dict (from CompliancePrivacyEngine.log_to_dict()).
        explanations: Optional list of explanation dicts (from ComplianceExplainer).
        regulations: List of regulation IDs covered.
    """

    _REGULATORY_MAPPING = [
        ("Section 2 — Model Architecture", "EU AI Act Art. 11 (Technical Documentation)"),
        ("Section 3 — Training Data", "EU AI Act Art. 10 (Data Governance); GDPR Art. 5"),
        ("Section 4 — Training Methodology", "EU AI Act Art. 9 (Risk Management)"),
        ("Section 5 — Privacy Assessment", "GDPR Art. 25 (Privacy by Design); GDPR Art. 32"),
        ("Section 6 — Explainability", "EU AI Act Art. 13 (Transparency)"),
        ("Section 7 — Fairness Assessment", "EU AI Act Art. 10 (Non-discrimination)"),
        ("Section 8 — Audit Trail", "EU AI Act Art. 12 (Record-Keeping)"),
        ("Section 9 — Human Oversight", "EU AI Act Art. 14 (Human Oversight)"),
    ]

    def __init__(
        self,
        model_introspection: ModelIntrospector,
        audit_chain,
        fairness_log: List[dict],
        training_config: dict,
        dataset_info: dict,
        dp_info: Optional[dict] = None,
        explanations: Optional[List[dict]] = None,
        regulations: Optional[List[str]] = None,
    ) -> None:
        self.mi = model_introspection
        self.audit_chain = audit_chain
        self.fairness_log = fairness_log
        self.training_config = training_config
        self.dataset_info = dataset_info
        self.dp_info = dp_info
        self.explanations = explanations or []
        self.regulations = regulations or ["eu_ai_act", "gdpr"]

    def save_pdf(self, filepath: str) -> None:
        """Generate and save the Annex IV PDF."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            HRFlowable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            spaceAfter=12,
        )
        h1_style = ParagraphStyle(
            "H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=6
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"], fontName="Helvetica", fontSize=10, spaceAfter=4
        )
        ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.grey,
        )

        ROW_EVEN = colors.white
        ROW_ODD = colors.Color(0.94, 0.94, 0.94)

        def _table(headers: list, rows: list) -> Table:
            data = [headers] + rows
            t = Table(data, repeatRows=1)
            style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.2)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_EVEN, ROW_ODD]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
            t.setStyle(TableStyle(style))
            return t

        story = []

        # ------------------------------------------------------------------
        # Title page
        # ------------------------------------------------------------------
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph("EU AI Act — Annex IV Technical Documentation", title_style))
        story.append(Paragraph("Auto-generated Compliance Report", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 0.2 * inch))
        story.append(
            _table(
                ["Field", "Value"],
                [
                    ["System name", self.dataset_info.get("system_name", "torchcomply pipeline")],
                    ["Developer", self.dataset_info.get("developer", "—")],
                    ["Version", self.dataset_info.get("version", "0.1.0")],
                    ["Generated", timestamp],
                    ["Purpose", self.dataset_info.get("purpose", "Machine learning inference")],
                    ["Risk level", self.dataset_info.get("risk_level", "High")],
                    [
                        "Regulations",
                        ", ".join(r.upper().replace("_", " ") for r in self.regulations),
                    ],
                ],
            )
        )
        story.append(PageBreak())

        # ------------------------------------------------------------------
        # Section 2 — Model Architecture
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 2 — Model Architecture", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(
                f"Architecture: <b>{self.mi.architecture}</b> | "
                f"Total parameters: <b>{self.mi.total_params:,}</b> | "
                f"Trainable: <b>{self.mi.trainable_params:,}</b>",
                body_style,
            )
        )
        story.append(Spacer(1, 0.1 * inch))
        layer_rows = [
            [lr["name"][:50], lr["type"], f"{lr['param_count']:,}"] for lr in self.mi.layers[:40]
        ]
        if len(self.mi.layers) > 40:
            layer_rows.append(["...", f"({len(self.mi.layers) - 40} more layers)", ""])
        story.append(_table(["Layer Name", "Type", "Parameters"], layer_rows))
        story.append(Spacer(1, 0.2 * inch))

        # ------------------------------------------------------------------
        # Section 3 — Training Data
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 3 — Training Data", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        ds_rows = [[str(k), str(v)] for k, v in self.dataset_info.items()]
        story.append(_table(["Attribute", "Value"], ds_rows))
        story.append(Spacer(1, 0.2 * inch))

        # ------------------------------------------------------------------
        # Section 4 — Training Methodology
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 4 — Training Methodology", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        tc_rows = [[str(k), str(v)] for k, v in self.training_config.items()]
        if tc_rows:
            story.append(_table(["Parameter", "Value"], tc_rows))
        else:
            story.append(Paragraph("No training configuration provided.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        # ------------------------------------------------------------------
        # Section 5 — Privacy Assessment (optional)
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 5 — Privacy Assessment", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        if self.dp_info:
            dp_rows = [[str(k), str(v)] for k, v in self.dp_info.items()]
            story.append(_table(["Attribute", "Value"], dp_rows))
        else:
            story.append(Paragraph("Differential privacy not applied in this run.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        # ------------------------------------------------------------------
        # Section 6 — Explainability
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 6 — Explainability (Captum)", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        if self.explanations:
            exp_rows = []
            for i, exp in enumerate(self.explanations[:10], start=1):
                tokens = exp.get("tokens", [])
                scores = exp.get("attribution_scores", [])
                top_tokens = ""
                if tokens and scores:
                    paired = sorted(zip(scores, tokens), key=lambda x: abs(x[0]), reverse=True)
                    top_tokens = ", ".join(f"'{t}'" for _, t in paired[:5])
                exp_rows.append(
                    [
                        str(i),
                        str(exp.get("predicted_class", "—")),
                        f"{exp.get('confidence', 0):.1%}",
                        top_tokens[:60],
                    ]
                )
            story.append(
                _table(
                    ["#", "Predicted Class", "Confidence", "Top Attributed Tokens"],
                    exp_rows,
                )
            )
        else:
            story.append(Paragraph("No Captum explanations logged.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        # ------------------------------------------------------------------
        # Section 7 — Fairness Assessment
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 7 — Fairness Assessment", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        if self.fairness_log:
            fair_rows = [
                [
                    str(e["epoch"]),
                    f"{e['disparity']:.4f}",
                    f"{e['threshold']:.4f}",
                    e["status"].upper(),
                ]
                for e in self.fairness_log
            ]
            story.append(_table(["Epoch", "Disparity", "Threshold", "Status"], fair_rows))
        else:
            story.append(Paragraph("No fairness checks logged.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        # ------------------------------------------------------------------
        # Section 8 — Audit Trail
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 8 — Audit Trail Summary", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        summary = self.audit_chain.summary()
        audit_rows = [
            ["Total entries", str(summary.get("total_entries", 0))],
            ["Unique operator types", str(summary.get("unique_operators", 0))],
            ["Chain integrity", "✅ VALID" if summary.get("chain_valid") else "❌ INVALID"],
        ]
        if self.audit_chain.entries:
            audit_rows.append(["First entry hash", self.audit_chain.entries[0].hash[:16] + "..."])
            audit_rows.append(["Last entry hash", self.audit_chain.entries[-1].hash[:16] + "..."])
        story.append(_table(["Attribute", "Value"], audit_rows))
        story.append(Spacer(1, 0.2 * inch))

        # ------------------------------------------------------------------
        # Section 9 — Regulatory Mapping
        # ------------------------------------------------------------------
        story.append(Paragraph("Section 9 — Regulatory Mapping", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(_table(["Report Section", "Regulation / Article"], self._REGULATORY_MAPPING))

        # ------------------------------------------------------------------
        # Footer on every page via onFirstPage / onLaterPages
        # ------------------------------------------------------------------
        def _add_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.grey)
            footer_text = f"Auto-generated by torchcomply — {timestamp} | Page {doc.page}"
            canvas.drawCentredString(A4[0] / 2.0, 0.5 * inch, footer_text)
            canvas.restoreState()

        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            leftMargin=inch,
            rightMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )
        doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
