"""
Art. 15 Evidence Pack — 2-page PDF handout.

WHAT IT PRODUCES
----------------
  storage/talk_assets/art15_evidence_pack.pdf

Page 1: Table of PyTorch entities that satisfy Art.15 (Accuracy, Robustness,
        Cybersecurity) with mapping_confidence >= 0.6.
        Columns: entity name, type, source file, confidence, rationale.

Page 2: Export survival matrix summary + gap report listing obligations
        with zero coverage.

USAGE
-----
    from src.assets.handout import HandoutAsset
    asset = HandoutAsset()
    path = asset.render(records, output_dir="storage/talk_assets")
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.assets.handout")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

_ART15_TAG          = "eu_ai_act_art_15"
_MIN_CONFIDENCE     = 0.6
_GITHUB_URL         = "https://github.com/pytorch-compliance/pytorch-compliance-toolkit"
_SPEAKER_NAME       = "Raja Gopal Hari Vijay"
_CONFERENCE         = "PyTorch Conference Europe 2026 — Paris"

# reportlab units: A4 page is 595 × 842 points.
_PAGE_W = 595
_PAGE_H = 842

# All EU AI Act articles to check for gap analysis.
_ALL_ARTICLES = [
    "eu_ai_act_art_9",
    "eu_ai_act_art_10",
    "eu_ai_act_art_11",
    "eu_ai_act_art_12",
    "eu_ai_act_art_13",
    "eu_ai_act_art_14",
    "eu_ai_act_art_15",
    "eu_ai_act_art_17",
    "eu_ai_act_art_43",
    "eu_ai_act_art_61",
    "eu_ai_act_art_72",
]

_ARTICLE_TITLES = {
    "eu_ai_act_art_9":  "Art. 9 — Risk Management",
    "eu_ai_act_art_10": "Art. 10 — Data Governance",
    "eu_ai_act_art_11": "Art. 11 — Technical Documentation",
    "eu_ai_act_art_12": "Art. 12 — Record-Keeping",
    "eu_ai_act_art_13": "Art. 13 — Transparency",
    "eu_ai_act_art_14": "Art. 14 — Human Oversight",
    "eu_ai_act_art_15": "Art. 15 — Accuracy / Robustness / Cybersecurity",
    "eu_ai_act_art_17": "Art. 17 — Quality Management",
    "eu_ai_act_art_43": "Art. 43 — Conformity Assessment",
    "eu_ai_act_art_61": "Art. 61 — Post-Market Monitoring",
    "eu_ai_act_art_72": "Art. 72 — Serious Incidents",
}


# ----------------------------------------------------------------------- #
# HandoutAsset
# ----------------------------------------------------------------------- #

class HandoutAsset:
    """Generates the 2-page Art.15 Evidence Pack PDF via reportlab."""

    def render(
        self,
        records: list["EntityRecord"],
        output_dir: str | Path = "storage/talk_assets",
    ) -> Path:
        """
        Build and save the evidence pack PDF.

        Parameters
        ----------
        records : list[EntityRecord]
            All annotated entity records.
        output_dir : str | Path
            Directory where the PDF is saved.

        Returns
        -------
        Path
            Full path of the written PDF file.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, HRFlowable,
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            logger.error(
                "HandoutAsset: reportlab is not installed. "
                "Run: pip install reportlab"
            )
            raise

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "art15_evidence_pack.pdf"

        # ── Styles ──────────────────────────────────────────────────────
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontSize=20,
            spaceAfter=6,
            textColor=colors.HexColor("#1a1a2e"),
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#636e72"),
            spaceAfter=4,
        )
        heading_style = ParagraphStyle(
            "Heading",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#2d3436"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
        )
        gap_style = ParagraphStyle(
            "Gap",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#e74c3c"),
        )
        covered_style = ParagraphStyle(
            "Covered",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#2ecc71"),
        )

        # ── Data preparation ────────────────────────────────────────────
        art15_entities = [
            r for r in records
            if _ART15_TAG in (r.compliance_tags or [])
            and float(r.mapping_confidence or 0.0) >= _MIN_CONFIDENCE
        ]
        art15_entities.sort(key=lambda r: -float(r.mapping_confidence or 0.0))

        # Gap analysis: articles with zero entities.
        covered_tags: set[str] = set()
        for r in records:
            covered_tags.update(r.compliance_tags or [])

        # Build export survival summary.
        survival_counts = _build_survival_counts(records)

        # ── PDF content ─────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )

        story = []

        # ── PAGE 1: Art.15 entity table ─────────────────────────────────
        story.append(Paragraph("Article 15 Evidence Pack", title_style))
        story.append(Paragraph(
            f"{_CONFERENCE}  ·  {_SPEAKER_NAME}  ·  {date.today().isoformat()}",
            subtitle_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#dfe6e9")))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            f"PyTorch entities mapped to Art. 15 (Accuracy, Robustness, "
            f"Cybersecurity) with mapping confidence ≥ {_MIN_CONFIDENCE:.0%}",
            body_style,
        ))
        story.append(Spacer(1, 0.3 * cm))

        if art15_entities:
            table_data = [
                ["Entity Name", "Type", "Source File", "Conf.", "Rationale"],
            ]
            for rec in art15_entities[:40]:   # cap at 40 rows for readability
                table_data.append([
                    Paragraph(rec.entity_name[:35], body_style),
                    rec.entity_type or "",
                    Paragraph(_short_path(rec.source_file or ""), body_style),
                    f"{float(rec.mapping_confidence or 0.0):.2f}",
                    Paragraph((rec.mapping_rationale or "")[:60], body_style),
                ])

            tbl = Table(
                table_data,
                colWidths=[3.8 * cm, 2.2 * cm, 4.0 * cm, 1.2 * cm, 5.8 * cm],
                repeatRows=1,
            )
            tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0),  9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f8f9fa"), colors.white]),
                ("FONTSIZE",     (0, 1), (-1, -1), 8),
                ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#dfe6e9")),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ]))
            story.append(tbl)
        else:
            story.append(Paragraph(
                "No Art.15 entities found with confidence ≥ 0.6. "
                "Run the LLM-enrich phase to improve coverage.",
                gap_style,
            ))

        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(
            f"Source: {_GITHUB_URL}",
            ParagraphStyle("Footer", parent=body_style,
                           textColor=colors.HexColor("#636e72"), fontSize=8),
        ))

        # ── PAGE 2: Export survival matrix + gap report ─────────────────
        story.append(PageBreak())
        story.append(Paragraph("Export Survival Matrix & Coverage Gap Report",
                                title_style))
        story.append(Paragraph(
            f"{_CONFERENCE}  ·  {_SPEAKER_NAME}",
            subtitle_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#dfe6e9")))
        story.append(Spacer(1, 0.3 * cm))

        # Export survival table.
        story.append(Paragraph("Export Survival Summary", heading_style))
        if survival_counts:
            surv_data = [["Export Format", "Survives", "Fails", "Partial"]]
            for fmt, counts in survival_counts.items():
                surv_data.append([
                    _COL_LABELS.get(fmt, fmt),
                    str(counts.get("yes", 0)),
                    str(counts.get("no", 0)),
                    str(counts.get("partial", 0)),
                ])
            surv_tbl = Table(
                surv_data,
                colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm],
            )
            surv_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2d3436")),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f8f9fa"), colors.white]),
                ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#dfe6e9")),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ]))
            story.append(surv_tbl)
        else:
            story.append(Paragraph(
                "No export survival data. Run --phase extract,annotate first.",
                gap_style,
            ))

        # Gap report.
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Compliance Coverage Gap Report", heading_style))
        story.append(Paragraph(
            "Articles with zero entity coverage require manual evidence collection.",
            body_style,
        ))
        story.append(Spacer(1, 0.2 * cm))

        for tag in _ALL_ARTICLES:
            title = _ARTICLE_TITLES.get(tag, tag)
            if tag in covered_tags:
                count = sum(1 for r in records if tag in (r.compliance_tags or []))
                story.append(Paragraph(f"✓  {title}  ({count} entities)", covered_style))
            else:
                story.append(Paragraph(f"✗  {title}  — NO COVERAGE", gap_style))

        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            f"Generated by PyTorch Compliance Toolkit · {_GITHUB_URL}",
            ParagraphStyle("Footer2", parent=body_style,
                           textColor=colors.HexColor("#636e72"), fontSize=8),
        ))

        doc.build(story)
        logger.info("HandoutAsset: saved %s (%d Art.15 entities)",
                    out_path, len(art15_entities))
        return out_path


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #

_COL_LABELS = {
    "onnx":    "ONNX",
    "ep":      "ExportedProgram",
    "ts":      "TorchScript",
    "compile": "torch.compile",
    "dcp":     "DCP",
}


def _build_survival_counts(records: list) -> dict[str, dict[str, int]]:
    """Aggregate export_survival data across all records."""
    counts: dict[str, dict[str, int]] = {
        fmt: {"yes": 0, "no": 0, "partial": 0}
        for fmt in ["onnx", "ep", "ts", "compile", "dcp"]
    }
    for rec in records:
        survival = getattr(rec, "export_survival", None) or {}
        for fmt in counts:
            val = survival.get(fmt)
            if val is True or val == "yes" or val == 1:
                counts[fmt]["yes"] += 1
            elif val is False or val == "no" or val == 0:
                counts[fmt]["no"] += 1
            elif val == "partial" or val == "conditional":
                counts[fmt]["partial"] += 1
    return counts


def _short_path(path: str) -> str:
    """Trim a file path to the last 2 components for readability."""
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else path
