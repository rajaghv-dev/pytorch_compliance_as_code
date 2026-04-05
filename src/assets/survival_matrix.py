"""
Export survival matrix visualisation — PNG for the conference talk.

WHAT IT PRODUCES
----------------
A 1920×1080 colour-coded table saved to
  `storage/talk_assets/survival_matrix.png`

Rows  = top compliance entities (hook points, operators, export APIs)
Cols  = export formats: ONNX | ExportedProgram | TorchScript | compile | DCP

Cell colours
  Green  (#2ecc71) — entity survives in this format
  Red    (#e74c3c) — entity does not survive
  Yellow (#f39c12) — partial / conditional survival

USAGE
-----
    from src.assets.survival_matrix import SurvivalMatrixAsset
    asset = SurvivalMatrixAsset()
    path = asset.render(records, output_dir="storage/talk_assets")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.assets.survival_matrix")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

# Output dimensions in pixels at 100 dpi.
_WIDTH_PX  = 1920
_HEIGHT_PX = 1080
_DPI        = 100

# Export format columns (matches export_survival dict keys in EntityRecord).
_COLUMNS = ["onnx", "ep", "ts", "compile", "dcp"]
_COL_LABELS = {
    "onnx":    "ONNX",
    "ep":      "ExportedProgram",
    "ts":      "TorchScript",
    "compile": "torch.compile",
    "dcp":     "DCP",
}

# Cell colours.
_GREEN  = "#2ecc71"
_RED    = "#e74c3c"
_YELLOW = "#f39c12"
_GREY   = "#bdc3c7"   # no data / not applicable

# How many rows to display (top N by number of compliance tags).
_MAX_ROWS = 30


# ----------------------------------------------------------------------- #
# SurvivalMatrixAsset
# ----------------------------------------------------------------------- #

class SurvivalMatrixAsset:
    """Renders the export survival matrix as a PNG conference slide."""

    def render(
        self,
        records: list["EntityRecord"],
        output_dir: str | Path = "storage/talk_assets",
    ) -> Path:
        """
        Build and save the survival matrix PNG.

        Parameters
        ----------
        records : list[EntityRecord]
            All annotated entity records.  Only records with a non-empty
            export_survival dict are included in the matrix.
        output_dir : str | Path
            Directory where the PNG is saved.

        Returns
        -------
        Path
            Full path of the written PNG file.
        """
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for server use
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "survival_matrix.png"

        # Select rows: entities with export_survival data, sorted by
        # compliance tag count (most-tagged first), then by name.
        candidates = [
            r for r in records
            if r.export_survival and isinstance(r.export_survival, dict)
        ]
        candidates.sort(
            key=lambda r: (-len(r.compliance_tags or []), r.entity_name)
        )
        rows = candidates[:_MAX_ROWS]

        if not rows:
            logger.warning(
                "SurvivalMatrixAsset: no records with export_survival data — "
                "generating placeholder PNG"
            )
            rows = []   # render empty table

        logger.info(
            "SurvivalMatrixAsset: rendering %d rows × %d cols at %dx%d …",
            len(rows),
            len(_COLUMNS),
            _WIDTH_PX,
            _HEIGHT_PX,
        )

        fig, ax = plt.subplots(
            figsize=(_WIDTH_PX / _DPI, _HEIGHT_PX / _DPI),
            dpi=_DPI,
        )
        ax.set_axis_off()
        fig.patch.set_facecolor("#1a1a2e")   # dark conference background

        # ── Title ──────────────────────────────────────────────────────
        fig.text(
            0.5, 0.97,
            "PyTorch Export Survival Matrix",
            ha="center", va="top",
            fontsize=28, fontweight="bold", color="white",
        )
        fig.text(
            0.5, 0.93,
            "Which PyTorch APIs survive each export format?",
            ha="center", va="top",
            fontsize=16, color="#95a5a6",
        )

        if not rows:
            fig.text(
                0.5, 0.5,
                "No export survival data available.\n"
                "Run the pipeline with --phase extract,annotate first.",
                ha="center", va="center",
                fontsize=18, color="#e74c3c",
            )
            fig.savefig(str(out_path), bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            logger.info("SurvivalMatrixAsset: placeholder saved to %s", out_path)
            return out_path

        # ── Table layout ───────────────────────────────────────────────
        n_rows = len(rows)
        n_cols = len(_COLUMNS)

        # Determine cell size and positioning.
        table_left   = 0.22   # fraction of figure width
        table_right  = 0.97
        table_top    = 0.88
        table_bottom = 0.08

        col_width  = (table_right - table_left) / n_cols
        row_height = (table_top - table_bottom) / (n_rows + 1)   # +1 for header

        # Draw column headers.
        for ci, col_key in enumerate(_COLUMNS):
            cx = table_left + (ci + 0.5) * col_width
            cy = table_top - 0.5 * row_height
            ax.text(
                cx, cy,
                _COL_LABELS[col_key],
                transform=fig.transFigure,
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="white",
            )

        # Draw rows.
        for ri, rec in enumerate(rows):
            cy = table_top - (ri + 1.5) * row_height
            bg_color = "#16213e" if ri % 2 == 0 else "#0f3460"

            # Row background.
            rect = mpatches.FancyBboxPatch(
                (0.01, cy - row_height * 0.45),
                0.98,
                row_height * 0.9,
                boxstyle="round,pad=0.01",
                transform=fig.transFigure,
                facecolor=bg_color,
                edgecolor="none",
                zorder=0,
            )
            fig.add_artist(rect)

            # Entity name label (left side).
            ax.text(
                0.02, cy,
                _truncate(rec.entity_name, 28),
                transform=fig.transFigure,
                ha="left", va="center",
                fontsize=11, color="#ecf0f1",
                fontfamily="monospace",
            )

            # Compliance tags label.
            tags_str = ", ".join(
                t.replace("eu_ai_act_art_", "Art.").replace("gdpr_art_", "G.")
                for t in (rec.compliance_tags or [])[:3]
            )
            ax.text(
                0.22, cy,
                tags_str,
                transform=fig.transFigure,
                ha="right", va="center",
                fontsize=9, color="#95a5a6",
            )

            # Survival cells.
            survival = rec.export_survival or {}
            for ci, col_key in enumerate(_COLUMNS):
                cx = table_left + (ci + 0.5) * col_width
                value = survival.get(col_key, None)
                cell_color, symbol = _survival_color(value)

                # Draw coloured circle.
                circle = plt.Circle(
                    (cx, cy),
                    row_height * 0.35,
                    color=cell_color,
                    transform=fig.transFigure,
                    zorder=1,
                )
                fig.add_artist(circle)

                # Symbol in centre of circle.
                ax.text(
                    cx, cy,
                    symbol,
                    transform=fig.transFigure,
                    ha="center", va="center",
                    fontsize=14, color="white", fontweight="bold",
                    zorder=2,
                )

        # ── Legend ─────────────────────────────────────────────────────
        legend_y = 0.03
        legend_items = [
            (_GREEN,  "✓", "Survives"),
            (_RED,    "✗", "Does not survive"),
            (_YELLOW, "~", "Partial / conditional"),
            (_GREY,   "?", "Not tested"),
        ]
        lx = 0.25
        for color, sym, label in legend_items:
            circle = plt.Circle((lx, legend_y), 0.012, color=color,
                                 transform=fig.transFigure, zorder=1)
            fig.add_artist(circle)
            ax.text(lx, legend_y, sym, transform=fig.transFigure,
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold", zorder=2)
            ax.text(lx + 0.02, legend_y, label, transform=fig.transFigure,
                    ha="left", va="center", fontsize=10, color="#bdc3c7")
            lx += 0.18

        fig.savefig(
            str(out_path),
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

        logger.info(
            "SurvivalMatrixAsset: saved %s (%d rows)",
            out_path,
            len(rows),
        )
        return out_path


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #

def _survival_color(value) -> tuple[str, str]:
    """Map an export_survival value to (cell_color, symbol)."""
    if value is True or value == "yes" or value == 1:
        return _GREEN,  "✓"
    if value is False or value == "no" or value == 0:
        return _RED,    "✗"
    if value == "partial" or value == "conditional":
        return _YELLOW, "~"
    return _GREY, "?"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"
