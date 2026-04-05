"""
Pre-rendered SPARQL query result screenshots for the conference talk.

WHAT IT PRODUCES
----------------
Renders the 5 SPARQL queries from `queries/*.rq` against the compiled RDF
graph and saves the results as formatted PNG images:

  storage/talk_assets/sparql_coverage_matrix.png
  storage/talk_assets/sparql_export_gaps.png
  storage/talk_assets/sparql_determinism_census.png
  storage/talk_assets/sparql_test_coverage.png
  storage/talk_assets/sparql_article14_gap.png

Each PNG shows the query text on the top half and the result table on the
bottom half, formatted for projection at 1280×720.

USAGE
-----
    from src.assets.screenshots import SparqlScreenshots
    asset = SparqlScreenshots()
    paths = asset.render_all(
        ttl_path="storage/rdf/compliance.ttl",
        queries_dir="queries/",
        output_dir="storage/talk_assets",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger("pct.assets.screenshots")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

_WIDTH_PX  = 1280
_HEIGHT_PX = 720
_DPI        = 100

# Map query file name → human-readable slide title.
_QUERY_TITLES = {
    "coverage_matrix.rq":     "Coverage Matrix — Entities per Article",
    "export_gaps.rq":         "Export Gaps — What Doesn't Survive",
    "determinism_census.rq":  "Determinism Census — Non-deterministic Operators",
    "test_coverage.rq":       "Test Coverage — Tests per Compliance Article",
    "article14_gap.rq":       "Art.14 Gap — Human Oversight Coverage",
}

# Maximum rows to display per screenshot.
_MAX_DISPLAY_ROWS = 20


# ----------------------------------------------------------------------- #
# SparqlScreenshots
# ----------------------------------------------------------------------- #

class SparqlScreenshots:
    """Renders pre-built SPARQL queries as PNG screenshots."""

    def render_all(
        self,
        ttl_path: str | Path = "storage/rdf/compliance.ttl",
        queries_dir: str | Path = "queries/",
        output_dir: str | Path = "storage/talk_assets",
    ) -> list[Path]:
        """
        Run all .rq query files and render each result as a PNG.

        Parameters
        ----------
        ttl_path : str | Path
            Path to the compiled Turtle ontology file.
        queries_dir : str | Path
            Directory containing .rq SPARQL files.
        output_dir : str | Path
            Directory where screenshot PNGs are saved.

        Returns
        -------
        list[Path]
            List of saved PNG paths (one per query file).
        """
        try:
            from rdflib import Graph
        except ImportError:
            logger.error("rdflib not installed. Run: pip install rdflib")
            raise

        ttl_path   = Path(ttl_path)
        queries_dir = Path(queries_dir)
        out_dir    = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not ttl_path.exists():
            logger.warning(
                "SparqlScreenshots: TTL file not found at %s — "
                "run the convert phase first", ttl_path
            )
            return []

        # Load the graph once; reuse for all queries.
        logger.info("SparqlScreenshots: loading graph from %s …", ttl_path)
        graph = Graph()
        graph.parse(str(ttl_path), format="turtle")
        logger.info("SparqlScreenshots: graph has %d triples", len(graph))

        saved_paths: list[Path] = []
        rq_files = sorted(queries_dir.glob("*.rq"))

        if not rq_files:
            logger.warning(
                "SparqlScreenshots: no .rq files found in %s", queries_dir
            )
            return []

        for rq_path in rq_files:
            try:
                out_path = self._render_one(
                    graph=graph,
                    rq_path=rq_path,
                    out_dir=out_dir,
                )
                saved_paths.append(out_path)
            except Exception as exc:
                logger.warning(
                    "SparqlScreenshots: failed to render %s: %s",
                    rq_path.name,
                    exc,
                )

        logger.info(
            "SparqlScreenshots: saved %d screenshots to %s",
            len(saved_paths),
            out_dir,
        )
        return saved_paths

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _render_one(
        self,
        graph,
        rq_path: Path,
        out_dir: Path,
    ) -> Path:
        """Execute one SPARQL query and render the result as a PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sparql_text = rq_path.read_text(encoding="utf-8")
        title = _QUERY_TITLES.get(rq_path.name, rq_path.stem)
        out_stem = rq_path.stem
        out_path = out_dir / f"sparql_{out_stem}.png"

        # Run the query.
        try:
            results = list(graph.query(sparql_text))
        except Exception as exc:
            logger.debug(
                "SparqlScreenshots: SPARQL error for %s: %s",
                rq_path.name,
                exc,
            )
            results = []

        # Determine column names.
        if results:
            try:
                col_names = [str(v) for v in results[0].labels]
            except AttributeError:
                col_names = [f"col{i}" for i in range(len(results[0]))]
        else:
            col_names = []

        # Convert rows to string lists.
        row_data = []
        for row in results[:_MAX_DISPLAY_ROWS]:
            row_data.append([
                _format_cell(row[i]) for i in range(len(col_names))
            ])

        # Render PNG.
        fig, axes = plt.subplots(
            2, 1,
            figsize=(_WIDTH_PX / _DPI, _HEIGHT_PX / _DPI),
            dpi=_DPI,
            gridspec_kw={"height_ratios": [1, 2.5]},
        )
        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.set_axis_off()

        # ── Top panel: query text ──────────────────────────────────────
        axes[0].set_title(title, fontsize=16, fontweight="bold",
                          color="white", pad=8)
        # Show the first 8 lines of the SPARQL query.
        query_preview = "\n".join(sparql_text.splitlines()[:8])
        axes[0].text(
            0.02, 0.5,
            query_preview,
            transform=axes[0].transAxes,
            fontsize=8.5,
            fontfamily="monospace",
            color="#a8e6cf",
            va="center",
            wrap=True,
        )

        # ── Bottom panel: results table ────────────────────────────────
        if row_data and col_names:
            tbl = axes[1].table(
                cellText=row_data,
                colLabels=col_names,
                loc="center",
                cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.4)

            # Style header row.
            for j in range(len(col_names)):
                cell = tbl[0, j]
                cell.set_facecolor("#2d3436")
                cell.set_text_props(color="white", fontweight="bold")

            # Alternate row colours.
            for i in range(1, len(row_data) + 1):
                bg = "#16213e" if i % 2 == 0 else "#0f3460"
                for j in range(len(col_names)):
                    cell = tbl[i, j]
                    cell.set_facecolor(bg)
                    cell.set_text_props(color="#ecf0f1")
                    cell.set_edgecolor("#2d3436")

            suffix = (
                f" (showing {len(row_data)} of {len(results)} rows)"
                if len(results) > _MAX_DISPLAY_ROWS
                else f" ({len(results)} rows)"
            )
            axes[1].set_title(
                f"Results{suffix}",
                fontsize=11,
                color="#95a5a6",
                pad=6,
            )
        else:
            axes[1].text(
                0.5, 0.5,
                "No results  (run pipeline first or check query syntax)",
                ha="center", va="center",
                fontsize=14, color="#e74c3c",
                transform=axes[1].transAxes,
            )

        fig.tight_layout(pad=0.5)
        fig.savefig(
            str(out_path),
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

        logger.info("SparqlScreenshots: saved %s", out_path)
        return out_path


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #

def _format_cell(value) -> str:
    """Format a SPARQL result cell for display."""
    if value is None:
        return ""
    s = str(value)
    # Trim URIs to just the fragment / last path component.
    if "#" in s:
        s = s.split("#")[-1]
    elif "/" in s and s.startswith("http"):
        s = s.rstrip("/").split("/")[-1]
    return s[:40]   # cap width
