"""
Compliance density heatmap — treemap / bar chart by PyTorch module.

WHAT IT PRODUCES
----------------
A 1920×1080 treemap (or horizontal bar chart if squarify is not installed)
showing compliance entity concentration by top-level PyTorch module.

  torch.nn       — hot  (many compliance-relevant entities)
  torch.utils    — cold (few compliance-relevant entities)

Saved to `storage/talk_assets/compliance_heatmap.png`.

COLOUR SCALE
------------
  Low density  → cool blue  (#1e3799)
  High density → hot red    (#e84118)

USAGE
-----
    from src.assets.heatmap import ComplianceHeatmapAsset
    asset = ComplianceHeatmapAsset()
    path = asset.render(records, output_dir="storage/talk_assets")
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.assets.heatmap")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

_WIDTH_PX  = 1920
_HEIGHT_PX = 1080
_DPI        = 100

# Colour gradient endpoints (low → high density).
_COLD_COLOR = "#1e3799"
_HOT_COLOR  = "#e84118"


# ----------------------------------------------------------------------- #
# ComplianceHeatmapAsset
# ----------------------------------------------------------------------- #

class ComplianceHeatmapAsset:
    """Renders the compliance density heatmap PNG."""

    def render(
        self,
        records: list["EntityRecord"],
        output_dir: str | Path = "storage/talk_assets",
    ) -> Path:
        """
        Build and save the compliance density heatmap.

        Parameters
        ----------
        records : list[EntityRecord]
            All annotated entity records.
        output_dir : str | Path
            Directory where the PNG is saved.

        Returns
        -------
        Path
            Full path of the written PNG file.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "compliance_heatmap.png"

        # Aggregate: count tagged entities per top-level module.
        module_stats = _aggregate_by_module(records)

        if not module_stats:
            logger.warning(
                "ComplianceHeatmapAsset: no module data — placeholder PNG"
            )
            _save_placeholder(out_path, _WIDTH_PX, _HEIGHT_PX, _DPI)
            return out_path

        # Sort by compliance density (tagged / total) descending.
        sorted_modules = sorted(
            module_stats.items(),
            key=lambda kv: kv[1]["density"],
            reverse=True,
        )

        # Try squarify for treemap; fall back to horizontal bar chart.
        try:
            import squarify
            _render_treemap(fig_path=out_path, sorted_modules=sorted_modules,
                            width_px=_WIDTH_PX, height_px=_HEIGHT_PX, dpi=_DPI)
        except ImportError:
            logger.debug(
                "squarify not installed — rendering bar chart instead. "
                "Install with: pip install squarify"
            )
            _render_barchart(fig_path=out_path, sorted_modules=sorted_modules,
                             width_px=_WIDTH_PX, height_px=_HEIGHT_PX, dpi=_DPI)

        logger.info(
            "ComplianceHeatmapAsset: saved %s (%d modules)",
            out_path,
            len(sorted_modules),
        )
        return out_path


# ----------------------------------------------------------------------- #
# Rendering helpers
# ----------------------------------------------------------------------- #

def _aggregate_by_module(
    records: list["EntityRecord"],
) -> dict[str, dict]:
    """
    Group records by top-level PyTorch module and compute density.

    Returns
    -------
    dict mapping module_name → {"total": int, "tagged": int, "density": float}
    """
    totals: Counter = Counter()
    tagged: Counter = Counter()

    for rec in records:
        module = _top_module(rec)
        totals[module] += 1
        if rec.compliance_tags:
            tagged[module] += 1

    result = {}
    for module, total in totals.items():
        t = tagged[module]
        result[module] = {
            "total":   total,
            "tagged":  t,
            "density": t / total if total > 0 else 0.0,
        }
    return result


def _top_module(rec: "EntityRecord") -> str:
    """Extract the top-level module name from a record's module_path or source_file."""
    # Try module_path first (e.g. "torch.nn.modules.module").
    mp = rec.module_path or ""
    if mp:
        top = mp.split(".")[0]
        if top:
            return top

    # Fall back to source_file (e.g. "torch/nn/modules/module.py").
    src = rec.source_file or ""
    parts = src.replace("\\", "/").split("/")
    if parts:
        return parts[0]

    return "unknown"


def _density_color(density: float) -> str:
    """Interpolate between cold and hot colour based on density in [0, 1]."""
    import matplotlib.colors as mcolors
    cold = mcolors.to_rgb(_COLD_COLOR)
    hot  = mcolors.to_rgb(_HOT_COLOR)
    d = max(0.0, min(1.0, density))
    rgb = tuple(c + (h - c) * d for c, h in zip(cold, hot))
    return mcolors.to_hex(rgb)


def _render_treemap(
    fig_path: Path,
    sorted_modules: list[tuple[str, dict]],
    width_px: int,
    height_px: int,
    dpi: int,
) -> None:
    """Render a squarify treemap PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import squarify

    labels = [m for m, _ in sorted_modules]
    sizes  = [max(s["total"], 1) for _, s in sorted_modules]
    colors = [_density_color(s["density"]) for _, s in sorted_modules]

    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi), dpi=dpi
    )
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    squarify.plot(
        sizes=sizes,
        label=[
            f"{lab}\n{s['tagged']}/{s['total']}\n({s['density']:.0%})"
            for lab, (_, s) in zip(labels, sorted_modules)
        ],
        color=colors,
        ax=ax,
        text_kwargs={"color": "white", "fontsize": 9, "fontweight": "bold"},
        pad=True,
    )

    ax.set_axis_off()
    ax.set_title(
        "PyTorch Compliance Entity Density by Module",
        fontsize=24, fontweight="bold", color="white", pad=20,
    )

    # Legend.
    import matplotlib.patches as mpatches
    low_patch  = mpatches.Patch(color=_COLD_COLOR, label="Low compliance density")
    high_patch = mpatches.Patch(color=_HOT_COLOR,  label="High compliance density")
    ax.legend(
        handles=[low_patch, high_patch],
        loc="lower right", fontsize=12,
        facecolor="#16213e", labelcolor="white", edgecolor="none",
    )

    fig.savefig(str(fig_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_barchart(
    fig_path: Path,
    sorted_modules: list[tuple[str, dict]],
    width_px: int,
    height_px: int,
    dpi: int,
) -> None:
    """Render a horizontal bar chart as fallback when squarify is absent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels     = [m for m, _ in sorted_modules[:25]]   # top 25
    densities  = [s["density"] for _, s in sorted_modules[:25]]
    totals_v   = [s["total"]   for _, s in sorted_modules[:25]]
    colors     = [_density_color(d) for d in densities]

    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi), dpi=dpi
    )
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(range(len(labels)), densities, color=colors, height=0.7)

    # Annotate bars with entity count.
    for i, (bar, total) in enumerate(zip(bars, totals_v)):
        ax.text(
            bar.get_width() + 0.005, i,
            f"{total} entities",
            va="center", fontsize=9, color="#bdc3c7",
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, color="white")
    ax.set_xlabel("Compliance Density (tagged / total)", fontsize=13, color="#bdc3c7")
    ax.set_title(
        "PyTorch Compliance Entity Density by Module",
        fontsize=22, fontweight="bold", color="white", pad=15,
    )
    ax.tick_params(axis="x", colors="#bdc3c7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#4a4a6a")
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    )

    fig.tight_layout()
    fig.savefig(str(fig_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_placeholder(path: Path, w: int, h: int, dpi: int) -> None:
    """Write a placeholder PNG when no data is available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_axis_off()
    ax.text(
        0.5, 0.5,
        "No compliance data yet.\nRun the pipeline first.",
        ha="center", va="center", fontsize=24,
        color="#e74c3c", transform=ax.transAxes,
    )
    fig.savefig(str(path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
