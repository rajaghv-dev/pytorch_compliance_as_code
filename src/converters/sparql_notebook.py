"""
SPARQL notebook generator for the compliance ontology.

Generates `notebooks/compliance_queries.ipynb` containing pre-built SPARQL
query cells that analysts can run interactively against the Turtle (.ttl)
knowledge graph produced by the RDF converter.

NOTEBOOK STRUCTURE
------------------
  Cell 0: Markdown — introduction and setup instructions
  Cell 1: Python — load ontology from .ttl file using rdflib
  Cell 2: Python — Query 1: entity count per EU AI Act article
  Cell 3: Python — Query 2: hooks that survive torch.compile
  Cell 4: Python — Query 3: non-deterministic operators census
  Cell 5: Python — Query 4: Art.14 (Human Oversight) coverage
  Cell 6: Python — Query 5: high-confidence entities (>= 0.8) by article

USAGE
-----
    from src.converters.sparql_notebook import SparqlNotebookConverter

    converter = SparqlNotebookConverter()
    converter.convert(output_path="notebooks/compliance_queries.ipynb",
                      ttl_path="storage/rdf/compliance.ttl")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("pct.converters.sparql_notebook")

# ----------------------------------------------------------------------- #
# Notebook cell content
# ----------------------------------------------------------------------- #

# Each entry is (cell_type, source_lines).
# cell_type is "markdown" or "code".

_INTRO_MARKDOWN = """\
# PyTorch Compliance-as-Code — SPARQL Query Notebook

This notebook queries the RDF/OWL knowledge graph produced by the
**PyTorch Compliance Toolkit** pipeline.  The graph maps PyTorch source
entities (hooks, operators, modules, tests) to EU AI Act and GDPR articles.

## Prerequisites

```bash
pip install rdflib
```

Run the pipeline first to produce the `.ttl` file:

```bash
pct --repo /path/to/pytorch --phase catalog,extract,annotate,organize,convert
```

Then set `TTL_PATH` in Cell 1 to point at `storage/rdf/compliance.ttl`.
"""

_CELL_LOAD_ONTOLOGY = '''\
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, OWL, XSD

# ── Configuration ──────────────────────────────────────────────────────────
# Update this path to wherever your pipeline wrote the Turtle file.
TTL_PATH = "../storage/rdf/compliance.ttl"

# ── Load the graph ─────────────────────────────────────────────────────────
g = Graph()
g.parse(TTL_PATH, format="turtle")

# Namespace shortcuts for queries below.
PCT  = Namespace("https://pytorch-compliance.org/ontology#")
PCTI = Namespace("https://pytorch-compliance.org/instance#")
EU   = Namespace("https://eu-ai-act.org/ontology#")
GDPR = Namespace("https://gdpr.org/ontology#")

print(f"Loaded {len(g):,} triples from {TTL_PATH}")
'''

_CELL_QUERY1 = '''\
# ── Query 1: Entity count per EU AI Act article ────────────────────────────
sparql_q1 = """
PREFIX ptc:  <https://pytorch-compliance.org/ontology#>
PREFIX eu:   <https://eu-ai-act.org/ontology#>

SELECT ?article (COUNT(DISTINCT ?entity) AS ?count)
WHERE {
    ?entity ptc:implements ?article .
    FILTER(STRSTARTS(STR(?article), STR(eu:)))
}
GROUP BY ?article
ORDER BY DESC(?count)
"""

print("Article                        | Entity Count")
print("-" * 45)
for row in g.query(sparql_q1):
    article_label = str(row.article).split("#")[-1]
    print(f"{article_label:<30} | {int(row['count'])}")
'''

_CELL_QUERY2 = '''\
# ── Query 2: Hooks that survive torch.compile ─────────────────────────────
sparql_q2 = """
PREFIX ptc: <https://pytorch-compliance.org/ontology#>

SELECT ?entity ?name ?sourceFile
WHERE {
    ?entity a ptc:HookPoint ;
            ptc:entityName ?name ;
            ptc:sourceFile ?sourceFile ;
            ptc:survives   ptc:TorchCompile .
}
ORDER BY ?name
"""

print("Hook Name                          | Source File")
print("-" * 70)
results = list(g.query(sparql_q2))
print(f"Found {len(results)} hooks that survive torch.compile\\n")
for row in results[:20]:   # show first 20
    name = str(row["name"])
    src  = str(row["sourceFile"]).split("/")[-1]
    print(f"{name:<35} | {src}")
'''

_CELL_QUERY3 = '''\
# ── Query 3: Non-deterministic operators census ───────────────────────────
sparql_q3 = """
PREFIX ptc: <https://pytorch-compliance.org/ontology#>

SELECT ?entity ?name ?determinism ?confidence
WHERE {
    ?entity ptc:entityName    ?name ;
            ptc:hasDeterminism ?determinism .
    OPTIONAL { ?entity ptc:extractionConfidence ?confidence }
}
ORDER BY ?name
"""

print("Operator Name                   | Determinism        | Confidence")
print("-" * 65)
results = list(g.query(sparql_q3))
print(f"Found {len(results)} operators with determinism annotations\\n")
for row in results[:30]:
    name = str(row["name"])
    det  = str(row["determinism"]).split("#")[-1]
    conf = f"{float(row['confidence']):.2f}" if row.get("confidence") else "N/A"
    print(f"{name:<30} | {det:<18} | {conf}")
'''

_CELL_QUERY4 = '''\
# ── Query 4: Art.14 (Human Oversight) coverage ────────────────────────────
# Art.14 requires that high-risk AI systems allow human intervention.
# We expect low coverage here — this is a gap report.
sparql_q4 = """
PREFIX ptc: <https://pytorch-compliance.org/ontology#>
PREFIX eu:  <https://eu-ai-act.org/ontology#>

SELECT ?entity ?name ?mappingConf ?rationale
WHERE {
    ?entity ptc:implements         eu:Art14 ;
            ptc:entityName         ?name ;
            ptc:mappingConfidence  ?mappingConf .
    OPTIONAL { ?entity ptc:mappingRationale ?rationale }
}
ORDER BY DESC(?mappingConf)
"""

results = list(g.query(sparql_q4))
print(f"Art.14 (Human Oversight) coverage: {len(results)} entities\\n")

if not results:
    print("⚠  GAP DETECTED: No entities mapped to Art.14.")
    print("   This likely means PyTorch has no explicit human-oversight hooks.")
    print("   Consider adding a custom ComplianceGate layer for deployment.")
else:
    print("Entity Name                     | Confidence | Rationale")
    print("-" * 75)
    for row in results[:20]:
        name = str(row["name"])
        conf = f"{float(row['mappingConf']):.2f}"
        rat  = str(row.get("rationale", ""))[:40]
        print(f"{name:<30} | {conf:<10} | {rat}")
'''

_CELL_QUERY5 = '''\
# ── Query 5: High-confidence entities (>= 0.8) grouped by article ─────────
sparql_q5 = """
PREFIX ptc: <https://pytorch-compliance.org/ontology#>
PREFIX eu:  <https://eu-ai-act.org/ontology#>

SELECT ?article (COUNT(DISTINCT ?entity) AS ?highConfCount)
WHERE {
    ?entity ptc:implements        ?article ;
            ptc:mappingConfidence ?conf .
    FILTER(?conf >= 0.8)
    FILTER(STRSTARTS(STR(?article), STR(eu:)))
}
GROUP BY ?article
ORDER BY DESC(?highConfCount)
"""

print("Article          | High-Confidence Entities (>= 0.8)")
print("-" * 50)
total = 0
for row in g.query(sparql_q5):
    article_label = str(row.article).split("#")[-1]
    count = int(row["highConfCount"])
    total += count
    bar = "█" * min(count // 2, 40)
    print(f"{article_label:<16} | {count:>4}  {bar}")
print(f"\\nTotal high-confidence mappings: {total}")
'''


# ----------------------------------------------------------------------- #
# SparqlNotebookConverter
# ----------------------------------------------------------------------- #

class SparqlNotebookConverter:
    """
    Generates a Jupyter notebook with pre-built SPARQL compliance queries.

    Uses nbformat to write a valid .ipynb file that can be opened in
    JupyterLab or VS Code.

    Parameters
    ----------
    kernel_name : str
        Jupyter kernel to declare in the notebook metadata.
    """

    def __init__(self, kernel_name: str = "python3") -> None:
        self.kernel_name = kernel_name

    def convert(
        self,
        output_path: str | Path = "notebooks/compliance_queries.ipynb",
        ttl_path: str = "../storage/rdf/compliance.ttl",
    ) -> Path:
        """
        Write the SPARQL query notebook to output_path.

        Parameters
        ----------
        output_path : str | Path
            Destination .ipynb file path.
        ttl_path : str
            Path to the Turtle file (used in the load cell's TTL_PATH comment).

        Returns
        -------
        Path
            The path of the written notebook file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Inject the ttl_path into the load cell.
        load_cell_src = _CELL_LOAD_ONTOLOGY.replace(
            '"../storage/rdf/compliance.ttl"',
            f'"{ttl_path}"',
        )

        nb = self._build_notebook([
            ("markdown", _INTRO_MARKDOWN),
            ("code",     load_cell_src),
            ("code",     _CELL_QUERY1),
            ("code",     _CELL_QUERY2),
            ("code",     _CELL_QUERY3),
            ("code",     _CELL_QUERY4),
            ("code",     _CELL_QUERY5),
        ])

        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        logger.info(
            "SPARQL notebook written to %s (%d cells)",
            out,
            len(nb["cells"]),
        )
        return out

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_notebook(
        self, cells: list[tuple[str, str]]
    ) -> dict:
        """
        Build a nbformat v4-compatible notebook dict.

        Parameters
        ----------
        cells : list of (cell_type, source) tuples
            "markdown" or "code" cells.
        """
        nb_cells = []
        for cell_type, source in cells:
            if cell_type == "markdown":
                nb_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": source,
                })
            else:
                nb_cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                })

        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": self.kernel_name,
                },
                "language_info": {
                    "name": "python",
                    "version": "3.12.0",
                },
            },
            "cells": nb_cells,
        }
