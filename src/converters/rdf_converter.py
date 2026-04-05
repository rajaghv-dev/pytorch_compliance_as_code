"""
RDF/Turtle knowledge graph converter for EntityRecords.

Writes the compliance evidence as an RDF graph in Turtle (.ttl) format
using a custom PyTorch Compliance Ontology namespace (pct:).

WHY RDF?
    RDF enables SPARQL queries over the compliance evidence.  A single query
    can answer questions like:

        "Which non-deterministic operators are used in training
         and do NOT survive torch.export?"

    That kind of cross-dimensional question would require multiple JOIN
    operations in SQL, but is natural in SPARQL.

NAMESPACES
    pct:  — PyTorch Compliance Toolkit ontology terms
    eu:   — EU AI Act article URIs
    gdpr: — GDPR article URIs

HOW TO QUERY
    After generating the .ttl file, you can query it with:
        - Python: rdflib + SPARQLWrapper
        - Command line: Apache Jena's arq utility
        - GUI: Protégé ontology editor

EXAMPLE SPARQL
    SELECT ?name ?article WHERE {
        ?entity pct:entityName ?name ;
                pct:hasComplianceTag ?article ;
                pct:lifecyclePhase "training_only" .
    }

OUTPUT
    storage/output/compliance_graph.ttl
"""

from __future__ import annotations

import logging
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.converters.rdf")


# ---------------------------------------------------------------------------
# Namespace declarations
# ---------------------------------------------------------------------------

# Base URI for our custom ontology terms (properties, classes).
PCT = Namespace("http://purl.org/pytorch-compliance/ont#")

# Base URI for EU AI Act article nodes.
EU = Namespace("http://purl.org/pytorch-compliance/regulation/eu-ai-act/")

# Base URI for GDPR article nodes.
GDPR = Namespace("http://purl.org/pytorch-compliance/regulation/gdpr/")


class RdfConverter:
    """Converts EntityRecords to an RDF/Turtle knowledge graph."""

    def convert(self, records: list[EntityRecord], output_path: Path) -> None:
        """
        Build an RDF graph from *records* and write it to *output_path* in
        Turtle format.

        Parameters
        ----------
        records : list[EntityRecord]
            The entity records to serialise.
        output_path : Path
            Destination .ttl file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "RDF: building graph for %d records → %s",
            len(records), output_path,
        )

        g = self._build_graph(records)

        try:
            g.serialize(destination=str(output_path), format="turtle")
            logger.info(
                "RDF: done — %d triples written to %s", len(g), output_path
            )
        except Exception as exc:
            logger.error("RDF: failed to write '%s': %s", output_path, exc)
            raise

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def _build_graph(self, records: list[EntityRecord]) -> Graph:
        """Build and return an rdflib Graph from *records*."""
        g = Graph()

        # Bind namespace prefixes so the .ttl file is human-readable.
        g.bind("pct", PCT)
        g.bind("eu", EU)
        g.bind("gdpr", GDPR)
        g.bind("rdfs", RDFS)

        # Add the ontology schema triples (class and property declarations).
        self._add_schema(g)

        # Add one subject per entity record.
        for rec in records:
            self._add_record(g, rec)

        return g

    def _add_schema(self, g: Graph) -> None:
        """
        Declare the classes and properties used in the graph.

        This makes the Turtle file self-describing and readable by tools
        like Protégé without a separate schema file.
        """
        # --- Classes ---
        g.add((PCT.Entity,        RDF.type,  RDFS.Class))
        g.add((PCT.Article,       RDF.type,  RDFS.Class))
        g.add((PCT.ExportSurvival, RDF.type, RDFS.Class))

        # --- Object properties ---
        g.add((PCT.hasComplianceTag,  RDF.type,    RDF.Property))
        g.add((PCT.hasExportSurvival, RDF.type,    RDF.Property))

        # --- Data properties ---
        for prop_name in (
            "entityName", "entityType", "subcategory", "language",
            "sourceFile", "modulePath", "qualifiedName",
            "startLine", "docstring", "lifecyclePhase",
            "executionLevel", "distributedSafety",
            "mappingConfidence", "extractionConfidence", "mappingRationale",
            "exportFormat", "survivalStatus",
        ):
            g.add((PCT[prop_name], RDF.type, RDF.Property))

    def _add_record(self, g: Graph, rec: EntityRecord) -> None:
        """
        Add all triples for a single EntityRecord to graph *g*.

        Each entity becomes a subject URI of the form:
            pct:entity/<record_id>
        """
        if not rec.id:
            # A record with no ID is invalid; skip it silently.
            return

        subject = PCT[f"entity/{rec.id}"]

        # Declare the type.
        g.add((subject, RDF.type, PCT.Entity))

        # ---- Identity fields ----
        _add_str(g, subject, PCT.entityName,    rec.entity_name)
        _add_str(g, subject, PCT.entityType,    rec.entity_type)
        _add_str(g, subject, PCT.subcategory,   rec.subcategory)
        _add_str(g, subject, PCT.language,      rec.language)
        _add_str(g, subject, PCT.sourceFile,    rec.source_file)
        _add_str(g, subject, PCT.modulePath,    rec.module_path)
        _add_str(g, subject, PCT.qualifiedName, rec.qualified_name)

        if rec.start_line:
            g.add((
                subject, PCT.startLine,
                Literal(rec.start_line, datatype=XSD.integer),
            ))

        # Docstring — truncated to 500 chars so the .ttl stays manageable.
        if rec.docstring:
            g.add((subject, PCT.docstring, Literal(rec.docstring[:500])))

        # ---- Compliance annotation fields ----
        _add_str(g, subject, PCT.lifecyclePhase,    rec.lifecycle_phase)
        _add_str(g, subject, PCT.executionLevel,    rec.execution_level)
        _add_str(g, subject, PCT.distributedSafety, rec.distributed_safety)
        _add_str(g, subject, PCT.mappingRationale,  rec.mapping_rationale)

        # Compliance tags — link to regulation article URIs.
        for tag in rec.compliance_tags:
            article_uri = _tag_to_uri(tag)
            if article_uri:
                g.add((subject, PCT.hasComplianceTag, article_uri))

        # ---- Confidence scores ----
        g.add((
            subject, PCT.mappingConfidence,
            Literal(round(rec.mapping_confidence, 4), datatype=XSD.decimal),
        ))
        g.add((
            subject, PCT.extractionConfidence,
            Literal(round(rec.extraction_confidence, 4), datatype=XSD.decimal),
        ))

        # ---- Export survival ----
        # Each export format becomes a separate blank-node triple.
        for fmt, status in (rec.export_survival or {}).items():
            export_node = PCT[f"export/{rec.id}/{fmt}"]
            g.add((subject,      PCT.hasExportSurvival, export_node))
            g.add((export_node,  RDF.type,              PCT.ExportSurvival))
            g.add((export_node,  PCT.exportFormat,      Literal(fmt)))
            g.add((export_node,  PCT.survivalStatus,    Literal(str(status))))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _add_str(
    g: Graph, subject: URIRef, predicate: URIRef, value: str
) -> None:
    """Add a string literal triple to *g* only if *value* is non-empty."""
    if value:
        g.add((subject, predicate, Literal(value)))


def _tag_to_uri(tag: str) -> URIRef | None:
    """
    Convert an internal compliance tag string to a regulation article URI.

    Returns None for tags that don't match the known patterns, so unknown
    tags are silently skipped rather than producing broken URIs.
    """
    if tag.startswith("eu_ai_act_art_"):
        article_num = tag.removeprefix("eu_ai_act_art_")
        return EU[f"Art{article_num}"]
    if tag.startswith("gdpr_art_"):
        article_num = tag.removeprefix("gdpr_art_")
        return GDPR[f"Art{article_num}"]
    return None
