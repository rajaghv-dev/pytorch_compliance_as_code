"""
Cross-reference graph builder for EntityRecords.

Builds a graph of relationships between entities from three sources:
1. AST-derived relations already present on each entity
2. Name-based references (entity A's raw_text mentions entity B's qualified_name)
3. Compliance co-occurrence (entities sharing the same compliance tag)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..extractors.base import EntityRecord

logger = logging.getLogger("pct.organizer.cross_references")


@dataclass
class CrossRefGraph:
    """Cross-reference graph over EntityRecords."""

    edges: list[dict] = field(default_factory=list)
    by_source: dict = field(default_factory=dict)       # source_id -> list of edges
    by_target: dict = field(default_factory=dict)       # target_id -> list of edges
    co_implements: dict = field(default_factory=dict)   # article -> list of entity_ids

    def to_dict(self) -> dict:
        """Serialise the graph to a JSON-safe dictionary."""
        return {
            "edges": self.edges,
            "by_source": self.by_source,
            "by_target": self.by_target,
            "co_implements": self.co_implements,
        }


class CrossReferenceBuilder:
    """Builds a cross-reference graph from entity relations."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("pct.organizer.cross_references")

    def build(self, records: list[EntityRecord]) -> CrossRefGraph:
        """
        Build the full cross-reference graph from the given records.

        Parameters
        ----------
        records : list[EntityRecord]
            Deduplicated entity records.

        Returns
        -------
        CrossRefGraph
            The completed graph with all edge types.
        """
        graph = CrossRefGraph()

        # Build lookup structures for efficient name matching
        id_by_qualified_name: dict[str, str] = {}
        id_by_entity_name: dict[str, str] = {}
        records_by_id: dict[str, EntityRecord] = {}

        for rec in records:
            records_by_id[rec.id] = rec
            if rec.qualified_name:
                id_by_qualified_name[rec.qualified_name] = rec.id
            if rec.entity_name:
                id_by_entity_name[rec.entity_name] = rec.id

        self.logger.info(
            "Building cross-references for %d records (%d qualified names)",
            len(records), len(id_by_qualified_name),
        )

        # 1. AST-derived edges from existing relations
        ast_count = self._add_ast_edges(records, graph)
        self.logger.info("Added %d AST-derived edges", ast_count)

        # 2. Name-based references (raw_text contains qualified_name)
        name_count = self._add_name_based_edges(records, id_by_qualified_name, graph)
        self.logger.info("Added %d name-based reference edges", name_count)

        # 3. Compliance co-occurrence
        self._add_compliance_co_occurrence(records, graph)
        self.logger.info(
            "Compliance co-occurrence: %d articles", len(graph.co_implements)
        )

        # Build the by_source and by_target indexes
        self._build_indexes(graph)

        self.logger.info("Cross-reference graph complete: %d total edges", len(graph.edges))
        return graph

    def _add_ast_edges(
        self, records: list[EntityRecord], graph: CrossRefGraph
    ) -> int:
        """Extract edges from entity.relations (calls, inherits, imports)."""
        count = 0
        for rec in records:
            for rel in rec.relations:
                if not isinstance(rel, dict):
                    continue
                edge = {
                    "type": rel.get("type", "unknown"),
                    "source_id": rec.id,
                    "target_id": rel.get("target", ""),
                    "source": "ast_derived",
                }
                graph.edges.append(edge)
                count += 1
        return count

    def _add_name_based_edges(
        self,
        records: list[EntityRecord],
        qname_lookup: dict[str, str],
        graph: CrossRefGraph,
    ) -> int:
        """
        Add edges when entity A's raw_text contains entity B's qualified_name.

        Uses a token inverted-index to avoid O(N×M) brute-force scanning.
        Only indexes qualified names from compliance-tagged records (the
        semantically meaningful targets). For 227K records this reduces
        inner-loop work from ~42B to ~2M operations.
        """
        import re as _re
        from collections import defaultdict

        # Step 1: build candidate qname set — only compliance-tagged targets
        # (or keep all if the set is small enough)
        compliance_qnames: dict[str, str] = {
            qn: eid for qn, eid in qname_lookup.items()
            if len(qn) >= 8  # skip trivially short names
        }

        # If still too large, further restrict to records with compliance tags
        if len(compliance_qnames) > 20000:
            tagged_ids = {
                rec.id for rec in records if rec.compliance_tags
            }
            compliance_qnames = {
                qn: eid for qn, eid in compliance_qnames.items()
                if eid in tagged_ids
            }

        if not compliance_qnames:
            self.logger.info("  Name-based matching: no compliance-tagged targets, skipping")
            return 0

        self.logger.info(
            "  Name-based matching: %d candidate qnames (compliance-tagged)",
            len(compliance_qnames),
        )

        # Step 2: build inverted token index
        # token (≥4 chars from splitting on . and _) → set of qnames containing it
        token_index: dict[str, set[str]] = defaultdict(set)
        for qn in compliance_qnames:
            for token in _re.split(r'[._]', qn):
                if len(token) >= 4:
                    token_index[token].add(qn)

        # Step 3: for each source record, look up candidates via token index
        count = 0
        _findall = _re.findall
        _token_pat = r'[a-zA-Z_]\w{3,}'

        for i, rec in enumerate(records):
            if not rec.raw_text:
                continue

            if (i + 1) % 10000 == 0:
                self.logger.info(
                    "  Name-based matching progress: %d/%d records",
                    i + 1, len(records),
                )

            # Gather candidate qnames via shared tokens
            candidates: set[str] = set()
            for token in _findall(_token_pat, rec.raw_text):
                bucket = token_index.get(token)
                if bucket:
                    candidates.update(bucket)

            # Verify full qname match only on candidates (not all 187K)
            for qname in candidates:
                target_id = compliance_qnames.get(qname)
                if target_id and target_id != rec.id and qname in rec.raw_text:
                    graph.edges.append({
                        "type": "references",
                        "source_id": rec.id,
                        "target_id": target_id,
                        "source": "name_match",
                    })
                    count += 1

        return count

    def _add_compliance_co_occurrence(
        self, records: list[EntityRecord], graph: CrossRefGraph
    ) -> None:
        """Group entities by shared compliance tags."""
        article_to_ids: dict[str, list[str]] = {}
        for rec in records:
            for tag in rec.compliance_tags:
                article_to_ids.setdefault(tag, []).append(rec.id)

        graph.co_implements = article_to_ids

        # Also add co_implements edges between entities sharing a tag
        for article, entity_ids in article_to_ids.items():
            if len(entity_ids) > 1:
                edge = {
                    "type": "co_implements",
                    "article": article,
                    "entity_ids": entity_ids,
                    "source": "compliance_tag",
                }
                graph.edges.append(edge)

    def _build_indexes(self, graph: CrossRefGraph) -> None:
        """Build by_source and by_target lookup dictionaries."""
        for edge in graph.edges:
            source_id = edge.get("source_id", "")
            target_id = edge.get("target_id", "")

            if source_id:
                graph.by_source.setdefault(source_id, []).append(edge)
            if target_id:
                graph.by_target.setdefault(target_id, []).append(edge)

    def write_results(self, graph: CrossRefGraph, output_path: Path) -> None:
        """
        Write the cross-reference graph to a JSON file atomically.

        Before writing:
        - Filters out edges where source_id or target_id contains Jinja-style
          template placeholders (``{{`` or ``}}``), which are noise from RST
          doc directives parsed as cross-reference targets.
        - Builds a ``nodes`` dict from all entity IDs that appear in edges so
          that SPARQL/graph analysis tools have a usable node inventory.

        Parameters
        ----------
        graph : CrossRefGraph
            The graph to persist.
        output_path : Path
            Destination JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fix #39: filter edges with unfilled template placeholders
        original_count = len(graph.edges)
        clean_edges = [
            e for e in graph.edges
            if "{{" not in e.get("source_id", "")
            and "}}" not in e.get("source_id", "")
            and "{{" not in e.get("target_id", "")
            and "}}" not in e.get("target_id", "")
        ]
        filtered_count = original_count - len(clean_edges)
        if filtered_count:
            self.logger.info(
                "Filtered %d edges with template placeholders ({{ }})",
                filtered_count,
            )
        graph.edges = clean_edges

        # Fix #41: build nodes dict from IDs that appear in edges
        node_ids: set[str] = set()
        for edge in graph.edges:
            src = edge.get("source_id", "")
            tgt = edge.get("target_id", "")
            if src:
                node_ids.add(src)
            if tgt:
                node_ids.add(tgt)
        nodes: dict[str, dict] = {nid: {"id": nid} for nid in node_ids}
        self.logger.info(
            "Cross-reference nodes dict: %d unique node IDs from edges",
            len(nodes),
        )

        # Rebuild indexes after filtering
        graph.by_source = {}
        graph.by_target = {}
        self._build_indexes(graph)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        )
        try:
            data = graph.to_dict()
            data["nodes"] = nodes
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(output_path))
            self.logger.info("Wrote cross-reference graph to %s", output_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
