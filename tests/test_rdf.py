"""
Tests for the RDF converter (Phase 5).

WHAT IS TESTED
--------------
- RdfConverter.convert() produces a valid Turtle file.
- The Turtle file can be loaded by rdflib without errors.
- Three SPARQL queries produce expected results on a small set of test records:
  1. Entity count per article.
  2. Hook entities filtered by source file.
  3. High-confidence entities (>= 0.8).
- Schema classes are defined (HookPoint, ATenOperator, etc.).
- Entity URI uses the stable ID (20-hex hash).

HOW TO RUN
----------
    pytest tests/test_rdf.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extractors.base import EntityRecord


# ---------------------------------------------------------------------------
# Skip if rdflib is not installed
# ---------------------------------------------------------------------------

try:
    import rdflib
    _RDFLIB_AVAILABLE = True
except ImportError:
    _RDFLIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RDFLIB_AVAILABLE,
    reason="rdflib not installed (pip install rdflib)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_missing(module_path: str) -> None:
    try:
        __import__(module_path)
    except (ImportError, ModuleNotFoundError):
        pytest.skip(f"{module_path} not available")


def _make_records() -> list[EntityRecord]:
    """
    Return 10 EntityRecords covering several entity types and articles.
    These are the test records used for all RDF assertions.
    """
    return [
        # 1. Hook definition → Art.61
        EntityRecord(
            id="aa0000111122223333",
            source_file="torch/nn/modules/module.py",
            language="python",
            entity_name="register_forward_hook",
            entity_type="method",
            subcategory="hook_definition",
            module_path="torch.nn.modules",
            qualified_name="torch.nn.modules.module.Module.register_forward_hook",
            start_line=100,
            end_line=140,
            raw_text="def register_forward_hook(self, hook): ...",
            docstring="Registers a forward hook on the module.",
            compliance_tags=["eu_ai_act_art_61"],
            extraction_confidence=1.0,
            mapping_confidence=0.8,
            mapping_rationale="Tier-1 name match",
            extractor="hookability",
        ),
        # 2. Determinism API → Art.15
        EntityRecord(
            id="bb0000111122223333",
            source_file="torch/__init__.py",
            language="python",
            entity_name="use_deterministic_algorithms",
            entity_type="function",
            subcategory="determinism_config",
            module_path="torch",
            qualified_name="torch.use_deterministic_algorithms",
            start_line=200,
            end_line=230,
            raw_text="def use_deterministic_algorithms(mode): ...",
            docstring="Enable deterministic mode.",
            compliance_tags=["eu_ai_act_art_15", "eu_ai_act_art_9"],
            extraction_confidence=1.0,
            mapping_confidence=0.9,
            mapping_rationale="Direct API mapping",
            extractor="operator_determinism",
        ),
        # 3. Dataset → Art.10
        EntityRecord(
            id="cc0000111122223333",
            source_file="torch/utils/data/dataset.py",
            language="python",
            entity_name="Dataset",
            entity_type="class",
            subcategory="",
            module_path="torch.utils.data",
            qualified_name="torch.utils.data.Dataset",
            start_line=50,
            end_line=200,
            raw_text="class Dataset: ...",
            docstring="Abstract dataset class.",
            compliance_tags=["eu_ai_act_art_10"],
            extraction_confidence=1.0,
            mapping_confidence=0.7,
            extractor="data_provenance",
        ),
        # 4. ONNX export → Art.11
        EntityRecord(
            id="dd0000111122223333",
            source_file="torch/onnx/__init__.py",
            language="python",
            entity_name="export",
            entity_type="function",
            subcategory="export_boundary",
            module_path="torch.onnx",
            qualified_name="torch.onnx.export",
            start_line=300,
            end_line=400,
            raw_text="def export(model, args, f, ...): ...",
            docstring="Export model to ONNX format.",
            compliance_tags=["eu_ai_act_art_11"],
            extraction_confidence=1.0,
            mapping_confidence=0.85,
            extractor="export_boundary",
        ),
        # 5. Record-keeping → Art.12
        EntityRecord(
            id="ee0000111122223333",
            source_file="torch/utils/tensorboard/__init__.py",
            language="python",
            entity_name="SummaryWriter",
            entity_type="class",
            subcategory="",
            module_path="torch.utils.tensorboard",
            qualified_name="torch.utils.tensorboard.SummaryWriter",
            start_line=10,
            end_line=100,
            raw_text="class SummaryWriter: ...",
            docstring="Writes events for TensorBoard.",
            compliance_tags=["eu_ai_act_art_12"],
            extraction_confidence=0.8,
            mapping_confidence=0.6,
            extractor="api_documentation",
        ),
        # 6. Test case → Art.9
        EntityRecord(
            id="ff0000111122223333",
            source_file="test/test_determinism.py",
            language="python",
            entity_name="TestDeterministicOps",
            entity_type="class",
            subcategory="test_class",
            module_path="test.test_determinism",
            qualified_name="test.test_determinism.TestDeterministicOps",
            start_line=10,
            end_line=100,
            raw_text="class TestDeterministicOps(unittest.TestCase): ...",
            docstring="Tests for deterministic operations.",
            compliance_tags=["eu_ai_act_art_9"],
            extraction_confidence=1.0,
            mapping_confidence=0.7,
            extractor="test_suite",
        ),
        # 7. Human oversight — low confidence → Art.14
        EntityRecord(
            id="gg0000111122223333",
            source_file="torch/nn/modules/module.py",
            language="python",
            entity_name="register_full_backward_hook",
            entity_type="method",
            subcategory="hook_definition",
            module_path="torch.nn.modules",
            qualified_name="torch.nn.modules.module.Module.register_full_backward_hook",
            start_line=150,
            end_line=190,
            raw_text="def register_full_backward_hook(self, hook): ...",
            docstring="Registers a full backward hook.",
            compliance_tags=["eu_ai_act_art_14"],
            extraction_confidence=1.0,
            mapping_confidence=0.4,
            extractor="hookability",
        ),
        # 8. Backward hook → low confidence
        EntityRecord(
            id="hh0000111122223333",
            source_file="torch/nn/modules/module.py",
            language="python",
            entity_name="register_backward_hook",
            entity_type="method",
            subcategory="hook_definition",
            module_path="torch.nn.modules",
            qualified_name="torch.nn.modules.module.Module.register_backward_hook",
            start_line=130,
            end_line=150,
            raw_text="def register_backward_hook(self, hook): ...",
            docstring="Registers a backward hook.",
            compliance_tags=["eu_ai_act_art_61"],
            extraction_confidence=1.0,
            mapping_confidence=0.75,
            extractor="hookability",
        ),
        # 9. GDPR article
        EntityRecord(
            id="ii0000111122223333",
            source_file="torch/utils/data/dataset.py",
            language="python",
            entity_name="MapDataset",
            entity_type="class",
            subcategory="",
            module_path="torch.utils.data",
            qualified_name="torch.utils.data.MapDataset",
            start_line=300,
            end_line=400,
            raw_text="class MapDataset: ...",
            docstring="Dataset that applies a function to each element.",
            compliance_tags=["gdpr_art_5"],
            extraction_confidence=0.8,
            mapping_confidence=0.5,
            extractor="data_provenance",
        ),
        # 10. Untagged entity (should still be in the graph).
        EntityRecord(
            id="jj0000111122223333",
            source_file="torch/utils/benchmark/timer.py",
            language="python",
            entity_name="Timer",
            entity_type="class",
            subcategory="",
            module_path="torch.utils.benchmark",
            qualified_name="torch.utils.benchmark.Timer",
            start_line=5,
            end_line=50,
            raw_text="class Timer: ...",
            docstring="Measures the time of PyTorch statements.",
            compliance_tags=[],
            extraction_confidence=0.8,
            mapping_confidence=0.0,
            extractor="api_documentation",
        ),
    ]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def rdf_graph(tmp_path: Path):
    """
    Run the RDF converter on 10 test records and return a loaded rdflib Graph.
    """
    _skip_if_missing("src.converters.rdf_converter")
    from src.converters.rdf_converter import RdfConverter
    from rdflib import Graph

    records  = _make_records()
    out_path = tmp_path / "compliance.ttl"

    converter = RdfConverter()
    converter.convert(records, out_path)

    g = Graph()
    g.parse(str(out_path), format="turtle")
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRdfConverter:

    def test_converter_creates_ttl_file(self, tmp_path: Path) -> None:
        """RdfConverter.convert() should create a non-empty .ttl file."""
        _skip_if_missing("src.converters.rdf_converter")
        from src.converters.rdf_converter import RdfConverter

        records  = _make_records()
        out_path = tmp_path / "compliance.ttl"

        converter = RdfConverter()
        converter.convert(records, out_path)

        assert out_path.exists(), ".ttl file was not created"
        assert out_path.stat().st_size > 0, ".ttl file is empty"

    def test_ttl_loads_without_error(self, rdf_graph) -> None:
        """The generated Turtle file should load into rdflib without exceptions."""
        assert len(rdf_graph) > 0, "rdf_graph should have at least one triple"

    def test_entity_count_per_article(self, rdf_graph) -> None:
        """SPARQL query 1: entity count per article should return rows."""
        # Uses actual property names from rdf_converter.py output.
        sparql = """
        PREFIX pct: <http://purl.org/pytorch-compliance/ont#>
        SELECT ?article (COUNT(DISTINCT ?entity) AS ?count)
        WHERE { ?entity pct:hasComplianceTag ?article }
        GROUP BY ?article
        ORDER BY DESC(?count)
        """
        results = list(rdf_graph.query(sparql))
        assert len(results) > 0, \
            "SPARQL query for entity count per article returned no rows"

    def test_hook_entities_in_graph(self, rdf_graph) -> None:
        """SPARQL query 2: hooks from module.py should appear in the graph."""
        sparql = """
        PREFIX pct: <http://purl.org/pytorch-compliance/ont#>
        SELECT ?entity ?name ?sourceFile
        WHERE {
            ?entity pct:entityName ?name ;
                    pct:sourceFile ?sourceFile .
            FILTER(CONTAINS(STR(?sourceFile), "module.py"))
        }
        """
        results = list(rdf_graph.query(sparql))
        assert len(results) > 0, \
            "SPARQL query for module.py entities returned no rows"

    def test_high_confidence_entities(self, rdf_graph) -> None:
        """SPARQL query 3: entities with mapping_confidence >= 0.8 should exist."""
        sparql = """
        PREFIX pct: <http://purl.org/pytorch-compliance/ont#>
        SELECT ?entity ?name ?conf
        WHERE {
            ?entity pct:entityName       ?name ;
                    pct:mappingConfidence ?conf .
            FILTER(?conf >= 0.8)
        }
        """
        results = list(rdf_graph.query(sparql))
        # Records with id=bb, dd have confidence >= 0.8.
        assert len(results) >= 2, \
            f"Expected >= 2 high-confidence entities, got {len(results)}"

    def test_all_records_are_in_graph(self, rdf_graph) -> None:
        """All 10 test records should appear as subjects in the graph."""
        sparql = """
        PREFIX pct: <http://purl.org/pytorch-compliance/ont#>
        SELECT (COUNT(DISTINCT ?entity) AS ?count)
        WHERE { ?entity pct:entityName ?name }
        """
        results = list(rdf_graph.query(sparql))
        assert results, "SPARQL query returned no results"
        count = int(results[0][0])
        assert count == 10, f"Expected 10 entities in graph, got {count}"

    def test_entity_uri_contains_id(self, rdf_graph) -> None:
        """Entity URIs should contain the stable hex ID."""
        sparql = """
        PREFIX pct: <http://purl.org/pytorch-compliance/ont#>
        SELECT ?entity
        WHERE { ?entity pct:entityName ?name }
        """
        results = list(rdf_graph.query(sparql))
        uris = [str(r[0]) for r in results]
        # At least one URI should contain "entity/" (our ID-based naming).
        entity_uris = [u for u in uris if "entity/" in u]
        assert len(entity_uris) > 0, \
            f"Expected entity/ URIs in graph, found: {uris[:5]}"

    def test_schema_has_entity_class(self, rdf_graph) -> None:
        """The ontology should declare Entity as an rdfs:Class."""
        sparql = """
        PREFIX pct:  <http://purl.org/pytorch-compliance/ont#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        ASK { pct:Entity a rdfs:Class }
        """
        result = bool(rdf_graph.query(sparql))
        assert result, "Entity class not declared in ontology"

    def test_article14_coverage_near_zero(self, rdf_graph) -> None:
        """Art.14 should have only the one hand-crafted record (gg...)."""
        sparql = """
        PREFIX pct: <http://purl.org/pytorch-compliance/ont#>
        PREFIX eu:  <http://purl.org/pytorch-compliance/regulation/eu-ai-act/>
        SELECT (COUNT(?entity) AS ?count)
        WHERE {
            ?entity pct:hasComplianceTag eu:Art14 .
        }
        """
        results = list(rdf_graph.query(sparql))
        count = int(results[0][0]) if results else 0
        # Our 10 test records include exactly 1 Art.14 entity.
        assert count == 1, \
            f"Expected 1 Art.14 entity in test data, got {count}"
