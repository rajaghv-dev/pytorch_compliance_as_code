"""
PyTorch Compliance Toolkit — extractors subpackage.

HOW THE REGISTRY WORKS
-----------------------
EXTRACTOR_REGISTRY maps a short string name (used in the YAML config and
CLI) to the extractor class that handles that concern.

The CLI (src/cli.py) reads config.extractors.enabled, looks each name up
in this registry, and instantiates the class with (repo_path, output_path).

HOW TO ADD A NEW EXTRACTOR
---------------------------
1. Create  src/extractors/my_extractor.py
   containing  class MyExtractor(BaseExtractor)  with an extract() method.

2. Add one line to EXTRACTOR_REGISTRY below:
       "my_name": MyExtractor,

3. Enable it in configs/default.yaml under:
       extractors:
         enabled:
           - my_name

That is the ONLY change required outside your new file.  Nothing else
in the codebase needs to be modified.

EXTRACTOR CONSTRUCTOR SIGNATURE
---------------------------------
All concrete extractors use:
    def __init__(self, repo_path: Path, output_path: Path) -> None:
        super().__init__("extractor_name", repo_path, output_path)
"""

from .api_documentation import APIDocumentationExtractor
from .commit_history import CommitHistoryExtractor
from .compliance_tools import ComplianceToolsExtractor
from .data_provenance import DataProvenanceExtractor
from .export_boundary import ExportBoundaryExtractor
from .hookability import HookabilityExtractor
from .module_hierarchy import ModuleHierarchyExtractor
from .operator_determinism import OperatorDeterminismExtractor
from .sphinx_notes import SphinxNotesExtractor
from .supply_chain import SupplyChainExtractor
from .test_suite import TestSuiteExtractor

# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------
# Maps config / CLI name → extractor class.
# The CLI uses this dict to instantiate extractors without any if/elif chain.
# To add a new extractor: add one line here (see module docstring above).
EXTRACTOR_REGISTRY: dict[str, type] = {
    "hookability":          HookabilityExtractor,
    "operator_determinism": OperatorDeterminismExtractor,
    "export_boundary":      ExportBoundaryExtractor,
    "data_provenance":      DataProvenanceExtractor,
    "module_hierarchy":     ModuleHierarchyExtractor,
    "supply_chain":         SupplyChainExtractor,
    "api_documentation":    APIDocumentationExtractor,
    "sphinx_notes":         SphinxNotesExtractor,
    "test_suite":           TestSuiteExtractor,
    "compliance_tools":     ComplianceToolsExtractor,
    "commit_history":       CommitHistoryExtractor,
}

__all__ = ["EXTRACTOR_REGISTRY"] + list(EXTRACTOR_REGISTRY.keys())
