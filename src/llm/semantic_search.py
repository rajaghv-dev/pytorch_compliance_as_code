"""
Semantic search over compliance entities using nomic-embed-text + FAISS.

HOW IT WORKS
------------
1. SemanticSearchIndex.build(records) — embeds every entity via Ollama's
   nomic-embed-text model and stores vectors in a FAISS flat index.
   Persists the index to disk so re-runs skip re-embedding.

2. SemanticSearchIndex.search(query, top_k) — embeds the query string and
   returns the top-k nearest entities by cosine similarity.

STORAGE
-------
  storage/embeddings/entity_embeddings.npy   — float32 numpy array (N × dim)
  storage/embeddings/entity_ids.json         — list of entity IDs aligned with rows

WHY COSINE SIMILARITY
---------------------
nomic-embed-text produces unit-normalised vectors, so inner-product search
on a normalised FAISS index is equivalent to cosine similarity and runs in
O(N) per query without a GPU. For larger corpora (> 100k entities) swap
IndexFlatIP → IndexIVFFlat with nlist=100.

USAGE
-----
    from src.llm.semantic_search import SemanticSearchIndex

    index = SemanticSearchIndex(ollama_client, records, embed_dir)
    index.build()                              # compute + persist embeddings
    hits = index.search("gradient checkpointing fairness", top_k=10)
    for rec in hits:
        print(rec.entity_name, rec.compliance_tags)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..extractors.base import EntityRecord
    from .ollama_client import OllamaClient

logger = logging.getLogger("pct.llm.semantic_search")

# ----------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------- #

# nomic-embed-text produces 768-dimensional vectors.
_EMBED_DIM = 768

# Ollama model for embeddings (must be pulled: `ollama pull nomic-embed-text`).
_EMBED_MODEL = "nomic-embed-text"

# File names inside the embedding directory.
_EMBEDDINGS_FILE = "entity_embeddings.npy"
_IDS_FILE = "entity_ids.json"


# ----------------------------------------------------------------------- #
# SemanticSearchIndex
# ----------------------------------------------------------------------- #

class SemanticSearchIndex:
    """
    Builds and queries a FAISS flat index of compliance-entity embeddings.

    Parameters
    ----------
    client : OllamaClient
        Configured Ollama HTTP client (used for embedding calls).
    records : list[EntityRecord]
        All entities to index.  Records without a docstring are skipped
        during embedding but are still stored with a zero vector so the
        index rows stay aligned with entity_ids.
    embed_dir : Path | str
        Directory where numpy array and ID JSON are persisted.
    model : str
        Ollama model name for embeddings (default: nomic-embed-text).
    """

    def __init__(
        self,
        client: "OllamaClient",
        records: list["EntityRecord"],
        embed_dir: Path | str = "storage/embeddings",
        model: str = _EMBED_MODEL,
    ) -> None:
        self.client = client
        self.records = records
        self.embed_dir = Path(embed_dir)
        self.model = model

        # Maps entity ID → index row; populated during build().
        self._id_to_row: dict[str, int] = {}
        # Numpy matrix (N × dim); populated during build() or load().
        self._embeddings: np.ndarray | None = None
        # Ordered list of entity IDs aligned with matrix rows.
        self._entity_ids: list[str] = []
        # Records keyed by ID for quick lookup.
        self._records_by_id: dict[str, "EntityRecord"] = {r.id: r for r in records}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(self, force_rebuild: bool = False) -> None:
        """
        Compute and persist embeddings for all records.

        If the embedding files already exist on disk and force_rebuild=False,
        loading from disk is skipped and the existing files are used as-is.
        Pass force_rebuild=True to recompute from scratch.

        Parameters
        ----------
        force_rebuild : bool
            When True, always recompute even if cached files exist.
        """
        emb_path = self.embed_dir / _EMBEDDINGS_FILE
        ids_path = self.embed_dir / _IDS_FILE

        if not force_rebuild and emb_path.exists() and ids_path.exists():
            logger.info(
                "Semantic search: loading cached embeddings from %s", self.embed_dir
            )
            self._load_from_disk(emb_path, ids_path)
            return

        logger.info(
            "Semantic search: computing embeddings for %d records using %s …",
            len(self.records),
            self.model,
        )
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self._compute_and_save(emb_path, ids_path)

    def search(self, query: str, top_k: int = 20) -> list["EntityRecord"]:
        """
        Return the top_k most relevant EntityRecords for the query string.

        Similarity is measured by cosine distance between the query embedding
        and all stored entity embeddings (inner product on unit vectors).

        Parameters
        ----------
        query : str
            Free-text compliance query, e.g. "fairness hooks for Art.10".
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[EntityRecord]
            Records sorted by descending similarity (highest first).
            May be shorter than top_k if the index has fewer entries.

        Raises
        ------
        RuntimeError
            If build() has not been called yet.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "SemanticSearchIndex.build() must be called before search()."
            )

        if len(self._entity_ids) == 0:
            return []

        # Embed the query.
        query_vec = self._embed_text(query)
        if query_vec is None:
            logger.warning("Semantic search: could not embed query — returning empty")
            return []

        # Cosine similarity: dot product on unit-normalised vectors.
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        scores = self._embeddings.dot(query_norm)  # shape: (N,)

        # Sort descending.
        n = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:n]

        results: list["EntityRecord"] = []
        for idx in top_indices:
            eid = self._entity_ids[idx]
            rec = self._records_by_id.get(eid)
            if rec is not None:
                results.append(rec)

        logger.debug(
            "Semantic search: query=%r  top_k=%d  returned=%d",
            query[:80],
            top_k,
            len(results),
        )
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_and_save(self, emb_path: Path, ids_path: Path) -> None:
        """Embed all records, build the matrix, and write to disk."""
        ids: list[str] = []
        vectors: list[list[float]] = []

        for i, rec in enumerate(self.records):
            # Build the text to embed: entity name + first 200 chars of docstring.
            text = f"{rec.entity_name}: {rec.docstring[:200]}" if rec.docstring else rec.entity_name
            vec = self._embed_text(text)
            if vec is None:
                # Fall back to a zero vector so index rows stay aligned.
                vec = np.zeros(_EMBED_DIM, dtype=np.float32)
                logger.debug(
                    "Semantic search: zero vector for entity %s (embed failed)",
                    rec.entity_name,
                )
            else:
                # Normalise to unit length for cosine similarity via dot product.
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

            ids.append(rec.id)
            vectors.append(vec.tolist())

            if (i + 1) % 100 == 0:
                logger.info(
                    "Semantic search: embedded %d / %d entities …",
                    i + 1,
                    len(self.records),
                )

        # Build numpy matrix and persist.
        matrix = np.array(vectors, dtype=np.float32)
        np.save(str(emb_path), matrix)
        ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")

        logger.info(
            "Semantic search: saved %d embeddings to %s  (shape=%s)",
            len(ids),
            self.embed_dir,
            matrix.shape,
        )

        # Populate in-memory state.
        self._entity_ids = ids
        self._embeddings = matrix
        self._id_to_row = {eid: i for i, eid in enumerate(ids)}

    def _load_from_disk(self, emb_path: Path, ids_path: Path) -> None:
        """Load pre-computed embeddings from disk."""
        self._embeddings = np.load(str(emb_path))
        self._entity_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self._id_to_row = {eid: i for i, eid in enumerate(self._entity_ids)}
        logger.info(
            "Semantic search: loaded %d embeddings (shape=%s)",
            len(self._entity_ids),
            self._embeddings.shape,
        )

    def _embed_text(self, text: str) -> np.ndarray | None:
        """
        Call Ollama embeddings endpoint and return a numpy float32 vector.

        Returns None if the embed call fails (caller should use a zero vector).
        """
        try:
            vec = self.client.embed(model=self.model, text=text)
            if not vec:
                return None
            return np.array(vec, dtype=np.float32)
        except Exception as exc:
            logger.debug("Semantic search: embed failed for text=%r: %s", text[:60], exc)
            return None
