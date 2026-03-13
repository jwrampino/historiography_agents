"""
FAISS IVF-PQ index for approximate nearest-neighbor search over the corpus.
Supports:
  - Building the index from scratch
  - Incremental additions
  - Saving / loading from disk
  - Querying by embedding vector
  - Mapping between FAISS integer IDs and corpus source_ids
"""

from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sources.config.settings import (
    FAISS_INDEX_PATH,
    FAISS_ID_MAP_PATH,
    EMBEDDING_DIM,
    FAISS_NLIST,
    FAISS_M,
    FAISS_NBITS,
)

logger = logging.getLogger(__name__)


class CorpusIndex:
    """
    Manages a FAISS IVF-PQ index over corpus embeddings.

    ID mapping:
      FAISS assigns sequential integer IDs (0, 1, 2, …) to each added vector.
      We maintain a bidirectional map:
        int_id → source_id (UUID string)
        source_id → int_id
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        id_map_path: Path = FAISS_ID_MAP_PATH,
        dim: int = EMBEDDING_DIM,
        nlist: int = FAISS_NLIST,
        m: int = FAISS_M,
        nbits: int = FAISS_NBITS,
    ):
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits

        self._index = None                    # faiss.Index
        self._int_to_source: Dict[int, str] = {}   # int_id → source_id
        self._source_to_int: Dict[str, int] = {}   # source_id → int_id
        self._next_id: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, source_ids: List[str]) -> None:
        """
        Build the index from scratch.
        Requires at least nlist * 39 vectors for IVF training.
        Falls back to a flat IndexFlatIP index for small corpora.

        Args:
            embeddings: float32 array (N, EMBEDDING_DIM)
            source_ids: list of N UUID strings matching rows
        """
        import faiss

        assert embeddings.shape[1] == self.dim, \
            f"Embedding dim mismatch: {embeddings.shape[1]} != {self.dim}"
        assert len(source_ids) == embeddings.shape[0]

        n = embeddings.shape[0]
        embeddings = embeddings.astype(np.float32)

        # Normalize for inner-product ≈ cosine similarity
        faiss.normalize_L2(embeddings)

        if n < self.nlist * 39:
            logger.warning(
                f"Corpus too small ({n}) for IVF training (need {self.nlist * 39}). "
                f"Using flat IndexFlatIP instead."
            )
            self._index = faiss.IndexFlatIP(self.dim)
        else:
            # IVF-PQ: fast approximate search
            quantizer = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFPQ(
                quantizer, self.dim, self.nlist, self.m, self.nbits
            )
            logger.info(f"Training IVF-PQ index on {n} vectors …")
            self._index.train(embeddings)
            logger.info("Training complete")

        self._index.add(embeddings)

        # Build ID maps
        self._int_to_source = {i: sid for i, sid in enumerate(source_ids)}
        self._source_to_int = {sid: i for i, sid in enumerate(source_ids)}
        self._next_id = n

        logger.info(f"Index built: {n} vectors, dim={self.dim}")

    def add(self, embeddings: np.ndarray, source_ids: List[str]) -> None:
        """
        Add new vectors to an existing index.
        Note: IVF-PQ does not support dynamic additions after training.
        For small batches, use a flat index or rebuild periodically.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Index not initialised — call build() or load() first")

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        self._index.add(embeddings)
        for sid in source_ids:
            self._int_to_source[self._next_id] = sid
            self._source_to_int[sid] = self._next_id
            self._next_id += 1

    def save(self) -> None:
        """Persist index and ID map to disk."""
        import faiss

        if self._index is None:
            raise RuntimeError("No index to save")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))

        # Save ID map as JSON (str keys for JSON compatibility)
        with open(self.id_map_path, "w") as f:
            json.dump(
                {
                    "int_to_source": {str(k): v for k, v in self._int_to_source.items()},
                    "next_id": self._next_id,
                },
                f,
                indent=2,
            )

        logger.info(f"Index saved: {self.index_path} ({self._next_id} vectors)")

    def load(self) -> None:
        """Load index and ID map from disk."""
        import faiss

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        self._index = faiss.read_index(str(self.index_path))

        with open(self.id_map_path) as f:
            data = json.load(f)

        self._int_to_source = {int(k): v for k, v in data["int_to_source"].items()}
        self._source_to_int = {v: k for k, v in self._int_to_source.items()}
        self._next_id = data.get("next_id", len(self._int_to_source))

        logger.info(f"Index loaded: {self._next_id} vectors from {self.index_path}")

    # ── Query ──────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        nprobe: int = 32,
    ) -> List[dict]:
        """
        Search for the top_k nearest vectors.

        Args:
            query_vector: 1D float32 array of length EMBEDDING_DIM
            top_k:        Number of results to return
            nprobe:       IVF probe count (higher = more accurate, slower)

        Returns:
            List of dicts: {source_id, similarity_score, rank}
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Index not initialised — call build() or load() first")

        query = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        # Set nprobe for IVF indexes
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = nprobe

        scores, indices = self._index.search(query, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:   # FAISS returns -1 for unfilled slots
                continue
            source_id = self._int_to_source.get(int(idx))
            if source_id:
                results.append(
                    {
                        "source_id": source_id,
                        "similarity_score": float(score),
                        "rank": rank + 1,
                    }
                )

        return results

    def search_by_source_id(
        self, source_id: str, top_k: int = 10
    ) -> List[dict]:
        """
        Find items similar to a known corpus item (by its source_id).
        Retrieves the stored vector and searches from it.
        """
        import faiss

        int_id = self._source_to_int.get(source_id)
        if int_id is None:
            raise KeyError(f"source_id not found in index: {source_id}")

        # Reconstruct vector from index
        vec = np.zeros((1, self.dim), dtype=np.float32)
        self._index.reconstruct(int_id, vec[0])
        return self.search(vec[0], top_k=top_k + 1)[1:]  # exclude self

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._next_id

    @property
    def is_trained(self) -> bool:
        return self._index is not None and (
            not hasattr(self._index, "is_trained") or self._index.is_trained
        )

    def source_id_exists(self, source_id: str) -> bool:
        return source_id in self._source_to_int
