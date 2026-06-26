"""
src/retrieval/embedding_index.py
---------------------------------
Local FAISS vector index using sentence-transformers for embedding.

No API calls are made for embeddings — everything runs on CPU via the
all-MiniLM-L6-v2 sentence-transformer model (fast, ~80 MB).

Similarity metric: cosine (implemented as inner product after L2 normalisation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class EmbeddingIndex:
    """
    Embeds document chunks with a local sentence-transformer model and stores
    them in a FAISS flat inner-product index for fast cosine similarity search.

    Usage::

        index = EmbeddingIndex()
        index.build_index(chunks)          # chunks: list[dict] with "text" key
        results = index.retrieve("query")  # -> list[(chunk_dict, score)]

    Args:
        model_name: A sentence-transformers model identifier.
                    Defaults to ``all-MiniLM-L6-v2`` — 384-dim, fast, ~80 MB.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._index = None           # faiss.Index, populated by build_index()
        self._chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds *chunks* and populates the FAISS index.

        Args:
            chunks: List of chunk dicts, each containing at minimum a ``"text"``
                    key.  Any extra metadata (source, filename, …) is preserved
                    and returned verbatim from ``retrieve()``.
        """
        import faiss  # lazy import — not needed until build time

        if not chunks:
            raise ValueError("chunks list is empty — nothing to index.")

        self._chunks = chunks
        texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]

        embeddings: np.ndarray = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=64,
        ).astype(np.float32)

        # L2-normalise so that inner product == cosine similarity.
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Returns the top-*k* chunks most similar to *query*.

        Args:
            query: Natural-language search query.
            k:     Number of results to return.  Capped at the index size.

        Returns:
            List of ``(chunk_dict, cosine_score)`` tuples, sorted descending
            by score.  ``cosine_score`` is in ``[-1.0, 1.0]``; higher is more
            similar.

        Raises:
            RuntimeError: If ``build_index()`` has not been called yet.
        """
        import faiss  # lazy import

        if self._index is None:
            raise RuntimeError(
                "FAISS index is not built.  Call build_index(chunks) first."
            )

        k_capped = min(k, len(self._chunks))
        q_emb: np.ndarray = self._model.encode(
            [query],
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(q_emb)

        scores, indices = self._index.search(q_emb, k_capped)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for padding when k > n_docs
                results.append((self._chunks[int(idx)], float(score)))

        return results
