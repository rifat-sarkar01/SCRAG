"""
src/retrieval/document_store.py
--------------------------------
Loads .txt and .md corpus files from a directory and splits them into
overlapping chunks using a recursive character-based splitting strategy
that mirrors LangChain's RecursiveCharacterTextSplitter logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class DocumentStore:
    """
    Loads plain-text and Markdown documents from a directory and chunks them
    using recursive character splitting.

    Chunk boundaries are attempted in priority order:
      paragraph break → line break → sentence boundary → word boundary → character

    Args:
        chunk_size:    Maximum character length of each chunk.
        chunk_overlap: Number of characters from the end of one chunk that are
                       prepended to the next (sliding-window context).
    """

    # Separator priority list: try to split at the coarsest level first.
    _SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_documents(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Recursively reads every .txt and .md file under *data_dir*.

        Args:
            data_dir: Path to the directory containing corpus files.

        Returns:
            List of document dicts:
            ``{"text": str, "source": str, "filename": str}``
        """
        documents: List[Dict[str, Any]] = []
        root = Path(data_dir)

        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        for filepath in sorted(root.rglob("*")):
            if filepath.suffix.lower() in {".txt", ".md"} and filepath.is_file():
                try:
                    text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError as exc:
                    # Skip unreadable files; log for debugging.
                    print(f"[DocumentStore] Warning: could not read {filepath}: {exc}")
                    continue

                if text:
                    documents.append({
                        "text": text,
                        "source": str(filepath),
                        "filename": filepath.name,
                    })

        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits each document into overlapping chunks.

        Args:
            documents: List of document dicts (as returned by ``load_documents``).

        Returns:
            Flat list of chunk dicts:
            ``{"text": str, "source": str, "filename": str, "chunk_index": int}``
        """
        all_chunks: List[Dict[str, Any]] = []

        for doc in documents:
            raw_chunks = self._split(doc["text"])
            for i, chunk_text in enumerate(raw_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "source": doc.get("source", ""),
                    "filename": doc.get("filename", ""),
                    "chunk_index": i,
                })

        return all_chunks

    # ------------------------------------------------------------------
    # Internal splitting logic
    # ------------------------------------------------------------------

    def _split(self, text: str) -> List[str]:
        """Entry point for recursive splitting; returns list of clean chunks."""
        raw = self._recursive_split(text, self._SEPARATORS)
        return self._apply_overlap(raw)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively tries each separator in *separators*.  If the text fits
        within chunk_size it is returned as-is.  If no separator works the
        text is hard-split at chunk_size.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            # Hard cut as last resort.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep, *rest_seps = separators

        if sep not in text:
            return self._recursive_split(text, rest_seps)

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # Part itself might be too large — recurse.
                if len(part) > self.chunk_size:
                    sub = self._recursive_split(part, rest_seps)
                    if sub:
                        chunks.extend(sub[:-1])
                        current = sub[-1]
                    else:
                        current = ""
                else:
                    current = part

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if c]

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Rebuilds the chunk list so that each chunk (except the first) is
        prepended with the last *chunk_overlap* characters of its predecessor,
        creating a sliding-window context window.
        """
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for chunk in chunks[1:]:
            tail = result[-1][-self.chunk_overlap :]
            merged = tail + (" " if tail and not tail.endswith(" ") else "") + chunk
            result.append(merged if len(merged) <= self.chunk_size else chunk)

        return result
