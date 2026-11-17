"""The plagiarism reference corpus: metadata, MinHash index and dense index.

A corpus is a directory holding ``meta.json`` (list of documents), ``lsh.pkl`` (a
MinHash LSH over passages), ``embeddings.npy`` (float32 passage vectors) and
``embeddings_ids.json``.  All of it is small enough to ship inside the artifact
bundle, and the same class works for the bundled 30 sample documents, the
10,000-article Wikipedia set and anything the user adds at runtime.

The bundled corpus is deliberately described honestly in the UI: coverage is
limited compared to commercial services, and the numbers shown are computed
against what is actually indexed.
"""

from __future__ import annotations

import json
import pickle
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from .shingling import tokenize_words

log = get_logger(__name__)

META_FILE = "meta.json"
LSH_FILE = "lsh.pkl"
EMB_FILE = "embeddings.npy"
EMB_IDS_FILE = "embeddings_ids.json"


@dataclass
class CorpusDocument:
    """One source document in the corpus."""

    id: str
    title: str
    url: str | None = None
    text: str = ""
    source: str = "bundled"
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CorpusStats:
    documents: int = 0
    passages: int = 0
    articles: int = 0
    updated_at: str | None = None


class PlagiarismCorpus:
    """Load, query and extend the plagiarism corpus."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or (settings.artifact_path / "corpus")
        self.documents: list[CorpusDocument] = []
        self.passage_ids: list[str] = []
        self.passage_docs: list[str] = []
        self.passage_texts: list[str] = []
        self._lsh: object | None = None
        self._embeddings: np.ndarray | None = None
        self._df: dict[int, int] = {}
        self._passage_shingles: list[set[int]] = []
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Read the index from disk. Missing files simply mean an empty corpus."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            meta_path = self.dir / META_FILE
            if not meta_path.exists():
                log.info("no plagiarism corpus at %s; the engine runs with bundled samples only", self.dir)
                return
            try:
                raw = json.loads(meta_path.read_text())
                self.documents = [CorpusDocument(**item) for item in raw.get("documents", [])]
                self.passage_ids = raw.get("passage_ids", [])
                self.passage_docs = raw.get("passage_docs", [])
                self.passage_texts = raw.get("passage_texts", [])
                self._df = {int(k): v for k, v in raw.get("document_frequency", {}).items()}
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning("corpus metadata unreadable (%s); corpus treated as empty", exc)
                return
            lsh_path = self.dir / LSH_FILE
            if lsh_path.exists():
                try:
                    with lsh_path.open("rb") as handle:
                        self._lsh = pickle.load(handle)
                except Exception as exc:  # noqa: BLE001 - corrupt cache is recoverable
                    log.warning("LSH index unreadable: %s", exc)
            emb_path = self.dir / EMB_FILE
            if emb_path.exists():
                try:
                    self._embeddings = np.load(emb_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning("embedding index unreadable: %s", exc)
            log.info(
                "corpus loaded: %s documents, %s passages, %s passages embedded",
                len(self.documents),
                len(self.passage_texts),
                len(self._embeddings) if self._embeddings is not None else 0,
            )

    # ------------------------------------------------------------------ #
    @property
    def lsh(self) -> object | None:
        self.load()
        return self._lsh

    @property
    def embeddings(self) -> np.ndarray | None:
        self.load()
        return self._embeddings

    @property
    def document_frequency(self) -> dict[int, int]:
        self.load()
        return self._df

    def stats(self) -> CorpusStats:
        self.load()
        return CorpusStats(
            documents=len(self.documents),
            passages=len(self.passage_texts),
            articles=sum(1 for d in self.documents if d.source.startswith("wikipedia")),
            updated_at=(self.dir / META_FILE).stat().st_mtime_ns.__str__()
            if (self.dir / META_FILE).exists()
            else None,
        )

    def document(self, doc_id: str) -> CorpusDocument | None:
        for document in self.documents:
            if document.id == doc_id:
                return document
        return None

    def source_text(self, doc_id: str) -> str:
        document = self.document(doc_id)
        return document.text if document else ""

    # ------------------------------------------------------------------ #
    def add_document(self, title: str, text: str, url: str | None = None, source: str = "upload") -> CorpusDocument:
        """Append a document at runtime and update both indexes in memory."""
        self.load()
        with self._lock:
            doc = CorpusDocument(
                id=f"doc-{len(self.documents):05d}",
                title=title,
                url=url,
                text=text,
                source=source,
            )
            self.documents.append(doc)
            new_ids, new_texts = self._passages_for(doc)
            offset = len(self.passage_texts)
            self.passage_texts.extend(new_texts)
            self.passage_ids.extend(new_ids)
            self.passage_docs.extend([doc.id] * len(new_texts))
            self._update_indexes(offset, new_ids, new_texts)
            return doc

    def save(self) -> None:
        """Persist metadata and indexes so the next start reuses them."""
        self.load()
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [asdict(d) for d in self.documents],
            "passage_ids": self.passage_ids,
            "passage_docs": self.passage_docs,
            "passage_texts": self.passage_texts,
            "document_frequency": {str(k): v for k, v in self._df.items()},
            "settings": {"minhash_perms": settings.minhash_perms, "lsh_threshold": settings.lsh_threshold},
        }
        (self.dir / META_FILE).write_text(json.dumps(payload))
        if self._lsh is not None:
            with (self.dir / LSH_FILE).open("wb") as handle:
                pickle.dump(self._lsh, handle)
        if self._embeddings is not None:
            np.save(self.dir / EMB_FILE, self._embeddings)
            (self.dir / EMB_IDS_FILE).write_text(json.dumps(self.passage_ids))

    # ------------------------------------------------------------------ #
    def _passages_for(self, doc: CorpusDocument) -> tuple[list[str], list[str]]:
        from ..text.segment import passages

        chunks = passages(doc.text)
        ids = [f"{doc.id}:p{i}" for i in range(len(chunks))]
        return ids, [c.text for c in chunks]

    def _update_indexes(self, offset: int, ids: list[str], texts: list[str]) -> None:
        """Insert new passages into LSH, the dense index and the df table."""
        from datasketch import MinHash, MinHashLSH

        from .shingling import shingle_set

        if self._lsh is None:
            self._lsh = MinHashLSH(threshold=settings.lsh_threshold, num_perm=settings.minhash_perms)
        for passage_id, text in zip(ids, texts):
            shingles = shingle_set(tokenize_words(text), settings.shingle_size)
            for shingle in shingles:
                self._df[shingle] = self._df.get(shingle, 0) + 1
            mh = MinHash(num_perm=settings.minhash_perms)
            for shingle in shingles:
                mh.update(str(shingle).encode())
            try:
                self._lsh.insert(passage_id, mh)
            except ValueError:  # duplicate id, e.g. re-adding the same document
                pass

    def query_lsh(self, minhash: object, top_k: int = 20) -> list[str]:
        self.load()
        if self._lsh is None:
            return []
        try:
            return list(self._lsh.query(minhash))[:200]  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            log.debug("lsh query failed: %s", exc)
            return []

    def query_dense(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Cosine top-k against the passage embeddings (exact, chunked)."""
        self.load()
        matrix = self._embeddings
        if matrix is None or matrix.size == 0:
            return []
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        query = vector / norm
        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1.0
        scores = (matrix @ query) / matrix_norms
        k = min(top_k, scores.size)
        idx = np.argpartition(-scores, k - 1)[:k] if k < scores.size else np.arange(scores.size)
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]
