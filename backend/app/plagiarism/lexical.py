"""Stage A (lexical): candidate retrieval with MinHash LSH.

Each query passage is converted to a MinHash signature over its word shingles and
looked up in the corpus LSH.  Shingles whose corpus document frequency exceeds
``COMMON_PHRASE_DF`` are dropped first: they are boilerplate ("in conclusion it can
be said that") and matching on them produces unfair reports.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..logging_setup import get_logger
from .corpus import PlagiarismCorpus
from .shingling import tokenize_words

log = get_logger(__name__)


@dataclass
class Candidate:
    """A corpus passage that may match the query passage."""

    passage_id: str | None
    document_id: str
    score: float
    kind: str  # lexical | dense


def filtered_shingles(text: str, corpus: PlagiarismCorpus, k: int | None = None) -> set[int]:
    """Shingle hashes with corpus-common shingles removed."""
    from .shingling import shingle_set

    hashes = shingle_set(tokenize_words(text), k or settings.shingle_size)
    df = corpus.document_frequency
    total_docs = max(1, len(corpus.documents))
    limit = settings.common_phrase_df * total_docs
    return {h for h in hashes if df.get(h, 0) <= limit or total_docs < 50}


def minhash_for(text: str, corpus: PlagiarismCorpus):  # noqa: ANN201 - datasketch type
    """Build a MinHash signature over the filtered shingles."""
    from datasketch import MinHash

    mh = MinHash(num_perm=settings.minhash_perms)
    for shingle in filtered_shingles(text, corpus):
        mh.update(str(shingle).encode())
    return mh


def retrieve(query_passages: list[str], corpus: PlagiarismCorpus) -> dict[int, list[Candidate]]:
    """Map passage index -> candidate corpus passages (lexical stage)."""
    corpus.load()
    by_passage_id = {pid: idx for idx, pid in enumerate(corpus.passage_ids)}
    out: dict[int, list[Candidate]] = {}
    for index, passage in enumerate(query_passages):
        mh = minhash_for(passage, corpus)
        hits = corpus.query_lsh(mh)
        candidates: list[Candidate] = []
        for passage_id in hits:
            corpus_idx = by_passage_id.get(passage_id)
            if corpus_idx is None:
                continue
            score = mh.jaccard(_cached_minhash(corpus, corpus_idx))
            if score <= 0:
                continue
            candidates.append(
                Candidate(
                    passage_id=passage_id,
                    document_id=corpus.passage_docs[corpus_idx],
                    score=float(score),
                    kind="lexical",
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        # One candidate per source document keeps the aligner focused.
        seen: set[str] = set()
        unique: list[Candidate] = []
        for candidate in candidates:
            if candidate.document_id in seen:
                continue
            seen.add(candidate.document_id)
            unique.append(candidate)
        out[index] = unique
    return out


def _cached_minhash(corpus: PlagiarismCorpus, index: int):  # noqa: ANN201
    """Build (and memoise on the corpus) a MinHash for a corpus passage."""
    cache = getattr(corpus, "_mh_cache", None)
    if cache is None:
        cache = {}
        setattr(corpus, "_mh_cache", cache)
    if index in cache:
        return cache[index]
    mh = minhash_for(corpus.passage_texts[index], corpus)
    cache[index] = mh
    return mh
