"""The plagiarism pipeline: document -> flagged, scored similarity report.

Stages, in order (spec section 10.1):
    A  candidate retrieval   lexical (MinHash LSH) + dense (embeddings)
    B  seed-and-extend alignment with exact character spans
    C  paraphrase verification with a cross-encoder
    D  exclusions computed per match and exposed as flags
    E  merge, score and attribute to sources

Anything that cannot run (missing semantic model, unreachable online sources) is
recorded in ``warnings`` and reflected in the result, never silently skipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import settings
from ..logging_setup import get_logger
from ..schemas import PlagMatch, PlagiarismResult
from ..text.exclusions import tag_sections
from ..text.segment import passages
from .align import Match, align, align_sentences
from .corpus import PlagiarismCorpus
from .lexical import retrieve
from .merge import MergeInput, merge
from .semantic import Embedder, Reranker

log = get_logger(__name__)

MAX_SOURCE_DOCS = 8
MAX_QUERY_PASSAGES = 120


@dataclass
class PlagiarismOutput:
    result: PlagiarismResult
    elapsed_ms: float


def _to_schema(match: Match, corpus: PlagiarismCorpus, query_text: str, source_text: str) -> PlagMatch:
    document = corpus.document(match.source_id)
    title = document.title if document else match.source_id
    url = document.url if document else None
    return PlagMatch(
        query_span=(match.query_start, match.query_end),
        source_span=(match.source_start, match.source_end),
        query_text=query_text[match.query_start : match.query_end],
        source_text=source_text[match.source_start : match.source_end],
        source_id=match.source_id,
        source_title=title,
        source_url=url,
        match_type=match.match_type,  # type: ignore[arg-type]
        score=round(match.score, 4),
        words=match.words,
    )


def _candidate_documents(query_passages: list[str], corpus: PlagiarismCorpus) -> list[str]:
    """Union of documents proposed by the lexical retriever, most frequent first."""
    if not query_passages:
        return []
    candidates = retrieve(query_passages, corpus)
    counts: dict[str, float] = {}
    for hits in candidates.values():
        for hit in hits:
            counts[hit.document_id] = counts.get(hit.document_id, 0.0) + hit.score
    ordered = sorted(counts, key=counts.get, reverse=True)
    return ordered[:MAX_SOURCE_DOCS]


def _dense_paraphrases(
    query_passages: list[str],
    corpus: PlagiarismCorpus,
    embedder: Embedder,
    reranker: Reranker,
    corpus_size: int,
) -> tuple[list[PlagMatch], list[str]]:
    """Find paraphrase matches that lexical retrieval cannot see."""
    if not embedder.available:
        return [], ["paraphrase detection unavailable: " + (embedder.reason or "no model")]
    query_vectors = embedder.encode(query_passages)
    if query_vectors is None:
        return [], ["paraphrase detection unavailable: embedding failed"]

    triples: list[tuple[int, int, float]] = []
    for index, vector in enumerate(query_vectors):
        for corpus_index, cosine in corpus.query_dense(vector, settings.semantic_top_k):
            if cosine >= settings.semantic_min_cosine - 0.05:
                triples.append((index, corpus_index, cosine))
    if not triples:
        return [], []

    from .semantic import verify_paraphrases

    outcome = verify_paraphrases(
        query_passages=query_passages,
        query_vectors=query_vectors,
        corpus_passages=corpus.passage_texts,
        corpus_vectors=corpus.embeddings,
        corpus_indices=[(q, c) for q, c, _ in triples],
        cosines=[cos for _, _, cos in triples],
        reranker=reranker,
    )
    matches: list[PlagMatch] = []
    for hit in outcome.hits:
        passage = passages(query_passages[hit.query_index])[0]
        source_doc = corpus.passage_docs[hit.corpus_index]
        document = corpus.document(source_doc)
        matches.append(
            PlagMatch(
                query_span=(passage.start, passage.end),
                source_span=None,
                query_text=passage.text,
                source_text=corpus.passage_texts[hit.corpus_index],
                source_id=source_doc,
                source_title=document.title if document else source_doc,
                source_url=document.url if document else None,
                match_type="paraphrase",
                score=round(hit.rerank_score or hit.cosine, 4),
                words=passage.words,
            )
        )
    return matches, outcome.warnings


def analyse(
    text: str,
    corpus: PlagiarismCorpus,
    options: dict[str, object] | None = None,
) -> PlagiarismOutput:
    """Run the full plagiarism pipeline for one document."""
    started = time.perf_counter()
    options = options or {}
    warnings: list[str] = []
    corpus.load()
    stats = corpus.stats()
    if stats.documents == 0:
        warnings.append(
            "the plagiarism corpus is empty on this server; similarity can only be computed "
            "for documents compared within this request"
        )

    sections = tag_sections(text)
    query_passages = [p.text for p in passages(text)][:MAX_QUERY_PASSAGES]
    document_ids = _candidate_documents(query_passages, corpus)

    lexical: list[PlagMatch] = []
    for doc_id in document_ids:
        source_text = corpus.source_text(doc_id)
        if not source_text:
            continue
        for match in align(text, source_text, doc_id):
            lexical.append(_to_schema(match, corpus, text, source_text))
        for match in align_sentences(text, source_text, doc_id):
            lexical.append(_to_schema(match, corpus, text, source_text))

    embedder = Embedder()
    reranker = Reranker()
    paraphrase, dense_warnings = ([], []) if not query_passages else _dense_paraphrases(
        query_passages, corpus, embedder, reranker, stats.documents
    )
    warnings.extend(dense_warnings)

    online_matches: list[PlagMatch] = []
    online_snippets: list[str] = []
    if options.get("online"):
        from .online import check_online

        online_matches, online_snippets, online_warnings = check_online(text, sections)
        warnings.extend(online_warnings)

    result = merge(
        MergeInput(
            text=text,
            matches=lexical + paraphrase + online_matches,
            sections=sections,
            corpus_size=stats.documents,
            warnings=warnings,
            online_snippets=online_snippets,
        ),
        exclusions=dict(options.get("exclusions") or {}),  # type: ignore[arg-type]
    )
    elapsed = (time.perf_counter() - started) * 1000
    log.info(
        "plagiarism: %s matches, %.1f%% (filtered %.1f%%) in %.0f ms",
        len(result.matches), result.similarity_percent, result.similarity_percent_filtered, elapsed,
    )
    return PlagiarismOutput(result=result, elapsed_ms=elapsed)
