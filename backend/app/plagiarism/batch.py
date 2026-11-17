"""Batch cross-comparison: every pair of submitted documents, in both directions.

The pairwise comparison uses the same aligner as the corpus path, so a match found
here means exactly what a match found against the corpus means.  Output includes the
N×N matrix (percent of row document found in column document), the strongest aligned
segments for the top pairs and graph data for the similarity view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging_setup import get_logger
from ..schemas import BatchPair, BatchResult
from .align import align

log = get_logger(__name__)

MAX_PAIRS = 200
TOP_SEGMENTS = 5


@dataclass
class BatchDocument:
    """A labelled input for the comparison."""

    id: str
    label: str
    text: str
    words: int = 0
    verdict: str = "uncertain"
    meta: dict[str, object] = field(default_factory=dict)


def _covered_percent(text_length: int, spans: list[tuple[int, int]]) -> float:
    if not spans or text_length <= 0:
        return 0.0
    ordered = sorted(spans)
    total, start, end = 0, ordered[0][0], ordered[0][1]
    for s, e in ordered[1:]:
        if s <= end:
            end = max(end, e)
        else:
            total += end - start
            start, end = s, e
    total += end - start
    return round(100.0 * total / text_length, 2)


def compare(documents: list[BatchDocument]) -> BatchResult:
    """Compare all pairs and return the matrix, pairs and graph data."""
    labels = [d.label for d in documents]
    n = len(documents)
    matrix = [[0.0] * n for _ in range(n)]
    pairs: list[BatchPair] = []
    compared = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 100.0
                continue
            if compared >= MAX_PAIRS and j < i:
                continue
            matches = align(documents[i].text, documents[j].text, documents[j].id, min_words=6)
            percent = _covered_percent(len(documents[i].text), [m.query_span for m in matches])
            matrix[i][j] = percent
            if i < j:
                reverse = _covered_percent(
                    len(documents[j].text), [m.query_span for m in matches]
                )
                pairs.append(
                    BatchPair(
                        a=labels[i],
                        b=labels[j],
                        percent_ab=percent,
                        percent_ba=reverse if matches else 0.0,
                        segments=[
                            {
                                "query_span": list(m.query_span),
                                "source_span": list(m.source_span),
                                "words": m.words,
                                "type": m.match_type,
                            }
                            for m in sorted(matches, key=lambda m: m.words, reverse=True)[:TOP_SEGMENTS]
                        ],
                    )
                )
                compared += 1

    nodes = [
        {
            "id": d.id,
            "label": d.label,
            "words": d.words,
            "verdict": d.verdict,
        }
        for d in documents
    ]
    edges = [
        {"a": p.a, "b": p.b, "weight": round(max(p.percent_ab, p.percent_ba), 2)}
        for p in pairs
        if max(p.percent_ab, p.percent_ba) >= 5.0
    ]
    return BatchResult(matrix=matrix, labels=labels, pairs=pairs, nodes=nodes, edges=edges)
