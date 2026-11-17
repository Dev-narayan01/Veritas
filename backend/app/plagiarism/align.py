"""Stage B: seed-and-extend alignment with precise spans.

Fingerprints shared by the query and a candidate source are seeds.  Seeds that are
consistent in order and close enough together (``SEED_GAP_WORDS``) are merged into
maximal segments; segments below ``MIN_MATCH_WORDS`` are dropped and the rest are
kept only when the query segment's shingles are sufficiently contained in the
source segment.  A sentence-level pass catches sentences that were moved rather
than copied in place.  The output carries character spans on *both* sides, which is
what the side-by-side evidence view needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..logging_setup import get_logger
from ..text.segment import split_sentences
from .shingling import (
    Fingerprint,
    containment,
    shingle_set,
    tokenize_words,
    winnow,
    word_spans,
)

log = get_logger(__name__)


@dataclass
class Match:
    """A verified textual overlap between the query and one source."""

    query_start: int
    query_end: int
    source_start: int
    source_end: int
    match_type: str  # exact | near | paraphrase | online
    score: float
    words: int
    source_id: str
    method: str = "alignment"  # alignment | sentence | semantic

    @property
    def query_span(self) -> tuple[int, int]:
        """Character span in the query document.

        Batch comparison and report rendering both need the span as a pair; keeping the
        stored form as two named fields makes the aligner's output readable in tests, so the
        tuple is exposed as a property rather than duplicated as a field.
        """
        return (self.query_start, self.query_end)

    @property
    def source_span(self) -> tuple[int, int]:
        """Character span in the source document."""
        return (self.source_start, self.source_end)


def _char_span(spans: list[tuple[int, int]], start_word: int, end_word: int, length: int) -> tuple[int, int]:
    """Convert word indices to character spans, clamped to the text length."""
    if not spans:
        return 0, length
    lo = spans[min(start_word, len(spans) - 1)][0]
    hi = spans[min(end_word, len(spans) - 1)][1]
    return lo, max(hi, lo)


def _merge_seeds(seeds: list[tuple[Fingerprint, Fingerprint]]) -> list[tuple[int, int, int, int]]:
    """Merge seeds into maximal segments along consistent diagonals.

    Two seeds belong to the same copied region when their *offset* (source
    position minus query position) is nearly equal: that is what "copied from the
    same place" means geometrically.  Grouping by diagonal (rather than naively by
    document order) keeps repeated boilerplate from fragmenting a genuine long
    copy, and the ``SEED_GAP_WORDS`` limit stops a diagonal from jumping across a
    large rewritten section.
    """
    if not seeds:
        return []
    buckets: dict[int, list[tuple[Fingerprint, Fingerprint]]] = {}
    for qf, sf in seeds:
        diagonal = sf.position - qf.position
        # 8-word tolerance absorbs equal-size edits that shift the diagonal
        # slightly; wider than that is a different copy site.
        key = diagonal // 8
        buckets.setdefault(key, []).append((qf, sf))

    groups: list[list[tuple[Fingerprint, Fingerprint]]] = []
    for key in sorted(buckets):
        items = sorted(buckets[key], key=lambda pair: pair[0].position)
        current: list[tuple[Fingerprint, Fingerprint]] = []
        for pair in items:
            if not current:
                current = [pair]
                continue
            dq = pair[0].position - current[-1][0].position
            ds = pair[1].position - current[-1][1].position
            if -2 <= dq <= settings.seed_gap_words and -2 <= ds <= settings.seed_gap_words:
                current.append(pair)
            else:
                groups.append(current)
                current = [pair]
        if current:
            groups.append(current)

    spans = [
        (g[0][0].position, g[-1][0].position, g[0][1].position, g[-1][1].position) for g in groups
    ]
    return sorted(spans, key=lambda span: span[0])


def align(
    query_text: str,
    source_text: str,
    source_id: str,
    min_words: int | None = None,
    min_containment: float | None = None,
) -> list[Match]:
    """Align one query document against one source document."""
    k = settings.winnow_k
    window = settings.winnow_w
    min_words = settings.min_match_words if min_words is None else min_words
    min_containment = (
        settings.lexical_min_containment if min_containment is None else min_containment
    )

    query_words = tokenize_words(query_text)
    source_words = tokenize_words(source_text)
    if len(query_words) < k or len(source_words) < k:
        return []
    q_spans = word_spans(query_text)
    s_spans = word_spans(source_text)
    q_prints = winnow(query_words, k, window)
    s_prints = winnow(source_words, k, window)
    if not q_prints or not s_prints:
        return []

    source_index: dict[int, list[int]] = {}
    for fp in s_prints:
        source_index.setdefault(fp.hash, []).append(fp.position)

    seeds: list[tuple[Fingerprint, Fingerprint]] = []
    for qf in q_prints:
        for position in source_index.get(qf.hash, []):
            seeds.append((qf, Fingerprint(hash=qf.hash, position=position)))

    matches: list[Match] = []
    for q_start, q_end, s_start, s_end in _merge_seeds(seeds):
        q_word_span = (q_start, q_end + k)
        s_word_span = (s_start, s_end + k)
        words = q_word_span[1] - q_word_span[0]
        if words < min_words:
            continue
        q_slice = query_words[q_word_span[0] : q_word_span[1]]
        s_slice = source_words[s_word_span[0] : s_word_span[1]]
        contain = containment(shingle_set(q_slice, k), shingle_set(s_slice, k))
        if contain < min_containment:
            continue
        match_type = "exact" if contain >= 0.9 else "near"
        matches.append(
            Match(
                query_start=_char_span(q_spans, q_word_span[0], q_word_span[1] - 1, len(query_text))[0],
                query_end=_char_span(q_spans, q_word_span[0], q_word_span[1] - 1, len(query_text))[1],
                source_start=_char_span(s_spans, s_word_span[0], s_word_span[1] - 1, len(source_text))[0],
                source_end=_char_span(s_spans, s_word_span[0], s_word_span[1] - 1, len(source_text))[1],
                match_type=match_type,
                score=round(contain, 4),
                words=words,
                source_id=source_id,
                method="alignment",
            )
        )
    return _dedupe(matches)


def align_sentences(
    query_text: str,
    source_text: str,
    source_id: str,
    min_words: int = 10,
    min_containment: float | None = None,
) -> list[Match]:
    """Sentence-level pass: catches moved or lightly edited sentences."""
    threshold = settings.lexical_min_containment if min_containment is None else min_containment
    k = settings.shingle_size
    source_sentences = split_sentences(source_text)
    if not source_sentences:
        return []
    source_sets = [(s, shingle_set(tokenize_words(s.text), k)) for s in source_sentences]
    matches: list[Match] = []
    for sentence in split_sentences(query_text):
        if sentence.words < min_words:
            continue
        q_set = shingle_set(tokenize_words(sentence.text), k)
        if not q_set:
            continue
        best = max(source_sets, key=lambda item: containment(q_set, item[1]))
        score = containment(q_set, best[1])
        if score >= max(threshold, 0.6):
            matches.append(
                Match(
                    query_start=sentence.start,
                    query_end=sentence.end,
                    source_start=best[0].start,
                    source_end=best[0].end,
                    match_type="exact" if score >= 0.9 else "near",
                    score=round(score, 4),
                    words=sentence.words,
                    source_id=source_id,
                    method="sentence",
                )
            )
    return _dedupe(matches)


def _dedupe(matches: list[Match]) -> list[Match]:
    """Drop matches whose query span is already covered by a stronger match."""
    ordered = sorted(matches, key=lambda m: (m.query_start, -m.score))
    kept: list[Match] = []
    for match in ordered:
        overlaps = [
            other
            for other in kept
            if not (match.query_end <= other.query_start or match.query_start >= other.query_end)
        ]
        if not overlaps:
            kept.append(match)
            continue
        # Keep the longer/stronger span and merge overlapping identical-type spans.
        best = max(overlaps, key=lambda m: (m.query_end - m.query_start) * m.score)
        if (match.query_end - match.query_start) * match.score > (
            best.query_end - best.query_start
        ) * best.score:
            kept = [m for m in kept if m not in overlaps] + [match]
    return sorted(kept, key=lambda m: m.query_start)
