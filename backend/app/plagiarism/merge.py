"""Stage E: merge all matches, score the document and attribute sources.

The headline similarity is the share of *characters* in the document covered by at
least one matched span.  That definition is transparent, additive per source, and
impossible to inflate by counting overlapping matches twice.  Separate percentages
are reported for lexical, paraphrase and online matches so the user can see what
kind of similarity it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import PlagMatch, PlagiarismResult, SourceBreakdown
from ..text.exclusions import classify_span

PALETTE_SIZE = 12


@dataclass
class MergeInput:
    """Everything the merge stage needs."""

    text: str
    matches: list[PlagMatch]
    sections: list = field(default_factory=list)
    corpus_size: int = 0
    warnings: list[str] = field(default_factory=list)
    online_snippets: list[str] = field(default_factory=list)


def _covered_length(spans: list[tuple[int, int]]) -> int:
    """Union length of (possibly overlapping) character spans."""
    if not spans:
        return 0
    ordered = sorted(spans)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def _colour_index(source_id: str, assigned: dict[str, int]) -> int:
    if source_id not in assigned:
        assigned[source_id] = len(assigned) % PALETTE_SIZE
    return assigned[source_id]


def merge(payload: MergeInput, exclusions: dict[str, object] | None = None) -> PlagiarismResult:
    """Build the plagiarism result, applying exclusion flags to a filtered score."""
    exclusions = exclusions or {}
    exclude_quotes = bool(exclusions.get("quotes", True))
    exclude_bibliography = bool(exclusions.get("bibliography", True))
    min_words = int(exclusions.get("min_words", 8) or 8)

    text_length = max(1, len(payload.text))
    colours: dict[str, int] = {}
    kept: list[PlagMatch] = []

    for match in payload.matches:
        span = (match.query_span[0], match.query_span[1])
        kind = classify_span(span, payload.sections) if payload.sections else "body"
        match.in_quote = kind == "quote"
        match.in_bibliography = kind in {"bibliography", "code"}
        match.color_index = _colour_index(match.source_id, colours)
        kept.append(match)

    def _spans(predicate) -> list[tuple[int, int]]:
        return [m.query_span for m in kept if predicate(m)]

    all_spans = _spans(lambda m: True)
    filtered_spans = _spans(
        lambda m: m.words >= min_words
        and not (exclude_quotes and m.in_quote)
        and not (exclude_bibliography and m.in_bibliography)
    )

    def pct(spans: list[tuple[int, int]]) -> float:
        return round(100.0 * _covered_length(spans) / text_length, 2)

    per_source: dict[str, dict[str, object]] = {}
    for match in kept:
        entry = per_source.setdefault(
            match.source_id,
            {
                "title": match.source_title,
                "url": match.source_url,
                "matches": 0,
                "spans": [],
            },
        )
        entry["matches"] = int(entry["matches"]) + 1  # type: ignore[arg-type]
        entry["spans"].append(match.query_span)  # type: ignore[union-attr]

    sources = [
        SourceBreakdown(
            id=source_id,
            title=str(data["title"]),
            url=data["url"],  # type: ignore[arg-type]
            color_index=_colour_index(source_id, colours),
            percent=pct(data["spans"]),  # type: ignore[arg-type]
            matches=int(data["matches"]),  # type: ignore[arg-type]
        )
        for source_id, data in per_source.items()
    ]
    sources.sort(key=lambda s: s.percent, reverse=True)

    result = PlagiarismResult(
        similarity_percent=pct(all_spans),
        similarity_percent_filtered=pct(filtered_spans),
        lexical_percent=pct(_spans(lambda m: m.match_type in {"exact", "near"})),
        paraphrase_percent=pct(_spans(lambda m: m.match_type == "paraphrase")),
        online_percent=pct(_spans(lambda m: m.match_type == "online")),
        sources=sources,
        matches=sorted(kept, key=lambda m: m.query_span[0])[:25],
        corpus_size=payload.corpus_size,
        online_snippets_sent=payload.online_snippets,
        warnings=payload.warnings,
    )
    return result


def filter_matches(matches: list[PlagMatch], exclusions: dict[str, object]) -> list[PlagMatch]:
    """Client-side re-filtering logic (mirrored in the frontend and tested with vitest)."""
    min_words = int(exclusions.get("min_words", 8) or 8)
    hide_below = float(exclusions.get("hide_below_percent", 0.5) or 0.0)
    exclude_quotes = bool(exclusions.get("quotes", True))
    exclude_bibliography = bool(exclusions.get("bibliography", True))
    by_source: dict[str, int] = {}
    for match in matches:
        by_source[match.source_id] = by_source.get(match.source_id, 0) + (match.query_span[1] - match.query_span[0])
    total = max(1, sum(by_source.values()))
    keep: list[PlagMatch] = []
    for match in matches:
        if match.words < min_words:
            continue
        if exclude_quotes and match.in_quote:
            continue
        if exclude_bibliography and match.in_bibliography:
            continue
        share = 100.0 * by_source[match.source_id] / total
        if share < hide_below:
            continue
        keep.append(match)
    return keep
