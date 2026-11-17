"""Opt-in online source check (Wikipedia, arXiv, OpenAlex).

Only 6-8 short, distinctive sentences are ever sent, and the response tells the
user exactly which sentences left their machine.  This is the single network path
that carries document text, and it is off unless the user ticks the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from ..logging_setup import get_logger
from ..schemas import PlagMatch
from ..text.segment import split_sentences
from .align import align
from .shingling import tokenize_words

log = get_logger(__name__)

MAX_SNIPPETS = 8
MAX_SNIPPET_CHARS = 200
USER_AGENT = "VeritasSourceCheck/1.0 (academic project; contact via repository)"
_TIMEOUT = 8


@dataclass
class FetchedSource:
    title: str
    url: str
    text: str
    provider: str


def _distinctive_sentences(text: str, sections: list, count: int = MAX_SNIPPETS) -> list[str]:
    """Pick long, information-dense sentences that are not quotes."""
    if sections:
        from ..text.exclusions import classify_span

        allowed = [
            s for s in split_sentences(text)
            if classify_span((s.start, s.end), sections) == "body"
        ]
    else:
        allowed = split_sentences(text)
    counts: dict[str, int] = {}
    for sentence in allowed:
        for word in set(tokenize_words(sentence.text)):
            counts[word] = counts.get(word, 0) + 1
    scored = []
    for sentence in allowed:
        if sentence.words < 6:
            continue
        words = set(tokenize_words(sentence.text))
        if not words:
            continue
        rarity = sum(1.0 / max(1, counts.get(w, 1)) for w in words) / len(words)
        scored.append((rarity * sentence.words, sentence.text))
    scored.sort(reverse=True)
    return [text[:MAX_SNIPPET_CHARS] for _, text in scored[:count]]


def _wikipedia(session: requests.Session, snippet: str) -> list[FetchedSource]:
    sources: list[FetchedSource] = []
    try:
        search = session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": snippet,
                "format": "json", "srlimit": 3,
            },
            timeout=_TIMEOUT,
        )
        search.raise_for_status()
        titles = [item["title"] for item in search.json()["query"]["search"]]
        if not titles:
            return []
        extracts = session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts", "explaintext": 1,
                "titles": "|".join(titles), "format": "json", "exlimit": 3,
            },
            timeout=_TIMEOUT,
        )
        extracts.raise_for_status()
        pages = extracts.json()["query"]["pages"]
        for page in pages.values():
            if page.get("extract"):
                sources.append(
                    FetchedSource(
                        title=page.get("title", ""),
                        url=f"https://en.wikipedia.org/wiki/{page.get('title', '').replace(' ', '_')}",
                        text=page["extract"],
                        provider="wikipedia",
                    )
                )
    except (requests.RequestException, KeyError, ValueError) as exc:
        log.warning("wikipedia lookup failed: %s", exc)
    return sources


def _arxiv(session: requests.Session, snippet: str) -> list[FetchedSource]:
    sources: list[FetchedSource] = []
    try:
        response = session.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{snippet[:120]}", "max_results": 3},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.text, "xml")
        for entry in soup.find_all("entry"):
            title = (entry.title.text or "").strip()
            summary = (entry.summary.text or "").strip()
            link = entry.id.text if entry.id else ""
            if summary:
                sources.append(
                    FetchedSource(title=title, url=link, text=summary, provider="arxiv")
                )
    except (requests.RequestException, AttributeError) as exc:
        log.warning("arxiv lookup failed: %s", exc)
    return sources


def _openalex(session: requests.Session, snippet: str) -> list[FetchedSource]:
    sources: list[FetchedSource] = []
    try:
        response = session.get(
            "https://api.openalex.org/works",
            params={"search": snippet[:120], "per-page": 3},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        for work in response.json().get("results", []):
            abstract = _rebuild_abstract(work.get("abstract_inverted_index"))
            if abstract:
                sources.append(
                    FetchedSource(
                        title=work.get("title") or "Untitled",
                        url=work.get("id", ""),
                        text=abstract,
                        provider="openalex",
                    )
                )
    except (requests.RequestException, KeyError, ValueError) as exc:
        log.warning("openalex lookup failed: %s", exc)
    return sources


def _rebuild_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, indices in inverted.items():
        for index in indices:
            positions[index] = word
    return " ".join(positions[i] for i in sorted(positions))


def check_online(text: str, sections: list) -> tuple[list[PlagMatch], list[str], list[str]]:
    """Query the three providers with a few sentences; return (matches, sent, warnings)."""
    warnings: list[str] = []
    snippets = _distinctive_sentences(text, sections)
    if not snippets:
        return [], [], ["online check skipped: the document has no distinctive sentences"]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    matches: list[PlagMatch] = []
    seen: set[str] = set()
    for snippet in snippets:
        for provider in (_wikipedia, _arxiv, _openalex):
            for source in provider(session, snippet):
                key = f"{source.provider}:{source.title}"
                if key in seen:
                    continue
                seen.add(key)
                for match in align(text, source.text, key, min_words=6):
                    matches.append(
                        PlagMatch(
                            query_span=(match.query_start, match.query_end),
                            source_span=(match.source_start, match.source_end),
                            query_text=text[match.query_start : match.query_end],
                            source_text=source.text[match.source_start : match.source_end],
                            source_id=key,
                            source_title=f"[{source.provider}] {source.title}"[:200],
                            source_url=source.url,
                            match_type="online",
                            score=round(match.score, 4),
                            words=match.words,
                        )
                    )
    if not matches:
        warnings.append("online check found no matching passages")
    return matches, snippets, warnings
