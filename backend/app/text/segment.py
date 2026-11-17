"""Sentence and passage segmentation over normalised text.

Sentences come from NLTK's Punkt tokenizer when the model is available locally;
otherwise a conservative regex splitter is used so the pipeline never depends on a
runtime download.  Passages are overlapping word windows used by the plagiarism
engine: window ``PASSAGE_WORDS`` words, advancing ``PASSAGE_STRIDE`` words, with
character offsets retained so a match can be highlighted precisely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "fig", "eq", "no", "vol", "pp", "al", "inc", "ltd", "co", "approx", "dept",
}

_punkt_ready = False
_TOKENIZER = None


def _load_punkt() -> None:
    """Try to load the Punkt model once; fall back to the regex splitter."""
    global _punkt_ready, _TOKENIZER
    if _punkt_ready:
        return
    _punkt_ready = True
    try:  # pragma: no cover - environment dependent
        import nltk

        try:
            _TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            nltk.download("punkt", quiet=True)
            _TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
        log.info("sentence tokenizer: nltk punkt")
    except Exception as exc:  # noqa: BLE001 - optional dependency, degrade gracefully
        _TOKENIZER = None
        log.warning("punkt unavailable (%s); using regex sentence splitter", exc)


@dataclass
class Sentence:
    """A sentence with its span in the cleaned text."""

    index: int
    text: str
    start: int
    end: int
    words: int


def _regex_split(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        candidate = text[start : match.start()]
        # Do not split after a known abbreviation ("Dr. Smith") or initials.
        tail = candidate.rstrip().rsplit(" ", 1)[-1].rstrip(".").lower()
        if tail in _ABBREVIATIONS or (len(tail) == 1 and tail.isalpha()):
            continue
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def split_sentences(text: str, min_chars: int = 1) -> list[Sentence]:
    """Return sentences with cleaned-text character offsets."""
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    _load_punkt()
    if _TOKENIZER is not None:
        try:  # pragma: no cover - depends on punkt availability
            offset = 0
            for sent in _TOKENIZER.tokenize(text):
                pos = text.find(sent, offset)
                if pos < 0:
                    continue
                spans.append((pos, pos + len(sent)))
                offset = pos + len(sent)
        except Exception as exc:  # noqa: BLE001
            log.warning("punkt failed (%s); falling back to regex", exc)
            spans = []
    if not spans:
        spans = _regex_split(text)

    out: list[Sentence] = []
    for start, end in spans:
        piece = text[start:end]
        if len(piece.strip()) < min_chars:
            continue
        lead = len(piece) - len(piece.lstrip())
        piece = piece.strip()
        out.append(
            Sentence(
                index=len(out),
                text=piece,
                start=start + lead,
                end=start + lead + len(piece),
                words=len(piece.split()),
            )
        )
    return out


@dataclass
class Passage:
    """An overlapping word window of the document."""

    index: int
    text: str
    start: int
    end: int
    words: int


def passages(text: str) -> list[Passage]:
    """Slice ``text`` into overlapping word windows with character offsets."""
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not tokens:
        return []
    size = settings.passage_words
    stride = settings.passage_stride
    out: list[Passage] = []
    pos = 0
    while pos < len(tokens):
        window = tokens[pos : pos + size]
        if len(window) < min(8, size) and out:
            break
        start, end = window[0][0], window[-1][1]
        out.append(
            Passage(
                index=len(out),
                text=text[start:end],
                start=start,
                end=end,
                words=len(window),
            )
        )
        if pos + size >= len(tokens):
            break
        pos += stride
    return out


def context_window(text: str, sentences: list[Sentence], idx: int) -> tuple[int, int]:
    """Character span of ``sentences[idx]`` plus one sentence on each side."""
    lo = max(0, idx - 1)
    hi = min(len(sentences), idx + 2)
    return sentences[lo].start, sentences[hi - 1].end
