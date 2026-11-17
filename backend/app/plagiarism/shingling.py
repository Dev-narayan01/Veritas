"""Shingling, winnowing and containment helpers shared by the plagiarism engine.

Everything here is deterministic across processes: shingles are hashed with
``xxhash`` rather than Python's salted ``hash()``, so a persisted index stays valid
after a restart.  Positions are tracked in *word* units, which lets the aligner
chain seeds without re-tokenising the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import xxhash

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens without punctuation."""
    return [w.lower() for w in _WORD.findall(text)]


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the word tokens, aligned with :func:`tokenize_words`."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def hash_token(token: str) -> int:
    """Stable 64-bit hash of a token."""
    return xxhash.xxh64(token.encode("utf-8")).intdigest()


def shingles(words: list[str], k: int) -> list[tuple[int, int]]:
    """Return (shingle_hash, start_word_index) for every k-word window."""
    if len(words) < k:
        return []
    out: list[tuple[int, int]] = []
    for start in range(len(words) - k + 1):
        joined = " ".join(words[start : start + k])
        out.append((xxhash.xxh64(joined.encode("utf-8")).intdigest(), start))
    return out


def shingle_set(words: list[str], k: int) -> set[int]:
    """Just the hashes, for containment and Jaccard calculations."""
    return {h for h, _ in shingles(words, k)}


def containment(query: set[int], source: set[int]) -> float:
    """Share of the query's shingles that also occur in the source."""
    if not query:
        return 0.0
    return len(query & source) / len(query)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class Fingerprint:
    """One winnowed fingerprint and where it occurs."""

    hash: int
    position: int


def winnow(words: list[str], k: int, window: int) -> list[Fingerprint]:
    """Winnowing (Schleimer et al., 2003): keep the minimum hash per window.

    Winnowing makes the fingerprint set robust to insertions and deletions while
    guaranteeing that any shared substring longer than ``k + window - 1`` words
    produces at least one shared fingerprint.
    """
    hashed = shingles(words, k)
    if not hashed:
        return []
    out: list[Fingerprint] = []
    last_emitted: int | None = None
    for start in range(max(1, len(hashed) - window + 1)):
        chunk = hashed[start : start + window]
        minimum = min(chunk, key=lambda item: item[0])
        if minimum[0] != last_emitted:
            out.append(Fingerprint(hash=minimum[0], position=minimum[1]))
            last_emitted = minimum[0]
    return out


def shingle_containment_text(query: str, source: str, k: int) -> float:
    """Containment of ``query`` shingles inside ``source`` (both plain text)."""
    return containment(
        shingle_set(tokenize_words(query), k), shingle_set(tokenize_words(source), k)
    )
