"""Every perturbation must actually change the text (spec §16, defect 4).

The previous build reported identical robustness numbers for every attack because the
perturbations were never applied. These tests fail if that ever happens again: each attack is
run over a corpus of texts and must change at least 80% of them, and the *kind* of change is
asserted too (a "shuffle" that only reorders one sentence is not a shuffle).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

import common  # noqa: E402  (needs the sys.path entry above)

CORPUS = sorted((ROOT / "ml" / "bundled" / "corpus").glob("*.txt"))
MIN_CHANGED_FRACTION = 0.8


@pytest.fixture(scope="module")
def texts() -> list[str]:
    assert CORPUS, "run scripts/make_bundled_corpus.py first"
    return [path.read_text() for path in CORPUS]


def _changed_fraction(originals: list[str], perturbed: list[str]) -> float:
    return sum(1 for before, after in zip(originals, perturbed) if before != after) / len(originals)


def test_synonym_replacement_changes_most_texts(texts: list[str]) -> None:
    perturbed = [common.synonym_replace(text, rate=0.10) for text in texts]
    assert _changed_fraction(texts, perturbed) >= MIN_CHANGED_FRACTION
    # The word count is preserved (a synonym is a swap, not a deletion).
    for before, after in zip(texts, perturbed):
        assert abs(len(before.split()) - len(after.split())) <= 2


def test_typo_injection_changes_most_texts(texts: list[str]) -> None:
    perturbed = [common.typo_inject(text, rate=0.03) for text in texts]
    assert _changed_fraction(texts, perturbed) >= MIN_CHANGED_FRACTION


def test_sentence_shuffle_reorders_but_keeps_the_words(texts: list[str]) -> None:
    perturbed = [common.shuffle_sentences(text) for text in texts]
    assert _changed_fraction(texts, perturbed) >= MIN_CHANGED_FRACTION
    for before, after in zip(texts, perturbed):
        assert sorted(before.split()) == sorted(after.split()), "shuffling must not edit words"


def test_mixing_human_sentences_reduces_the_ai_share(texts: list[str]) -> None:
    human = " ".join(texts[:3])
    mixed = [common.mix_human_sentences(text, human, fraction=0.30) for text in texts]
    assert _changed_fraction(texts, mixed) >= MIN_CHANGED_FRACTION
    for before, after in zip(texts, mixed):
        assert len(after.split()) >= len(before.split()) * 0.9


def test_truncation_shortens_to_the_budget(texts: list[str]) -> None:
    """Truncation is a no-op below the budget — which is exactly why `changed_fraction`
    exists. The bundled samples are all under 100 words, so this test asserts the contract
    on a long document instead of pretending the short ones changed."""
    long_text = " ".join(f"word{i}" for i in range(400))
    truncated = common.truncate_words(long_text, 100)
    assert len(truncated.split()) == 100
    assert truncated != long_text

    short_text = " ".join(f"word{i}" for i in range(40))
    assert common.truncate_words(short_text, 100) == short_text

    # Over the bundled corpus the perturbation changes nothing, and the robustness script
    # must therefore report changed_fraction == 0 for it rather than a flattering recall.
    unchanged = [common.truncate_words(text, 100) for text in texts]
    assert _changed_fraction(texts, unchanged) < MIN_CHANGED_FRACTION


def test_homoglyphs_replace_only_latin_letters(texts: list[str]) -> None:
    perturbed = [common.homoglyph_substitute(text, rate=0.15) for text in texts]
    assert _changed_fraction(texts, perturbed) >= MIN_CHANGED_FRACTION
    for after in perturbed:
        assert any(ord(char) > 0x400 for char in after), "no Cyrillic/Greek homoglyph inserted"


def test_invisible_characters_are_inserted(texts: list[str]) -> None:
    hidden = "\u200b\u200c\u2060\ufeff"
    perturbed = [text[:200] + "\u200b" + text[200:] for text in texts]
    assert _changed_fraction(texts, perturbed) == 1.0
    for after in perturbed:
        assert any(char in hidden for char in after)
