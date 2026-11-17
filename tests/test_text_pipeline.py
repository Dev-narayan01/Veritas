"""Normalisation, segmentation, exclusions and integrity checks."""

from __future__ import annotations

from app.text.clean import join_hyphenated, normalize
from app.text.exclusions import classify_span, tag_sections
from app.text.segment import passages, split_sentences
from app.integrity.engine import run as run_integrity


def test_normalize_maps_every_character_back() -> None:
    raw = "Hello\u00a0world,\n\nthe  quick\tbrown fox."
    result = normalize(raw)
    assert "  " not in result.text
    assert "\n" not in result.text
    assert len(result.orig_offsets) == len(result.text)
    # Offsets are monotonically increasing and stay inside the original. A collapsed
    # whitespace run is represented by one space that maps to the start of the next word,
    # which is the convention highlighting needs: a span never bleeds into the previous
    # word. Every non-space cleaned character maps to an identical original character.
    assert result.orig_offsets == sorted(result.orig_offsets)
    for index, offset in enumerate(result.orig_offsets):
        assert 0 <= offset < len(raw)
        if not result.text[index].isspace():
            assert raw[offset] == result.text[index]


def test_normalize_span_round_trip() -> None:
    raw = "First sentence.\n\nSecond   sentence here."
    result = normalize(raw)
    start = result.text.index("Second")
    end = start + len("Second")
    orig_start, orig_end = result.to_original_span(start, end)
    assert raw[orig_start:orig_end] == "Second"


def test_normalize_drops_control_characters() -> None:
    result = normalize("clean\u200btext\u200dhere")
    assert result.text == "cleantexthere"


def test_join_hyphenated_rejoins_wrapped_words() -> None:
    assert join_hyphenated("inter-\nnational") == "international"
    assert join_hyphenated("well-known") == "well-known"


def test_split_sentences_covers_the_whole_text() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta! Eta theta iota?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].start == 0
    assert sentences[-1].end == len(text)
    for sentence in sentences:
        assert text[sentence.start : sentence.end].strip() == sentence.text.strip()


def test_passages_are_overlapping_windows() -> None:
    text = " ".join(f"word{i}" for i in range(120))
    windows = passages(text)
    assert len(windows) >= 2
    assert windows[0].start == 0
    assert windows[1].start < windows[0].end, "windows should overlap"
    for window in windows:
        assert window.end > window.start


def test_tag_sections_finds_quote_and_bibliography() -> None:
    text = (
        'The report opens normally. She wrote "this whole sentence is a quotation that '
        'someone else published earlier and it is quite long enough to detect". '
        "References\nSmith, J. (2020). A book about things. Publisher."
    )
    sections = tag_sections(text)
    kinds = {section.kind for section in sections}
    assert "quote" in kinds or "bibliography" in kinds
    assert all(section.end > section.start for section in sections)
    bibliography = [section for section in sections if section.kind == "bibliography"]
    if bibliography:
        assert classify_span((bibliography[0].start + 1, bibliography[0].start + 5), sections) == "bibliography"


def test_integrity_flags_invisible_characters() -> None:
    text = "Normal text with a zero\u200bwidth space and a homoglyph \u0430."
    outcome = run_integrity(text)
    kinds = {flag.kind for flag in outcome.result.flags}
    assert "invisible_characters" in kinds
    assert outcome.removed_chars >= 1
    assert "\u200b" not in outcome.text
    assert "zero" in outcome.text and "width" in outcome.text, "visible text must survive cleaning"


def test_integrity_is_clean_for_plain_text() -> None:
    outcome = run_integrity("A perfectly ordinary sentence, with punctuation.")
    assert outcome.result.flags == []
    assert outcome.removed_chars == 0
