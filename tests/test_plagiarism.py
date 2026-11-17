"""Plagiarism internals: shingling, alignment, merge/exclusions, code similarity."""

from __future__ import annotations

from app.plagiarism import align, code_sim, merge, shingling
from app.schemas import PlagMatch
from app.text.exclusions import tag_sections


def test_winnowing_finds_the_shared_passage() -> None:
    shared = " ".join(f"shared{i}" for i in range(20))
    left = f"Prefix text here. {shared} Suffix text here."
    right = f"Completely different opening. {shared} A different ending."
    left_prints = shingling.winnow(left.split(), 5, 4)
    right_prints = shingling.winnow(right.split(), 5, 4)
    hashes_left = {print_.hash for print_ in left_prints}
    hashes_right = {print_.hash for print_ in right_prints}
    assert hashes_left & hashes_right, "identical passages must share winnowed hashes"


def test_containment_is_asymmetric() -> None:
    long_text = " ".join(f"w{i}" for i in range(300))
    short_text = " ".join(f"w{i}" for i in range(20, 140))
    source = {print_.hash for print_ in shingling.winnow(long_text.split(), 5, 4)}
    query = {print_.hash for print_ in shingling.winnow(short_text.split(), 5, 4)}
    assert shingling.containment(query, source) > 0.5
    assert shingling.containment(source, query) < shingling.containment(query, source)


def test_jaccard_is_symmetric_and_bounded() -> None:
    a = {1, 2, 3, 4}
    b = {3, 4, 5, 6}
    assert shingling.jaccard(a, b) == shingling.jaccard(b, a)
    assert 0.0 <= shingling.jaccard(a, b) <= 1.0
    assert shingling.jaccard(a, a) == 1.0


def test_align_returns_spans_inside_the_query() -> None:
    sentence = (
        "the quick brown fox jumps over the lazy dog while the tired farmer watches "
        "from the gate and the dog keeps running"
    )
    query = f"Completely unrelated opening sentence about weather. {sentence}"
    source = f"A different start about something else. {sentence}"
    matches = align.align(query, source, source_id="s1", min_words=6)
    assert matches, "an identical passage must produce a match"
    for match in matches:
        assert 0 <= match.query_start < match.query_end <= len(query)
        assert query[match.query_start : match.query_end].strip()
        assert match.source_id == "s1"
        assert match.words >= 6


def test_align_ignores_unrelated_text() -> None:
    query = "totally unrelated content about nothing at all in particular"
    source = "completely different material regarding other topics entirely"
    assert align.align(query, source, source_id="s1", min_words=6) == []


def test_merge_marks_quotes_and_bibliography() -> None:
    # The bibliography heading has to sit in the last 30% of the document for the tagger
    # to accept it, so the body is padded first — this mirrors a real paper.
    body = "This opening section discusses the background of the study in several sentences. " * 4
    text = (
        body
        + 'She wrote "A quotation that is long enough to be flagged by the engine because it runs '
        'on for several words and is attributed to somebody else entirely." The argument continues '
        'after the quotation with more of the author\'s own prose. References\nSmith, J. 2020. A Book.'
    )
    quote_span = (text.index('"') + 1, text.index('"') + 60)
    biblio_span = (text.index("Smith"), text.index("Smith") + 12)
    payload = merge.MergeInput(
        text=text,
        matches=[
            PlagMatch(
                query_span=quote_span,
                query_text=text[quote_span[0] : quote_span[1]],
                source_id="s1",
                source_title="Source one",
                match_type="exact",
                score=1.0,
                words=10,
            ),
            PlagMatch(
                query_span=biblio_span,
                query_text=text[biblio_span[0] : biblio_span[1]],
                source_id="s2",
                source_title="Source two",
                match_type="exact",
                score=1.0,
                words=3,
            ),
        ],
        corpus_size=10,
        sections=tag_sections(text),
    )
    result = merge.merge(payload, {"quotes": True, "bibliography": True, "min_words": 3})
    assert result.matches, "matches must survive merging"
    assert any(match.in_quote for match in result.matches)
    assert result.similarity_percent_filtered <= result.similarity_percent


def test_filter_matches_respects_every_exclusion() -> None:
    text = "x" * 400
    payload = merge.MergeInput(
        text=text,
        matches=[
            PlagMatch(query_span=(0, 100), query_text=text[0:100], source_id="s1", source_title="one", match_type="exact", score=1.0, words=20),
            PlagMatch(query_span=(100, 200), query_text=text[100:200], source_id="s2", source_title="two", match_type="exact", score=1.0, words=3),
        ],
        corpus_size=10,
    )
    result = merge.merge(payload, {"quotes": False, "bibliography": False, "min_words": 8})
    kept = merge.filter_matches(result.matches, {"quotes": False, "bibliography": False, "min_words": 8})
    assert all(match.words >= 8 for match in kept)


def test_code_similarity_ignores_comments_and_renaming() -> None:
    left = (
        "def total(values):\n"
        "    # add them up\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result = result + value\n"
        "    return result\n"
    )
    right = (
        "def summarise(items):\n"
        "    # different comment\n"
        "    output = 0\n"
        "    for item in items:\n"
        "        output = output + item\n"
        "    return output\n"
    )
    tokens_a, lines_a, funcs_a = code_sim._python_tokens(left)
    tokens_b, lines_b, funcs_b = code_sim._python_tokens(right)
    a = code_sim.CodeFile(name="a.py", language="python", text=left, tokens=tokens_a, token_lines=lines_a, functions=funcs_a)
    b = code_sim.CodeFile(name="b.py", language="python", text=right, tokens=tokens_b, token_lines=lines_b, functions=funcs_b)
    assert code_sim.similarity(a, b) > 0.5


def test_code_similarity_between_unrelated_code_is_low() -> None:
    left = "def f(rows):\n    return [row for row in rows if row]\n"
    right = "class Widget:\n    colour = 'blue'\n    def paint(self):\n        return self.colour\n"
    tokens_a, lines_a, funcs_a = code_sim._python_tokens(left)
    tokens_b, lines_b, funcs_b = code_sim._python_tokens(right)
    a = code_sim.CodeFile(name="a.py", language="python", text=left, tokens=tokens_a, token_lines=lines_a, functions=funcs_a)
    b = code_sim.CodeFile(name="b.py", language="python", text=right, tokens=tokens_b, token_lines=lines_b, functions=funcs_b)
    assert code_sim.similarity(a, b) <= 0.5
