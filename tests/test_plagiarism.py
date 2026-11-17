"""Plagiarism internals: shingling, alignment, merge/exclusions, code similarity."""

from __future__ import annotations

from app.plagiarism import align, batch, code_sim, merge, shingling
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


# --------------------------------------------------------------------------- #
# Batch comparison (spec §19: "batch matrix, graph, code plagiarism ... work")
# --------------------------------------------------------------------------- #
# This path had no test and was broken in production: `batch.py` read `match.query_span`,
# an attribute the aligner never had, so every batch analysis ended in
# "internal_error: 'Match' object has no attribute 'query_span'".

BATCH_A = (
    "Library automation replaced paper catalogues with searchable records, and readers could "
    "find a book without walking to the shelf. Staff could finally see what a shelf held. " * 3
)
BATCH_B = (
    "A library catalogue is a description of a collection. Automation made that description "
    "queryable, which changed how collections are managed and how their gaps become visible. " * 3
)


def _batch_doc(name: str, text: str) -> batch.BatchDocument:
    return batch.BatchDocument(id=name, label=name, text=text, words=len(text.split()))


def test_match_exposes_spans_as_pairs() -> None:
    """The attribute batch comparison depends on must exist and stay consistent."""
    match = align.Match(
        query_start=10, query_end=40, source_start=5, source_end=35,
        match_type="exact", score=1.0, words=5, source_id="src",
    )
    assert match.query_span == (10, 40)
    assert match.source_span == (5, 35)
    assert match.query_span == (match.query_start, match.query_end)


def test_batch_compares_every_pair() -> None:
    documents = [_batch_doc("a", BATCH_A), _batch_doc("b", BATCH_A), _batch_doc("c", BATCH_B)]
    result = batch.compare(documents)
    assert len(result.matrix) == 3 and all(len(row) == 3 for row in result.matrix)
    # A document is 100% similar to itself and the diagonal is not a "finding".
    assert [result.matrix[i][i] for i in range(3)] == [100.0, 100.0, 100.0]
    labels = {(pair.a, pair.b) for pair in result.pairs}
    assert labels == {("a", "b"), ("a", "c"), ("b", "c")}


def test_batch_finds_identical_documents_and_not_unrelated_ones() -> None:
    documents = [_batch_doc("a", BATCH_A), _batch_doc("b", BATCH_A), _batch_doc("c", BATCH_B)]
    result = batch.compare(documents)
    pairs = {(pair.a, pair.b): pair for pair in result.pairs}
    assert pairs[("a", "b")].percent_ab > 90, "identical documents must score near 100%"
    assert pairs[("a", "b")].segments, "the matched segments are the evidence"
    assert pairs[("a", "b")].percent_ab > pairs[("a", "c")].percent_ab
    for segment in pairs[("a", "b")].segments:
        assert segment["query_span"][1] > segment["query_span"][0]
        assert segment["source_span"][1] > segment["source_span"][0]


def test_batch_edges_respect_the_threshold() -> None:
    documents = [_batch_doc("a", BATCH_A), _batch_doc("b", BATCH_A), _batch_doc("c", BATCH_B)]
    result = batch.compare(documents)
    edge_pairs = {(edge["a"], edge["b"]) for edge in result.edges}
    assert ("a", "b") in edge_pairs, "a near-identical pair must be an edge"
    for edge in result.edges:
        assert edge["weight"] >= 5, "below the 5% edge threshold nothing should be drawn"
    assert all(node["words"] > 0 for node in result.nodes)


def test_batch_handles_a_single_document() -> None:
    result = batch.compare([_batch_doc("solo", BATCH_A)])
    assert result.matrix == [[100.0]]
    assert result.pairs == []
    assert result.edges == []


def test_code_match_reports_renamed_function_pairs(tmp_path) -> None:
    """A renamed copy must say which name became which.

    The whole point of the renamed-variable case is that a reader cannot find the reported
    name in the copied file. Reporting only the original name would look like the copy
    kept it, so both sides are returned.
    """
    body = "\n".join(
        f"def task_{i}(items):\n"
        f"    running = 0\n"
        f"    for entry in items:\n"
        f"        if entry % {i + 2} == 0:\n"
        f"            running += entry * {i + 1}\n"
        f"    return running\n"
        for i in range(4)
    )
    original = tmp_path / "original.py"
    renamed = tmp_path / "renamed.py"
    original.write_text(body)
    renamed.write_text(body.replace("task_", "calc_"))

    matches = code_sim.compare_all([code_sim.load_code(original), code_sim.load_code(renamed)])
    assert matches, "a renamed copy must still be detected"
    match = matches[0]
    assert match.similarity > 0.9
    assert match.functions == ["task_0", "task_1", "task_2", "task_3"]
    assert match.function_pairs[0] == ("task_0", "calc_0")
    assert all(original_name != copy_name for original_name, copy_name in match.function_pairs)


def test_code_matches_become_graph_edges() -> None:
    """A renamed copy must not vanish from the similarity graph.

    The text aligner finds 0% overlap between two copies of a program whose identifiers were
    all renamed. Without this merge the graph shows two unrelated files while the code card
    reports a 100% match — a contradiction the reader has to resolve on their own.
    """
    result = batch.compare([_batch_doc("a.py", BATCH_A), _batch_doc("b.py", BATCH_B)])
    assert result.edges == [] or all(edge["kind"] == "text" for edge in result.edges)

    from app.schemas import CodeMatch

    match = CodeMatch(a="a.py", b="b.py", similarity=1.0)
    merged = batch.merge_code_edges(result, [match])
    kinds = {(edge["a"], edge["b"]): edge for edge in merged.edges}
    assert ("a.py", "b.py") in kinds
    assert kinds[("a.py", "b.py")]["kind"] == "code"
    assert kinds[("a.py", "b.py")]["weight"] == 100.0


def test_code_edges_do_not_duplicate_a_text_edge() -> None:
    from app.schemas import CodeMatch

    result = batch.compare([_batch_doc("a", BATCH_A), _batch_doc("b", BATCH_A)])
    before = len(result.edges)
    merged = batch.merge_code_edges(result, [CodeMatch(a="a", b="b", similarity=1.0)])
    assert len(merged.edges) == before, "a pair already drawn must not get a second edge"
