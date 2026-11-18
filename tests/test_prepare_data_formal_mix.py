"""The formal-text mix in split_records.

Background: with the default ``mix_formal=0.0`` every arXiv abstract goes to
``test_ood`` and none is trained on. A model fine-tuned that way scored 0.9885
balanced accuracy on the in-domain val split and a 99.0% human false-positive rate on
``test_ood``, because formal academic prose was entirely outside its training
distribution and it had learned "formal implies AI".

These tests pin the three properties the fix depends on:

1. ``mix_formal=0.0`` still behaves exactly as before (no regression, no surprise).
2. A positive value actually moves rows into the in-domain splits.
3. The moved-in rows and the held-out rows never overlap, so no document is in both
   train and test.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
# prepare_data does `from common import ...`, i.e. it expects ml/ on the path as a
# script directory, so both entries are needed when importing it as ml.prepare_data.
sys.path.insert(0, str(_ROOT / "ml"))

from common import RunConfig  # noqa: E402
from prepare_data import (  # noqa: E402
    OOD_DOMAIN,
    OOD_FORMAL_SOURCE,
    Record,
    split_records,
)


def _corpus(n_formal: int = 200, n_hc3: int = 400) -> list[Record]:
    """Enough rows that every split is non-empty after the 60/10/10/20 cut."""
    records: list[Record] = []
    for i in range(n_formal):
        text = f"arxiv abstract number {i} about a narrow technical subject"
        records.append(
            Record(
                id=f"arxiv-{i:04d}",
                text=text,
                label=0,
                domain="formal_academic",
                generator="human",
                source=OOD_FORMAL_SOURCE,
                group=f"arxiv-{i:04d}",
            )
        )
    for i in range(n_hc3):
        label = i % 2
        text = f"hc3 question {i} with answer {'generated' if label else 'written'}"
        records.append(
            Record(
                id=f"hc3-{i:04d}",
                text=text,
                label=label,
                domain=OOD_DOMAIN if i % 20 == 0 else "open_qa",
                generator="chatgpt" if label else "human",
                source="hc3",
                group=f"hc3-{i:04d}",
            )
        )
    return records


def _ids(rows: list[Record]) -> set[str]:
    return {r.id for r in rows}


def test_default_keeps_every_arxiv_row_out_of_training() -> None:
    """0.0 is the original behaviour and must not change silently."""
    splits = split_records(_corpus(), RunConfig(smoke=False))
    formal_in_train = [r for r in splits["train"] if r.source == OOD_FORMAL_SOURCE]
    assert formal_in_train == []
    in_ood = [r for r in splits["test_ood"] if r.source == OOD_FORMAL_SOURCE]
    assert len(in_ood) == 200


def test_mixing_moves_formal_rows_into_training() -> None:
    splits = split_records(_corpus(), RunConfig(smoke=False), mix_formal=0.5)
    formal_in_train = [r for r in splits["train"] if r.source == OOD_FORMAL_SOURCE]
    assert formal_in_train, "no arXiv rows reached train, so the mix did nothing"


def test_held_out_and_mixed_in_formal_rows_are_disjoint() -> None:
    """No abstract may be both trained on and tested on."""
    splits = split_records(_corpus(), RunConfig(smoke=False), mix_formal=0.5)
    in_train = _ids([r for r in splits["train"] if r.source == OOD_FORMAL_SOURCE])
    in_ood = _ids([r for r in splits["test_ood"] if r.source == OOD_FORMAL_SOURCE])
    assert in_train and in_ood, "both sides should be non-empty at 0.5"
    assert not (in_train & in_ood), f"{len(in_train & in_ood)} abstracts leaked across"


def test_every_document_lands_in_exactly_one_split() -> None:
    corpus = _corpus()
    splits = split_records(corpus, RunConfig(smoke=False), mix_formal=0.5)
    seen: list[str] = []
    for rows in splits.values():
        seen.extend(_ids(rows))
    assert len(seen) == len(set(seen)), "a document appears in more than one split"
    assert set(seen) == _ids(corpus), "a document went missing"


def test_mixing_is_deterministic() -> None:
    """A rebuild must produce the same split, or results are not reproducible."""
    a = split_records(_corpus(), RunConfig(smoke=False), mix_formal=0.5)
    b = split_records(_corpus(), RunConfig(smoke=False), mix_formal=0.5)
    for name in a:
        assert _ids(a[name]) == _ids(b[name]), f"{name} differs between runs"


def test_extreme_values_do_not_crash() -> None:
    for value in (0.0, 1.0, 2.0, -1.0):
        splits = split_records(_corpus(), RunConfig(smoke=False), mix_formal=value)
        assert "train" in splits and "test_ood" in splits
