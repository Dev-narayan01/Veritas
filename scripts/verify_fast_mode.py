#!/usr/bin/env python3
"""Prove the deployed perplexity port matches the trained implementation.

The Vercel functions cannot import ``app.detector.perplexity`` (a function ships as one
file), so ``api/_lib/core.py`` carries a hand-written port of the n-gram features and the
style features. A port that is "close" is worse than useless here: the stacker consumes
these numbers as logits, so a silent drift would quietly change every verdict.

This replays real held-out rows through both implementations and fails on any
disagreement beyond floating-point noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "api"))

TOL = 1e-9


def main() -> int:
    import joblib
    import pandas as pd

    from app.detector.perplexity import PerplexityArtifact  # the trained implementation
    from _lib.core import _PplBundle, _load_ppl

    artifact: PerplexityArtifact = joblib.load(REPO / "ml" / "artifacts" / "ppl_gpt2_model.pkl")
    # Load through the runtime's own loader rather than re-implementing it here. The loader
    # is what ships, and a divergent copy is how the ``_lib.core`` pickle reference got
    # past verification once already: this script unpickled a ``_PplBundle`` happily while
    # the deployed single-file function could not.
    bundle: _PplBundle | None = _load_ppl(REPO / "api" / "artifacts")
    if bundle is None:
        print("perplexity bundle not present or not loadable", file=sys.stderr)
        return 3

    print(f"bundle: order={bundle.order} vocab={bundle.vocab_size} total={bundle.total_tokens:,}")

    frame = pd.read_parquet(REPO / "data" / "processed" / "test.parquet").sample(n=60, random_state=11)
    worst_feature = 0.0
    worst_prob = 0.0
    failures: list[str] = []

    for _, row in frame.iterrows():
        text = row["text"]
        expected = artifact.features(text)
        actual = bundle.features(text)
        for name, value in expected.items():
            got = actual.get(name)
            if got is None:
                failures.append(f"{name}: missing from port")
                continue
            worst_feature = max(worst_feature, abs(float(value) - float(got)))

        p_expected = float(artifact.predict_proba([text])[0])
        p_actual = float(bundle.predict_proba([text])[0])
        worst_prob = max(worst_prob, abs(p_expected - p_actual))

    print(f"rows checked           : {len(frame)}")
    print(f"worst feature delta    : {worst_feature:.3e}")
    print(f"worst probability delta: {worst_prob:.3e}")
    if failures:
        for line in failures[:10]:
            print("  MISSING:", line)

    ok = not failures and worst_feature < 1e-6 and worst_prob < 1e-6
    print("RESULT:", "IDENTICAL" if ok else "DIVERGED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
