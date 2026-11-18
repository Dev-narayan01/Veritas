#!/usr/bin/env python3
"""Build the compact perplexity bundle the Vercel functions can actually ship.

Why a converter instead of shipping ``ppl_gpt2_model.pkl`` directly
-------------------------------------------------------------------
The trained artifact is ~26 MB and its classes live in ``app.detector.*``, which does not
exist inside a Vercel function (the builder ships one file per function and nothing else).
Unpickling it there also took ~9 s, which is most of the ``fast`` mode's 10 s budget.

So this script emits a *deployment* artifact:

* the trigram counts table becomes ``lm.json.gz`` (gzip, level 1 - a one-off cost, not a
  per-request one), loaded with ``json``/``gzip`` instead of pickle;
* the style features, the feature-name list and the classifier are pickled together with a
  ``_PplBundle`` class **defined in ``api/_lib/core.py``**, so unpickling never touches the
  application package.

Nothing is retrained or approximated: the classifier coefficients and the counts are the
ones ``ml/train_perplexity_model.py`` produced. ``scripts/verify_fast_mode.py`` replays the
same rows through both implementations and fails on any disagreement.

Usage:
    python scripts/build_ppl_bundle.py
"""

from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "ml" / "artifacts" / "ppl_gpt2_model.pkl"
OUT_DIR = REPO / "api" / "artifacts"
COUNTS = OUT_DIR / "ppl_lm.json.gz"
BUNDLE = OUT_DIR / "ppl_bundle.pkl"


def main() -> int:
    sys.path.insert(0, str(REPO / "backend"))
    sys.path.insert(0, str(REPO / "api"))

    if not SRC.exists():
        print(f"missing {SRC}: run ml/train_perplexity_model.py first", file=sys.stderr)
        return 1
    if OUT_DIR.exists() is False:
        OUT_DIR.mkdir(parents=True)

    import joblib

    from _lib.core import _PplBundle  # the class the function will unpickle into

    artifact = joblib.load(SRC)
    if getattr(artifact, "backend", None) != "ngram_lm":
        print(f"refusing to bundle backend={artifact.backend!r}: only ngram_lm is torch-free", file=sys.stderr)
        return 1
    lm = artifact.lm
    if lm is None:
        print("artifact carries no language model", file=sys.stderr)
        return 1

    # counts: keys are tuples of strings -> JSON needs lists. The order is preserved
    # because python dicts keep insertion order and JSON does the same.
    counts = {json.dumps(list(k)): dict(v) for k, v in lm.counts.items()}
    payload = {
        "order": lm.order,
        "vocab_size": lm.vocab_size,
        "total_tokens": lm.total_tokens,
        "counts": counts,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    with open(COUNTS, "wb") as handle:
        handle.write(gzip.compress(raw, 1))

    bundle = _PplBundle(
        classifier=artifact.classifier,
        feature_names=list(artifact.feature_names),
        style_reference=dict(getattr(artifact, "style_reference", {}) or {}),
        order=lm.order,
        vocab_size=lm.vocab_size,
        total_tokens=lm.total_tokens,
    )
    with open(BUNDLE, "wb") as handle:
        pickle.dump(bundle, handle, protocol=4)

    print(f"wrote {COUNTS.relative_to(REPO)} ({COUNTS.stat().st_size / 1048576:.1f} MB)")
    print(f"wrote {BUNDLE.relative_to(REPO)} ({BUNDLE.stat().st_size / 1024:.0f} KB)")
    print(f"  order={lm.order} vocab={lm.vocab_size} total_tokens={lm.total_tokens:,}")
    print(f"  features={len(artifact.feature_names)} classifier={type(artifact.classifier).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
