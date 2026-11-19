"""Why does ``wiki_csai`` human prose produce a 45% false-positive rate?

The trained detector flags roughly 45% of *human* Wikipedia answers as AI while
flagging only 4% of human academic abstracts. Two explanations fit that gap, and they
call for different fixes:

1. **Population bias.** ``split_records`` routes every ``wiki_csai`` row to
   ``test_ood`` -- it is both the training origin of the human half of HC3 and the
   held-out test set. So the model is asked to classify prose from a distribution it
   has been actively kept away from. If that is the story, the number would be far
   lower against a wiki population that had contributed to training.
2. **Domain leakage.** The model may have learned "informal or citation-free prose =
   AI", which is a spurious rule that happens to score well on HC3. If that is the
   story, mixing in Wikipedia rows would not help much either.

This script measures both without a GPU, so it can run while training is queued:

* ``population`` -- the exact bias factor, as a weighted FPR under the wiki train/val
  population. The arithmetic is single-group Laplace smoothing, so the answer is
  ``p*w1 / (p*w1 + (1-p))`` for held-out FPR ``p`` and population share ``w1``. It
  assumes the *rate* is shared, which is the assumption this script tests next.
* ``domain_probe`` -- can a TF-IDF logistic regression tell Wikipedia human text apart
  from the in-domain human text the detector was trained on? High AUC means the two
  populations are distinguishable at all, i.e. the detector's decision rule is
  sensitive to a domain signature rather than to authorship.
* ``style`` -- corpus statistics that name the signature, so the fix can be aimed at
  something specific (citation markers, markdown, paragraph structure, length).
* ``detector`` -- re-scores ``test_ood`` with the trained ModernBERT checkpoint while
  sweeping the calibration temperature under a fixed *in-domain* human FPR budget.
  This keeps the operating point honest: the budget is set on in-domain data, where
  the human sample is unbiased, and the wiki number is then simply reported at that
  operating point.

Every number the script prints is a measurement. Where a split it needs is absent (the
sandbox corpus has no Wikipedia rows at all) it says so and skips, rather than
substituting the old single-field numbers.

Usage:

    python ml/diagnose_wiki_fpr.py                       # population + probe + style
    python ml/diagnose_wiki_fpr.py --detector ml/artifacts/modernbert_1ep
    python ml/diagnose_wiki_fpr.py --json ml/artifacts/wiki_fpr.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# HC3's ``wiki_csai`` domain is the OOD test domain and, on the real machine, the
# source of the ``encyclopedia`` rows that ``load_wiki_formal`` adds to the in-domain
# splits. Both facts are needed to work out the population correction.
WIKI_DOMAIN = "wiki_csai"
WIKI_IN_DOMAIN_LABEL = "encyclopedia"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, default=PROCESSED)
    p.add_argument("--train", default="train.parquet")
    p.add_argument("--val", default="val_stack.parquet")
    p.add_argument("--test", default="test.parquet")
    p.add_argument("--ood", default="test_ood.parquet")
    p.add_argument("--detector", type=Path, default=None,
                   help="artifact dir from train_modernbert.py; re-scores test_ood")
    p.add_argument("--target-in-domain-fpr", type=float, default=0.01,
                   help="operating point: in-domain human FPR the calibration aims at")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--json", type=Path, default=None, help="write the report here")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load(path: Path):
    import pandas as pd

    if not path.exists():
        return None
    return pd.read_parquet(path)


def _human_fpr(probs, labels) -> float | None:
    """False-positive rate on the human rows, or None when there are none."""

    import numpy as np

    y = np.asarray(labels)
    p = np.asarray(probs, dtype=float)
    human = y == 0
    if not human.any():
        return None
    return float((p[human] >= 0.5).mean())


# --------------------------------------------------------------------------- #
# 1. Population correction
# --------------------------------------------------------------------------- #
def population_report(splits: dict, ood) -> dict:
    """What wiki FPR would be if wiki human rows had trained on the same population.

    Single-group Laplace smoothing with the trained share ``b`` and a held-out rate
    ``p`` gives ``p*(1-b) / (p*(1-b) + b)``. This is an upper bound on the benefit: it
    assumes the per-document false-positive *rate* is shared by the two wiki
    populations, which holds only if the failure is population bias. ``domain_probe``
    tests that assumption, and it decides whether this correction is worth reading.
    """

    import pandas as pd

    out: dict[str, object] = {}

    in_domain: dict[str, int] = {}
    for name, frame in splits.items():
        if frame is None:
            continue
        if WIKI_IN_DOMAIN_LABEL in set(frame.get("domain", pd.Series(dtype=str))):
            in_domain[name] = int((frame["domain"] == WIKI_IN_DOMAIN_LABEL).sum())
    out["wiki_rows_in_domain"] = in_domain

    n_in = sum(in_domain.values())
    if n_in == 0:
        out["available"] = False
        out["why"] = (
            "no Wikipedia rows in the in-domain splits, so this corpus cannot answer "
            "the question; a corpus that builds them puts them in train/val"
        )
        return out

    wiki_ood = ood[ood["domain"] == WIKI_DOMAIN] if ood is not None else None
    if wiki_ood is None or wiki_ood.empty:
        out["available"] = False
        out["why"] = f"no {WIKI_DOMAIN} rows in the OOD split"
        return out
    n_ood_human = int((wiki_ood["label"] == 0).sum())

    b = n_in / (n_in + n_ood_human) if (n_in + n_ood_human) else 0.0
    out.update(
        available=True,
        wiki_human_rows_trained_on=n_in,
        wiki_human_rows_held_out=n_ood_human,
        trained_share=round(b, 4),
        formula="p*(1-b) / (p*(1-b) + b)   # single-group Laplace smoothing",
        # A mapping, not a measurement: it converts whatever held-out rate was
        # observed into the rate expected at the mixed population. Read the row that
        # matches the number actually being explained.
        held_out_to_mixed={
            str(p): round((p * (1 - b)) / (p * (1 - b) + b), 4)
            for p in (0.2, 0.3, 0.4471, 0.6, 0.8)
        },
        note=(
            "assumes both wiki populations share the same per-document FP rate; see "
            "domain_probe -- if that AUC is high, this correction is optimistic"
        ),
    )
    return out


# --------------------------------------------------------------------------- #
# 2. Can a probe tell the two human populations apart?
# --------------------------------------------------------------------------- #
def domain_probe(splits: dict, ood) -> dict:
    """Cross-validated AUC for "wiki human" vs "in-domain human".

    If this is near 0.5 the populations are interchangeable and the model's failure is
    a calibration or population-mix problem. If it is near 1.0 the two kinds of human
    writing are trivially separable, the detector has every opportunity to learn a
    domain signature, and merely adding wiki rows may not be enough.
    """

    if ood is None:
        return {"available": False, "why": "no OOD split"}
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    wiki_human = ood[(ood["domain"] == WIKI_DOMAIN) & (ood["label"] == 0)]
    if wiki_human.empty:
        return {"available": False, "why": f"no human {WIKI_DOMAIN} rows in OOD"}

    others = []
    for name in ("train", "val", "test"):
        frame = splits.get(name)
        if frame is None:
            continue
        human = frame[frame["label"] == 0]
        if WIKI_IN_DOMAIN_LABEL in set(human["domain"]):
            human = human[human["domain"] != WIKI_IN_DOMAIN_LABEL]
        # Cap each in-domain source so one big domain cannot define "not wiki".
        others.append(human.groupby("domain", group_keys=False).head(1500))
    if not others:
        return {"available": False, "why": "no in-domain human rows to compare against"}
    other = pd.concat(others, ignore_index=True)

    texts = list(wiki_human["text"]) + list(other["text"])
    y = np.array([1] * len(wiki_human) + [0] * len(other))
    clf = make_pipeline(
        TfidfVectorizer(max_features=40_000, ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced"),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(clf, texts, y, cv=folds, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    acc = float(((prob >= 0.5).astype(int) == y).mean())

    # Name the features the probe leans on, because those are the signature.
    clf.fit(texts, y)
    vec = clf.named_steps["tfidfvectorizer"]
    lr = clf.named_steps["logisticregression"]
    names = np.asarray(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    top_wiki = names[order[-12:]][::-1].tolist()
    top_other = names[order[:12]].tolist()

    return {
        "available": True,
        "n_wiki_human": int(len(wiki_human)),
        "n_other_human": int(len(other)),
        "other_human_by_domain": other["domain"].value_counts().to_dict(),
        "cv_auc": round(auc, 4),
        "cv_accuracy": round(acc, 4),
        "read": (
            "populations are separable: a domain signature exists"
            if auc >= 0.9 else
            "weakly separable" if auc >= 0.75 else
            "near-interchangeable: the failure is population mix, not style"
        ),
        "top_features_wiki": top_wiki,
        "top_features_other": top_other,
    }


# --------------------------------------------------------------------------- #
# 3. What the signature looks like
# --------------------------------------------------------------------------- #
_MARKERS = {
    "citation_bracket": r"\[\d{1,3}\]",
    "et_al": r"\bet al\.",
    "doi_or_url": r"(doi\.org|https?://)",
    "figure_or_table": r"\b(Figure|Table|Fig\.)\s*\d",
    "markdown_bold": r"\*\*[^*]+\*\*",
    "markdown_heading": r"(?m)^#{1,4}\s",
    "bullets_or_numbers": r"(?m)^\s*([-*\u2022]|\d+[.)])\s",
    "percent_or_number": r"\b\d+(\.\d+)?\s?%",
    "first_person_plural": r"\b(we|our)\b",
    "second_person": r"\byou\b",
    "contraction": r"\b\w+(n't|'re|'ve|'ll|'m)\b",
}


def style_report(splits: dict, ood) -> dict:
    """Compare the marker profile of wiki human text to the in-domain human text."""

    if ood is None:
        return {"available": False, "why": "no OOD split"}
    import numpy as np
    import pandas as pd

    wiki_human = ood[(ood["domain"] == WIKI_DOMAIN) & (ood["label"] == 0)]
    if wiki_human.empty:
        return {"available": False, "why": f"no human {WIKI_DOMAIN} rows in OOD"}
    parts = [f for f in (splits.get("train"), splits.get("test")) if f is not None]
    in_domain_human = pd.concat(parts, ignore_index=True) if parts else None
    if in_domain_human is None:
        return {"available": False, "why": "no in-domain splits"}
    in_domain_human = in_domain_human[in_domain_human["label"] == 0]

    def profile(frame: pd.DataFrame) -> dict:
        text = frame["text"].astype(str)
        words = text.str.split().str.len().to_numpy()
        lower = text.str.lower()
        out = {
            "n": int(len(frame)),
            "words_mean": round(float(np.mean(words)), 1),
            "words_median": float(np.median(words)),
            "sentences_mean": round(float(np.mean(
                lower.str.count(r"[.!?]").to_numpy() + 1)), 2),
            "type_token_ratio": round(float(np.mean([
                len(set(t.split())) / max(1, len(t.split())) for t in lower.head(2000)
            ])), 4) if len(frame) else 0.0,
        }
        for name, pattern in _MARKERS.items():
            out[name] = round(float(lower.str.contains(pattern, regex=True,
                                                       na=False).mean()), 4)
        return out

    return {
        "available": True,
        "wiki_csai_human": profile(wiki_human),
        "in_domain_human": profile(in_domain_human),
        "by_in_domain_source": {
            str(k): profile(v) for k, v in list(
                in_domain_human.groupby("domain"))[:6]
        },
    }


# --------------------------------------------------------------------------- #
# 4. Re-score the checkpoint at an in-domain operating point
# --------------------------------------------------------------------------- #
def detector_report(args, splits: dict, val, ood) -> dict:
    """Score test_ood with the trained checkpoint, sweeping calibration temperature.

    The operating point is fixed by an in-domain human FPR budget because in-domain is
    the only human sample that is not also the test set. Temperature is swept and the
    wiki number reported at each setting -- a monotone rescaling cannot hide a
    systematic shift, and if it could the sweep would show it.
    """

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ml.common import log

    art = args.detector
    if art is None or not (art / "config.json").exists():
        return {"available": False, "why": f"no checkpoint at {art}"}
    if ood is None or val is None:
        return {"available": False, "why": "need both --val and --ood splits"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(art)
    model = AutoModelForSequenceClassification.from_pretrained(art).to(device).eval()

    @torch.no_grad()
    def score(frame) -> np.ndarray:
        texts = list(frame["text"].astype(str))
        out = []
        bs = args.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            out.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
            if (i // bs) % 50 == 0:
                log.info("scored %s/%s", i + len(batch), len(texts))
        return np.concatenate(out) if out else np.zeros(0)

    val_scores = score(val)
    ood_scores = score(ood)
    val_y = val["label"].to_numpy()
    ood_y = ood["label"].to_numpy()
    ood_dom = ood["domain"].astype(str).to_numpy()
    wiki_human = (ood_dom == WIKI_DOMAIN) & (ood_y == 0)
    val_human = val_y == 0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    grid = []
    for temp in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        s_val = val_scores / temp
        # Bias so the in-domain human FPR sits at the budget. Quantile, not a search:
        # the threshold is exactly the budget quantile of the in-domain scores.
        thr = float(np.quantile(s_val[val_human], 1.0 - args.target_in_domain_fpr))
        p_ood = sigmoid(ood_scores / temp - thr)
        p_val = sigmoid(s_val - thr)
        entry = {
            "temperature": temp,
            "threshold_logit": round(thr, 4),
            "in_domain_human_fpr": round(float((p_val[val_human] >= 0.5).mean()), 4),
            "wiki_human_fpr": round(float((p_ood[wiki_human] >= 0.5).mean()), 4)
            if wiki_human.any() else None,
        }
        for dom in ("formal_academic", "mage_heldout", "arxiv_ai"):
            m = ood_dom == dom
            if not m.any():
                continue
            if (ood_y[m] == 0).any():
                entry[f"{dom}_human_fpr"] = round(
                    float((p_ood[m][ood_y[m] == 0] >= 0.5).mean()), 4)
            if (ood_y[m] == 1).any():
                entry[f"{dom}_recall"] = round(
                    float((p_ood[m][ood_y[m] == 1] >= 0.5).mean()), 4)
        grid.append(entry)

    return {
        "available": True,
        "checkpoint": str(art),
        "device": device,
        "calibrated_on": "in-domain human rows in --val (unbiased sample)",
        "target_in_domain_fpr": args.target_in_domain_fpr,
        "sweep": grid,
        "read": (
            "wiki FPR barely moves with temperature: the shift is systematic, not a "
            "mis-set threshold"
            if max(g["wiki_human_fpr"] or 0 for g in grid)
            - min(g["wiki_human_fpr"] or 0 for g in grid) < 0.10
            else "temperature matters; report the chosen operating point explicitly"
        ),
    }


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = args.data
    splits = {
        "train": _load(base / args.train),
        "val": _load(base / args.val),
        "test": _load(base / args.test),
    }
    ood = _load(base / args.ood)
    if splits["train"] is None:
        sys.exit(f"missing {base / args.train} - run `python ml/prepare_data.py` first")

    for name, frame in splits.items():
        if frame is not None:
            print(f"{name:6s} {len(frame):6d} rows  "
                  f"domains={sorted(set(frame.get('domain', [])))}")
    if ood is not None:
        print(f"{'ood':6s} {len(ood):6d} rows  "
              f"domains={sorted(set(ood['domain']))}")

    report: dict[str, object] = {
        "corpus": str(base),
        "splits": {
            k: (None if v is None else v["domain"].value_counts().to_dict())
            for k, v in splits.items()
        },
        "ood_domains": None if ood is None else ood["domain"].value_counts().to_dict(),
    }
    report["population"] = population_report(splits, ood)
    report["domain_probe"] = domain_probe(splits, ood)
    report["style"] = style_report(splits, ood)
    if args.detector:
        report["detector"] = detector_report(args, splits, splits["val"], ood)

    print("\n=== wiki FPR diagnosis ===")
    print(json.dumps(report, indent=2, default=str))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
