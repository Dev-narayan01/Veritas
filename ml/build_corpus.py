"""Build the plagiarism reference corpus index.

Sources, in order:

1. files under ``data/corpus/**`` already supplied by the user;
2. the first ``CORPUS_ARTICLES`` articles of simple Wikipedia (streamed from the Hugging
   Face Hub when it is reachable; otherwise extracted from ``data/corpus/wiki/*.jsonl``
   if such a dump was supplied);
3. the 30 bundled sample documents (always kept, so the engine has something honest to
   compare against offline).

Everything is written under ``ml/artifacts/corpus/``: ``meta.json`` (documents,
passages, document frequency), ``lsh.pkl`` (MinHash LSH), ``embeddings.npy`` and
``embeddings_ids.json`` (dense index when a sentence-transformer is available).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import ARTIFACT_DIR, DATA_DIR, REPO_ROOT, RunConfig, log, seed_everything, timer, write_json

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.config import settings  # noqa: E402
from app.plagiarism.corpus import PlagiarismCorpus  # noqa: E402
from app.text.clean import normalize  # noqa: E402
from app.text.exclusions import strip_bibliography  # noqa: E402

BUNDLED = REPO_ROOT / "ml" / "bundled" / "corpus"


def _add_file_corpus(corpus: PlagiarismCorpus, directory: Path) -> int:
    if not directory.exists():
        return 0
    added = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = strip_bibliography(normalize(path.read_text(errors="replace")).text)
        if len(text.split()) < 60:
            continue
        corpus.add_document(title=path.stem.replace("_", " ")[:120], text=text,
                            url=None, source=directory.name)
        added += 1
    return added


def _add_wikipedia(corpus: PlagiarismCorpus, limit: int) -> int:
    """Stream simple Wikipedia articles when the Hub is reachable."""
    try:  # pragma: no cover - needs the Hub
        from datasets import load_dataset

        dataset = load_dataset(
            "wikimedia/wikipedia", "20231101.simple", split="train", streaming=True
        )
        added = 0
        for row in dataset:
            text = strip_bibliography(normalize(str(row.get("text", ""))[:8000]).text)
            if len(text.split()) < 80:
                continue
            corpus.add_document(
                title=str(row.get("title", "Untitled"))[:120], text=text,
                url=f"https://simple.wikipedia.org/wiki/{str(row.get('title', '')).replace(' ', '_')}",
                source="wikipedia",
            )
            added += 1
            if added >= limit:
                break
        log.info("added %s Wikipedia articles", added)
        return added
    except Exception as exc:  # noqa: BLE001 - offline is the normal case
        dumped = DATA_DIR / "corpus" / "wiki"
        if dumped.exists():
            added = _add_file_corpus(corpus, dumped)
            log.info("added %s articles from the local Wikipedia dump", added)
            return added
        log.warning(
            "Wikipedia corpus unavailable (%s); the corpus will contain only the %s bundled "
            "sample documents and any files in data/corpus/. The UI states the real size.",
            exc, 30,
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything()

    target = ARTIFACT_DIR / "corpus"
    if (target / "meta.json").exists() and not args.force:
        log.info("corpus index already exists; use --force to rebuild")
        return 0

    with timer("build_corpus"):
        corpus = PlagiarismCorpus(target)
        corpus.documents = []
        corpus.passage_texts, corpus.passage_ids, corpus.passage_docs = [], [], []
        bundled = _add_file_corpus(corpus, BUNDLED) if BUNDLED.exists() else 0
        log.info("bundled sample documents: %s", bundled)
        file_corpus = _add_file_corpus(corpus, DATA_DIR / "corpus")
        log.info("user corpus documents: %s", file_corpus)
        limit = 300 if cfg.smoke else settings.corpus_articles
        _add_wikipedia(corpus, limit)
        corpus.save()

        embedder = None
        try:
            from app.plagiarism.semantic import Embedder

            embedder = Embedder()
            if not embedder.available:
                log.warning(
                    "no sentence-transformer available: the dense index is not built, so "
                    "paraphrase retrieval will report itself unavailable"
                )
                embedder = None
        except Exception as exc:  # noqa: BLE001
            log.warning("embedder unavailable: %s", exc)

        if embedder is not None and corpus.passage_texts:
            import numpy as np

            vectors = []
            batch = 256
            for start in range(0, len(corpus.passage_texts), batch):
                encoded = embedder.encode(corpus.passage_texts[start : start + batch])
                if encoded is None:
                    break
                vectors.append(encoded)
            if vectors:
                matrix = np.vstack(vectors).astype(np.float32)
                corpus._embeddings = matrix  # noqa: SLF001 - builder owns the index
                corpus.save()
                log.info("dense index: %s passages", matrix.shape[0])

        stats = corpus.stats()
        write_json(
            ARTIFACT_DIR / "corpus_stats.json",
            {"documents": stats.documents, "passages": stats.passages,
             "wikipedia_articles": stats.articles,
             "dense_index": corpus.embeddings is not None,
             "run": cfg.label},
        )
        log.info("corpus built: %s documents, %s passages", stats.documents, stats.passages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
