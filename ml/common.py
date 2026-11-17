"""Shared helpers for every ``ml`` script.

Each script is runnable on its own and logs its settings and runtime.  This module
holds the pieces they all need: paths, seeding, JSON artifacts, determinism, and the
perturbation functions (which live here so the unit tests and the robustness
evaluation exercise exactly the same code).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.logging_setup import get_logger  # noqa: E402

SEED = 42
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR = ARTIFACT_DIR / "plots"

log = get_logger(__name__)


def prepare_dirs() -> None:
    for path in (ARTIFACT_DIR, PROCESSED_DIR, PLOTS_DIR, DATA_DIR / "raw", DATA_DIR / "samples"):
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    """Fix every random source that could make a run irreproducible."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default))


def read_json(path: Path, default: object = None) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _default(value: object) -> object:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


@contextmanager
def timer(label: str):
    """Log the wall-clock duration of a step."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        log.info("%s finished in %.1fs", label, elapsed)


@dataclass
class RunConfig:
    """How large a run is allowed to be."""

    smoke: bool = False
    full: bool = False

    @property
    def classical_cap(self) -> int:
        return 500 if self.smoke else (60_000 if self.full else 20_000)

    @property
    def ppl_cap(self) -> int:
        return 300 if self.smoke else 10_000

    @property
    def transformer_cap(self) -> int:
        return 200 if self.smoke else 40_000

    @property
    def zeroshot_cap(self) -> int:
        return 150 if self.smoke else 4_000

    @property
    def ood_cap(self) -> int:
        return 150 if self.smoke else 6_000

    @property
    def label(self) -> str:
        return "smoke" if self.smoke else ("full" if self.full else "cpu")


def bootstrap_ci(values, statistic=None, n_boot: int = 1000, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for a scalar statistic.

    ``values`` is a 2-D array (rows to resample) or a 1-D array; ``statistic`` maps a
    resample to a scalar.  Returns (low, high) or (None, None) when undefined.
    """
    import numpy as np

    data = np.asarray(values)
    if data.size == 0:
        return None, None
    rng = np.random.RandomState(SEED)
    estimates = []
    for _ in range(n_boot):
        idx = rng.randint(0, data.shape[0], data.shape[0])
        try:
            if statistic is None:
                estimates.append(float(np.mean(data[idx])))
            else:
                value = statistic(data[idx])
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    estimates.append(float(value))
        except Exception:  # noqa: BLE001 - a resample may be degenerate
            continue
    if len(estimates) < 20:
        return None, None
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


# --------------------------------------------------------------------------- #
# Perturbations (shared with the unit tests)
# --------------------------------------------------------------------------- #
_WORDNET_SYNONYMS: dict[str, list[str]] | None = None

# A curated offline table so the synonym attack (and its unit test) works without
# downloading WordNet.  One or two unambiguous substitutes per entry.
_BUILTIN_SYNONYMS: dict[str, list[str]] = {
    "good": ["fine", "decent"], "bad": ["poor", "flawed"], "big": ["large", "sizeable"],
    "small": ["little", "minor"], "fast": ["quick", "rapid"], "slow": ["gradual"],
    "happy": ["glad", "content"], "sad": ["unhappy", "gloomy"], "important": ["significant", "key"],
    "difficult": ["hard", "tough"], "easy": ["simple", "straightforward"],
    "increase": ["raise", "grow"], "decrease": ["reduce", "lower"], "result": ["outcome", "finding"],
    "problem": ["issue", "difficulty"], "method": ["approach", "technique"],
    "study": ["research", "investigation"], "analysis": ["examination", "assessment"],
    "data": ["figures", "information"], "model": ["framework", "structure"],
    "system": ["framework", "arrangement"], "process": ["procedure", "operation"],
    "change": ["alter", "modify"], "create": ["build", "produce"], "use": ["employ", "apply"],
    "show": ["demonstrate", "reveal"], "find": ["discover", "identify"], "help": ["assist", "aid"],
    "make": ["produce", "construct"], "many": ["numerous", "several"], "people": ["individuals", "persons"],
    "different": ["distinct", "various"], "similar": ["comparable", "alike"],
    "possible": ["feasible", "achievable"], "clear": ["obvious", "evident"],
    "strong": ["powerful", "robust"], "weak": ["fragile", "feeble"], "start": ["begin", "commence"],
    "end": ["finish", "conclude"], "improve": ["enhance", "better"], "reduce": ["lessen", "cut"],
    "provide": ["offer", "supply"], "require": ["need", "demand"], "consider": ["regard", "weigh"],
    "allow": ["permit", "enable"], "ensure": ["guarantee", "secure"], "develop": ["build", "evolve"],
    "support": ["back", "sustain"], "explain": ["describe", "clarify"], "understand": ["comprehend", "grasp"],
    "begin": ["commence", "start"], "however": ["nevertheless"], "therefore": ["consequently"],
    "because": ["since"], "although": ["though"], "often": ["frequently"],
    "quickly": ["rapidly", "swiftly"], "very": ["extremely"], "also": ["additionally"],
    "say": ["state", "mention"], "said": ["stated", "noted"], "think": ["believe", "consider"],
    "example": ["instance", "illustration"], "reason": ["cause", "justification"],
    "effect": ["impact", "consequence"], "purpose": ["aim", "goal"], "part": ["section", "portion"],
    "whole": ["entire"], "main": ["primary", "principal"], "new": ["novel", "recent"],
    "old": ["former", "previous"], "large": ["substantial", "considerable"],
    "students": ["learners", "pupils"], "research": ["study", "inquiry"],
    "approach": ["method", "strategy"], "benefit": ["advantage", "gain"],
    "challenge": ["difficulty", "obstacle"], "technology": ["technique", "innovation"],
    "language": ["tongue", "speech"], "writing": ["composition", "prose"],
    "text": ["passage", "document"], "authors": ["writers", "creators"],
    "human": ["person", "individual"], "machine": ["computer", "device"],
    "artificial": ["synthetic", "simulated"], "intelligence": ["intellect", "reasoning"],
    "detection": ["identification", "discovery"], "evidence": ["proof", "indication"],
    "conclusion": ["finding", "deduction"], "opinion": ["view", "belief"],
    "strongly": ["firmly"], "clearly": ["evidently"], "mainly": ["chiefly"],
    "almost": ["nearly"], "enough": ["sufficient"], "several": ["various"],
    "whether": ["if"], "within": ["inside"], "across": ["throughout"],
}


def _synonym_table() -> dict[str, list[str]]:
    """WordNet synonym table, downloaded once; empty when unavailable."""
    global _WORDNET_SYNONYMS
    if _WORDNET_SYNONYMS is not None:
        return _WORDNET_SYNONYMS
    table: dict[str, list[str]] = {}
    try:  # pragma: no cover - needs the WordNet corpus
        import nltk
        from nltk.corpus import wordnet

        try:
            wordnet.ensure_loaded()
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            wordnet.ensure_loaded()
        for word in (
            "good bad big small fast slow happy sad important difficult easy "
            "increase decrease result problem method study analysis data model "
            "system process change create use show find help make many people "
            "different similar possible clear strong weak start end"
        ).split():
            synonyms = {lemma.name().replace("_", " ") for syn in wordnet.synsets(word)
                        for lemma in syn.lemmas()}
            synonyms.discard(word)
            if synonyms:
                table[word] = sorted(synonyms)[:6]
    except Exception as exc:  # noqa: BLE001 - corpus unavailable offline
        log.warning("WordNet unavailable (%s); synonym perturbation uses a small table", exc)
        table = _BUILTIN_SYNONYMS
    _WORDNET_SYNONYMS = table
    return table


def synonym_replace(text: str, rate: float = 0.10, seed: int = SEED) -> str:
    """Replace ~rate of *all* words with a synonym where one exists.

    The denominator is every word in the text (not just the words we happen to have a
    synonym for), which is what "20% synonym replacement" has to mean for the
    robustness comparison to be honest.  Words without a synonym are left alone.
    """
    rng = random.Random(seed)
    table = _synonym_table()
    words = text.split()
    if not words:
        return text
    # Every word gets a chance proportional to the rate, so the realised fraction is
    # close to the requested one instead of saturating on a small synonym table.
    for index, raw in enumerate(words):
        if rng.random() >= rate:
            continue
        key = raw.lower().strip(".,;:!?\"'")
        if key not in table:
            key = re.sub(r"(ing|ed|es|s|ly)$", "", key)
        if key not in table:
            continue
        replacement = rng.choice(table[key])
        if raw[:1].isupper():
            replacement = replacement.capitalize()
        words[index] = raw.replace(key, replacement.lower() if raw.islower() else replacement, 1)
    return " ".join(words)


def typo_inject(text: str, rate: float = 0.03, seed: int = SEED) -> str:
    """Introduce small character-level typos in ~rate of the words."""
    rng = random.Random(seed + 1)
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 4]
    rng.shuffle(candidates)
    for index in candidates[: int(len(candidates) * rate)]:
        word = words[index]
        pos = rng.randrange(1, len(word) - 1)
        operation = rng.choice(("swap", "delete", "double"))
        if operation == "swap":
            words[index] = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
        elif operation == "delete":
            words[index] = word[:pos] + word[pos + 1 :]
        else:
            words[index] = word[:pos] + word[pos] + word[pos:]
    return " ".join(words)


def shuffle_sentences(text: str, seed: int = SEED) -> str:
    """Shuffle sentence order deterministically."""
    from app.text.segment import split_sentences

    sentences = [s.text for s in split_sentences(text)]
    if len(sentences) < 3:
        sentences = [part for part in text.split(". ") if part.strip()]
    rng = random.Random(seed + 2)
    rng.shuffle(sentences)
    return " ".join(sentences)


def mix_human_sentences(ai_text: str, human_text: str, fraction: float = 0.30, seed: int = SEED) -> str:
    """Interleave human sentences into AI text until ``fraction`` of words are human."""
    from app.text.segment import split_sentences

    ai_sentences = [s.text for s in split_sentences(ai_text)]
    human_sentences = [s.text for s in split_sentences(human_text)]
    if not ai_sentences or not human_sentences:
        return ai_text
    total_words = sum(len(s.split()) for s in ai_sentences)
    needed = int(total_words * fraction / max(1e-6, 1 - fraction))
    chosen: list[str] = []
    rng = random.Random(seed + 3)
    rng.shuffle(human_sentences)
    words = 0
    for sentence in human_sentences:
        if words >= needed:
            break
        chosen.append(sentence)
        words += len(sentence.split())
    merged = list(ai_sentences)
    for sentence in chosen:
        merged.insert(rng.randrange(len(merged) + 1), sentence)
    return " ".join(merged)


def truncate_words(text: str, words: int = 100) -> str:
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words])


def homoglyph_substitute(text: str, rate: float = 0.15, seed: int = SEED) -> str:
    """Swap Latin letters for Cyrillic lookalikes inside ~rate of the long words."""
    from app.integrity.homoglyph import CONFUSABLES

    reverse = {v: k for k, v in CONFUSABLES.items() if len(v) == 1 and v.islower()}
    rng = random.Random(seed + 4)
    words = text.split()
    for index, word in enumerate(words):
        if len(word) > 3 and rng.random() < rate:
            chars = list(word)
            for pos, char in enumerate(chars):
                if char in reverse:
                    chars[pos] = reverse[char]
                    break
            words[index] = "".join(chars)
    return " ".join(words)


def insert_invisible(text: str, seed: int = SEED) -> str:
    """Insert zero-width spaces every few words (``seed`` kept for a uniform signature)."""
    del seed
    words = text.split()
    for index in range(0, len(words), max(1, len(words) // 25)):
        words[index] = words[index] + "\u200b"
    return " ".join(words)


PERTURBATIONS = {
    "synonym_10": lambda t, h: synonym_replace(t, 0.10),
    "synonym_20": lambda t, h: synonym_replace(t, 0.20),
    "typo_3": lambda t, h: typo_inject(t, 0.03),
    "shuffle_sentences": lambda t, h: shuffle_sentences(t),
    "mix_human_30": lambda t, h: mix_human_sentences(t, h, 0.30),
    "truncate_100": lambda t, h: truncate_words(t, 100),
    "homoglyph": lambda t, h: homoglyph_substitute(t),
    "invisible_chars": lambda t, h: insert_invisible(t),
}
