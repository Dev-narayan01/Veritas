"""Central configuration for Veritas.

Every tunable number lives here (or in a tuned-artifact JSON produced by the ML
scripts) so that no magic constants are scattered through the code base.  Values
can be overridden with environment variables, which is how the Hugging Face
Space and the Vercel Lite functions are configured without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    # --- limits -----------------------------------------------------------
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 20)
    max_files_per_request: int = _env_int("MAX_FILES_PER_REQUEST", 50)
    max_zip_total_mb: int = _env_int("MAX_ZIP_TOTAL_MB", 100)
    max_chars_per_doc: int = _env_int("MAX_CHARS_PER_DOC", 250_000)
    rate_limit_per_min: int = _env_int("RATE_LIMIT_PER_MIN", 30)
    retention_days: int = _env_int("RETENTION_DAYS", 30)

    # --- detector ---------------------------------------------------------
    default_mode: str = _env_str("DEFAULT_MODE", "standard")
    min_words_for_detection: int = _env_int("MIN_WORDS_FOR_DETECTION", 50)
    low_confidence_words: int = _env_int("LOW_CONFIDENCE_WORDS", 150)
    target_human_fpr: float = _env_float("TARGET_HUMAN_FPR", 0.01)
    ai_miss_for_human_verdict: float = _env_float("AI_MISS_FOR_HUMAN_VERDICT", 0.05)

    ppl_model: str = _env_str("PPL_MODEL", "gpt2")
    binoculars_observer: str = _env_str("BINOCULARS_OBSERVER", "Qwen/Qwen2.5-0.5B")
    binoculars_performer: str = _env_str("BINOCULARS_PERFORMER", "Qwen/Qwen2.5-0.5B-Instruct")
    fastdetect_model: str = _env_str("FASTDETECT_MODEL", "Qwen/Qwen2.5-0.5B")
    transformer_model_gpu: str = _env_str("TRANSFORMER_MODEL_GPU", "roberta-base")
    transformer_model_cpu: str = _env_str("TRANSFORMER_MODEL_CPU", "distilroberta-base")
    transformer_max_len: int = _env_int("TRANSFORMER_MAX_LEN", 256)
    deep_cpu_max_tokens: int = _env_int("DEEP_CPU_MAX_TOKENS", 1024)

    # --- passages / plagiarism -------------------------------------------
    shingle_size: int = _env_int("SHINGLE_SIZE", 5)
    passage_words: int = _env_int("PASSAGE_WORDS", 60)
    passage_stride: int = _env_int("PASSAGE_STRIDE", 30)
    minhash_perms: int = _env_int("MINHASH_PERMS", 128)
    lsh_threshold: float = _env_float("LSH_THRESHOLD", 0.25)
    winnow_k: int = _env_int("WINNOW_K", 5)
    winnow_w: int = _env_int("WINNOW_W", 4)
    seed_gap_words: int = _env_int("SEED_GAP_WORDS", 12)
    min_match_words: int = _env_int("MIN_MATCH_WORDS", 8)
    lexical_min_containment: float = _env_float("LEXICAL_MIN_CONTAINMENT", 0.50)
    semantic_model: str = _env_str("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    semantic_min_cosine: float = _env_float("SEMANTIC_MIN_COSINE", 0.80)
    semantic_top_k: int = _env_int("SEMANTIC_TOP_K", 5)
    rerank_model: str = _env_str("RERANK_MODEL", "cross-encoder/quora-distilroberta-base")
    rerank_min_score: float = _env_float("RERANK_MIN_SCORE", 0.60)
    common_phrase_df: float = _env_float("COMMON_PHRASE_DF", 0.02)
    corpus_articles: int = _env_int("CORPUS_ARTICLES", 10_000)
    code_winnow_k: int = _env_int("CODE_WINNOW_K", 5)
    code_winnow_w: int = _env_int("CODE_WINNOW_W", 4)
    code_match_threshold: float = _env_float("CODE_MATCH_THRESHOLD", 0.50)
    hide_sources_below_percent: float = _env_float("HIDE_SOURCES_BELOW_PERCENT", 0.5)

    # --- artifacts / deployment ------------------------------------------
    hub_repo: str = _env_str("VERITAS_HUB_REPO", "")
    artifact_dir: str = _env_str("VERITAS_ARTIFACT_DIR", "ml/artifacts")
    hf_token: str = _env_str("HF_TOKEN", "")
    cors_origins: tuple[str, ...] = tuple(
        _env_list("CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"])
    )
    ollama_url: str = _env_str("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = _env_str("OLLAMA_MODEL", "llama3.2")
    debug: bool = _env_bool("VERITAS_DEBUG", False)
    data_dir: str = _env_str("VERITAS_DATA_DIR", "data")
    db_url: str = _env_str("VERITAS_DB_URL", "")

    # --- misc -------------------------------------------------------------
    max_url_bytes: int = _env_int("MAX_URL_BYTES", 5 * 1024 * 1024)
    url_timeout_s: int = _env_int("URL_TIMEOUT_S", 10)
    worker_poll_s: float = _env_float("WORKER_POLL_S", 0.05)

    @property
    def artifact_path(self) -> Path:
        """Absolute artifact directory (relative paths are resolved from the repo root)."""
        p = Path(self.artifact_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def uploads_path(self) -> Path:
        return self.data_path / "uploads"

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        return f"sqlite:///{(self.data_path / 'veritas.sqlite3').as_posix()}"

    @property
    def work_dir(self) -> Path:
        """Scratch directory for runtime caches (URL fetches, hub snapshots)."""
        return self.data_path / "cache"


settings = Settings()

DISCLAIMER = (
    "AI-text detection is probabilistic and must never be the sole basis for an accusation."
)

# The four detector operating modes, ordered from cheapest to most expensive.
MODES: tuple[str, ...] = ("lite", "fast", "standard", "deep")
MODE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "lite": ("classical",),
    "fast": ("classical", "ppl_gpt2"),
    "standard": ("classical", "ppl_gpt2", "transformer"),
    "deep": ("classical", "ppl_gpt2", "transformer", "binoculars", "fastdetect"),
}
# Components that need torch/transformers weights; used for availability checks.
NEURAL_COMPONENTS: frozenset[str] = frozenset({"transformer", "binoculars", "fastdetect"})

VERDICT_COLORS: dict[str, str] = {
    "likely_human": "#34D399",
    "uncertain": "#FBBF24",
    "likely_ai": "#F87171",
    "too_short": "#8B96AD",
}
