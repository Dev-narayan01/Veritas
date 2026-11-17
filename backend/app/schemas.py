"""Pydantic v2 request/response models for the Veritas API.

These schemas are the contract between the backend and the React client.  The AI
result block deliberately exposes *which* components ran and which did not: a
mode must never be presented as available when a component was skipped.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Verdict = Literal["likely_human", "uncertain", "likely_ai", "too_short"]
MatchType = Literal["exact", "near", "paraphrase", "online"]


class Exclusions(BaseModel):
    """Client-side exclusion filters for the similarity report."""

    quotes: bool = Field(True, description="Exclude quoted passages from the score")
    bibliography: bool = Field(True, description="Exclude the bibliography section")
    min_words: int = Field(8, ge=1, le=200, description="Smallest match to report")
    hide_below_percent: float = Field(0.5, ge=0.0, le=100.0)


class AnalysisOptions(BaseModel):
    """Everything the user chose on the Analyze page."""

    mode: Literal["lite", "fast", "standard", "deep"] | None = None
    ai: bool = True
    plagiarism: bool = True
    batch: bool = False
    explain: bool = True
    online: bool = False
    exclusions: Exclusions = Field(default_factory=Exclusions)
    retain: bool = True


class AnalyzeResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"] = "queued"


class ProgressEvent(BaseModel):
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    message: str = ""


class ComponentScore(BaseModel):
    name: str
    probability: float | None = None
    raw: dict[str, float] = Field(default_factory=dict)
    available: bool = True
    reason: str | None = None


class SentenceScore(BaseModel):
    index: int
    start: int
    end: int
    text: str
    ai_prob: float
    flagged_tokens: list[int] = Field(default_factory=list)


class Segment(BaseModel):
    start: int
    end: int
    label: Literal["human", "ai"]
    confidence: float


class TokenScore(BaseModel):
    token: str
    start: int
    end: int
    surprisal: float
    rank: int | None = None
    flagged: bool = False


class Explanation(BaseModel):
    source: Literal["rule_based", "local_llm"]
    text: str
    factors: list[dict[str, Any]] = Field(default_factory=list)


class AIResult(BaseModel):
    mode_requested: str
    mode_used: str | None = None
    components_run: list[str] = Field(default_factory=list)
    components_unavailable: list[dict[str, str]] = Field(default_factory=list)
    probability: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    verdict: Verdict = "too_short"
    thresholds: dict[str, float | None] = Field(default_factory=dict)
    band_collapsed: bool = False
    word_count: int = 0
    low_confidence: bool = False
    pass_count: int = 0
    components: list[ComponentScore] = Field(default_factory=list)
    sentences: list[SentenceScore] = Field(default_factory=list)
    tokens: list[TokenScore] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    ai_fraction: float | None = None
    explanation: Explanation | None = None
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class SourceBreakdown(BaseModel):
    id: str
    title: str
    url: str | None = None
    color_index: int = 0
    percent: float = 0.0
    matches: int = 0


class PlagMatch(BaseModel):
    query_span: tuple[int, int]
    source_span: tuple[int, int] | None = None
    query_text: str
    source_text: str = ""
    source_id: str
    source_title: str
    source_url: str | None = None
    color_index: int = 0
    match_type: MatchType
    score: float
    words: int
    in_quote: bool = False
    in_bibliography: bool = False


class PlagiarismResult(BaseModel):
    similarity_percent: float = 0.0
    similarity_percent_filtered: float = 0.0
    lexical_percent: float = 0.0
    paraphrase_percent: float = 0.0
    online_percent: float = 0.0
    sources: list[SourceBreakdown] = Field(default_factory=list)
    matches: list[PlagMatch] = Field(default_factory=list)
    corpus_size: int = 0
    online_snippets_sent: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IntegrityFlag(BaseModel):
    kind: str
    count: int
    detail: str = ""
    examples: list[str] = Field(default_factory=list)


class IntegrityResult(BaseModel):
    flags: list[IntegrityFlag] = Field(default_factory=list)
    cleaned_chars: int = 0
    hidden_text_removed: int = 0


class BatchPair(BaseModel):
    a: str
    b: str
    percent_ab: float
    percent_ba: float
    segments: list[dict[str, Any]] = Field(default_factory=list)


class BatchResult(BaseModel):
    matrix: list[list[float]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    pairs: list[BatchPair] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class CodeMatch(BaseModel):
    a: str
    b: str
    similarity: float
    lines_a: list[int] = Field(default_factory=list)
    lines_b: list[int] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)


class DocumentResult(BaseModel):
    id: str
    filename: str
    kind: str
    words: int
    pages: int | None = None
    text: str = ""
    warnings: list[str] = Field(default_factory=list)
    integrity: IntegrityResult = Field(default_factory=IntegrityResult)
    ai: AIResult | None = None
    plagiarism: PlagiarismResult | None = None


class AnalysisResult(BaseModel):
    id: str
    created_at: str
    status: Literal["queued", "running", "done", "error"]
    options: AnalysisOptions
    documents: list[DocumentResult] = Field(default_factory=list)
    batch: BatchResult | None = None
    code_matches: list[CodeMatch] = Field(default_factory=list)
    error: str | None = None
    exit_notes: list[str] = Field(default_factory=list)
    disclaimer: str = ""


class AnalysisSummary(BaseModel):
    id: str
    created_at: str
    status: str
    filenames: list[str]
    mode: str | None = None
    verdicts: list[str] = Field(default_factory=list)
    similarity: float | None = None


class ModeAvailability(BaseModel):
    mode: str
    available: bool
    reason: str | None = None
    components: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "loading"]
    version: str
    modes: list[ModeAvailability]
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    corpus_size: int = 0
    ocr: bool = False
    llm: bool = False
    gpu: bool = False
    artifact_source: Literal["kaggle", "cpu", "none"] = "none"
    artifact_notes: list[str] = Field(default_factory=list)
    data_source: str = "unknown"
    default_mode: str = "standard"
    disclaimer: str = ""


class CorpusStats(BaseModel):
    documents: int
    passages: int
    articles: int = 0
    updated_at: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
