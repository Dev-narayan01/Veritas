"""Progress publishing for analyses (SSE with a polling fallback).

Progress is *real*: every call to :meth:`Progress.stage` corresponds to a stage the
worker actually reached, and ``progress`` is computed from completed stages, never
from a timer.  Events are persisted so a client that connects late, or polls instead
of using SSE, sees the same sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import db
from .logging_setup import get_logger

log = get_logger(__name__)

# Ordered pipeline stages and the fraction of the bar each completes.
STAGE_ORDER: tuple[str, ...] = ("parse", "integrity", "ai", "plagiarism", "batch", "report")
_STAGE_WEIGHT = 1.0 / len(STAGE_ORDER)


@dataclass
class Progress:
    """Writes progress events for one analysis."""

    analysis_id: str
    completed: set[str] = field(default_factory=set)

    def stage(self, stage: str, message: str = "", fraction: float = 0.0) -> None:
        """Mark a stage as started ``fraction`` of the way through."""
        if stage not in STAGE_ORDER:
            log.warning("unknown progress stage %s", stage)
            return
        index = STAGE_ORDER.index(stage)
        base = index * _STAGE_WEIGHT
        value = min(1.0, base + _STAGE_WEIGHT * max(0.0, min(1.0, fraction)))
        db.add_event(self.analysis_id, stage, value, message)

    def done(self, stage: str) -> None:
        self.completed.add(stage)
        index = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER) - 1
        value = min(1.0, (index + 1) * _STAGE_WEIGHT)
        db.add_event(self.analysis_id, stage, value, f"{stage} complete")

    def finish(self, message: str = "analysis complete") -> None:
        db.add_event(self.analysis_id, "done", 1.0, message)

    def fail(self, message: str) -> None:
        db.add_event(self.analysis_id, "error", 1.0, message)
