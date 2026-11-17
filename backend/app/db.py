"""SQLite persistence for analyses and progress events.

Two tables only: ``analyses`` (the JSON result, status, retention flags) and
``events`` (the SSE stage log, kept so a client that connects late still sees the
history).  SQLAlchemy 2 is used for portability, but nothing here needs a server.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AnalysisRow(Base):
    """One analysis request and its result."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    filenames: Mapped[list] = mapped_column(JSON, default=list)
    verdicts: Mapped[list] = mapped_column(JSON, default=list)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retain: Mapped[bool] = mapped_column(Boolean, default=True)
    artifacts_purged: Mapped[bool] = mapped_column(Boolean, default=False)


class EventRow(Base):
    """One progress event of one analysis."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(36), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_engine = None
_Session: sessionmaker[Session] | None = None


def get_engine():  # noqa: ANN201 - SQLAlchemy engine type
    """Create the engine (and the data directory) on first use."""
    global _engine, _Session
    if _engine is None:
        settings.data_path.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False}
            if settings.database_url.startswith("sqlite")
            else {},
            future=True,
        )
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def session() -> Session:
    """A short-lived session; callers use it as a context manager."""
    get_engine()
    assert _Session is not None
    return _Session()


def add_event(analysis_id: str, stage: str, progress: float, message: str = "") -> None:
    """Append a progress event, numbering it monotonically per analysis."""
    with session() as db:
        last = db.execute(
            select(EventRow.seq)
            .where(EventRow.analysis_id == analysis_id)
            .order_by(EventRow.seq.desc())
            .limit(1)
        ).scalar()
        seq = (last or 0) + 1
        db.add(
            EventRow(
                analysis_id=analysis_id,
                seq=seq,
                payload={"stage": stage, "progress": round(progress, 4), "message": message},
            )
        )
        db.commit()


def get_events(analysis_id: str, after: int = 0) -> list[dict[str, Any]]:
    with session() as db:
        rows = db.execute(
            select(EventRow)
            .where(EventRow.analysis_id == analysis_id, EventRow.seq > after)
            .order_by(EventRow.seq)
        ).scalars().all()
        return [
            {"seq": row.seq, "created_at": row.created_at.isoformat(), **row.payload} for row in rows
        ]


def purge_expired() -> int:
    """Delete analyses older than ``RETENTION_DAYS`` (called on startup)."""
    cutoff = utcnow() - timedelta(days=settings.retention_days)
    with session() as db:
        rows = db.execute(select(AnalysisRow).where(AnalysisRow.created_at < cutoff)).scalars().all()
        for row in rows:
            db.delete(row)
        db.execute(delete(EventRow).where(EventRow.created_at < cutoff))
        db.commit()
        return len(rows)


def delete_analysis(analysis_id: str) -> None:
    with session() as db:
        db.execute(delete(EventRow).where(EventRow.analysis_id == analysis_id))
        row = db.get(AnalysisRow, analysis_id)
        if row is not None:
            db.delete(row)
        db.commit()


def dump_json(value: Any) -> str:
    """Compact JSON for logging and exports (never used for document HTML)."""
    return json.dumps(value, ensure_ascii=False)
