"""FastAPI application entry point for Veritas.

Run locally with ``uvicorn app.main:app --reload`` from ``backend/``.  In production
the same app runs on the Hugging Face Space (full backend) while the Vercel ``api/``
functions serve a torch-free ``lite`` fallback.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db
from .api import analyses, corpus, health, metrics
from .artifacts import registry
from .config import DISCLAIMER, settings
from .logging_setup import get_logger
from .schemas import ErrorBody, ErrorResponse

log = get_logger(__name__)


def _warm_artifacts() -> None:
    """Load artifacts in the background so ``/health`` answers immediately."""
    try:
        registry.load()
        log.info("artifact warm-up finished (source=%s)", registry.source)
    except Exception as exc:  # noqa: BLE001 - never kill the server
        log.exception("artifact warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201 - FastAPI lifespan signature
    """Start background loading and purge expired analyses."""
    thread = threading.Thread(target=_warm_artifacts, name="artifact-warmup", daemon=True)
    thread.start()
    try:
        removed = db.purge_expired()
        if removed:
            log.info("purged %s analyses past the retention window", removed)
    except Exception as exc:  # noqa: BLE001
        log.warning("retention purge failed: %s", exc)
    yield


app = FastAPI(
    title="Veritas API",
    version="1.0.0",
    description=(
        "AI-text detection and plagiarism intelligence. "
        "AI-text detection is probabilistic and must never be the sole basis for an accusation."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):  # noqa: ANN001, ANN201
    """Add the security headers the spec requires and log slow requests."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=()")
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a stack trace to a client; log it and return a clean error."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorBody(code="internal_error", message="Something went wrong on the server")
        ).model_dump(),
    )


app.include_router(health.router, prefix="/api")
app.include_router(analyses.router, prefix="/api")
app.include_router(corpus.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    """A tiny landing document so a bare visit is not a 404."""
    return {
        "name": "Veritas API",
        "docs": "/docs",
        "health": "/api/health",
        "disclaimer": DISCLAIMER,
    }
