"""Logging bootstrap shared by the API server and the ``ml`` command-line scripts."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, in a single human-readable format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.environ.get("VERITAS_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved, logging.INFO))
    root.addHandler(handler)
    # Third-party loggers are noisy at INFO; keep them at WARNING unless debugging.
    for noisy in ("urllib3", "matplotlib", "PIL", "nltk", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)
