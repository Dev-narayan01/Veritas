"""Shared Lite helpers for the Vercel functions.

This file exists so ``_lib`` is a regular package rather than a namespace package.
Vercel's Python builder traces the import graph to decide which files to bundle into
each function; a directory without ``__init__.py`` is not reliably followed, so
``from _lib.core import ...`` failed at runtime with an ImportError even though the
file was in the repository.
"""
