"""ZIP handling with traversal, symlink and zip-bomb protection.

All members are extracted into memory and validated *before* anything is parsed:
no absolute paths, no ``..``, no symlinks, no nested archives, bounded member count
and bounded uncompressed size, plus a compression-ratio guard.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

from ..config import settings
from .base import ALL_EXT, ParseError, safe_name

# A member whose compressed size is tiny but expansion is huge is a bomb.
_MAX_COMPRESSION_RATIO = 100


@dataclass
class ArchiveMember:
    """One validated file from an uploaded archive."""

    name: str
    data: bytes
    suffix: str


def extract(data: bytes) -> tuple[list[ArchiveMember], list[str]]:
    """Validate and extract supported files; return (members, warnings)."""
    warnings: list[str] = []
    members: list[ArchiveMember] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ParseError("invalid_zip", "The uploaded file is not a valid ZIP archive.") from exc

    infos = archive.infolist()
    if len(infos) > settings.max_files_per_request:
        raise ParseError(
            "too_many_files",
            f"Archives may contain at most {settings.max_files_per_request} files.",
            413,
        )

    total_uncompressed = sum(max(0, info.file_size) for info in infos)
    if total_uncompressed > settings.max_zip_total_mb * 1024 * 1024:
        raise ParseError("zip_too_large", "Uncompressed archive is too large.", 413)

    for info in infos:
        name = info.filename.replace("\\", "/")
        if name.endswith("/"):
            continue  # directory entry
        if name.startswith("/") or ".." in name.split("/") or ":" in name.split("/")[0]:
            raise ParseError("unsafe_path", f"Unsafe path in archive: {name}")
        # Symlinks are stored with the Unix mode in the high bits.
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ParseError("unsafe_path", f"Symlinks are not allowed: {name}")
        safe = safe_name(name)
        suffix = "." + safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
        if suffix in {".zip", ".gz", ".tar", ".7z", ".rar"}:
            warnings.append(f"skipped nested archive {safe}")
            continue
        if suffix not in ALL_EXT:
            warnings.append(f"skipped unsupported file {safe}")
            continue
        if info.file_size > 0 and info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > _MAX_COMPRESSION_RATIO and info.file_size > 5 * 1024 * 1024:
                raise ParseError("zip_bomb", f"Suspicious compression ratio in {name}.", 413)
        if info.file_size > settings.max_upload_mb * 1024 * 1024:
            warnings.append(f"skipped oversized file {safe}")
            continue
        with archive.open(info) as handle:
            payload = handle.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(payload) > settings.max_upload_mb * 1024 * 1024:
            warnings.append(f"skipped oversized file {safe}")
            continue
        members.append(ArchiveMember(name=safe, data=payload, suffix=suffix))

    if not members:
        raise ParseError("empty_zip", "No supported documents were found in the archive.")
    return members, warnings
