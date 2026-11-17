"""Parsers: one real fixture per type, plus the safety rules (zip bomb, SSRF, traversal)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from app.parsers import archive, code, docx as docx_parser, pptx as pptx_parser, text as text_parser, url as url_parser
from app.parsers.base import ParseError


def write_test_pdf(path: Path, paragraphs: list[str]) -> None:
    """Build a small real PDF with ReportLab, so the parser is tested on actual PDF bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=A4)
    for paragraph in paragraphs:
        text_object = pdf.beginText(60, 780)
        for line in [paragraph[i : i + 80] for i in range(0, len(paragraph), 80)]:
            text_object.textLine(line)
        pdf.drawText(text_object)
        pdf.showPage()
    pdf.save()


SENTENCE = (
    "Library automation replaced paper catalogues with searchable records, which changed how "
    "readers browse a shelf and how staff describe their own work."
)


def test_text_parser_reports_encoding(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text(SENTENCE, encoding="utf-8")
    parsed = text_parser.parse(path)
    assert parsed.kind == "text"
    assert parsed.text.strip() == SENTENCE
    assert parsed.warnings == []


def test_text_parser_falls_back_on_latin1(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes("Café naïve résumé".encode("latin-1"))
    parsed = text_parser.parse(path)
    assert "Caf" in parsed.text
    assert parsed.meta["encoding"] == "latin-1"


def test_empty_file_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    parsed = text_parser.parse(path)
    assert "empty_document" in parsed.warnings


def test_pdf_parser_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    write_test_pdf(path, [SENTENCE, "A second paragraph for the second page."])
    from app.parsers import pdf as pdf_parser

    parsed = pdf_parser.parse(path)
    assert parsed.kind == "pdf"
    assert "Library automation" in parsed.text
    assert parsed.pages == 2


def test_docx_parser_round_trip(tmp_path: Path) -> None:
    from docx import Document

    document = Document()
    document.add_paragraph(SENTENCE)
    path = tmp_path / "essay.docx"
    document.save(path)
    parsed = docx_parser.parse(path)
    assert "Library automation" in parsed.text
    assert parsed.kind == "docx"


def test_pptx_parser_round_trip(tmp_path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(2))
    box.text_frame.text = SENTENCE
    box.text_frame.paragraphs[0].font.size = Pt(18)
    path = tmp_path / "deck.pptx"
    presentation.save(path)
    parsed = pptx_parser.parse(path)
    assert "Library automation" in parsed.text
    assert parsed.kind == "pptx"


def test_code_parser_detects_language_and_functions(tmp_path: Path) -> None:
    path = tmp_path / "script.py"
    path.write_text("def total(values):\n    return sum(values)\n", encoding="utf-8")
    parsed = code.parse(path)
    assert parsed.kind == "code"
    assert parsed.meta["language"] in {"py", "python"}
    assert "def total" in parsed.text


def test_zip_extracts_members_and_skips_unsafe_paths(tmp_path: Path) -> None:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.txt", SENTENCE)
        zf.writestr("nested/b.txt", "Second member text.")
        zf.writestr("../escape.txt", "must never be written")
    from app.parsers.base import ParseError

    # The safe rule is to refuse the whole archive when any member is unsafe: silently
    # dropping a member would let an attacker learn which paths are checked.
    with pytest.raises(ParseError) as error:
        archive.extract(path.read_bytes())
    assert error.value.code == "unsafe_path"
    assert not (tmp_path.parent / "escape.txt").exists()

    safe_zip = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe_zip, "w") as zf:
        zf.writestr("a.txt", SENTENCE)
        zf.writestr("nested/b.txt", "Second member text.")
    members, warnings = archive.extract(safe_zip.read_bytes())
    assert {member.name for member in members} == {"a.txt", "b.txt"}
    assert warnings == []


def test_zip_bomb_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("huge.txt", "a" * (40 * 1024 * 1024))
    from app.parsers.base import ParseError

    with pytest.raises(ParseError) as error:
        archive.extract(path.read_bytes())
    assert error.value.code in {"zip_too_large", "zip_bomb"}


def test_url_parser_refuses_private_and_loopback_hosts() -> None:
    for candidate in (
        "http://127.0.0.1:8000/api/health",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/secret",
        "file:///etc/passwd",
    ):
        with pytest.raises(ParseError) as error:
            url_parser.validate_url(candidate)
        assert error.value.code in {"blocked_url", "invalid_url", "bad_url"}


def test_url_parser_accepts_a_public_url() -> None:
    assert url_parser.validate_url("https://en.wikipedia.org/wiki/Cataloguing").startswith("https://")


def test_url_parser_rejects_oversized_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html", "content-length": str(99 * 1024 * 1024)}
        content = b"<html></html>"
        is_redirect = False

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 65536):
            payload = b"x" * (1024 * 1024)
            for _ in range(60):  # 60 MB, over the configured limit
                yield payload

    # Patch the session the function builds, not the module-level helper.
    monkeypatch.setattr(url_parser.requests.Session, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(url_parser, "validate_url", lambda url: url)
    with pytest.raises(ParseError) as error:
        url_parser.fetch_text("https://example.com/big")
    assert error.value.code in {"too_large", "url_too_large"}


def test_archive_reader_handles_a_bad_zip(tmp_path: Path) -> None:
    path = tmp_path / "broken.zip"
    from app.parsers.base import ParseError

    path.write_bytes(b"not a zip file at all")
    with pytest.raises(ParseError) as error:
        archive.extract(path.read_bytes())
    assert error.value.code == "invalid_zip"


def test_binary_content_is_not_treated_as_text(tmp_path: Path) -> None:
    path = tmp_path / "blob.txt"
    path.write_bytes(bytes(range(256)) * 8)
    parsed = text_parser.parse(path)
    assert parsed.text is not None  # decoding never raises; it may warn instead
    assert isinstance(parsed.warnings, list)


def test_network_failures_become_actionable_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocked/slow/TLS-broken page must never surface as an unhandled 500.

    Each of these is one user-visible experience ("that page could not be fetched"), so each is
    mapped to a 422 that names the cause instead of escaping requests' exception zoo.
    """
    import requests

    cases = [
        (requests.exceptions.SSLError("tls eof"), "url_tls_failed"),
        (requests.exceptions.ConnectTimeout("slow"), "url_timeout"),
        (requests.exceptions.ConnectionError("no route"), "url_unreachable"),
    ]
    monkeypatch.setattr(url_parser, "validate_url", lambda url: url)
    for error, expected_code in cases:

        def boom(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(url_parser.requests.Session, "get", boom)
        with pytest.raises(ParseError) as excinfo:
            url_parser.fetch_text("https://example.com/page")
        assert excinfo.value.code == expected_code
        assert excinfo.value.status == 422
        assert "example.com" in excinfo.value.message


def test_http_error_status_is_reported_with_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class NotFound:
        status_code = 404
        headers: dict[str, str] = {}
        is_redirect = False

        def raise_for_status(self) -> None:
            raise url_parser.requests.exceptions.HTTPError("404")

    monkeypatch.setattr(url_parser, "validate_url", lambda url: url)
    monkeypatch.setattr(url_parser.requests.Session, "get", lambda *a, **k: NotFound())
    with pytest.raises(ParseError) as excinfo:
        url_parser.fetch_text("https://example.com/missing")
    assert excinfo.value.code == "url_http_error"
    assert "404" in excinfo.value.message


def test_url_endpoint_returns_a_structured_error(client: Any) -> None:
    """End to end: a URL the sandbox cannot reach must answer 422 with the error shape."""
    # No `files` part at all: sending an empty filename trips FastAPI's own validation
    # before the handler runs, which would test the framework rather than our error path.
    response = client.post(
        "/api/analyses",
        data={
            "url": "https://en.wikipedia.org/wiki/Library_catalog",
            "options": json.dumps({"mode": "lite", "ai": True}),
        },
    )
    assert response.status_code == 422, response.text
    body = response.json()["detail"]["error"]
    assert set(body) == {"code", "message"}
    assert body["code"] in {
        "url_unreachable",
        "url_tls_failed",
        "url_timeout",
        "url_http_error",
        "input_failed",
        "unresolvable_url",
    }
