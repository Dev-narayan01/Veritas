"""Safe URL fetching with SSRF protection.

The user may paste a URL to analyse a web page.  Because the server fetches it, a
naive implementation would let an attacker read internal services
(``http://169.254.169.254/latest/meta-data``).  Every hostname is therefore
resolved and every candidate address is checked against private, loopback,
link-local, multicast and reserved ranges, both before the request *and* after
each redirect.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests

from ..config import settings
from ..logging_setup import get_logger
from .base import ParseError, ParsedDoc

log = get_logger(__name__)

USER_AGENT = "VeritasBot/1.0 (+academic plagiarism and AI-text research project)"
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}
_MAX_REDIRECTS = 5


def _is_public_address(raw: str) -> bool:
    """True when an IP literal is globally routable."""
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_url(url: str) -> str:
    """Validate scheme/host/port and resolve the host to a public address."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ParseError("invalid_url", "Only http and https URLs are supported.", 422)
    host = (parsed.hostname or "").lower()
    if not host or host in _BLOCKED_HOSTNAMES:
        raise ParseError("blocked_url", "That host is not allowed.", 422)
    if parsed.port is not None and parsed.port not in {80, 443, 8080, 8000, 3000}:
        port_ok = 1024 <= parsed.port <= 65535
        if not port_ok:
            raise ParseError("blocked_url", "That port is not allowed.", 422)
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ParseError("unresolvable_url", f"Could not resolve {host}.", 422) from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or not all(_is_public_address(addr) for addr in addresses):
        raise ParseError("blocked_url", "That address is not public.", 422)
    return url.strip()


def _get(session: requests.Session, url: str) -> requests.Response:
    """One bounded GET, with every network failure turned into an actionable message.

    ``requests`` raises a dozen exception types for what the user experiences as one thing —
    "that page could not be fetched". Letting them escape produced a 500 with no explanation,
    so they are all mapped to a 422 that names the cause.
    """
    try:
        return session.get(
            url, timeout=settings.url_timeout_s, stream=True, allow_redirects=False
        )
    except requests.exceptions.SSLError as exc:
        raise ParseError(
            "url_tls_failed",
            f"Could not establish a secure connection to {urlparse(url).hostname}.",
            422,
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ParseError(
            "url_timeout",
            f"{urlparse(url).hostname} did not respond within {settings.url_timeout_s}s.",
            422,
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ParseError(
            "url_unreachable",
            f"Could not fetch {urlparse(url).hostname}: {type(exc).__name__}.",
            422,
        ) from exc


def fetch_text(url: str) -> ParsedDoc:
    """Download a page (bounded in size and time) and extract its main text."""
    validate_url(url)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,text/plain"})
    response = _get(session, url)
    redirects = 0
    while response.is_redirect:
        redirects += 1
        if redirects > _MAX_REDIRECTS:
            raise ParseError("too_many_redirects", "Too many redirects.", 422)
        target = response.headers.get("Location", "")
        target = requests.compat.urljoin(url, target)
        validate_url(target)  # re-check every hop: prevents redirect-based SSRF
        url = target
        response = _get(session, url)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise ParseError(
            "url_http_error",
            f"{urlparse(url).hostname} returned HTTP {response.status_code}.",
            422,
        ) from exc

    chunks: list[bytes] = []
    size = 0
    try:
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > settings.max_url_bytes:
                raise ParseError("url_too_large", "Page exceeds the size limit.", 413)
            chunks.append(chunk)
    except requests.exceptions.RequestException as exc:
        raise ParseError(
            "url_interrupted", "The download was interrupted before it finished.", 422
        ) from exc
    body = b"".join(chunks)
    content_type = response.headers.get("content-type", "").lower()

    if "html" in content_type or b"<html" in body[:2000].lower():
        text = _extract_html(body, url)
    else:
        text = body.decode("utf-8", errors="replace")
    return ParsedDoc(
        filename=url,
        kind="text",
        text=text.strip(),
        pages=None,
        warnings=[] if text.strip() else ["empty_document"],
        meta={"url": url, "content_type": content_type, "bytes": size},
    )


def _extract_html(body: bytes, url: str) -> str:
    """Prefer trafilatura's main-content extraction, fall back to BeautifulSoup."""
    try:  # optional dependency
        import trafilatura

        extracted = trafilatura.extract(body, include_comments=False, include_tables=True)
        if extracted and len(extracted.split()) > 40:
            return extracted
    except Exception as exc:  # noqa: BLE001 - optional dependency
        log.warning("trafilatura extraction failed: %s", exc)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)
