"""Structure-aware code plagiarism detection.

Python is parsed with ``ast`` and reduced to a stream of node types, discarding
identifier names, literals, docstrings and comments — so renaming every variable
does not help.  Other languages go through ``pygments`` with identifiers → ``ID``,
strings → ``STR``, numbers → ``NUM`` and comments dropped.  Winnowing over that
stream gives fingerprints robust to reformatting, and the similarity is the overlap
divided by the smaller fingerprint set (so a small file copied into a large one is
still found).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..logging_setup import get_logger
from ..schemas import CodeMatch
from .shingling import winnow

log = get_logger(__name__)


@dataclass
class CodeFile:
    """A source file reduced to a structural token stream."""

    name: str
    language: str
    text: str
    tokens: list[str] = field(default_factory=list)
    token_lines: list[int] = field(default_factory=list)
    functions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def lines(self) -> int:
        return self.text.count("\n") + 1


def _python_tokens(source: str) -> tuple[list[str], list[int], dict[str, list[str]]]:
    """Node-type stream for Python, with per-function fingerprints."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        log.warning("python parse failed: %s", exc)
        return _generic_tokens(source, "python")
    tokens: list[str] = []
    lines: list[int] = []
    functions: dict[str, list[str]] = {}

    def walk(node: ast.AST, sink: list[str] | None = None) -> None:
        name = type(node).__name__
        if isinstance(node, (ast.Expr,)) and isinstance(getattr(node, "value", None), ast.Constant):
            if isinstance(node.value.value, str):
                return  # docstring
        sink = sink if sink is not None else tokens
        sink.append(name)
        if sink is tokens:
            lines.append(getattr(node, "lineno", 0))
        for child in ast.iter_child_nodes(node):
            walk(child, sink)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body: list[str] = []
            for child in node.body:
                walk(child, body)
            functions[node.name] = body
    walk(tree)
    return tokens, lines, functions


def _generic_tokens(source: str, language: str) -> tuple[list[str], list[int], dict[str, list[str]]]:
    """Language-agnostic structural stream via pygments token classes."""
    try:
        from pygments import lex
        from pygments.lexers import get_lexer_by_name
        from pygments.token import Comment, Name, Number, String

        lexer = get_lexer_by_name(language)
        tokens: list[str] = []
        lines: list[int] = []
        line = 1
        for kind, value in lex(source, lexer):
            line += value.count("\n")
            if kind in Comment:
                continue
            if kind in Name:
                tokens.append("ID")
            elif kind in String:
                tokens.append("STR")
            elif kind in Number:
                tokens.append("NUM")
            elif kind in Comment or not value.strip():
                continue
            else:
                tokens.append(str(kind).split(".")[-1])
            lines.append(line)
        return tokens, lines, {}
    except Exception as exc:  # noqa: BLE001 - unknown language
        log.warning("pygments failed for %s: %s", language, exc)
        return source.split(), list(range(len(source.split()))), {}


def load_code(path: Path, language: str | None = None) -> CodeFile:
    """Read a source file and build its structural fingerprint material."""
    text = path.read_text(errors="replace")
    lang = (language or path.suffix.lstrip(".")).lower()
    if lang == "py":
        lang = "python"
    if lang == "python":
        tokens, lines, functions = _python_tokens(text)
    else:
        tokens, lines, functions = _generic_tokens(text, lang)
    return CodeFile(
        name=path.name, language=lang, text=text, tokens=tokens, token_lines=lines, functions=functions
    )


def fingerprint(code: CodeFile) -> set[int]:
    """Winnowed fingerprint set of the structural token stream."""
    prints = winnow(code.tokens, settings.code_winnow_k, settings.code_winnow_w)
    return {p.hash for p in prints}


def similarity(a: CodeFile, b: CodeFile) -> float:
    """|A ∩ B| / min(|A|, |B|) over structural fingerprints."""
    fa, fb = fingerprint(a), fingerprint(b)
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / min(len(fa), len(fb))


def compare_all(code_files: list[CodeFile], threshold: float | None = None) -> list[CodeMatch]:
    """Report every pair at or above the match threshold, with line ranges."""
    limit = settings.code_match_threshold if threshold is None else threshold
    results: list[CodeMatch] = []
    for i in range(len(code_files)):
        for j in range(i + 1, len(code_files)):
            a, b = code_files[i], code_files[j]
            score = similarity(a, b)
            if score < limit:
                continue
            words_a, words_b = set(a.tokens), set(b.tokens)
            shared_functions = [
                name for name, body in a.functions.items()
                if body and any(body[: min(20, len(body))] == other[: min(20, len(other))]
                                for other in b.functions.values())
            ]
            del words_a, words_b
            results.append(
                CodeMatch(
                    a=a.name,
                    b=b.name,
                    similarity=round(float(score), 4),
                    lines_a=_matched_lines(a, b),
                    lines_b=_matched_lines(b, a),
                    functions=shared_functions,
                )
            )
    results.sort(key=lambda m: m.similarity, reverse=True)
    return results


def _matched_lines(target: CodeFile, other: CodeFile, max_lines: int = 40) -> list[int]:
    """Line numbers in ``target`` whose structural tokens also occur in ``other``."""
    other_tokens = set(other.tokens)
    lines: list[int] = []
    for token, line in zip(target.tokens, target.token_lines):
        if token in other_tokens and line and line not in lines:
            lines.append(line)
        if len(lines) >= max_lines:
            break
    return sorted(lines)
