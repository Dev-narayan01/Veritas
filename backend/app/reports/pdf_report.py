"""PDF report generation with reportlab.

The report states the verdict, the thresholds in force, which components actually
ran, the plagiarism sources and a permanent disclaimer.  Every number is copied from
the analysis result; nothing is recomputed or embellished.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..config import DISCLAIMER
from ..schemas import AnalysisResult
from . import charts

_TEXT = colors.HexColor("#1A1F2B")
_MUTED = colors.HexColor("#5B6478")
_ACCENT = colors.HexColor("#7C5CFF")
_RULE = colors.HexColor("#D8DEEA")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, textColor=_TEXT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9, textColor=_MUTED, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12.5, textColor=_TEXT, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.2, textColor=_TEXT, leading=13),
        "small": ParagraphStyle("sm", parent=base["BodyText"], fontSize=7.8, textColor=_MUTED, leading=10.5),
    }


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_report(result: AnalysisResult) -> bytes:
    """Render the full report and return the PDF bytes."""
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Veritas report {result.id}",
        author="Veritas",
    )
    story: list = []
    story.append(Paragraph("Veritas — analysis report", styles["title"]))
    story.append(
        Paragraph(
            f"Analysis {result.id} · created {result.created_at} · "
            f"mode requested: {_escape(str(result.options.mode or 'server default'))}",
            styles["sub"],
        )
    )

    for document in result.documents:
        story.append(Paragraph(_escape(document.filename), styles["h2"]))
        story.append(
            Paragraph(
                f"{document.words} words · kind {_escape(document.kind)}"
                + (f" · {document.pages} pages" if document.pages else ""),
                styles["small"],
            )
        )
        story.append(Spacer(1, 4))
        ai = document.ai
        if ai is not None and ai.probability is not None:
            thresholds = ai.thresholds or {}
            image = charts.probability_gauge(
                ai.probability,
                float(thresholds.get("t_human") or 0.0),
                float(thresholds.get("t_ai") or 1.0),
                ai.verdict,
            )
            story.append(Image(io.BytesIO(image), width=150 * mm, height=61 * mm))
            rows = [
                ["Verdict", ai.verdict.replace("_", " ")],
                ["Mode used", ai.mode_used or "none"],
                ["Components run", ", ".join(ai.components_run) or "none"],
                [
                    "Thresholds",
                    f"human ≤ {float(thresholds.get('t_human') or 0):.3f}, "
                    f"AI ≥ {float(thresholds.get('t_ai') or 0):.3f}",
                ],
                ["95% interval", f"{ai.ci_low:.3f} – {ai.ci_high:.3f}" if ai.ci_low is not None else "n/a"],
            ]
            if ai.ai_fraction is not None:
                rows.append(["Mixed-authorship estimate", f"{ai.ai_fraction:.0%} machine-like"])
            table = Table(rows, colWidths=[45 * mm, 115 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                        ("TEXTCOLOR", (0, 0), (0, -1), _MUTED),
                        ("TEXTCOLOR", (1, 0), (1, -1), _TEXT),
                        ("LINEBELOW", (0, 0), (-1, -2), 0.25, _RULE),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(table)
            if ai.explanation and ai.explanation.text:
                story.append(Spacer(1, 5))
                story.append(
                    Paragraph(
                        f"<b>Explanation ({ai.explanation.source.replace('_', ' ')}):</b> "
                        + _escape(ai.explanation.text),
                        styles["body"],
                    )
                )
            if ai.sentences:
                story.append(Spacer(1, 6))
                image = charts.sentence_heatmap([s.model_dump() for s in ai.sentences])
                story.append(Image(io.BytesIO(image), width=170 * mm, height=39 * mm))
            for warning in ai.warnings[:6]:
                story.append(Paragraph("• " + _escape(warning), styles["small"]))
        elif ai is not None:
            story.append(Paragraph(f"No probability: {_escape(ai.verdict.replace('_', ' '))}", styles["body"]))
            for warning in ai.warnings:
                story.append(Paragraph("• " + _escape(warning), styles["small"]))

        plagiarism = document.plagiarism
        if plagiarism is not None:
            story.append(Paragraph("Plagiarism", styles["h2"]))
            story.append(
                Paragraph(
                    f"Similarity {plagiarism.similarity_percent:.1f}% "
                    f"(after exclusions {plagiarism.similarity_percent_filtered:.1f}%) · "
                    f"corpus of {plagiarism.corpus_size} documents · "
                    f"lexical {plagiarism.lexical_percent:.1f}%, paraphrase "
                    f"{plagiarism.paraphrase_percent:.1f}%, online "
                    f"{plagiarism.online_percent:.1f}%",
                    styles["body"],
                )
            )
            if plagiarism.sources:
                image = charts.sources_bar([s.model_dump() for s in plagiarism.sources[:8]])
                story.append(Image(io.BytesIO(image), width=150 * mm, height=65 * mm))
                for source in plagiarism.sources[:8]:
                    story.append(
                        Paragraph(
                            f"{_escape(source.title)} — {source.percent:.1f}% "
                            f"({source.matches} matches)",
                            styles["small"],
                        )
                    )
            for warning in plagiarism.warnings[:4]:
                story.append(Paragraph("• " + _escape(warning), styles["small"]))

        flags = document.integrity.flags
        if flags:
            story.append(Paragraph("Integrity checks", styles["h2"]))
            for flag in flags:
                story.append(
                    Paragraph(
                        f"{_escape(flag.kind.replace('_', ' '))}: {flag.count} "
                        f"— {_escape(flag.detail)}",
                        styles["small"],
                    )
                )
        if document.warnings:
            story.append(Paragraph("Document warnings", styles["h2"]))
            for warning in document.warnings[:10]:
                story.append(Paragraph("• " + _escape(warning), styles["small"]))
        story.append(PageBreak())

    if result.batch is not None:
        story.append(Paragraph("Batch comparison", styles["h2"]))
        header = [""] + [label[:18] for label in result.batch.labels]
        table_rows = [header]
        for label, row in zip(result.batch.labels, result.batch.matrix):
            table_rows.append([label[:18]] + [f"{value:.1f}" for value in row])
        table = Table(table_rows)
        table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, _RULE)]))
        story.append(table)
        story.append(PageBreak())

    story.append(Paragraph("How to read this report", styles["h2"]))
    story.append(
        Paragraph(
            "AI-text detection combines several statistical signals and a calibrated "
            "decision threshold aimed at a low false-positive rate. The verdict has three "
            "states: likely human, uncertain (abstain) and likely AI. Plagiarism similarity "
            "is the share of characters matched against the reference corpus that was "
            "actually indexed on this server; it is not a comparison with the whole "
            "internet unless the online check was enabled and reported its snippets. "
            + DISCLAIMER,
            styles["body"],
        )
    )
    story.append(Paragraph(_escape(result.disclaimer or DISCLAIMER), styles["small"]))
    story.append(Paragraph(f"Veritas · analysis {result.id}", styles["small"]))
    doc.build(story)
    return buffer.getvalue()
