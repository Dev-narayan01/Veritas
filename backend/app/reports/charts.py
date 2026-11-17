"""Charts for the PDF report (matplotlib, Agg backend, no GUI dependencies)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported anywhere

import matplotlib.pyplot as plt  # noqa: E402

VERDICT_COLORS = {
    "likely_human": "#34D399",
    "uncertain": "#FBBF24",
    "likely_ai": "#F87171",
    "too_short": "#8B96AD",
}
ACCENT = "#7C5CFF"


@dataclass
class ChartStyle:
    figure_size: tuple[float, float] = (6.4, 2.6)
    dpi: int = 150


def probability_gauge(probability: float, t_human: float, t_ai: float, verdict: str) -> bytes:
    """A horizontal bar with the two thresholds marked."""
    style = ChartStyle()
    fig, ax = plt.subplots(figsize=style.figure_size, dpi=style.dpi)
    ax.barh([0], [1.0], color="#1F2A44", height=0.5)
    ax.barh([0], [probability], color=VERDICT_COLORS.get(verdict, ACCENT), height=0.5)
    for value, label in ((t_human, "human ≤"), (t_ai, "AI ≥")):
        ax.axvline(value, color="#E8ECF6", linestyle="--", linewidth=0.8)
        ax.text(value, 0.32, f"{label} {value:.2f}", fontsize=7, ha="center", color="#E8ECF6")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("calibrated probability of machine authorship")
    ax.set_title(f"P(machine) = {probability:.1%}", fontsize=9)
    _finish(fig, ax)
    return _to_png(fig)


def sentence_heatmap(sentences: list[dict]) -> bytes:
    """Sentence-level scores as a strip, one cell per sentence."""
    style = ChartStyle(figure_size=(7.0, 1.6))
    fig, ax = plt.subplots(figsize=style.figure_size, dpi=style.dpi)
    values = [float(s.get("ai_prob", 0.0)) for s in sentences] or [0.0]
    ax.imshow([values], aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_yticks([])
    ax.set_xlabel("sentences in document order (red = more machine-like)")
    _finish(fig, ax)
    return _to_png(fig)


def sources_bar(sources: list[dict]) -> bytes:
    """Similarity contribution per source."""
    style = ChartStyle(figure_size=(6.4, 2.8))
    fig, ax = plt.subplots(figsize=style.figure_size, dpi=style.dpi)
    labels = [str(s.get("title", ""))[:28] for s in sources][::-1] or ["no sources"]
    values = [float(s.get("percent", 0.0)) for s in sources][::-1] or [0.0]
    ax.barh(range(len(labels)), values, color=ACCENT)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("% of document matched")
    _finish(fig, ax)
    return _to_png(fig)


def _finish(fig, ax) -> None:  # noqa: ANN001 - matplotlib objects
    fig.patch.set_facecolor("#0E1424")
    ax.set_facecolor("#0E1424")
    for spine in ax.spines.values():
        spine.set_color("#1F2A44")
    ax.tick_params(colors="#8B96AD", labelsize=7)
    ax.xaxis.label.set_color("#8B96AD")
    ax.title.set_color("#E8ECF6")
    fig.tight_layout()


def _to_png(fig) -> bytes:  # noqa: ANN001
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()
