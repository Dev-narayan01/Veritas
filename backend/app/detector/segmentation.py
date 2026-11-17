"""Mixed-authorship segmentation with a two-state HMM.

A document that is partly the student's own and partly machine-written is the
common real case (paste-assisted editing).  We model the sentence sequence as a
hidden Markov chain with states *human* and *AI*, emission probability taken from
the sentence's AI score, a strong stay probability (0.9) so segments are coherent,
and decode with Viterbi.  The output is a list of contiguous segments plus the
fraction of the document attributed to the AI state.
"""

from __future__ import annotations


import numpy as np

from ..schemas import SentenceScore, Segment

STAY_PROBABILITY = 0.9
_EPS = 1e-9


def viterbi(scores: list[float], stay: float = STAY_PROBABILITY) -> list[int]:
    """Return the most likely state sequence (0=human, 1=AI)."""
    if not scores:
        return []
    p = np.clip(np.asarray(scores, dtype=float), 0.01, 0.99)
    # Log emission probabilities: state AI favours high scores, human favours low.
    emissions = np.vstack([np.log(1.0 - p + _EPS), np.log(p + _EPS)])
    switch = 1.0 - stay
    transition = np.log(np.array([[stay, switch], [switch, stay]]) + _EPS)

    n = p.size
    delta = np.zeros((n, 2))
    back = np.zeros((n, 2), dtype=int)
    delta[0] = np.log(0.5) + emissions[:, 0]
    for t in range(1, n):
        for state in range(2):
            candidates = delta[t - 1] + transition[:, state]
            back[t, state] = int(np.argmax(candidates))
            delta[t, state] = candidates[back[t, state]] + emissions[state, t]
    path = [int(np.argmax(delta[-1]))]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return list(reversed(path))


def segment_document(
    sentence_scores: list[SentenceScore],
    *_: object,
) -> tuple[list[Segment], float]:
    """Build contiguous segments and the AI fraction of the document."""
    if not sentence_scores:
        return [], 0.0
    scores = [s.ai_prob for s in sentence_scores]
    states = viterbi(scores)

    segments: list[Segment] = []
    weights = [max(1, len(s.text.split())) for s in sentence_scores]
    total_words = sum(weights) or 1
    ai_words = 0

    start_idx = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start_idx]:
            label = "ai" if states[start_idx] == 1 else "human"
            segment_scores = scores[start_idx:i]
            segments.append(
                Segment(
                    start=sentence_scores[start_idx].start,
                    end=sentence_scores[i - 1].end,
                    label=label,  # type: ignore[arg-type]
                    confidence=round(float(np.mean(segment_scores)), 4),
                )
            )
            if label == "ai":
                ai_words += sum(weights[start_idx:i])
            start_idx = i
    return segments, round(ai_words / total_words, 4)
