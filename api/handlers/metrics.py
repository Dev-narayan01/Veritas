"""Request handling for GET /api/metrics.

Lite ships no evaluation artifacts. Rather than a page of invented numbers, the endpoint
says so, and the UI renders that as "metrics unavailable on this deployment" with a link
to the full server. ``test_ood`` stays in the payload shape so the Evaluation page's
ordering logic is unchanged.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

def _load() -> dict:
    """Serve the real evaluation numbers when the bundle carries them.

    The previous version hardcoded "Lite ships no evaluation artifacts", which was true
    of the smoke run but is now false: a full training run produces ``metrics.json``, and
    hiding it would make the page say "no data" about numbers that exist. If the file is
    absent or unreadable the endpoint still degrades honestly.
    """
    directory = find_artifact_dir()
    empty = {
        "artifact_source": "none",
        "data_source": "unknown",
        "metrics": None,
        "manifest": {"deployment": "lite"},
        "robustness": None,
        "bias": None,
        "cross_generator": None,
        "baseline": None,
        "plag_thresholds": None,
        "plag_benchmark_results": None,
        "corpus_stats": None,
        "embedding_projection": None,
        "plots": [],
        "plots_base": "/metrics/plots",
        "thresholds": None,
        "manifest_notes": [
            "This is the Lite deployment and no evaluation artifacts are bundled with it. "
            "The full analysis server evaluates on held-out data; nothing is fabricated here."
        ],
        "deployment": "lite",
    }
    if directory is None:
        return empty
    path = directory / "metrics.json"
    if not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - a corrupt file is reported, not raised
        return empty

    notes = list(payload.get("notes") or [])
    notes.append(
        "Lite deployment: these numbers come from the CPU baseline bundle shipped with "
        "the functions. The full server adds perplexity, transformer and zero-shot "
        "components, so its figures differ."
    )
    return {
        "artifact_source": "cpu",
        "data_source": payload.get("data_source", "hc3"),
        "metrics": payload,
        "manifest": {"deployment": "lite", "run": payload.get("run")},
        "robustness": None,
        "bias": None,
        "cross_generator": None,
        "baseline": None,
        "plag_thresholds": None,
        "plag_benchmark_results": None,
        "corpus_stats": None,
        "embedding_projection": None,
        "plots": payload.get("plots") or [],
        "plots_base": "/metrics/plots",
        "thresholds": _thresholds(directory),
        "manifest_notes": notes,
        "deployment": "lite",
    }


def _thresholds(directory) -> dict | None:
    """The bands actually in force, per mode.

    Only ``lite`` used to be returned, keyed at the top level. Now that ``fast`` is deployed
    it runs a different band, and the metrics page must not show the tuner's in-domain band
    (0.9308/0.9784) as if it were live: that band is the one measured to be worse than lite
    out-of-domain and was deliberately not shipped. ``fast_deployed`` is what decides a
    verdict, so that is what this reports, alongside the rejected band for comparison.
    """
    try:
        data = json.loads((directory / "thresholds.json").read_text())
        lite = data.get("lite") or None
        deployed = data.get("fast_deployed") or None
        rejected = data.get("fast_rejected_tuned") or None
        if deployed:
            fast = {
                "t_human": deployed.get("t_human"),
                "t_ai": deployed.get("t_ai"),
                "source": "test_ood (conservative: high precision, low coverage)",
                "measured": deployed.get("measured"),
                "rejected_tuned_band": rejected,
            }
        else:
            fast = data.get("fast") or None
        return {
            "lite": lite,
            "fast": fast,
            # keep the old shape readable for the frontend, which reads t_ai/t_human here
            **(lite or {}),
        }
    except Exception:  # noqa: BLE001
        return None


BODY = _load()


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel's expected class name
    """Vercel's Python runtime looks for this class."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        payload = json.dumps(BODY).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(payload)
