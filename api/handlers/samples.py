"""Request handling for GET /api/samples: short texts embedded in the function.

They are original sample paragraphs written for this project, not scraped documents.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

HUMAN = (
    "I tried to keep a sourdough starter alive through a heatwave and it went badly. "
    "The first two days it looked promising, all bubbles and a clean sour smell. Then the "
    "kitchen hit thirty-five degrees and the thing turned into a grey soup that smelled like "
    "old paint. I fed it anyway, twice a day, because everyone online insisted that patience "
    "was the whole trick. By Thursday I had two jars of failure and a stubborn refusal to "
    "admit that the cheap flour was the problem. The third attempt finally worked, sort of, "
    "and the first loaf came out flat enough to use as a chopping board. My neighbour said it "
    "tasted fine, which is what people say when they are being kind about bread."
)
AI = (
    "Maintaining a sourdough starter requires consistent attention to temperature, hydration "
    "and feeding frequency. It is important to note that fluctuations in ambient conditions "
    "can significantly affect fermentation activity. A starter kept at approximately 24 °C "
    "with regular feeding intervals typically develops a stable microbial community within "
    "seven to ten days. When the environment becomes too warm, fermentation accelerates and "
    "the culture may exhaust its available nutrients before the next feeding. In conclusion, "
    "careful monitoring and a consistent routine are essential components of successful "
    "starter maintenance. These practices help ensure reliable results over time."
)


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel's expected class name
    """Vercel's Python runtime looks for this class."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        payload = json.dumps(
            [
                {"label": "Sample: human-written", "text": HUMAN},
                {"label": "Sample: AI-written", "text": AI},
            ]
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(payload)
