"""
jina_fetcher.py — Fetches trending YouTube data as clean Markdown.

How it works:
  Jina AI's r.jina.ai service acts as a universal web-to-markdown converter.
  Prepending any URL with https://r.jina.ai/ returns that page's content
  as clean, LLM-readable markdown — no HTML parsing needed on our end.

  We fetch trends24.in/youtube which aggregates YouTube trending data
  updated roughly every hour.

Usage:
    from src.tools.jina_fetcher import fetch_trending_markdown
    markdown = fetch_trending_markdown()

Environment variables:
    JINA_API_KEY   — optional but raises rate limits significantly
    JINA_TARGET_URL — override the default target (useful for testing)
"""

from __future__ import annotations

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_TARGET_URL = "https://youtube.trends24.in/"
JINA_BASE          = "https://r.jina.ai/"
REQUEST_TIMEOUT    = 30.0  # seconds — Jina can be slow on first fetch

# Jina returns cleaner output with these headers
_JINA_HEADERS = {
    "Accept":            "text/plain",      # ask for plain text, not HTML
    "X-Return-Format":   "markdown",        # explicit markdown mode
    "X-No-Cache":        "true",            # always get fresh trending data
    "X-Remove-Selector": "header,footer,nav,script,style,aside",  # strip chrome
}


# ── Main function ─────────────────────────────────────────────────────────────

def fetch_trending_markdown(target_url: str | None = None) -> str:
    """
    Fetch a trending page and return it as clean Markdown.

    Args:
        target_url: Override the default trends24.in URL.
                    Useful for pointing at a different trending source.

    Returns:
        Markdown string ready to pass to the Search Agent.

    Raises:
        httpx.HTTPError: if the request fails after retries.
        ValueError: if the response is empty or too short to be useful.
    """
    target  = target_url or os.getenv("JINA_TARGET_URL", DEFAULT_TARGET_URL)
    api_key = os.getenv("JINA_API_KEY", "")

    fetch_url = f"{JINA_BASE}{target}"
    headers   = dict(_JINA_HEADERS)

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        print("  ⚠  JINA_API_KEY not set — using free tier (rate limited)")

    print(f"  🌐 Fetching: {fetch_url}")

    with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        response = client.get(fetch_url, headers=headers)
        response.raise_for_status()

    markdown = response.text.strip()

    if len(markdown) < 200:
        raise ValueError(
            f"Jina returned suspiciously short response ({len(markdown)} chars). "
            f"The target page may have blocked the request or changed structure.\n"
            f"Response preview: {markdown[:200]}"
        )

    print(f"  ✅ Got {len(markdown):,} chars of markdown from {target}")
    return markdown


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run directly to test the fetcher in isolation:
        uv run python -m src.tools.jina_fetcher
    """
    print("Testing jina_fetcher...\n")
    try:
        md = fetch_trending_markdown()
        print(f"\nFirst 1000 chars of response:\n{'─' * 50}")
        print(md[:1000])
        print(f"{'─' * 50}")
        print(f"\nTotal length: {len(md):,} chars")

        # Check the markdown looks like it contains YouTube video data
        youtube_signals = ["youtube.com", "youtu.be", "watch?v=", "/watch/"]
        found = [s for s in youtube_signals if s in md.lower()]
        if found:
            print(f"✅ YouTube URL signals found: {found}")
        else:
            print("⚠  No YouTube URL signals found — check the markdown above")
            print("   The Search Agent prompt may need adjusting for this format.")

    except Exception as exc:
        print(f"❌ Failed: {exc}")
