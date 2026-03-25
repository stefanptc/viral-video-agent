"""
test_agents.py — Quick smoke tests. Run before wiring the graph.

Usage:
    uv run python test_agents.py

Tests:
  1. LM Studio connectivity (ping)
  2. Search Agent — does it return valid JSON with the right keys?
  3. Editor Chief Agent — does APPROVE/REJECT logic fire correctly?
"""

from __future__ import annotations

import json
import sys

from dotenv import load_dotenv
load_dotenv()

from src.agents.llm_factory import ping_lm_studio, get_llm
from src.agents.prompts import AGENT_PROMPTS
from langchain_core.messages import SystemMessage, HumanMessage


def test_ping() -> bool:
    print("\n── Test 1: LM Studio ping ──────────────────────────────")
    result = ping_lm_studio()
    if not result:
        print("FAIL — cannot continue without LM Studio.\n")
    return result


def test_search_agent() -> bool:
    print("\n── Test 2: Search Agent JSON output ────────────────────")
    llm = get_llm(temperature=0.1)

    fake_markdown = """\
# YouTube Trending

1. [How I Built a Tesla in My Garage](https://www.youtube.com/watch?v=abc123) — 4.2M views
2. [AI Destroyed My Business (True Story)](https://www.youtube.com/watch?v=def456) — 2.1M views
3. [Gordon Ramsay Reacts to Prison Food](https://www.youtube.com/watch?v=ghi789) — 8.9M views
"""

    messages = [
        SystemMessage(content=AGENT_PROMPTS["search"]["system"]),
        HumanMessage(content=AGENT_PROMPTS["search"]["human"].format(
            markdown_content=fake_markdown
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        print(f"Raw output:\n{raw}\n")
        parsed = json.loads(raw)
        assert "trending_videos" in parsed, "Missing 'trending_videos' key"
        assert isinstance(parsed["trending_videos"], list), "'trending_videos' is not a list"
        assert len(parsed["trending_videos"]) > 0, "Empty trending_videos list"
        first = parsed["trending_videos"][0]
        assert "rank" in first and "title" in first and "url" in first, \
            f"Missing keys in first item: {first}"
        print(f"✅  PASS — got {len(parsed['trending_videos'])} videos, first: {first['title']}")
        return True
    except Exception as exc:
        print(f"❌  FAIL — {exc}")
        return False


def test_editor_chief_approve() -> bool:
    print("\n── Test 3: Editor Chief — APPROVE path ─────────────────")
    llm = get_llm(temperature=0.1)

    good_formula = json.dumps({
        "core_emotion": "awe",
        "pacing_rules": [
            "Cut every 1.5 seconds during the first 6 seconds",
            "Hold the reveal shot for exactly 2 seconds at the 10-second mark"
        ],
        "visual_mandates": [
            "Extreme close-up on subject's eyes in frame 1",
            "Dutch angle (15°) during tension build, clips 2 and 3",
            "Pure white background for the payoff shot"
        ],
        "audio_cues": [
            "Hard bass drop synced to the first cut",
            "Silence for 0.5 seconds immediately before the reveal"
        ],
        "hook_blueprint": "0–0.5s: extreme close-up of hands doing something unexpected; 0.5–2s: quick-cut montage at 1.5s intervals showing scale; 2–3s: freeze frame with text pop-in"
    }, indent=2)

    messages = [
        SystemMessage(content=AGENT_PROMPTS["editor_chief"]["system"]),
        HumanMessage(content=AGENT_PROMPTS["editor_chief"]["human"].format(
            formula_json=good_formula
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        print(f"Raw output:\n{raw}\n")
        parsed = json.loads(raw)
        assert "decision" in parsed, "Missing 'decision' key"
        assert parsed["decision"] in ("APPROVED", "REJECTED"), \
            f"decision must be APPROVED or REJECTED, got: {parsed['decision']}"
        print(f"✅  PASS — decision: {parsed['decision']}")
        return True
    except Exception as exc:
        print(f"❌  FAIL — {exc}")
        return False


def test_editor_chief_reject() -> bool:
    print("\n── Test 4: Editor Chief — REJECT path ──────────────────")
    llm = get_llm(temperature=0.1)

    bad_formula = json.dumps({
        "core_emotion": "positive vibes",
        "pacing_rules": ["edit fast", "keep it engaging"],
        "visual_mandates": ["use bright colors", "make it exciting"],
        "audio_cues": ["good music"],
        "hook_blueprint": "start with something cool"
    }, indent=2)

    messages = [
        SystemMessage(content=AGENT_PROMPTS["editor_chief"]["system"]),
        HumanMessage(content=AGENT_PROMPTS["editor_chief"]["human"].format(
            formula_json=bad_formula
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        print(f"Raw output:\n{raw}\n")
        parsed = json.loads(raw)
        assert parsed.get("decision") == "REJECTED", \
            f"Expected REJECTED for vague formula, got: {parsed.get('decision')}"
        assert len(parsed.get("feedback", "")) > 20, \
            "Feedback too short — should contain specific corrections"
        print(f"✅  PASS — correctly rejected. Feedback preview: {parsed['feedback'][:120]}...")
        return True
    except Exception as exc:
        print(f"❌  FAIL — {exc}")
        return False


if __name__ == "__main__":
    results = []

    ok = test_ping()
    results.append(ok)
    if not ok:
        sys.exit(1)

    results.append(test_search_agent())
    results.append(test_editor_chief_approve())
    results.append(test_editor_chief_reject())

    passed = sum(results)
    total  = len(results)
    print(f"\n{'─'*55}")
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("Some tests failed. Check LM Studio logs for clues.")
        sys.exit(1)
    else:
        print("All good — ready to build graph/nodes.py 🎬")