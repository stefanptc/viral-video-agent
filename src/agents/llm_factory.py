"""
llm_factory.py — Single place to create LLM clients.

LM Studio mimics the OpenAI API, so we use ChatOpenAI with a custom
base_url. Two factory functions are exposed:

  get_llm()            — standard text/JSON completions (all 5 text agents)
  get_multimodal_llm() — same endpoint but with vision enabled (Breakdown Agent)
  get_creative_llm()   — higher temperature for the Film Director agent

JSON correctness is NOT enforced via response_format (LM Studio only supports
"json_schema" or "text", not "json_object"). Instead it is enforced by:
  1. Inline output schema in every system prompt
  2. Pydantic validation + retry loops in graph/nodes.py
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL:    str = os.getenv("LM_STUDIO_MODEL",    "local-model")

# LM Studio ignores the api_key value but langchain_openai requires it non-empty.
_DUMMY_KEY = "lm-studio"


# ── Factories ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm(
    temperature: float = 0.3,
    max_tokens:  int   = 4096,
) -> ChatOpenAI:
    """
    Cached ChatOpenAI instance for structured-output agents.

    temperature=0.3 keeps output deterministic. JSON validity is enforced
    by prompt schema + Pydantic validation in the graph, not by the API.
    """
    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=_DUMMY_KEY,
        model=LM_STUDIO_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_multimodal_llm(
    temperature: float = 0.2,
    max_tokens:  int   = 4096,
) -> ChatOpenAI:
    """
    Cached ChatOpenAI instance for vision tasks (Breakdown Agent).

    Slightly lower temperature than get_llm() — visual analysis
    should be precise. Message payloads include image frames.
    """
    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=_DUMMY_KEY,
        model=LM_STUDIO_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm_high_tokens(
    temperature: float = 0.3,
    max_tokens:  int   = 50000,
    disable_thinking: bool = False,
) -> ChatOpenAI:
    """
    Uncached LLM with a higher token ceiling.
    Use for nodes that produce large JSON payloads (Social Science, Film Director).

    disable_thinking=True passes budget_tokens=0 to suppress Qwen 3.5's
    extended thinking mode, preventing it from consuming all tokens on reasoning
    and leaving none for the actual JSON output.
    """
    kwargs = {}
    if disable_thinking:
        # LM Studio passes extra_body fields through to llama.cpp
        # budget_tokens=0 disables chain-of-thought for Qwen 3.5
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=_DUMMY_KEY,
        model=LM_STUDIO_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def get_creative_llm(
    temperature: float = 0.75,
    max_tokens:  int   = 2048,
) -> ChatOpenAI:
    """
    Higher-temperature LLM for the Film Director agent.

    Not cached — callers may want fresh instances with different temps.
    Prompt schema + Pydantic validation still enforce JSON correctness.
    """
    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=_DUMMY_KEY,
        model=LM_STUDIO_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ── Health-check helper ───────────────────────────────────────────────────────

def ping_lm_studio() -> bool:
    """
    Quick connectivity check. Call this at startup in main.py to fail fast
    before the graph runs.

    Returns True if LM Studio responds with a loaded model, False otherwise.
    """
    import httpx

    try:
        resp = httpx.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5.0)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        loaded = [m["id"] for m in models]
        if not loaded:
            print("⚠️  LM Studio is running but no model is loaded.")
            return False
        print(f"✅  LM Studio connected. Loaded model(s): {loaded}")
        return True
    except Exception as exc:
        print(f"❌  Cannot reach LM Studio at {LM_STUDIO_BASE_URL}: {exc}")
        return False
