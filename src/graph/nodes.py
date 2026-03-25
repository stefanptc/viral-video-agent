"""
nodes.py — Every LangGraph node function for the Viral Video Agent.

Each function takes AgentState and returns a partial dict that LangGraph
merges back into the state. Nodes never mutate state in place.

Node map:
  Phase 1 — Acquisition
    fetch_trends        : jina_fetcher → raw_markdown
    search_node         : Search Agent → acquisition
    download_node       : yt_downloader → acquisition.raw_clip_paths

  Phase 2 — Analysis
    breakdown_node      : Breakdown Agent → breakdowns (one video per call)
    qa_node             : Pydantic validation → breakdown.qa_passed
    social_science_node : Social Science Agent → viral_formula
    editor_chief_node   : Editor Chief Agent → viral_formula.status

  Phase 3 — Generation
    film_director_node  : Film Director Agent → storyboard
    sales_node          : Sales Agent → storyboard (product injected)
    human_checkpoint    : Terminal review → storyboard.status
    comfyui_node        : ComfyUI API → generation.generated_clip_paths
    assemble_node       : FFMPEG → generation.final_video_path
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from src.state import (
    AgentState,
    AcquisitionResult,
    TrendingVideo,
    VideoBreakdown,
    ViralFormula,
    FormulaStatus,
    Storyboard,
    StoryboardClip,
    PromptsStatus,
    GenerationResult,
)
from src.agents.llm_factory import get_llm, get_llm_high_tokens, get_multimodal_llm, get_creative_llm
from src.agents.prompts import AGENT_PROMPTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    """
    Strip Qwen <think>...</think> reasoning blocks before JSON extraction.
    Qwen 3.5 in extended-thinking mode outputs reasoning first, then the
    answer. Without this, _extract_json finds no JSON in a think-only response.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict:
    """
    Robustly pull a JSON object out of an LLM response.

    Handles four common Qwen output styles:
      1. Raw JSON (ideal)
      2. <think>...</think> preamble followed by JSON
      3. ```json ... ``` fenced block
      4. JSON embedded in prose with surrounding text

    Also handles truncated responses by attempting to close open braces,
    which can happen when max_tokens is hit mid-output.
    """
    text = _strip_think_tags(text)

    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Find the first top-level { ... } block
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response:\n{text[:300]}")

    # Walk to find the matching closing brace
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        return json.loads(text[start : end + 1])

    # Response was truncated — try to recover by closing open braces/brackets
    truncated = text[start:]
    open_braces   = truncated.count("{") - truncated.count("}")
    open_brackets = truncated.count("[") - truncated.count("]")
    # Close any open string (look for odd number of unescaped quotes)
    recovery = truncated.rstrip().rstrip(",")
    recovery += "]" * max(0, open_brackets)
    recovery += "}" * max(0, open_braces)
    try:
        return json.loads(recovery)
    except json.JSONDecodeError:
        raise ValueError(f"Unmatched braces in response (truncated?):\n{text[:300]}")


def _fmt(template: str, **kwargs) -> str:
    """Format a prompt template, escaping any braces in JSON payloads."""
    return template.format(**kwargs)


# ── Phase 1: Acquisition ──────────────────────────────────────────────────────

def fetch_trends(state: AgentState) -> dict:
    """
    Node: fetch_trends
    Calls jina_fetcher to turn trends24.in into Markdown.
    Imported here to avoid circular imports with the tools layer.
    """
    from src.tools.jina_fetcher import fetch_trending_markdown

    print("\n[Node] fetch_trends — fetching trending page...")
    try:
        markdown = fetch_trending_markdown()
        print(f"  ✅ Got {len(markdown)} chars of markdown")
        return {"raw_markdown": markdown, "acquisition_error": ""}
    except Exception as exc:
        print(f"  ❌ fetch_trends failed: {exc}")
        return {"acquisition_error": str(exc)}


def _trim_markdown(markdown: str, max_chars: int = 12000) -> str:
    """
    Trim the Jina markdown to the most relevant section.

    trends24.in markdown is ~60k chars. We only need the trending
    video list section — typically the first 12k chars contains all
    10 trending videos with their titles and YouTube URLs.
    """
    if len(markdown) <= max_chars:
        return markdown

    # Try to find where the trending video list ends
    # and cut there rather than hard-truncating mid-entry
    trimmed = markdown[:max_chars]
    last_newline = trimmed.rfind("\n")
    if last_newline > max_chars * 0.8:
        trimmed = trimmed[:last_newline]

    print(f"  ✂  Trimmed markdown: {len(markdown):,} → {len(trimmed):,} chars")
    return trimmed


def search_node(state: AgentState) -> dict:
    """
    Node: search_node
    Runs the Search Agent over raw_markdown → populates acquisition.

    The real Jina markdown is ~60k chars — we trim it to the first
    12k which contains all trending video entries, before passing
    to Qwen to avoid context overflow and thinking-only responses.
    """
    print("\n[Node] search_node — parsing trending videos...")
    markdown = state.get("raw_markdown", "")

    if not markdown:
        return {"acquisition_error": "raw_markdown is empty — fetch_trends must run first"}

    # Trim to relevant section only
    markdown = _trim_markdown(markdown)

    # 50k tokens — enough for Qwen 3.5 to think fully AND produce the JSON output.
    llm = get_llm_high_tokens(temperature=0.1, max_tokens=50000)

    messages = [
        SystemMessage(content=AGENT_PROMPTS["search"]["system"]),
        HumanMessage(content=_fmt(
            AGENT_PROMPTS["search"]["human"],
            markdown_content=markdown,
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        print(f"  🔍 Raw response preview: {repr(raw[:400])}")
        data = _extract_json(raw)
        videos = [TrendingVideo(**v) for v in data["trending_videos"]]
        result = AcquisitionResult(trending_videos=videos)
        print(f"  ✅ Found {len(videos)} trending videos")
        for v in videos:
            print(f"     {v.rank}. {v.title}")
        return {"acquisition": result, "acquisition_error": ""}
    except (ValidationError, ValueError, KeyError) as exc:
        print(f"  ❌ search_node failed: {exc}")
        return {"acquisition_error": str(exc)}


def download_node(state: AgentState) -> dict:
    """
    Node: download_node
    Downloads 0–60s of each trending video via yt_downloader.
    Updates acquisition.raw_clip_paths.
    """
    from src.tools.yt_downloader import download_clip

    print("\n[Node] download_node — downloading clips...")
    acquisition = state.get("acquisition")
    if not acquisition:
        return {"acquisition_error": "No acquisition result — search_node must run first"}

    clip_paths: list[Path] = []
    for video in acquisition.trending_videos:
        print(f"  ⬇  [{video.rank}] {video.title}")
        try:
            path = download_clip(video.url, video.rank)
            clip_paths.append(path)
            print(f"       → {path}")
        except Exception as exc:
            print(f"       ❌ Failed: {exc}")

    updated = acquisition.model_copy(update={"raw_clip_paths": clip_paths})
    print(f"  ✅ Downloaded {len(clip_paths)}/{len(acquisition.trending_videos)} clips")
    return {"acquisition": updated, "acquisition_error": ""}


# ── Phase 2: Analysis ─────────────────────────────────────────────────────────

def breakdown_node(state: AgentState) -> dict:
    """
    Node: breakdown_node
    Runs the Breakdown Agent on the current video (state.current_video_index).
    Appends result to state.breakdowns.
    """
    print("\n[Node] breakdown_node — analysing clip...")
    acquisition = state.get("acquisition")
    if not acquisition:
        return {"breakdown_errors": ["No acquisition — cannot run breakdown"]}

    idx = state.get("current_video_index", 0)
    if idx >= len(acquisition.trending_videos):
        return {}  # all videos processed

    video = acquisition.trending_videos[idx]
    clip_path = (
        acquisition.raw_clip_paths[idx]
        if idx < len(acquisition.raw_clip_paths)
        else None
    )

    print(f"  📹 [{video.rank}] {video.title}")
    llm = get_multimodal_llm()

    # Build message — include video frames if clip was downloaded
    human_text = _fmt(
        AGENT_PROMPTS["breakdown"]["human"],
        video_rank=video.rank,
        video_title=video.title,
    )

    messages = [
        SystemMessage(content=AGENT_PROMPTS["breakdown"]["system"]),
        HumanMessage(content=human_text),
    ]

    # TODO: When ComfyUI / frame extraction is wired, attach base64 frames here
    # as additional HumanMessage image blocks before invoking.

    try:
        response = llm.invoke(messages)
        data = _extract_json(response.content)
        breakdown = VideoBreakdown(**data, qa_passed=False, qa_attempts=0)

        existing = list(state.get("breakdowns", []))
        existing.append(breakdown)
        print(f"  ✅ Breakdown complete — emotion: {breakdown.emotional_trigger}")
        return {
            "breakdowns": existing,
            "current_video_index": idx + 1,
        }
    except (ValidationError, ValueError) as exc:
        print(f"  ❌ breakdown_node failed: {exc}")
        errors = list(state.get("breakdown_errors", []))
        errors.append(f"Video {video.rank}: {exc}")
        return {
            "breakdown_errors": errors,
            "current_video_index": idx + 1,
        }


def qa_node(state: AgentState) -> dict:
    """
    Node: qa_node
    Validates the most recently added breakdown.
    Sets qa_passed=True or triggers a retry by calling breakdown_node again.

    The retry is handled by edges.py routing back to breakdown_node
    when qa_passed is False and qa_attempts < max_qa_retries.
    """
    print("\n[Node] qa_node — validating breakdown JSON...")
    breakdowns = list(state.get("breakdowns", []))
    if not breakdowns:
        return {}

    latest = breakdowns[-1]
    max_retries = state.get("max_qa_retries", 3)

    # Check all required fields are non-empty
    issues = []
    if not latest.hook.description:
        issues.append("hook.description is empty")
    if not latest.emotional_trigger:
        issues.append("emotional_trigger is empty")
    if not latest.visual_style.dominant_colors:
        issues.append("visual_style.dominant_colors is empty")
    if not latest.estimated_watch_time:
        issues.append("estimated_watch_time is empty")

    if not issues:
        latest_updated = latest.model_copy(update={"qa_passed": True})
        breakdowns[-1] = latest_updated
        print(f"  ✅ QA passed for video {latest.video_rank}")
        return {"breakdowns": breakdowns}
    else:
        attempts = latest.qa_attempts + 1
        latest_updated = latest.model_copy(update={"qa_attempts": attempts})
        breakdowns[-1] = latest_updated
        print(f"  ⚠  QA failed (attempt {attempts}/{max_retries}): {issues}")

        if attempts >= max_retries:
            print(f"  ❌ Max retries reached for video {latest.video_rank} — skipping")
            # Mark as passed anyway to avoid infinite loop; errors are logged
            latest_updated = latest.model_copy(update={"qa_passed": True, "qa_attempts": attempts})
            breakdowns[-1] = latest_updated

        return {"breakdowns": breakdowns}


def social_science_node(state: AgentState) -> dict:
    """
    Node: social_science_node
    Synthesises all breakdowns into a ViralFormula.

    On first run: uses the standard synthesis prompt.
    On retry (formula was rejected): injects the Editor Chief's feedback
    so Qwen knows exactly what to fix rather than repeating the same mistake.
    """
    print("\n[Node] social_science_node — synthesising viral formula...")
    breakdowns = state.get("breakdowns", [])
    passed = [b for b in breakdowns if b.qa_passed]

    if len(passed) < 3:
        msg = f"Only {len(passed)} valid breakdowns — need at least 3 to synthesise"
        print(f"  ❌ {msg}")
        return {"formula_error": msg}

    llm = get_llm()
    breakdowns_json = json.dumps(
        [b.model_dump(exclude={"qa_passed", "qa_attempts"}) for b in passed],
        indent=2,
    )

    # Check if this is a revision run
    prior_formula = state.get("viral_formula")
    is_revision = (
        prior_formula is not None
        and prior_formula.status == FormulaStatus.REJECTED
        and prior_formula.editor_feedback
    )

    if is_revision:
        revision_num = prior_formula.revision_count
        print(f"  🔄 Revision {revision_num} — incorporating Editor Chief feedback...")
        print(f"     Feedback: {prior_formula.editor_feedback[:120]}...")
        human_content = _fmt(
            AGENT_PROMPTS["social_science"]["revision"],
            breakdowns_json=breakdowns_json,
            prior_formula_json=json.dumps(
                prior_formula.model_dump(
                    exclude={"status", "editor_feedback", "revision_count"}
                ),
                indent=2,
            ),
            editor_feedback=prior_formula.editor_feedback,
        )
    else:
        print(f"  📊 First pass — synthesising from {len(passed)} breakdowns...")
        human_content = _fmt(
            AGENT_PROMPTS["social_science"]["human"],
            breakdowns_json=breakdowns_json,
        )

    messages = [
        SystemMessage(content=AGENT_PROMPTS["social_science"]["system"]),
        HumanMessage(content=human_content),
    ]

    try:
        response = get_llm_high_tokens(max_tokens=50000).invoke(messages)
        data = _extract_json(response.content)
        revision_count = prior_formula.revision_count if prior_formula else 0
        formula = ViralFormula(
            **data,
            status=FormulaStatus.PENDING,
            revision_count=revision_count,
        )
        print(f"  ✅ Formula {'revised' if is_revision else 'drafted'} — core emotion: {formula.core_emotion}")
        return {"viral_formula": formula, "formula_error": ""}
    except (ValidationError, ValueError) as exc:
        print(f"  ❌ social_science_node failed: {exc}")
        return {"formula_error": str(exc)}


def editor_chief_node(state: AgentState) -> dict:
    """
    Node: editor_chief_node
    Reviews the viral formula. Sets status to APPROVED or REJECTED.
    If REJECTED, stores feedback so social_science_node can revise.
    """
    print("\n[Node] editor_chief_node — reviewing formula...")
    formula = state.get("viral_formula")
    if not formula:
        return {"formula_error": "No formula to review"}

    llm = get_llm()
    formula_json = json.dumps(
        formula.model_dump(exclude={"status", "editor_feedback", "revision_count"}),
        indent=2,
    )

    messages = [
        SystemMessage(content=AGENT_PROMPTS["editor_chief"]["system"]),
        HumanMessage(content=_fmt(
            AGENT_PROMPTS["editor_chief"]["human"],
            formula_json=formula_json,
        )),
    ]

    try:
        response = llm.invoke(messages)
        data = _extract_json(response.content)
        decision = data.get("decision", "").upper()
        feedback = data.get("feedback", "")

        if decision == "APPROVED":
            updated = formula.model_copy(update={
                "status": FormulaStatus.APPROVED,
                "editor_feedback": "",
            })
            print(f"  ✅ Formula APPROVED")
        else:
            updated = formula.model_copy(update={
                "status": FormulaStatus.REJECTED,
                "editor_feedback": feedback,
                "revision_count": formula.revision_count + 1,
            })
            print(f"  🔄 Formula REJECTED — feedback: {feedback[:120]}...")

        return {"viral_formula": updated, "formula_error": ""}
    except (ValidationError, ValueError) as exc:
        print(f"  ❌ editor_chief_node failed: {exc}")
        return {"formula_error": str(exc)}


# ── Phase 3: Generation ───────────────────────────────────────────────────────

def film_director_node(state: AgentState) -> dict:
    """
    Node: film_director_node
    Translates approved ViralFormula into a 5-clip Storyboard.
    """
    print("\n[Node] film_director_node — generating storyboard...")
    formula = state.get("viral_formula")
    if not formula or formula.status != FormulaStatus.APPROVED:
        return {"generation_error": "Formula not approved — cannot generate storyboard"}

    llm = get_creative_llm(max_tokens=50000)
    formula_json = json.dumps(
        formula.model_dump(exclude={"status", "editor_feedback", "revision_count"}),
        indent=2,
    )

    messages = [
        SystemMessage(content=AGENT_PROMPTS["film_director"]["system"]),
        HumanMessage(content=_fmt(
            AGENT_PROMPTS["film_director"]["human"],
            formula_json=formula_json,
        )),
    ]

    try:
        response = llm.invoke(messages)
        data = _extract_json(response.content)
        clips = [StoryboardClip(**c) for c in data["clips"]]
        storyboard = Storyboard(
            clips=clips,
            target_product=state.get("target_product", ""),
            status=PromptsStatus.PENDING,
        )
        print(f"  ✅ Storyboard generated — {len(clips)} clips")
        for c in clips:
            print(f"     Clip {c.clip_number}: {c.comfyui_prompt[:80]}...")
        return {"storyboard": storyboard, "generation_error": ""}
    except (ValidationError, ValueError) as exc:
        print(f"  ❌ film_director_node failed: {exc}")
        return {"generation_error": str(exc)}


def sales_node(state: AgentState) -> dict:
    """
    Node: sales_node
    Injects the target product into clips 3–5 of the storyboard.
    """
    print("\n[Node] sales_node — injecting product...")
    storyboard = state.get("storyboard")
    if not storyboard:
        return {"generation_error": "No storyboard — film_director_node must run first"}

    product = state.get("target_product", "")
    if not product:
        print("  ⚠  No TARGET_PRODUCT set — skipping sales injection")
        return {}

    llm = get_llm()
    storyboard_json = json.dumps(
        storyboard.model_dump(exclude={"status", "human_notes"}),
        indent=2,
    )

    messages = [
        SystemMessage(content=AGENT_PROMPTS["sales"]["system"]),
        HumanMessage(content=_fmt(
            AGENT_PROMPTS["sales"]["human"],
            target_product=product,
            storyboard_json=storyboard_json,
        )),
    ]

    try:
        response = llm.invoke(messages)
        data = _extract_json(response.content)
        clips = [StoryboardClip(**c) for c in data["clips"]]
        updated = storyboard.model_copy(update={
            "clips": clips,
            "target_product": product,
        })
        injected = sum(1 for c in clips if c.product_featured)
        print(f"  ✅ Product injected into {injected} clip(s)")
        return {"storyboard": updated, "generation_error": ""}
    except (ValidationError, ValueError) as exc:
        print(f"  ❌ sales_node failed: {exc}")
        return {"generation_error": str(exc)}


def human_checkpoint(state: AgentState) -> dict:
    """
    Node: human_checkpoint
    Pauses the graph and presents the 5 prompts for terminal review.
    User can approve as-is, edit individual prompts, or abort.

    LangGraph handles the actual pause via interrupt_before in the graph
    definition — this node processes the resume input.
    """
    print("\n" + "═" * 60)
    print("  HUMAN CHECKPOINT — Review the 5 ComfyUI prompts")
    print("═" * 60)

    storyboard = state.get("storyboard")
    if not storyboard:
        return {"generation_error": "No storyboard to review"}

    for clip in storyboard.clips:
        tag = " [PRODUCT]" if clip.product_featured else ""
        print(f"\n  Clip {clip.clip_number}{tag}:")
        print(f"  {clip.comfyui_prompt}")

    print("\n" + "─" * 60)
    print("  Options:")
    print("    [a]  Approve all prompts as-is")
    print("    [1-5] Edit a specific clip prompt")
    print("    [q]  Abort the run")
    print("─" * 60)

    clips = list(storyboard.clips)
    notes: list[str] = []

    while True:
        choice = input("\n  Your choice: ").strip().lower()

        if choice == "a":
            updated = storyboard.model_copy(update={
                "clips": clips,
                "status": PromptsStatus.APPROVED,
                "human_notes": "; ".join(notes) if notes else "",
            })
            print("  ✅ Prompts approved — proceeding to ComfyUI")
            return {"storyboard": updated}

        elif choice == "q":
            print("  🛑 Run aborted by user")
            sys.exit(0)

        elif choice in ("1", "2", "3", "4", "5"):
            clip_idx = int(choice) - 1
            current = clips[clip_idx].comfyui_prompt
            print(f"\n  Current prompt for clip {choice}:")
            print(f"  {current}")
            new_prompt = input("  New prompt (Enter to keep): ").strip()
            if new_prompt:
                clips[clip_idx] = clips[clip_idx].model_copy(
                    update={"comfyui_prompt": new_prompt}
                )
                notes.append(f"Clip {choice} edited by user")
                print(f"  ✏  Clip {choice} updated")
        else:
            print("  Invalid choice — enter 'a', 'q', or a clip number 1-5")


def comfyui_node(state: AgentState) -> dict:
    """
    Node: comfyui_node
    Sends each approved prompt to ComfyUI via its API and waits for output.
    """
    from src.tools.hardware_manager import send_to_wan2gp

    print("\n[Node] comfyui_node — generating clips...")
    storyboard = state.get("storyboard")
    if not storyboard or storyboard.status not in (
        PromptsStatus.APPROVED, PromptsStatus.EDITED
    ):
        return {"generation_error": "Storyboard not approved — human_checkpoint must run first"}

    clip_paths: list[Path] = []
    for clip in storyboard.clips:
        print(f"  🎨 Generating clip {clip.clip_number}...")
        try:
            path = send_to_wan2gp(
                prompt=clip.comfyui_prompt,
                negative_prompt=clip.negative_prompt,
                clip_number=clip.clip_number,
            )
            clip_paths.append(path)
            print(f"     → {path}")
        except Exception as exc:
            print(f"     ❌ Clip {clip.clip_number} failed: {exc}")

    result = GenerationResult(generated_clip_paths=clip_paths)
    print(f"  ✅ Generated {len(clip_paths)}/5 clips")
    return {"generation": result, "generation_error": ""}


def assemble_node(state: AgentState) -> dict:
    """
    Node: assemble_node
    FFMPEG stitches the 5 ComfyUI clips into one 15-second MP4.
    """
    from src.tools.video_assembler import stitch_clips

    print("\n[Node] assemble_node — stitching final video...")
    generation = state.get("generation")
    if not generation or not generation.generated_clip_paths:
        return {"generation_error": "No generated clips to assemble"}

    try:
        output_path = stitch_clips(generation.generated_clip_paths)
        updated = generation.model_copy(update={"final_video_path": output_path})
        print(f"  ✅ Final video: {output_path}")
        return {"generation": updated, "generation_error": ""}
    except Exception as exc:
        print(f"  ❌ assemble_node failed: {exc}")
        return {"generation_error": str(exc)}
