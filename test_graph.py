"""
test_graph.py — Dry-run the full graph with stubbed tools.

This test patches out all I/O (Jina, yt-dlp, ComfyUI, FFMPEG) with
fakes so the entire node/edge wiring can be verified without any
external services beyond LM Studio.

What it proves:
  ✅ All imports resolve correctly
  ✅ StateGraph compiles without errors
  ✅ Every node receives and returns state correctly
  ✅ QA loop fires and passes
  ✅ Editor Chief consensus loop fires and approves
  ✅ Human checkpoint can be auto-approved (non-interactive mode)
  ✅ Final state contains a video path

Run with:
    uv run python test_graph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from dotenv import load_dotenv
load_dotenv()

# ── Step 1: Verify imports ────────────────────────────────────────────────────

print("\n── Step 1: Imports ─────────────────────────────────────────")
try:
    from src.state import initial_state, AgentState
    from src.graph.builder import build_graph
    from src.agents.llm_factory import ping_lm_studio
    print("✅  All imports OK")
except ImportError as e:
    print(f"❌  Import failed: {e}")
    sys.exit(1)


# ── Step 2: LM Studio ping ────────────────────────────────────────────────────

print("\n── Step 2: LM Studio ping ──────────────────────────────────")
if not ping_lm_studio():
    print("❌  LM Studio not reachable — cannot continue")
    sys.exit(1)


# ── Step 3: Graph compiles ────────────────────────────────────────────────────

print("\n── Step 3: Graph compilation ───────────────────────────────")
try:
    graph = build_graph()
    print("✅  StateGraph compiled OK")
except Exception as e:
    print(f"❌  Graph compilation failed: {e}")
    sys.exit(1)


# ── Step 4: Full dry-run with stubs ──────────────────────────────────────────

print("\n── Step 4: Full pipeline dry-run (stubs active) ────────────")
print("    Tools patched: jina_fetcher, yt_downloader,")
print("                   hardware_manager, video_assembler")
print("    Human checkpoint: auto-approved\n")

# ── Stub: jina_fetcher ────────────────────────────────────────────────────────
FAKE_MARKDOWN = """\
# YouTube Trending Today

1. [Man Builds AI Robot That Cooks Breakfast](https://www.youtube.com/watch?v=aaaa001) — 9.1M views
2. [I Lived on $1 a Day for 30 Days](https://www.youtube.com/watch?v=bbbb002) — 7.4M views
3. [This Dog Learned to Talk (Not Clickbait)](https://www.youtube.com/watch?v=cccc003) — 6.8M views
4. [World's Loudest Car vs Speed Camera](https://www.youtube.com/watch?v=dddd004) — 5.2M views
5. [Gordon Ramsay Tries Gas Station Sushi](https://www.youtube.com/watch?v=eeee005) — 4.9M views
"""

def fake_fetch_trending_markdown():
    print("  [STUB] jina_fetcher → returning fake markdown")
    return FAKE_MARKDOWN

# ── Stub: yt_downloader ───────────────────────────────────────────────────────
def fake_download_clip(url: str, rank: int) -> Path:
    path = Path(f"data/1_raw_clips/video_{rank:02d}.mp4")
    print(f"  [STUB] yt_downloader → fake path {path}")
    return path

# ── Stub: hardware_manager (ComfyUI) ─────────────────────────────────────────
def fake_send_to_comfyui(prompt: str, negative_prompt: str, clip_number: int) -> Path:
    path = Path(f"data/3_final_outputs/clip_{clip_number:02d}.mp4")
    print(f"  [STUB] comfyui → fake clip {path}")
    return path

# ── Stub: video_assembler ─────────────────────────────────────────────────────
def fake_stitch_clips(clip_paths: list) -> Path:
    path = Path("data/3_final_outputs/final_video.mp4")
    print(f"  [STUB] video_assembler → fake output {path}")
    return path

# ── Stub: human_checkpoint (non-interactive auto-approve) ────────────────────
from src.state import PromptsStatus

def fake_human_checkpoint(state: AgentState) -> dict:
    storyboard = state.get("storyboard")
    if not storyboard:
        return {"generation_error": "No storyboard"}
    print("  [STUB] human_checkpoint → auto-approving all prompts")
    updated = storyboard.model_copy(update={"status": PromptsStatus.APPROVED})
    return {"storyboard": updated}


# ── Run with all patches active ───────────────────────────────────────────────
with (
    patch("src.tools.jina_fetcher.fetch_trending_markdown",  fake_fetch_trending_markdown),
    patch("src.tools.yt_downloader.download_clip",           fake_download_clip),
    patch("src.tools.hardware_manager.send_to_wan2gp",       fake_send_to_comfyui),
    patch("src.tools.video_assembler.stitch_clips",          fake_stitch_clips),
    patch("src.graph.nodes.human_checkpoint",                fake_human_checkpoint),
):
    # Rebuild graph inside patch context so node references are fresh
    graph = build_graph()
    state = initial_state(target_product="Test Sneaker Brand")

    try:
        final = graph.invoke(state)
    except Exception as exc:
        print(f"\n❌  Graph crashed: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)


# ── Step 5: Assert final state ────────────────────────────────────────────────

print("\n── Step 5: Final state assertions ──────────────────────────")
failures = []

def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  ✅  {label}")
    else:
        print(f"  ❌  {label}" + (f" — {detail}" if detail else ""))
        failures.append(label)

acquisition = final.get("acquisition")
check("acquisition exists",       acquisition is not None)
check("trending videos found",    acquisition and len(acquisition.trending_videos) >= 1)

breakdowns = final.get("breakdowns", [])
check("at least 1 breakdown",     len(breakdowns) >= 1)
check("all breakdowns qa_passed", all(b.qa_passed for b in breakdowns),
      f"{sum(1 for b in breakdowns if not b.qa_passed)} failed QA")

formula = final.get("viral_formula")
check("viral formula exists",     formula is not None)
check("formula approved",         formula and formula.status.value == "approved",
      f"status={formula.status.value if formula else 'None'}")

storyboard = final.get("storyboard")
check("storyboard exists",        storyboard is not None)
check("storyboard has 5 clips",   storyboard and len(storyboard.clips) == 5,
      f"got {len(storyboard.clips) if storyboard else 0} clips")
check("storyboard approved",      storyboard and storyboard.status.value in ("approved", "edited"))

generation = final.get("generation")
check("generation result exists", generation is not None)
check("5 clip paths generated",   generation and len(generation.generated_clip_paths) == 5,
      f"got {len(generation.generated_clip_paths) if generation else 0}")
check("final video path set",     generation and generation.final_video_path is not None)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─' * 55}")
if not failures:
    print(f"Results: {10 - len(failures)}/10 passed")
    print("\nAll checks passed — graph is wired correctly 🎬")
    print("Next step: build the tools layer (jina_fetcher, yt_downloader, etc.)")
else:
    passed = 10 - len(failures)
    print(f"Results: {passed}/10 passed")
    print(f"\nFailed checks: {failures}")
    sys.exit(1)