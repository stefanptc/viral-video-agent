"""
edges.py — All routing logic for the LangGraph state machine.

Each function is a conditional edge: it receives AgentState and returns
a string that LangGraph uses to pick the next node.

Routing map:
  after fetch_trends      → search_node | END (on error)
  after search_node       → download_node | END (on error)
  after download_node     → breakdown_node
  after breakdown_node    → qa_node
  after qa_node           → breakdown_node (retry) | breakdown_node (next video) | social_science_node
  after social_science_node → editor_chief_node | END (on error)
  after editor_chief_node → film_director_node (approved) | social_science_node (retry) | END (max revisions)
  after film_director_node → sales_node | END (on error)
  after sales_node        → human_checkpoint
  after human_checkpoint  → comfyui_node
  after comfyui_node      → assemble_node | END (on error)
"""

from __future__ import annotations

from src.state import AgentState, FormulaStatus, PromptsStatus


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def after_fetch_trends(state: AgentState) -> str:
    if state.get("acquisition_error"):
        print(f"\n[Edge] fetch_trends failed — aborting: {state['acquisition_error']}")
        return "end"
    return "search_node"


def after_search_node(state: AgentState) -> str:
    if state.get("acquisition_error"):
        print(f"\n[Edge] search_node failed — aborting: {state['acquisition_error']}")
        return "end"
    return "download_node"


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def after_qa_node(state: AgentState) -> str:
    """
    Three possible routes:
      1. Latest breakdown failed QA and has retries left → back to breakdown_node
         (current_video_index is NOT advanced so it re-processes the same video)
      2. More videos to process → breakdown_node (index was advanced in breakdown_node)
      3. All videos done → social_science_node
    """
    breakdowns = state.get("breakdowns", [])
    acquisition = state.get("acquisition")
    if not acquisition:
        return "social_science_node"

    total_videos = len(acquisition.trending_videos)
    max_retries = state.get("max_qa_retries", 3)

    # Check if the latest breakdown needs a retry
    if breakdowns:
        latest = breakdowns[-1]
        if not latest.qa_passed and latest.qa_attempts < max_retries:
            print(f"\n[Edge] QA retry for video {latest.video_rank} "
                  f"(attempt {latest.qa_attempts}/{max_retries})")
            # Step back the index so breakdown_node re-processes this video
            return "breakdown_node"

    # Check if there are more videos to process
    current_idx = state.get("current_video_index", 0)
    if current_idx < total_videos:
        print(f"\n[Edge] Moving to video {current_idx + 1}/{total_videos}")
        return "breakdown_node"

    print(f"\n[Edge] All {total_videos} videos processed → social_science_node")
    return "social_science_node"


def after_social_science_node(state: AgentState) -> str:
    if state.get("formula_error"):
        print(f"\n[Edge] social_science_node failed — aborting: {state['formula_error']}")
        return "end"
    return "editor_chief_node"


def after_editor_chief_node(state: AgentState) -> str:
    """
    Three possible routes:
      1. APPROVED → film_director_node
      2. REJECTED and revisions remaining → social_science_node
      3. REJECTED and max revisions hit → end
    """
    formula = state.get("viral_formula")
    max_revisions = state.get("max_formula_revisions", 3)

    if not formula:
        return "end"

    if formula.status == FormulaStatus.APPROVED:
        print("\n[Edge] Formula approved → film_director_node")
        return "film_director_node"

    if formula.revision_count >= max_revisions:
        print(f"\n[Edge] Max revisions ({max_revisions}) reached — aborting")
        return "end"

    print(f"\n[Edge] Formula rejected (revision {formula.revision_count}/{max_revisions}) "
          f"→ social_science_node")
    return "social_science_node"


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def after_film_director_node(state: AgentState) -> str:
    if state.get("generation_error"):
        print(f"\n[Edge] film_director_node failed — aborting: {state['generation_error']}")
        return "end"
    return "sales_node"


def after_sales_node(state: AgentState) -> str:
    # Sales injection is best-effort — we continue even if it fails
    if state.get("generation_error"):
        print(f"\n[Edge] sales_node error (non-fatal): {state['generation_error']}")
    return "human_checkpoint"


def after_comfyui_node(state: AgentState) -> str:
    generation = state.get("generation")
    if not generation or not generation.generated_clip_paths:
        print("\n[Edge] No clips generated — aborting")
        return "end"
    return "assemble_node"