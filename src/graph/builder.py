"""
builder.py — Assembles the LangGraph StateGraph from nodes and edges.

Checkpointing:
  The graph uses LangGraph's SQLite checkpointer to persist state after
  every node. This means:
    - Runs can be resumed from any point if they crash or are interrupted
    - You can re-run just Phase 3 without re-doing Phase 1 & 2
    - Each run gets a unique thread_id so runs don't overwrite each other

  Checkpoint database: data/checkpoints/pipeline.db
  Each run is identified by a thread_id (default: "run-{timestamp}")
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.state import AgentState
from src.graph.nodes import (
    fetch_trends,
    search_node,
    download_node,
    breakdown_node,
    qa_node,
    social_science_node,
    editor_chief_node,
    film_director_node,
    sales_node,
    human_checkpoint,
    comfyui_node,
    assemble_node,
)
from src.graph.edges import (
    after_fetch_trends,
    after_search_node,
    after_qa_node,
    after_social_science_node,
    after_editor_chief_node,
    after_film_director_node,
    after_sales_node,
    after_comfyui_node,
)

CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_DB  = CHECKPOINT_DIR / "pipeline.db"


def build_graph(use_checkpointing: bool = True):
    """
    Builds and compiles the full pipeline graph.

    Args:
        use_checkpointing: If True (default), attaches a SQLite checkpointer
                           so runs can be resumed. Set False for dry-run tests.

    Returns:
        (graph, checkpointer) tuple. checkpointer is None if not used.
    """
    g = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("fetch_trends",        fetch_trends)
    g.add_node("search_node",         search_node)
    g.add_node("download_node",       download_node)
    g.add_node("breakdown_node",      breakdown_node)
    g.add_node("qa_node",             qa_node)
    g.add_node("social_science_node", social_science_node)
    g.add_node("editor_chief_node",   editor_chief_node)
    g.add_node("film_director_node",  film_director_node)
    g.add_node("sales_node",          sales_node)
    g.add_node("human_checkpoint",    human_checkpoint)
    g.add_node("comfyui_node",        comfyui_node)
    g.add_node("assemble_node",       assemble_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("fetch_trends")

    # ── Phase 1 edges ─────────────────────────────────────────────────────────
    g.add_conditional_edges("fetch_trends",  after_fetch_trends,
                            {"search_node": "search_node", "end": END})
    g.add_conditional_edges("search_node",   after_search_node,
                            {"download_node": "download_node", "end": END})
    g.add_edge("download_node", "breakdown_node")

    # ── Phase 2 edges ─────────────────────────────────────────────────────────
    g.add_edge("breakdown_node", "qa_node")
    g.add_conditional_edges(
        "qa_node", after_qa_node,
        {
            "breakdown_node":      "breakdown_node",
            "social_science_node": "social_science_node",
        },
    )
    g.add_conditional_edges("social_science_node", after_social_science_node,
                            {"editor_chief_node": "editor_chief_node", "end": END})
    g.add_conditional_edges(
        "editor_chief_node", after_editor_chief_node,
        {
            "film_director_node":  "film_director_node",
            "social_science_node": "social_science_node",
            "end": END,
        },
    )

    # ── Phase 3 edges ─────────────────────────────────────────────────────────
    g.add_conditional_edges("film_director_node", after_film_director_node,
                            {"sales_node": "sales_node", "end": END})
    g.add_conditional_edges("sales_node", after_sales_node,
                            {"human_checkpoint": "human_checkpoint"})
    g.add_edge("human_checkpoint", "comfyui_node")
    g.add_conditional_edges("comfyui_node", after_comfyui_node,
                            {"assemble_node": "assemble_node", "end": END})
    g.add_edge("assemble_node", END)

    # ── Checkpointer ─────────────────────────────────────────────────────────
    if use_checkpointing:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        return g.compile(checkpointer=checkpointer), checkpointer

    return g.compile(), None
