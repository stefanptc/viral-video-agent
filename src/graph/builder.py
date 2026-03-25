"""
builder.py — Assembles the LangGraph StateGraph from nodes and edges.

Checkpointing:
  The graph uses LangGraph's SQLite checkpointer to persist state after
  every node. Falls back to MemorySaver if the sqlite package is not
  installed (run: uv add langgraph-checkpoint-sqlite to enable persistence).

  Checkpoint database: data/checkpoints/pipeline.db
  Each run is identified by a thread_id (default: "run-{timestamp}")
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.graph.nodes import (
    fetch_trends, search_node, download_node, breakdown_node, qa_node,
    social_science_node, editor_chief_node, film_director_node,
    sales_node, human_checkpoint, comfyui_node, assemble_node,
)
from src.graph.edges import (
    after_fetch_trends, after_search_node, after_qa_node,
    after_social_science_node, after_editor_chief_node,
    after_film_director_node, after_sales_node, after_comfyui_node,
)

CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_DB  = CHECKPOINT_DIR / "pipeline.db"


def _make_checkpointer(use_checkpointing: bool):
    """
    Returns the best available checkpointer.
    Tries SQLite first (persistent), falls back to MemorySaver (in-process only).
    """
    if not use_checkpointing:
        return None

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Try SQLite (requires: uv add langgraph-checkpoint-sqlite)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        print(f"  💾 Checkpointer: SQLite → {CHECKPOINT_DB}")
        return checkpointer
    except ImportError:
        pass

    # Try the newer package name
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        print(f"  💾 Checkpointer: SQLite → {CHECKPOINT_DB}")
        return checkpointer
    except ImportError:
        pass

    # Fall back to MemorySaver — works but state is lost between processes
    from langgraph.checkpoint.memory import MemorySaver
    print("  ⚠  SQLite checkpointer not installed — using MemorySaver")
    print("     Install for persistence: uv add langgraph-checkpoint-sqlite")
    return MemorySaver()


def build_graph(use_checkpointing: bool = True):
    """
    Builds and compiles the full pipeline graph.

    Returns:
        (compiled_graph, checkpointer) tuple.
        checkpointer is None if use_checkpointing=False.
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
    g.add_conditional_edges("fetch_trends", after_fetch_trends,
                            {"search_node": "search_node", "end": END})
    g.add_conditional_edges("search_node",  after_search_node,
                            {"download_node": "download_node", "end": END})
    g.add_edge("download_node", "breakdown_node")

    # ── Phase 2 edges ─────────────────────────────────────────────────────────
    g.add_edge("breakdown_node", "qa_node")
    g.add_conditional_edges(
        "qa_node", after_qa_node,
        {"breakdown_node": "breakdown_node", "social_science_node": "social_science_node"},
    )
    g.add_conditional_edges("social_science_node", after_social_science_node,
                            {"editor_chief_node": "editor_chief_node", "end": END})
    g.add_conditional_edges(
        "editor_chief_node", after_editor_chief_node,
        {"film_director_node": "film_director_node",
         "social_science_node": "social_science_node", "end": END},
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

    # ── Attach checkpointer ───────────────────────────────────────────────────
    checkpointer = _make_checkpointer(use_checkpointing)
    if checkpointer:
        return g.compile(checkpointer=checkpointer), checkpointer
    return g.compile(), None
