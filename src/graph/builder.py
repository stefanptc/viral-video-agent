"""
builder.py — Assembles the LangGraph StateGraph from nodes and edges.

Import and call build_graph() in main.py to get a compiled, runnable graph.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

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


def build_graph():
    """
    Builds and compiles the full pipeline graph.

    Returns a CompiledGraph ready to invoke with an initial AgentState.
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
    g.add_conditional_edges("fetch_trends",  after_fetch_trends,  {"search_node": "search_node",   "end": END})
    g.add_conditional_edges("search_node",   after_search_node,   {"download_node": "download_node", "end": END})
    g.add_edge("download_node", "breakdown_node")

    # ── Phase 2 edges ─────────────────────────────────────────────────────────
    g.add_edge("breakdown_node", "qa_node")
    g.add_conditional_edges(
        "qa_node",
        after_qa_node,
        {
            "breakdown_node":      "breakdown_node",    # retry or next video
            "social_science_node": "social_science_node",
        },
    )
    g.add_conditional_edges("social_science_node", after_social_science_node,
                            {"editor_chief_node": "editor_chief_node", "end": END})
    g.add_conditional_edges(
        "editor_chief_node",
        after_editor_chief_node,
        {
            "film_director_node":  "film_director_node",
            "social_science_node": "social_science_node",  # revision loop
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

    return g.compile()
