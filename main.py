"""
main.py — Entry point for the Viral Video Agent.

Run modes:
    uv run python main.py                    # fresh run with new thread_id
    uv run python main.py --resume           # resume last interrupted run
    uv run python main.py --resume <id>      # resume a specific run by thread_id
    uv run python main.py --list             # list all saved runs
    uv run python main.py --phase3 <id>      # re-run only Phase 3 from a saved run
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.agents.llm_factory import ping_lm_studio
from src.state import initial_state, PromptsStatus, FormulaStatus
from src.graph.builder import build_graph, CHECKPOINT_DB


def parse_args():
    parser = argparse.ArgumentParser(description="Viral Video Agent")
    parser.add_argument("--resume",  nargs="?", const="last",
                        help="Resume last run or a specific thread_id")
    parser.add_argument("--list",    action="store_true",
                        help="List all saved checkpoint runs")
    parser.add_argument("--phase3",  metavar="THREAD_ID",
                        help="Re-run Phase 3 only from a saved run")
    return parser.parse_args()


def list_runs(checkpointer):
    """Print all saved runs from the checkpoint database."""
    print("\n📋  Saved runs:")
    print("─" * 60)
    try:
        import sqlite3
        conn = sqlite3.connect(str(CHECKPOINT_DB))
        rows = conn.execute(
            "SELECT DISTINCT thread_id, MAX(checkpoint_ns) as latest "
            "FROM checkpoints GROUP BY thread_id ORDER BY latest DESC"
        ).fetchall()
        conn.close()
        if not rows:
            print("  No saved runs found.")
        for thread_id, ts in rows:
            print(f"  {thread_id}")
    except Exception as e:
        print(f"  Could not read checkpoint DB: {e}")
    print("─" * 60)


def get_last_thread_id():
    """Get the most recently created thread_id from the database."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(CHECKPOINT_DB))
        row = conn.execute(
            "SELECT thread_id FROM checkpoints "
            "ORDER BY checkpoint_ns DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def main():
    args = parse_args()
    product = os.getenv("TARGET_PRODUCT", "Your Product Here")

    print(f"\n🎬  Viral Video Agent")
    print(f"    Target product : {product}")
    print(f"    LM Studio URL  : {os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1')}")
    print(f"    Generation mode: {os.getenv('GENERATION_MODE', 'colab')}")

    # Build graph with checkpointing
    graph, checkpointer = build_graph(use_checkpointing=True)

    # ── --list mode ───────────────────────────────────────────────────────────
    if args.list:
        list_runs(checkpointer)
        return

    # ── Determine thread_id ───────────────────────────────────────────────────
    if args.resume == "last" or args.phase3:
        thread_id = args.phase3 or get_last_thread_id()
        if not thread_id:
            print("❌  No saved runs found to resume.")
            sys.exit(1)
        print(f"\n🔄  Resuming run: {thread_id}")
    elif args.resume and args.resume != "last":
        thread_id = args.resume
        print(f"\n🔄  Resuming run: {thread_id}")
    else:
        # Fresh run — generate a unique thread_id
        thread_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\n🆕  New run: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    # ── LM Studio check (skip for phase3-only if Qwen not needed) ─────────────
    if not args.phase3:
        if not ping_lm_studio():
            print("\n❌  Cannot start — fix LM Studio connection first.")
            sys.exit(1)

    # ── --phase3 mode: patch saved state to re-run from film_director ─────────
    if args.phase3:
        print("\n🎬  Phase 3 only — loading saved state...")
        saved = graph.get_state(config)
        if not saved or not saved.values:
            print(f"❌  No saved state found for thread_id: {thread_id}")
            sys.exit(1)

        saved_values = saved.values
        formula = saved_values.get("viral_formula")
        if not formula:
            print("❌  No viral formula in saved state — run full pipeline first.")
            sys.exit(1)

        print(f"  ✅ Loaded state — formula emotion: {formula.core_emotion}")
        print(f"  🔄 Clearing Phase 3 state and re-running from film_director_node...")

        # Clear Phase 3 state so it re-runs cleanly
        graph.update_state(config, {
            "storyboard":       None,
            "generation":       None,
            "generation_error": "",
        }, as_node="editor_chief_node")

        print("\n🚀  Re-running Phase 3...\n")
        final_state = graph.invoke(None, config)

    else:
        # ── Fresh or resume run ───────────────────────────────────────────────
        # Check if there's existing checkpoint state to resume from
        saved = graph.get_state(config)
        if saved and saved.values and saved.values.get("breakdowns"):
            breakdowns = saved.values.get("breakdowns", [])
            formula = saved.values.get("viral_formula")
            print(f"\n♻️  Resuming — found {len(breakdowns)} breakdowns already saved")
            if formula:
                print(f"     Formula: {formula.core_emotion} ({formula.status.value})")
            print("     Continuing from last checkpoint...\n")
            final_state = graph.invoke(None, config)
        else:
            state = initial_state(target_product=product)
            print("\n🚀  Starting pipeline...\n")
            final_state = graph.invoke(state, config)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)
    print(f"\n  Thread ID : {thread_id}")
    print(f"  Resume with: uv run python main.py --resume {thread_id}")

    generation = final_state.get("generation")
    if generation and generation.final_video_path:
        print(f"\n  ✅ Final video: {generation.final_video_path}")
    else:
        print("\n  ⚠  No final video produced.")
        err = final_state.get("generation_error", "")
        if err:
            print(f"     Error: {err}")

    formula = final_state.get("viral_formula")
    if formula:
        print(f"\n  Viral formula core emotion : {formula.core_emotion}")
        print(f"  Formula status             : {formula.status.value}")

    breakdowns = final_state.get("breakdowns", [])
    print(f"\n  Videos analysed : {len(breakdowns)}")
    print(f"  QA passed       : {sum(1 for b in breakdowns if b.qa_passed)}")


if __name__ == "__main__":
    main()
