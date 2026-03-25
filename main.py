"""
main.py — Entry point for the Viral Video Agent.

Run with:
    uv run python main.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.agents.llm_factory import ping_lm_studio
from src.state import initial_state
from src.graph.builder import build_graph


def main():
    product = os.getenv("TARGET_PRODUCT", "Your Product Here")
    print(f"\n🎬  Viral Video Agent")
    print(f"    Target product : {product}")
    print(f"    LM Studio URL  : {os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1')}")

    # Fail fast if LM Studio isn't ready
    if not ping_lm_studio():
        print("\n❌  Cannot start — fix LM Studio connection first.")
        sys.exit(1)

    graph = build_graph()
    state = initial_state(target_product=product)

    print("\n🚀  Starting pipeline...\n")
    final_state = graph.invoke(state)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)

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
