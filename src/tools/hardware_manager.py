"""
hardware_manager.py — Unloads Qwen from LM Studio, then handles video generation.

Supports two modes via GENERATION_MODE in .env:

  "colab"  (default) — writes prompts to a JSON file and prints clear
                        instructions for generating clips manually via
                        Google Colab. Free, no local GPU needed.

  "wan2gp" — calls local Wan2GP headless (requires Wan2GP installed
              and working — use once your ROCm setup is stable).

Environment variables (.env):
  GENERATION_MODE     — "colab" or "wan2gp" (default: "colab")
  WAN2GP_DIR          — path to Wan2GP install (wan2gp mode only)
  WAN2GP_MODEL        — CLI flag e.g. --t2v-14B (wan2gp mode only)
  WAN2GP_PROFILE      — memory profile 1-5 (wan2gp mode only)
  LM_STUDIO_BASE_URL  — used for the unload call

Usage:
    from src.tools.hardware_manager import send_to_wan2gp
    path = send_to_wan2gp(prompt="...", negative_prompt="...", clip_number=1)

Standalone test:
    uv run python -m src.tools.hardware_manager
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GENERATION_MODE    = os.getenv("GENERATION_MODE", "colab")   # "colab" or "wan2gp"
WAN2GP_DIR         = Path(os.getenv("WAN2GP_DIR", str(Path.home() / "AI/Wan2GP")))
WAN2GP_MODEL       = os.getenv("WAN2GP_MODEL",   "--t2v-14B")
WAN2GP_PROFILE     = os.getenv("WAN2GP_PROFILE", "4")
OUTPUT_DIR         = Path("data/3_final_outputs")
TEMP_DIR           = Path("data/_wan2gp_tmp")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
GENERATION_TIMEOUT = 600

COLAB_NOTEBOOK_URL = (
    "https://colab.research.google.com/github/Isi-dev/Google-Colab_Notebooks"
    "/blob/main/Wan2_1_14B_T2V_GGUF_Free.ipynb"
)


# ── Step 1: Unload LM Studio ──────────────────────────────────────────────────

def unload_lm_studio() -> bool:
    """Unload all models from LM Studio to free VRAM."""
    print("  🔌 Unloading LM Studio model to free VRAM...")
    try:
        resp = httpx.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5.0)
        resp.raise_for_status()
        models = resp.json().get("data", [])

        if not models:
            print("  ℹ  No model loaded in LM Studio — nothing to unload")
            return True

        for model in models:
            model_id = model["id"]
            unload = httpx.delete(
                f"{LM_STUDIO_BASE_URL}/models/{model_id}", timeout=30.0
            )
            if unload.status_code in (200, 204):
                print(f"  ✅ Unloaded: {model_id}")
            else:
                print(f"  ⚠  Status {unload.status_code} when unloading {model_id}")
        return True

    except httpx.ConnectError:
        print("  ⚠  LM Studio not reachable — skipping unload")
        return False
    except Exception as exc:
        print(f"  ⚠  Unload failed: {exc} — continuing anyway")
        return False


# ── Step 2: Generate video ────────────────────────────────────────────────────

def send_to_wan2gp(
    prompt: str,
    negative_prompt: str,
    clip_number: int,
) -> Path:
    """
    Generate a single 3-second video clip.
    Routes to Colab manual mode or local Wan2GP based on GENERATION_MODE.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if GENERATION_MODE == "wan2gp":
        return _generate_wan2gp(prompt, negative_prompt, clip_number)
    else:
        return _generate_colab(prompt, negative_prompt, clip_number)


# ── Colab manual mode ─────────────────────────────────────────────────────────

def _generate_colab(prompt: str, negative_prompt: str, clip_number: int) -> Path:
    """
    Colab mode: saves the prompt to a JSON file and waits for the user
    to generate the clip manually via Google Colab, then place the output
    file in the correct location.
    """
    output_path = OUTPUT_DIR / f"clip_{clip_number:02d}.mp4"

    if output_path.exists():
        # Validate the existing file is a real video, not a placeholder
        size_mb = output_path.stat().st_size / (1024 * 1024)
        if size_mb > 0.1:  # more than 100KB = real clip
            print(f"  ⏭  Clip {clip_number} already exists ({size_mb:.1f} MB) — skipping")
            return output_path
        else:
            print(f"  ⚠  Clip {clip_number} exists but is too small ({size_mb:.2f} MB) — regenerating")
            output_path.unlink()

    # Save prompt details for the user
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    prompt_file = TEMP_DIR / f"clip_{clip_number:02d}_prompt.json"
    prompt_file.write_text(json.dumps({
        "clip_number":     clip_number,
        "prompt":          prompt,
        "negative_prompt": negative_prompt,
        "resolution":      "832x480",
        "frames":          49,
        "steps":           20,
        "output_filename": str(output_path.resolve()),
    }, indent=2))

    # Print clear instructions
    print(f"\n{'═' * 60}")
    print(f"  📋 COLAB GENERATION — Clip {clip_number}/5")
    print(f"{'═' * 60}")
    print(f"\n  1. Open this Colab notebook:")
    print(f"     {COLAB_NOTEBOOK_URL}")
    print(f"\n  2. In the notebook, use this prompt:")
    print(f"\n     PROMPT: {prompt}")
    print(f"\n     NEGATIVE: {negative_prompt}")
    print(f"\n  3. Settings:")
    print(f"     Resolution : 832x480")
    print(f"     Frames     : 49  (= 3 seconds)")
    print(f"     Steps      : 20")
    print(f"\n  4. Download the generated .mp4 and save it to:")
    print(f"     {output_path.resolve()}")
    print(f"\n  Prompt details also saved to: {prompt_file}")
    print(f"{'─' * 60}")

    # Wait for the file to appear
    print(f"\n  ⏳ Waiting for {output_path.name} ...")
    print(f"     (Press Ctrl+C to skip this clip and continue)\n")

    try:
        while not output_path.exists():
            time.sleep(5)
            print(f"     Still waiting... place the file at:")
            print(f"     {output_path.resolve()}")
    except KeyboardInterrupt:
        print(f"\n  ⏭  Clip {clip_number} skipped by user")
        # Create a placeholder so the pipeline can continue
        _create_placeholder_clip(output_path)

    print(f"  ✅ Clip {clip_number} ready: {output_path}")
    return output_path


def _create_placeholder_clip(path: Path) -> None:
    """Create a black placeholder clip so the pipeline doesn't break."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:size=832x480:rate=16",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", "3",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", "-loglevel", "error",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    print(f"  ⚠  Placeholder clip created for {path.name}")


# ── Local Wan2GP mode ─────────────────────────────────────────────────────────

def _generate_wan2gp(prompt: str, negative_prompt: str, clip_number: int) -> Path:
    """Local Wan2GP headless generation."""
    if not WAN2GP_DIR.exists():
        raise FileNotFoundError(
            f"Wan2GP not found at: {WAN2GP_DIR}\n"
            f"Set WAN2GP_DIR in .env or switch to GENERATION_MODE=colab"
        )

    output_path = OUTPUT_DIR / f"clip_{clip_number:02d}.mp4"
    if output_path.exists():
        # Validate the existing file is a real video, not a placeholder
        size_mb = output_path.stat().st_size / (1024 * 1024)
        if size_mb > 0.1:  # more than 100KB = real clip
            print(f"  ⏭  Clip {clip_number} already exists ({size_mb:.1f} MB) — skipping")
            return output_path
        else:
            print(f"  ⚠  Clip {clip_number} exists but is too small ({size_mb:.2f} MB) — regenerating")
            output_path.unlink()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    settings = {
        "prompt":              prompt,
        "negative_prompt":     negative_prompt,
        "num_frames":          49,
        "width":               832,
        "height":              480,
        "num_inference_steps": 20,
        "guidance_scale":      5.0,
        "seed":                clip_number * 1000,
    }
    settings_path = TEMP_DIR / f"clip_{clip_number:02d}_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))

    python_bin = _find_wan2gp_python()
    cmd = [
        python_bin, str(WAN2GP_DIR / "wgp.py"),
        WAN2GP_MODEL,
        "--process",    str(settings_path.resolve()),
        "--output-dir", str(OUTPUT_DIR.resolve()),
        "--profile",    WAN2GP_PROFILE,
        "--verbose",    "1",
    ]

    print(f"  🎬 Running Wan2GP (clip {clip_number})...")
    start = time.time()
    result = subprocess.run(
        cmd, cwd=str(WAN2GP_DIR), timeout=GENERATION_TIMEOUT
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"Wan2GP failed for clip {clip_number}")

    output_file = _find_newest_mp4(OUTPUT_DIR, after=start)
    if output_file is None:
        raise RuntimeError(f"No .mp4 found after Wan2GP run for clip {clip_number}")

    if output_file != output_path:
        output_file.rename(output_path)

    print(f"  ✅ Clip {clip_number} done in {elapsed:.0f}s → {output_path}")
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_wan2gp_python() -> str:
    conda_base = Path(os.getenv("CONDA_PREFIX", "")).parent.parent
    for env_name in ("wangp", "wan2gp"):
        p = conda_base / "envs" / env_name / "bin" / "python"
        if p.exists():
            return str(p)
    venv = WAN2GP_DIR / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _find_newest_mp4(directory: Path, after: float) -> Path | None:
    candidates = [
        f for f in directory.glob("*.mp4")
        if f.stat().st_mtime > after
    ]
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generation mode: {GENERATION_MODE}\n")

    print("── Step 1: LM Studio unload ─────────────────────────────────")
    ok = unload_lm_studio()
    print(f"  Result: {'✅ OK' if ok else '⚠  Skipped'}")

    if GENERATION_MODE == "colab":
        print("\n── Step 2: Colab mode test ──────────────────────────────────")
        print("  In colab mode the pipeline will print prompts and wait.")
        print("  No actual generation happens in this test.")
        print("  ✅ Colab mode configured correctly")
        print(f"\n  Notebook URL:\n  {COLAB_NOTEBOOK_URL}")
    else:
        print("\n── Step 2: Wan2GP mode — run with --full to test generation")
