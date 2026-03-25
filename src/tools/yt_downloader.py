"""
yt_downloader.py — Downloads exactly 0:00–1:00 of a YouTube video.

Design decisions:
  - Uses yt-dlp's --download-sections to clip server-side where possible,
    falling back to ffmpeg post-processing when the format doesn't support it.
  - Enforces a hard 60-second ceiling — if the clip is longer for any reason,
    ffmpeg trims it before the file is returned.
  - Output goes to data/1_raw_clips/video_{rank:02d}.mp4
  - Skips re-download if the file already exists (safe to re-run).
  - All yt-dlp output is captured; only errors are surfaced to the caller.

Requirements (system-level, not pip):
    sudo apt install yt-dlp ffmpeg

Usage:
    from src.tools.yt_downloader import download_clip
    path = download_clip("https://youtube.com/watch?v=abc123", rank=1)

Run standalone to test a single URL:
    uv run python -m src.tools.yt_downloader <youtube_url> [rank]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR      = Path("data/1_raw_clips")
CLIP_DURATION   = 60          # seconds — hard ceiling
MAX_FILESIZE    = "150M"      # yt-dlp guard against accidentally huge downloads
# Best mp4 under 1080p — keeps file sizes manageable for Qwen's vision context
FORMAT_SELECTOR = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best"


# ── Main function ─────────────────────────────────────────────────────────────

def download_clip(url: str, rank: int) -> Path:
    """
    Download exactly the first 60 seconds of a YouTube video.

    Args:
        url:  Full YouTube URL (youtube.com or youtu.be).
        rank: Position in the trending list (used for the output filename).

    Returns:
        Path to the downloaded .mp4 file.

    Raises:
        RuntimeError: if yt-dlp fails or the output file is not created.
        FileNotFoundError: if yt-dlp or ffmpeg are not installed.
    """
    _check_dependencies()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"video_{rank:02d}.mp4"

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        if size_mb > 0.5:  # more than 500KB = real clip
            print(f"  ⏭  [{rank}] Already downloaded ({size_mb:.1f} MB) — skipping ({output_path})")
            return output_path
        else:
            print(f"  ⚠  [{rank}] Existing file too small ({size_mb:.2f} MB) — re-downloading")
            output_path.unlink()

    # yt-dlp writes to a temp name then renames; use a template
    output_template = str(OUTPUT_DIR / f"video_{rank:02d}.%(ext)s")

    cmd = [
        "yt-dlp",
        # ── Clip to first 60 seconds ──────────────────────────────────────
        "--download-sections", f"*0-{CLIP_DURATION}",
        "--force-keyframes-at-cuts",   # cleaner cut point
        # ── Format ───────────────────────────────────────────────────────
        "--format",    FORMAT_SELECTOR,
        "--merge-output-format", "mp4",
        # ── Safety limits ─────────────────────────────────────────────────
        "--max-filesize", MAX_FILESIZE,
        "--no-playlist",              # never download a whole playlist
        # ── Output ────────────────────────────────────────────────────────
        "--output",    output_template,
        "--no-warnings",
        "--quiet",
        url,
    ]

    print(f"  ⬇  [{rank}] Downloading 0–{CLIP_DURATION}s clip...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed for rank {rank} ({url}):\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    # yt-dlp may produce .mp4 or .webm depending on availability
    # Find whatever it created and rename to our canonical .mp4 name
    candidates = list(OUTPUT_DIR.glob(f"video_{rank:02d}.*"))
    if not candidates:
        raise RuntimeError(
            f"yt-dlp exited cleanly but no output file found for rank {rank}.\n"
            f"URL: {url}"
        )

    raw_file = candidates[0]
    if raw_file.suffix != ".mp4":
        # ffmpeg-convert to mp4 (rare but possible with some formats)
        _convert_to_mp4(raw_file, output_path)
        raw_file.unlink()
    elif raw_file != output_path:
        raw_file.rename(output_path)

    # Hard trim safety net — ensure file is no longer than CLIP_DURATION
    _trim_to_limit(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"       ✅ Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_dependencies() -> None:
    """Fail fast with a helpful message if yt-dlp or ffmpeg are missing."""
    for tool in ("yt-dlp", "ffmpeg"):
        result = subprocess.run(
            ["which", tool], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise FileNotFoundError(
                f"'{tool}' not found. Install with:\n"
                f"    sudo apt install {tool.replace('-', '')}"
            )


def _convert_to_mp4(src: Path, dst: Path) -> None:
    """Convert any video file to mp4 using ffmpeg."""
    print(f"       🔄 Converting {src.suffix} → .mp4...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "libx264", "-c:a", "aac",
        "-t", str(CLIP_DURATION),
        "-loglevel", "error",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr[:300]}")


def _trim_to_limit(path: Path) -> None:
    """
    Safety net: if the file is longer than CLIP_DURATION, trim it in-place.
    This catches cases where --download-sections didn't clip precisely.
    """
    # Check duration with ffprobe
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return  # can't probe — skip trim, not fatal

    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        return

    if duration <= CLIP_DURATION + 1:  # 1s tolerance
        return

    print(f"       ✂  Trimming {duration:.1f}s → {CLIP_DURATION}s...")
    tmp = path.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(path),
        "-t", str(CLIP_DURATION),
        "-c", "copy",
        "-loglevel", "error",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)  # trim failed — keep original, not fatal


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test a single download:
        uv run python -m src.tools.yt_downloader https://youtube.com/watch?v=xxx
        uv run python -m src.tools.yt_downloader https://youtube.com/watch?v=xxx 3
    """
    if len(sys.argv) < 2:
        print("Usage: uv run python -m src.tools.yt_downloader <url> [rank]")
        print("Example: uv run python -m src.tools.yt_downloader 'https://youtube.com/watch?v=dQw4w9WgXcQ' 1")
        sys.exit(1)

    test_url  = sys.argv[1]
    test_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"Testing yt_downloader...")
    print(f"  URL  : {test_url}")
    print(f"  Rank : {test_rank}\n")

    try:
        output = download_clip(test_url, test_rank)
        size   = output.stat().st_size / (1024 * 1024)

        # Verify duration with ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(output)],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip()) if probe.returncode == 0 else -1

        print(f"\n✅ Download complete")
        print(f"   File     : {output}")
        print(f"   Size     : {size:.1f} MB")
        print(f"   Duration : {duration:.1f}s (limit: {CLIP_DURATION}s)")

        if duration > CLIP_DURATION + 1:
            print(f"   ⚠  Duration exceeds limit — check _trim_to_limit()")
        else:
            print(f"   ✅ Duration within limit")

    except FileNotFoundError as e:
        print(f"\n❌ Dependency missing: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)
