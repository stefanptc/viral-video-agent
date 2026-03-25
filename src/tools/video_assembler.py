"""
video_assembler.py — Stitches 5 Wan2GP clips into one 15-second MP4.

How it works:
  FFmpeg concat demuxer reads a simple text manifest listing the 5 clip
  paths, then joins them in order with zero re-encoding (-c copy).
  This is near-instant and lossless — no quality degradation.

  If clips have mismatched codecs or resolutions (can happen if Wan2GP
  settings changed between clips), we fall back to a re-encode pass at
  a sane quality preset.

Output:
  data/3_final_outputs/final_video.mp4

Usage:
    from src.tools.video_assembler import stitch_clips
    output = stitch_clips([Path("clip_01.mp4"), ..., Path("clip_05.mp4")])

Standalone test:
    uv run python -m src.tools.video_assembler
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

OUTPUT_DIR   = Path("data/3_final_outputs")
FINAL_NAME   = "final_video.mp4"
EXPECTED_CLIPS = 5


# ── Main function ─────────────────────────────────────────────────────────────

def stitch_clips(clip_paths: list[Path]) -> Path:
    """
    Concatenate clip_paths into a single MP4 in OUTPUT_DIR.

    Args:
        clip_paths: Ordered list of .mp4 files (clip 1 → clip 5).

    Returns:
        Path to the stitched final_video.mp4.

    Raises:
        ValueError:      if fewer than 2 clips are provided.
        FileNotFoundError: if any clip path doesn't exist.
        RuntimeError:    if FFmpeg fails.
    """
    if len(clip_paths) < 2:
        raise ValueError(f"Need at least 2 clips to stitch, got {len(clip_paths)}")

    # Validate all inputs exist
    for p in clip_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Clip not found: {p}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / FINAL_NAME

    if len(clip_paths) != EXPECTED_CLIPS:
        print(f"  ⚠  Expected {EXPECTED_CLIPS} clips, got {len(clip_paths)} — stitching anyway")

    print(f"  🎞  Stitching {len(clip_paths)} clips → {output_path}")

    # Try fast lossless concat first, fall back to re-encode if it fails
    try:
        _concat_copy(clip_paths, output_path)
    except RuntimeError as exc:
        print(f"  ⚠  Lossless concat failed ({exc}) — falling back to re-encode...")
        _concat_reencode(clip_paths, output_path)

    # Verify output
    duration = _get_duration(output_path)
    size_mb   = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Final video: {output_path}")
    print(f"     Duration : {duration:.1f}s  |  Size: {size_mb:.1f} MB")

    return output_path


# ── FFmpeg strategies ─────────────────────────────────────────────────────────

def _write_concat_manifest(clip_paths: list[Path]) -> str:
    """
    Write a temporary FFmpeg concat manifest file.
    Returns the path as a string (caller manages cleanup via tempfile).
    """
    lines = []
    for p in clip_paths:
        # FFmpeg concat format requires absolute paths and escaped apostrophes
        abs_path = str(Path(p).resolve()).replace("'", "'\\''")
        lines.append(f"file '{abs_path}'")
    return "\n".join(lines)


def _concat_copy(clip_paths: list[Path], output: Path) -> None:
    """
    Fast path: concat with -c copy (no re-encode, lossless, near-instant).
    Fails if clips have incompatible streams.
    """
    manifest_content = _write_concat_manifest(clip_paths)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="concat_"
    ) as f:
        f.write(manifest_content)
        manifest_path = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f",       "concat",
        "-safe",    "0",
        "-i",       manifest_path,
        "-c",       "copy",
        "-movflags", "+faststart",   # optimise for web streaming
        "-loglevel", "error",
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(manifest_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr[:300])


def _concat_reencode(clip_paths: list[Path], output: Path) -> None:
    """
    Slow path: re-encode all clips to a common codec before joining.
    Used when clips have mismatched resolutions or codecs.
    Targets H.264 / AAC at CRF 18 (visually lossless).
    """
    manifest_content = _write_concat_manifest(clip_paths)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="concat_"
    ) as f:
        f.write(manifest_content)
        manifest_path = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f",       "concat",
        "-safe",    "0",
        "-i",       manifest_path,
        "-c:v",     "libx264",
        "-crf",     "18",
        "-preset",  "fast",
        "-c:a",     "aac",
        "-b:a",     "192k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(manifest_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Re-encode also failed:\n{result.stderr[:300]}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test with real clips if they exist, or generate 5 synthetic test clips.

    uv run python -m src.tools.video_assembler
    """
    print("Testing video_assembler...\n")

    # Check for real clips first
    real_clips = sorted(OUTPUT_DIR.glob("clip_0*.mp4"))
    if len(real_clips) >= 2:
        print(f"  Found {len(real_clips)} real clips — stitching those")
        clips_to_use = real_clips[:5]
    else:
        # Generate 5 synthetic 3-second colour clips with ffmpeg for testing
        print("  No real clips found — generating 5 synthetic test clips...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        clips_to_use = []
        colors = ["red", "blue", "green", "orange", "purple"]

        for i, color in enumerate(colors, 1):
            clip_path = OUTPUT_DIR / f"clip_{i:02d}_test.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c={color}:size=832x480:rate=16",
                "-f", "lavfi",
                "-i", "anullsrc=r=44100:cl=stereo",
                "-t", "3",
                "-c:v", "libx264", "-c:a", "aac",
                "-shortest",
                "-loglevel", "error",
                str(clip_path),
            ]
            subprocess.run(cmd, check=True)
            clips_to_use.append(clip_path)
            print(f"  🎨 Generated test clip {i}: {color}")

    print()
    try:
        output = stitch_clips(clips_to_use)
        duration = _get_duration(output)

        print(f"\n✅ Stitch complete")
        print(f"   Output   : {output}")
        print(f"   Duration : {duration:.1f}s (expected ~{len(clips_to_use) * 3}s)")

        if abs(duration - len(clips_to_use) * 3) > 1.0:
            print("   ⚠  Duration mismatch — check individual clip lengths")
        else:
            print("   ✅ Duration correct")

    except Exception as exc:
        print(f"\n❌ Failed: {exc}")
        sys.exit(1)
    finally:
        # Clean up synthetic test clips
        for p in OUTPUT_DIR.glob("clip_*_test.mp4"):
            p.unlink()
