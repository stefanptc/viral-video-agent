"""
state.py — Single source of truth for the entire LangGraph pipeline.

Every agent reads from and writes to this TypedDict. Pydantic models
enforce strict structure on all LLM-generated JSON so bad outputs are
caught at the QA nodes rather than blowing up downstream.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class FormulaStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"


class PromptsStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"   # human approved in terminal
    EDITED    = "edited"     # human approved with changes


# ──────────────────────────────────────────────
# Phase 1 — Acquisition models
# ──────────────────────────────────────────────

class TrendingVideo(BaseModel):
    """One entry returned by the Search Agent."""
    title: str = Field(..., description="YouTube video title as it appears on the trending page")
    url: str   = Field(..., description="Full YouTube URL, e.g. https://www.youtube.com/watch?v=...")
    rank: int  = Field(..., ge=1, le=10, description="Position on the trending list (1 = most viral)")

    @field_validator("url")
    @classmethod
    def must_be_youtube(cls, v: str) -> str:
        if "youtube.com" not in v and "youtu.be" not in v:
            raise ValueError(f"URL does not look like a YouTube link: {v}")
        return v


class AcquisitionResult(BaseModel):
    """Output of the full Phase 1 pipeline."""
    trending_videos: list[TrendingVideo] = Field(
        ..., min_length=1, max_length=10,
        description="Ordered list of trending videos (up to 10)"
    )
    raw_clip_paths: list[Path] = Field(
        default_factory=list,
        description="Absolute paths to the downloaded 60-second MP4 clips"
    )

    @field_validator("trending_videos")
    @classmethod
    def must_have_unique_ranks(cls, v: list[TrendingVideo]) -> list[TrendingVideo]:
        ranks = [video.rank for video in v]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Duplicate ranks found in trending_videos")
        return sorted(v, key=lambda x: x.rank)


# ──────────────────────────────────────────────
# Phase 2 — Analysis models
# ──────────────────────────────────────────────

class HookAnalysis(BaseModel):
    """What happens in the first 3 seconds of the video."""
    description: str = Field(..., description="Plain-language description of the opening hook")
    hook_type: str   = Field(..., description="e.g. 'shock cut', 'question', 'action', 'text reveal'")


class VisualStyle(BaseModel):
    cuts_per_minute: int  = Field(..., ge=0,    description="Estimated edit cuts per minute")
    has_text_overlays: bool
    dominant_colors: list[str] = Field(..., max_length=5, description="Hex or plain-color names")
    aspect_ratio: str           = Field(...,    description="e.g. '9:16', '16:9', '1:1'")


class AudioProfile(BaseModel):
    music_intensity: str  = Field(..., description="'none' | 'low' | 'medium' | 'high' | 'drop'")
    has_sound_effects: bool
    voice_tone: str       = Field(..., description="e.g. 'energetic', 'calm', 'whispering', 'none'")
    has_trending_audio: bool = Field(False, description="Whether audio uses a trending/viral sound")


class VideoBreakdown(BaseModel):
    """
    Structured analysis of a single 60-second clip.
    The QA node validates this before it moves to the Social Science Agent.
    """
    video_rank: int          = Field(..., ge=1, le=10)
    video_title: str
    hook: HookAnalysis
    visual_style: VisualStyle
    audio_profile: AudioProfile
    emotional_trigger: str   = Field(..., description="Primary emotion targeted: curiosity, fear, joy, etc.")
    estimated_watch_time: str = Field(..., description="Why viewers likely watch past 3s, in one sentence")

    # QA bookkeeping — set by the graph, not the LLM
    qa_passed: bool = False
    qa_attempts: int = 0


class ViralFormula(BaseModel):
    """
    Synthesised by the Social Science Agent from all 10 breakdowns.
    Must survive the Editor Chief Agent's review before Phase 3 starts.
    """
    core_emotion: str          = Field(..., description="The single dominant emotion driving virality")
    pacing_rules: list[str]    = Field(..., min_length=2, max_length=5)
    visual_mandates: list[str] = Field(..., min_length=2, max_length=5)
    audio_cues: list[str]      = Field(..., min_length=1, max_length=4)
    hook_blueprint: str        = Field(..., description="Exact recipe for the first 3 seconds")

    # Consensus loop bookkeeping
    status: FormulaStatus = FormulaStatus.PENDING
    editor_feedback: str  = ""     # populated when status == REJECTED
    revision_count: int   = 0


# ──────────────────────────────────────────────
# Phase 3 — Generation models
# ──────────────────────────────────────────────

class StoryboardClip(BaseModel):
    """One 3-second clip in the 5-clip storyboard."""
    clip_number: int    = Field(..., ge=1, le=5)
    duration_seconds: Literal[3] = 3
    comfyui_prompt: str = Field(
        ...,
        description=(
            "Comma-separated Stable Diffusion image prompt. "
            "Describe only what the camera sees: lighting, angle, subject, action, style."
        )
    )
    negative_prompt: str = Field(
        default="blurry, low quality, text, watermark, logo",
        description="Comma-separated negative prompt for ComfyUI"
    )
    product_featured: bool = Field(
        False,
        description="True if the Sales Agent injected the product into this clip"
    )


class Storyboard(BaseModel):
    """Five 3-second clips = 15-second video."""
    clips: list[StoryboardClip] = Field(..., min_length=5, max_length=5)
    target_product: str         = Field(..., description="Product being promoted")

    # Human checkpoint bookkeeping
    status: PromptsStatus    = PromptsStatus.PENDING
    human_notes: str         = ""   # reviewer comments from terminal

    @field_validator("clips")
    @classmethod
    def clips_must_be_sequential(cls, v: list[StoryboardClip]) -> list[StoryboardClip]:
        numbers = [c.clip_number for c in v]
        if sorted(numbers) != list(range(1, 6)):
            raise ValueError("clips must be numbered 1–5 with no gaps or duplicates")
        return sorted(v, key=lambda c: c.clip_number)


class GenerationResult(BaseModel):
    """Tracks ComfyUI output paths and final assembly."""
    generated_clip_paths: list[Path] = Field(
        default_factory=list,
        description="Paths to the 5 ComfyUI-generated MP4/GIF clips"
    )
    final_video_path: Path | None = Field(
        None,
        description="Path to the FFMPEG-stitched 15-second MP4"
    )


# ──────────────────────────────────────────────
# Master LangGraph State
# ──────────────────────────────────────────────

from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    """
    The single state object threaded through every LangGraph node.

    Naming convention:
      - snake_case fields map 1:1 to pipeline phases
      - Fields suffixed _error hold the last error string so the
        graph can branch to a retry or human-escalation node
    """

    # ── Internal message history (LangGraph managed) ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Phase 1: Acquisition ──
    raw_markdown: str                        # Jina-fetched markdown of trends24.in
    acquisition: Optional[AcquisitionResult]
    acquisition_error: str

    # ── Phase 2: Analysis ──
    breakdowns: list[VideoBreakdown]         # grows as each clip is processed
    breakdown_errors: list[str]             # per-video errors (index matches breakdowns)
    viral_formula: Optional[ViralFormula]
    formula_error: str

    # ── Phase 3: Generation ──
    storyboard: Optional[Storyboard]
    generation: Optional[GenerationResult]
    generation_error: str

    # ── Control flow ──
    current_video_index: int                # which of the 10 clips we're processing
    max_qa_retries: int                     # default 3, set in main.py
    max_formula_revisions: int              # default 3, set in main.py
    target_product: str                     # from .env TARGET_PRODUCT


# ──────────────────────────────────────────────
# Helper — build a clean initial state
# ──────────────────────────────────────────────

def initial_state(target_product: str) -> AgentState:
    """Call this in main.py to get a clean starting state."""
    return AgentState(
        messages=[],
        raw_markdown="",
        acquisition=None,
        acquisition_error="",
        breakdowns=[],
        breakdown_errors=[],
        viral_formula=None,
        formula_error="",
        storyboard=None,
        generation=None,
        generation_error="",
        current_video_index=0,
        max_qa_retries=3,
        max_formula_revisions=3,
        target_product=target_product,
    )