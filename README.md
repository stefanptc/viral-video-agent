# Viral Video Agent 🎬

An autonomous agentic pipeline that identifies real-time YouTube trends, deconstructs their psychological mechanics using a local LLM, and synthesises a 15-second AI-generated promotional video.

Built with LangGraph, Qwen 3.5 (via LM Studio), and Wan2GP on a fully local stack.

---

## What It Does

1. **Fetches** real trending YouTube data from youtube.trends24.in via Jina AI
2. **Downloads** the first 60 seconds of each trending video
3. **Analyses** each clip with Qwen — hook, visual style, audio profile, emotional trigger
4. **Synthesises** a "Viral Formula" connecting the common patterns across all videos
5. **Generates** a 5-clip storyboard optimised for the formula
6. **Injects** your target product naturally into the storyboard
7. **Pauses** for your review and approval of the prompts
8. **Generates** 5 video clips via Wan2GP (local) or Google Colab (free cloud)
9. **Stitches** them into a final 15-second MP4 with FFmpeg

---

## Architecture

```
viral-video-agent/
├── main.py                        # Entry point
├── src/
│   ├── state.py                   # Pydantic models + LangGraph AgentState
│   ├── agents/
│   │   ├── llm_factory.py         # LM Studio connection (ChatOpenAI wrapper)
│   │   └── prompts.py             # All 6 agent system prompts
│   ├── graph/
│   │   ├── nodes.py               # Every LangGraph node function
│   │   ├── edges.py               # All conditional routing logic
│   │   └── builder.py             # StateGraph assembly
│   └── tools/
│       ├── jina_fetcher.py        # Fetches trending page as Markdown
│       ├── yt_downloader.py       # Downloads 60s clips via yt-dlp
│       ├── hardware_manager.py    # Unloads LM Studio + drives Wan2GP
│       └── video_assembler.py     # FFmpeg concat of 5 clips → final MP4
├── data/
│   ├── 1_raw_clips/               # Downloaded 60s MP4s
│   ├── 2_breakdowns/              # JSON analyses (not committed)
│   └── 3_final_outputs/           # Generated clips + final video
├── comfy_workflows/
│   └── api_workflow.json          # Wan2GP settings reference
├── test_agents.py                 # Tests LM Studio connection + 4 agent calls
└── test_graph.py                  # Full dry-run with stubbed tools (10/10)
```

---

## The 6 Agents

| Agent | Role | Output |
|-------|------|--------|
| **Search Agent** | Extracts top 10 trending video titles + URLs from Markdown | `AcquisitionResult` |
| **Breakdown Agent** | Analyses a 60s clip — hook, visual style, audio, emotion | `VideoBreakdown` |
| **Social Science Agent** | Synthesises 10 breakdowns into a Viral Formula | `ViralFormula` |
| **Editor Chief Agent** | Approves or rejects the formula with specific feedback | `APPROVED / REJECTED` |
| **Film Director Agent** | Translates formula into 5 ComfyUI video prompts | `Storyboard` |
| **Sales Agent** | Injects target product into clips 3–5 | Updated `Storyboard` |

---

## Pipeline Graph

```
fetch_trends → search_node → download_node → breakdown_node ⟲ qa_node
                                                    ↓ (all videos done)
                                          social_science_node ⟲ editor_chief_node
                                                    ↓ (approved)
                                          film_director_node → sales_node
                                                    ↓
                                          human_checkpoint (terminal review)
                                                    ↓
                                          comfyui_node → assemble_node → END
```

**Loops:**
- `breakdown_node ↔ qa_node`: retries up to 3× if JSON is invalid
- `social_science_node ↔ editor_chief_node`: revises formula up to 3× if rejected, with feedback passed back

---

## Hardware & Stack

| Component | Spec |
|-----------|------|
| GPU | AMD Radeon RX 7900 XTX (24GB VRAM) |
| RAM | 64GB DDR5 |
| ROCm | 6.4.0 |
| LLM | Qwen 3.5 35B A3B (via LM Studio) |
| Video gen | Wan2GP v10 with Wan2.1 14B BF16 |
| Python | 3.12 (uv managed) |
| OS | Ubuntu 24.04 (Noble) |

---

## Prerequisites

### System packages
```bash
sudo apt install ffmpeg yt-dlp
```

### LM Studio
- Download from https://lmstudio.ai
- Load **Qwen 3.5 35B A3B** (or similar Qwen 3.x model)
- Enable local server on `http://localhost:1234`

### Wan2GP
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git ~/AI/Wan2GP
cd ~/AI/Wan2GP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.4
python -m pip install -r requirements.txt --no-deps
```

First launch (downloads ~15GB of model weights):
```bash
python wgp.py --t2v-14B --profile 4 --attention sdpa --steps 20
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/viral-video-agent.git
cd viral-video-agent
uv sync
cp .env.example .env   # then edit with your values
```

### Environment variables (`.env`)
```env
# LM Studio
LM_STUDIO_BASE_URL="http://localhost:1234/v1"
LM_STUDIO_MODEL="qwen3.5-35b-a3b"

# Jina AI (free tier works, get key at https://jina.ai)
JINA_API_KEY="your_key_here"
JINA_TARGET_URL="https://youtube.trends24.in/"

# Wan2GP
WAN2GP_DIR="/home/YOUR_USERNAME/AI/Wan2GP"
WAN2GP_MODEL="--t2v-14B"
WAN2GP_PROFILE="4"

# Generation mode: "wan2gp" (local) or "colab" (manual cloud)
GENERATION_MODE="wan2gp"

# Your product
TARGET_PRODUCT="Your Product Name Here"
```

### AMD GPU environment (add to `~/.bashrc`)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

---

## Running

### Quick smoke test (no GPU needed)
```bash
uv run python test_agents.py
```
Tests: LM Studio ping, Search Agent JSON output, Editor Chief approve/reject paths.

### Full dry-run (stubs all I/O, uses real LLM)
```bash
uv run python test_graph.py
```
Expected: 10/10 assertions pass. Takes ~5 minutes (real Qwen inference).

### Real pipeline
```bash
uv run python main.py
```

The pipeline runs automatically through Phases 1 and 2. It pauses at the **human checkpoint** where you can review and optionally edit the 5 video prompts before generation begins.

---

## Generation Modes

### Local (wan2gp mode)
Set `GENERATION_MODE=wan2gp` in `.env`. Wan2GP runs headless via `--process settings.json`.

Current performance on 7900 XTX + ROCm 6.4:
- ~22 minutes per 3-second clip at 832×480, 20 steps, BF16
- 5 clips = ~2 hours total

### Manual Colab mode
Set `GENERATION_MODE=colab` in `.env`. The pipeline prints each prompt and waits for you to generate the clip manually via:
```
https://colab.research.google.com/github/Isi-dev/Google-Colab_Notebooks/blob/main/Wan2_1_14B_T2V_GGUF_Free.ipynb
```
Save each output as `data/3_final_outputs/clip_0X.mp4` and the pipeline continues automatically.

---

## Known Issues & Limitations

| Issue | Status | Notes |
|-------|--------|-------|
| Qwen 3.5 extended thinking consumes tokens before JSON output | Workaround: `max_tokens=50000` | Model thinks fully then outputs JSON |
| Wan2GP ~22min/clip on ROCm 6.4 | Open | TeaCache may reduce to ~10min — needs testing |
| Sales Agent response truncation | Non-fatal | Pipeline continues with Film Director prompts |
| Breakdown Agent is text-only | Limitation | Video frames not yet attached to Qwen calls |

---

## Roadmap

- [ ] Attach actual video frames to Breakdown Agent calls (true multimodal analysis)
- [ ] Test TeaCache in Wan2GP to reduce generation time
- [ ] Add a README-generated summary video of each pipeline run
- [ ] Support multiple target products in one run
- [ ] Add a web UI for the human checkpoint (replace terminal input)
- [ ] Explore ROCm upgrade path for better flash attention performance

---

## Project Journal

This project was built incrementally as a learning exercise in:
- **LangGraph** state machines with conditional edges and retry loops
- **Pydantic v2** for strict LLM output validation
- **Local LLM** inference with LM Studio (Qwen 3.5)
- **AMD ROCm** GPU compute for video generation
- **Prompt engineering** for structured JSON output without API enforcement

Key lessons learned:
- `response_format: json_object` is OpenAI-specific — LM Studio doesn't support it
- Qwen 3.5 extended thinking mode fills token budgets with reasoning before JSON — set `max_tokens=50000`
- ROCm 6.2 has broken INT8 Triton kernels for RDNA3 — upgrade to 6.4
- Profile 1 (pin full model to RAM) crashes 64GB systems when other apps are running — use Profile 4
- The Editor Chief consensus loop is the most valuable architectural decision — it catches vague formulas before they reach the video generation stage

---

## License

MIT
