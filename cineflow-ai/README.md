# CineFlow AI — Multimodal Video Editing & Voiceover Platform

Production-ready full-stack AI video editing and voiceover studio engineered for single-click deployment on **Render.com**.

## Architecture Stack
- **Backend**: Python 3.12 + FastAPI + Uvicorn + Async SQLite (`aiosqlite` WAL) + Streaming File Ingestion (`aiofiles`).
- **Multimodal Video Intelligence**: Google Gemini (`google-genai` SDK, `gemini-2.5-flash` with `gemini-2.0-flash` fallback) via Gemini Files API + timestamped FFmpeg keyframe grid fallback for large uploads.
- **Voiceover & Script Engine**: Groq Cloud (`groq` SDK, `llama-3.3-70b-versatile` LPU pacing optimizer + `canopylabs/orpheus-v1-english` native TTS with automatic `<=185`-char WAV chunking and seamless fallback to Microsoft Edge Neural TTS via `edge-tts`).
- **Media Engine**: Asynchronous FFmpeg & FFprobe subprocess pipeline supporting smartphone rotation detection, silent-video stereo `anullsrc` synthesis, speed ramping (`setpts` / `atempo`), aspect-ratio smart cropping (`original`, `9:16`, `16:9`, `1:1`), transitions (`cut`, `fade`, `flash`), and dynamic `sidechaincompress` background audio ducking.
- **Frontend Studio**: Modern Vanilla ES2022 + Vite dark-mode NLE workstation with drag-and-drop XHR upload progress, live 5-stage pipeline telemetry, interactive 3-track timeline visualizer (V1 Video, A1 Voiceover, A2 SVG Ducking Curve), and synchronized Before/After comparison player.

---

## 1-Click Deployment on Render.com

1. Push this directory to a GitHub or GitLab repository.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** -> **Blueprint** and connect your repository.
3. Render automatically reads `render.yaml` and prompts you for:
   - `GEMINI_API_KEY` — Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - `GROQ_API_KEY` — Get from [Groq Console](https://console.groq.com/keys) *(Tip: open `canopylabs/orpheus-v1-english` once in Groq Playground to accept model terms, or rely on the built-in `edge-tts` neural fallback).*
4. Click **Apply**. Render builds the multi-stage `Dockerfile` (Node 22 Vite frontend bundle + Debian Bookworm Python 3.12 + native `ffmpeg`) and launches the service with `/api/health` monitoring.

---

## Local Docker Run

```bash
cp .env.example .env
# Edit .env with GEMINI_API_KEY and GROQ_API_KEY

docker build -t cineflow-ai .
docker run --rm -p 10000:10000 --env-file .env cineflow-ai
```

Open `http://localhost:10000`.

---

## Local Bare-Metal Development

Requires Python 3.11+, Node.js 20+, and `ffmpeg` / `ffprobe` on `PATH`:

```bash
# 1. Optional: Build optimized static frontend bundle (FastAPI also serves ./frontend directly in dev)
cd frontend && npm install && npm run build && cd ..

# 2. Install Python dependencies & launch server
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```
