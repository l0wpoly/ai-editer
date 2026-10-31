"""
CineFlow AI — Production FastAPI Multimedia & AI Video Editing Server (`main.py`).

Endpoints:
- `GET  /api/health`               : Container health check & provider readiness
- `POST /api/upload`               : Chunked streaming video upload + `ffprobe` metadata & poster
- `GET  /api/media/{id}/video`     : HTTP Range-capable stream of original source video
- `GET  /api/media/{id}/poster`    : Extracted JPEG poster thumbnail
- `POST /api/process`              : Enqueue AI video analysis, voiceover & FFmpeg rendering job
- `GET  /api/status/{job_id}`      : Real-time job telemetry, logs, timeline JSON & output links
- `GET  /api/download/{job_id}`    : Download (or inline stream) rendered MP4 output
- `GET  /api/jobs`                 : Recent job history stored in Async SQLite
- `DELETE /api/jobs/{job_id}`      : Remove rendered job artifacts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import aiofiles
import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services.gemini_service import GeminiServiceError, GeminiVideoService
from services.groq_service import VOICE_CATALOG, GroqServiceError, GroqVoiceoverService
from services.video_engine import MediaEngineError, RenderSegmentSpec, VideoEngine, VideoMetadata

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("cineflow.api")

# --- Paths & Runtime Settings ---
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage"))).resolve()
UPLOADS_DIR = STORAGE_DIR / "uploads"
WORKDIR_ROOT = STORAGE_DIR / "workdir"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
DB_PATH = STORAGE_DIR / "cineflow.sqlite3"

STATIC_BUILD_DIR = Path(os.getenv("STATIC_DIR", str(BASE_DIR / "static"))).resolve()
FRONTEND_SRC_DIR = (BASE_DIR / "frontend").resolve()

MAX_UPLOAD_MB = max(10, int(os.getenv("MAX_UPLOAD_MB", "300")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
JOB_RETENTION_HOURS = max(1, int(os.getenv("JOB_RETENTION_HOURS", "24")))
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB streaming upload chunks

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}

JobStatusLiteral = Literal[
    "Queued",
    "Analyzing",
    "Generating Voiceover",
    "Rendering FFmpeg",
    "Done",
    "Failed",
]


# --- API Request / Response Schemas ---
class ProcessRequest(BaseModel):
    media_id: str = Field(..., min_length=6, max_length=64)
    prompt: str = Field(
        ...,
        min_length=4,
        max_length=1500,
        description="Natural language editing & narration instructions.",
    )
    voice: Literal["auto", "troy", "autumn", "diana", "hannah", "austin", "daniel"] = Field(
        default="auto",
        description="Narrator voice persona ('auto' lets Gemini choose best match).",
    )
    aspect_ratio: Literal["original", "9:16", "16:9", "1:1"] = Field(
        default="original",
        description="Output frame aspect ratio.",
    )
    tts_engine: Literal["auto", "groq", "edge"] = Field(
        default="auto",
        description="Voice synthesis engine ('auto' uses Groq Orpheus with Edge Neural fallback).",
    )
    ducking_level: float = Field(
        default=0.18,
        ge=0.05,
        le=0.65,
        description="Background video audio level during active narration (0.18 ≈ -15 dB).",
    )
    max_duration: Optional[float] = Field(
        default=None,
        ge=4.0,
        le=300.0,
        description="Optional cap on final edited video duration in seconds.",
    )


# --- Singleton Services & Concurrency Primitives ---
video_engine = VideoEngine()
gemini_service = GeminiVideoService()
groq_service = GroqVoiceoverService()

job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
active_jobs_cache: dict[str, dict[str, Any]] = {}
background_tasks_registry: set[asyncio.Task[Any]] = set()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_filename(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "highlight").lower()).strip("-")
    return slug[:36] or "highlight"


# --- Async SQLite Persistence Layer ---
async def init_database() -> None:
    """Create storage directories and initialize Async SQLite tables with WAL mode."""
    for folder in (UPLOADS_DIR, WORKDIR_ROOT, OUTPUTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS media_assets (
                media_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                poster_path TEXT,
                mime_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_epoch REAL NOT NULL
            );
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS render_jobs (
                job_id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                stage_detail TEXT NOT NULL,
                prompt TEXT NOT NULL,
                options_json TEXT NOT NULL,
                logs_json TEXT NOT NULL,
                timeline_json TEXT,
                source_metadata_json TEXT NOT NULL,
                output_metadata_json TEXT,
                output_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_epoch REAL NOT NULL
            );
            """
        )
        await db.commit()


async def persist_job_record(job: dict[str, Any]) -> None:
    """Upsert job state to Async SQLite while keeping in-memory cache synchronized."""
    job["updated_at"] = utc_now_iso()
    active_jobs_cache[job["job_id"]] = job

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO render_jobs (
                job_id, media_id, status, progress, stage_detail, prompt,
                options_json, logs_json, timeline_json, source_metadata_json,
                output_metadata_json, output_path, error, created_at, updated_at, created_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status = excluded.status,
                progress = excluded.progress,
                stage_detail = excluded.stage_detail,
                logs_json = excluded.logs_json,
                timeline_json = excluded.timeline_json,
                output_metadata_json = excluded.output_metadata_json,
                output_path = excluded.output_path,
                error = excluded.error,
                updated_at = excluded.updated_at;
            """,
            (
                job["job_id"],
                job["media_id"],
                job["status"],
                int(job["progress"]),
                job["stage_detail"],
                job["prompt"],
                json.dumps(job["options"]),
                json.dumps(job["logs"]),
                json.dumps(job["timeline"]) if job.get("timeline") else None,
                json.dumps(job["source_metadata"]),
                json.dumps(job["output_metadata"]) if job.get("output_metadata") else None,
                job.get("output_path"),
                job.get("error"),
                job["created_at"],
                job["updated_at"],
                job["created_epoch"],
            ),
        )
        await db.commit()


async def fetch_job_record(job_id: str) -> Optional[dict[str, Any]]:
    """Retrieve job record from hot cache or Async SQLite."""
    if job_id in active_jobs_cache:
        return active_jobs_cache[job_id]

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM render_jobs WHERE job_id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None

            record = {
                "job_id": row["job_id"],
                "media_id": row["media_id"],
                "status": row["status"],
                "progress": row["progress"],
                "stage_detail": row["stage_detail"],
                "prompt": row["prompt"],
                "options": json.loads(row["options_json"]),
                "logs": json.loads(row["logs_json"]),
                "timeline": json.loads(row["timeline_json"]) if row["timeline_json"] else None,
                "source_metadata": json.loads(row["source_metadata_json"]),
                "output_metadata": (
                    json.loads(row["output_metadata_json"]) if row["output_metadata_json"] else None
                ),
                "output_path": row["output_path"],
                "error": row["error"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "created_epoch": row["created_epoch"],
            }
            active_jobs_cache[job_id] = record
            return record


async def fetch_media_asset(media_id: str) -> Optional[dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM media_assets WHERE media_id = ?", (media_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "media_id": row["media_id"],
                "original_filename": row["original_filename"],
                "file_path": row["file_path"],
                "poster_path": row["poster_path"],
                "mime_type": row["mime_type"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }


def serialize_job_response(job: dict[str, Any]) -> dict[str, Any]:
    """Format public API payload for `/api/status/{job_id}`."""
    job_id = job["job_id"]
    media_id = job["media_id"]
    is_done = job["status"] == "Done" and bool(job.get("output_path"))

    return {
        "job_id": job_id,
        "media_id": media_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage_detail": job["stage_detail"],
        "prompt": job["prompt"],
        "options": job["options"],
        "logs": job["logs"][-40:],
        "timeline": job.get("timeline"),
        "source_metadata": job["source_metadata"],
        "output_metadata": job.get("output_metadata"),
        "source_video_url": f"/api/media/{media_id}/video",
        "stream_url": f"/api/download/{job_id}?inline=1" if is_done else None,
        "download_url": f"/api/download/{job_id}" if is_done else None,
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


async def cleanup_expired_artifacts_loop() -> None:
    """Periodic janitor purging media/renders older than `JOB_RETENTION_HOURS`."""
    while True:
        await asyncio.sleep(1800)  # Check every 30 minutes
        cutoff = time.time() - (JOB_RETENTION_HOURS * 3600)
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT job_id, output_path FROM render_jobs WHERE created_epoch < ?",
                    (cutoff,),
                ) as cur:
                    old_jobs = await cur.fetchall()

                for row in old_jobs:
                    jid = row["job_id"]
                    if row["output_path"]:
                        Path(row["output_path"]).unlink(missing_ok=True)
                    shutil.rmtree(WORKDIR_ROOT / jid, ignore_errors=True)
                    active_jobs_cache.pop(jid, None)

                await db.execute("DELETE FROM render_jobs WHERE created_epoch < ?", (cutoff,))

                async with db.execute(
                    "SELECT media_id, file_path, poster_path FROM media_assets WHERE created_epoch < ?",
                    (cutoff,),
                ) as cur:
                    old_media = await cur.fetchall()

                for row in old_media:
                    Path(row["file_path"]).unlink(missing_ok=True)
                    if row["poster_path"]:
                        Path(row["poster_path"]).unlink(missing_ok=True)

                await db.execute("DELETE FROM media_assets WHERE created_epoch < ?", (cutoff,))
                await db.commit()

                if old_jobs or old_media:
                    logger.info(
                        "Janitor cleaned %d expired jobs and %d expired media uploads.",
                        len(old_jobs),
                        len(old_media),
                    )
        except Exception as exc:
            logger.warning("Storage cleanup loop encountered non-fatal error: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_database()
    janitor_task = asyncio.create_task(cleanup_expired_artifacts_loop())
    ff_info = video_engine.check_ffmpeg_available()
    logger.info(
        "CineFlow AI started | FFmpeg=%s | GeminiKey=%s | GroqKey=%s | Storage=%s",
        ff_info["available"],
        gemini_service.is_configured(),
        groq_service.is_configured(),
        STORAGE_DIR,
    )
    try:
        yield
    finally:
        janitor_task.cancel()
        for task in list(background_tasks_registry):
            task.cancel()


app = FastAPI(
    title="CineFlow AI — Video Editing & Voiceover Platform",
    version="1.0.0",
    description="AI-powered nonlinear video editor using Google Gemini, Groq LPU/TTS, and FFmpeg.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Background AI Editing & Rendering Pipeline ---
async def execute_editing_pipeline(job_id: str, media_asset: dict[str, Any]) -> None:
    """
    Orchestrates the end-to-end multimodal pipeline:
    `Queued` -> `Analyzing` -> `Generating Voiceover` -> `Rendering FFmpeg` -> `Done`
    """
    job = await fetch_job_record(job_id)
    if not job:
        return

    work_dir = WORKDIR_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    async def update_state(
        status: JobStatusLiteral,
        progress: int,
        detail: str,
        log_entry: Optional[str] = None,
    ) -> None:
        job["status"] = status
        job["progress"] = max(0, min(100, int(progress)))
        job["stage_detail"] = detail
        if log_entry:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            job["logs"].append(f"[{ts}] {log_entry}")
        await persist_job_record(job)

    async with job_semaphore:
        try:
            source_path = Path(media_asset["file_path"])
            source_meta = VideoMetadata.model_validate(media_asset["metadata"])
            options = job["options"]

            # ==================================================================
            # Stage 1: Gemini Multimodal Video Analysis & JSON Timeline Creation
            # ==================================================================
            await update_state(
                "Analyzing",
                10,
                "Initializing Gemini multimodal video analysis...",
                f"Started Gemini analysis on '{media_asset['original_filename']}' ({source_meta.duration:.1f}s).",
            )

            async def gemini_progress(pct: int, msg: str) -> None:
                await update_state("Analyzing", pct, msg)

            edit_plan = await gemini_service.analyze_and_plan(
                video_path=source_path,
                metadata=source_meta,
                user_prompt=job["prompt"],
                work_dir=work_dir,
                video_engine=video_engine,
                max_duration=options.get("max_duration"),
                progress_callback=gemini_progress,
            )

            chosen_voice = (
                edit_plan.recommended_voice
                if options.get("voice") in (None, "", "auto")
                else options["voice"]
            )
            if chosen_voice not in VOICE_CATALOG:
                chosen_voice = "troy"

            # Publish initial timeline immediately so UI renders cuts while VO & FFmpeg run
            job["timeline"] = {
                "title": edit_plan.title,
                "summary": edit_plan.summary,
                "narrative_tone": edit_plan.narrative_tone,
                "voice_persona": chosen_voice,
                "voice_label": VOICE_CATALOG[chosen_voice]["label"],
                "analysis_mode": edit_plan.analysis_mode,
                "model_used": edit_plan.model_used,
                "target_duration": edit_plan.target_duration,
                "segments": [s.model_dump(exclude={"voiceover_audio_path"}) for s in edit_plan.segments],
            }
            await update_state(
                "Analyzing",
                40,
                f"Gemini planned {len(edit_plan.segments)} cuts ('{edit_plan.title}').",
                f"Gemini ({edit_plan.model_used}, {edit_plan.analysis_mode}) created "
                f"{len(edit_plan.segments)} segments | Tone: {edit_plan.narrative_tone}.",
            )

            # ==================================================================
            # Stage 2: Groq Script Pacing Polish & Synchronous Voiceover TTS
            # ==================================================================
            await update_state(
                "Generating Voiceover",
                42,
                f"Polishing narration & generating '{VOICE_CATALOG[chosen_voice]['label']}' voiceover...",
                f"Starting Groq script optimization & TTS synthesis (voice={chosen_voice}).",
            )

            async def groq_progress(pct: int, msg: str) -> None:
                await update_state("Generating Voiceover", pct, msg)

            voiced_segments = await groq_service.polish_and_synthesize(
                segments=edit_plan.segments,
                user_prompt=job["prompt"],
                narrative_tone=edit_plan.narrative_tone,
                voice_persona=chosen_voice,
                work_dir=work_dir,
                video_engine=video_engine,
                tts_engine_preference=options.get("tts_engine", "auto"),
                progress_callback=groq_progress,
            )

            engines_used = sorted(
                {s.tts_engine_used for s in voiced_segments if s.tts_engine_used}
            )
            job["timeline"]["tts_engines_used"] = engines_used
            job["timeline"]["segments"] = [
                s.model_dump(exclude={"voiceover_audio_path"}) for s in voiced_segments
            ]

            await update_state(
                "Generating Voiceover",
                60,
                "Voiceover tracks synthesized and duration-calibrated.",
                f"Generated {sum(1 for s in voiced_segments if s.voiceover_audio_path)} voiceover clips "
                f"via {', '.join(engines_used) if engines_used else 'none (instrumental)'}.",
            )

            # ==================================================================
            # Stage 3: FFmpeg Trimming, Transitions, Ducking & Final MP4 Muxing
            # ==================================================================
            output_filename = f"cineflow-{slugify_filename(edit_plan.title)}-{job_id[:8]}.mp4"
            output_path = OUTPUTS_DIR / output_filename

            await update_state(
                "Rendering FFmpeg",
                62,
                "Launching FFmpeg filter-graph rendering & dynamic audio ducking...",
                f"Rendering {len(voiced_segments)} segments -> {options.get('aspect_ratio', 'original')} MP4.",
            )

            async def ffmpeg_progress(pct: int, msg: str) -> None:
                await update_state("Rendering FFmpeg", pct, msg)

            final_path, final_meta, final_segments = await video_engine.render_timeline(
                source_video=source_path,
                source_meta=source_meta,
                segments=voiced_segments,
                work_dir=work_dir,
                output_path=output_path,
                aspect_ratio=options.get("aspect_ratio", "original"),
                ducking_level=float(options.get("ducking_level", 0.18)),
                progress_callback=ffmpeg_progress,
            )

            job["output_path"] = str(final_path)
            job["output_metadata"] = final_meta.model_dump()
            job["timeline"]["segments"] = [
                s.model_dump(exclude={"voiceover_audio_path"}) for s in final_segments
            ]

            await update_state(
                "Done",
                100,
                f"Render complete! Final cut: {final_meta.duration:.1f}s ({final_meta.width}x{final_meta.height}).",
                f"Completed '{output_filename}' ({final_meta.size_bytes / (1024 * 1024):.2f} MB, {final_meta.duration:.2f}s).",
            )

        except (GeminiServiceError, GroqServiceError, MediaEngineError) as known_exc:
            logger.error("Job %s failed with domain error: %s", job_id, known_exc)
            job["error"] = str(known_exc)
            await update_state("Failed", job["progress"], f"Error: {known_exc}", f"FAILED: {known_exc}")
        except Exception as unexpected_exc:
            logger.exception("Job %s crashed with unexpected error", job_id)
            job["error"] = f"Unexpected processing error: {unexpected_exc}"
            await update_state(
                "Failed",
                job["progress"],
                "An unexpected processing error occurred.",
                f"CRITICAL: {unexpected_exc}",
            )
        finally:
            # Always reclaim ephemeral intermediate segments & uncompressed WAV files
            shutil.rmtree(work_dir, ignore_errors=True)


# --- API Routes ---
@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """System readiness & capabilities endpoint for Render health probes and frontend status bar."""
    ff_status = video_engine.check_ffmpeg_available()
    running_count = sum(
        1
        for j in active_jobs_cache.values()
        if j.get("status") in ("Queued", "Analyzing", "Generating Voiceover", "Rendering FFmpeg")
    )
    return {
        "status": "ok",
        "timestamp": utc_now_iso(),
        "ffmpeg": ff_status,
        "providers": {
            "gemini_configured": gemini_service.is_configured(),
            "gemini_model": gemini_service.primary_model,
            "groq_configured": groq_service.is_configured(),
            "groq_llm_model": groq_service.llm_model,
            "groq_tts_model": groq_service.tts_model,
            "edge_tts_fallback": True,
        },
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "active_jobs": running_count,
        },
        "voices": [
            {"id": vid, "label": cfg["label"], "edge_voice": cfg["edge_voice"]}
            for vid, cfg in VOICE_CATALOG.items()
        ],
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Stream a video upload to disk in 1 MiB chunks, validate format & byte ceiling,
    probe technical metadata via `ffprobe`, and generate a preview poster thumbnail.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext or 'none'}'. Supported formats: {allowed_list}.",
        )

    ff_status = video_engine.check_ffmpeg_available()
    if not ff_status["available"]:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg/FFprobe binaries are not installed on the server. Run via Docker or install FFmpeg.",
        )

    media_id = uuid.uuid4().hex
    saved_video_path = UPLOADS_DIR / f"{media_id}{ext}"
    poster_path = UPLOADS_DIR / f"{media_id}_poster.jpg"
    bytes_written = 0

    try:
        async with aiofiles.open(saved_video_path, "wb") as out_stream:
            while True:
                chunk = await file.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed upload limit of {MAX_UPLOAD_MB} MB.",
                    )
                await out_stream.write(chunk)
    except HTTPException:
        saved_video_path.unlink(missing_ok=True)
        raise
    except Exception as io_exc:
        saved_video_path.unlink(missing_ok=True)
        logger.exception("Streaming file upload failed")
        raise HTTPException(status_code=500, detail=f"Upload I/O error: {io_exc}") from io_exc
    finally:
        await file.close()

    if bytes_written < 1024:
        saved_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video file is empty or truncated.")

    try:
        metadata = await video_engine.probe_video(saved_video_path)
    except MediaEngineError as probe_exc:
        saved_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(probe_exc)) from probe_exc

    poster_exists = False
    try:
        poster_ts = min(1.0, metadata.duration * 0.15)
        await video_engine.extract_poster(saved_video_path, poster_path, timestamp=poster_ts)
        poster_exists = poster_path.exists()
    except MediaEngineError as poster_exc:
        logger.warning("Non-fatal: poster extraction skipped for %s (%s)", media_id, poster_exc)

    mime_map = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }
    resolved_mime = mime_map.get(ext, file.content_type or "video/mp4")
    created_iso = utc_now_iso()

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO media_assets (
                media_id, original_filename, file_path, poster_path,
                mime_type, metadata_json, created_at, created_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                media_id,
                file.filename,
                str(saved_video_path),
                str(poster_path) if poster_exists else None,
                resolved_mime,
                json.dumps(metadata.model_dump()),
                created_iso,
                time.time(),
            ),
        )
        await db.commit()

    return {
        "media_id": media_id,
        "filename": file.filename,
        "metadata": metadata.model_dump(),
        "video_url": f"/api/media/{media_id}/video",
        "poster_url": f"/api/media/{media_id}/poster" if poster_exists else None,
        "created_at": created_iso,
    }


@app.get("/api/media/{media_id}/video")
async def stream_source_video(media_id: str) -> FileResponse:
    """Serve uploaded source video with HTTP Range support for Before/After comparison playback."""
    asset = await fetch_media_asset(media_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Source media asset not found.")
    path = Path(asset["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Source media file is no longer on disk.")
    return FileResponse(path=path, media_type=asset["mime_type"])


@app.get("/api/media/{media_id}/poster")
async def get_source_poster(media_id: str) -> FileResponse:
    """Serve extracted JPEG poster thumbnail for uploaded source video."""
    asset = await fetch_media_asset(media_id)
    if not asset or not asset.get("poster_path"):
        raise HTTPException(status_code=404, detail="Poster thumbnail not found.")
    path = Path(asset["poster_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Poster file missing on disk.")
    return FileResponse(path=path, media_type="image/jpeg")


@app.post("/api/process")
async def start_processing_job(req: ProcessRequest) -> dict[str, Any]:
    """Validate configuration & media asset, enqueue a background render job, and return `job_id`."""
    if not gemini_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server. Set GEMINI_API_KEY in environment variables.",
        )

    if req.tts_engine == "groq" and not groq_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Set GROQ_API_KEY or switch TTS Engine to 'auto' / 'edge'.",
        )

    ff_status = video_engine.check_ffmpeg_available()
    if not ff_status["available"]:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg is not available on this host. Deploy using the provided Dockerfile or install FFmpeg.",
        )

    asset = await fetch_media_asset(req.media_id)
    if not asset or not Path(asset["file_path"]).exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded video asset not found or expired. Please re-upload your video.",
        )

    job_id = uuid.uuid4().hex
    now_iso = utc_now_iso()
    ts_short = datetime.now(timezone.utc).strftime("%H:%M:%S")

    job_record: dict[str, Any] = {
        "job_id": job_id,
        "media_id": req.media_id,
        "status": "Queued",
        "progress": 3,
        "stage_detail": "Job queued and waiting for render worker...",
        "prompt": req.prompt.strip(),
        "options": {
            "voice": req.voice,
            "aspect_ratio": req.aspect_ratio,
            "tts_engine": req.tts_engine,
            "ducking_level": req.ducking_level,
            "max_duration": req.max_duration,
        },
        "logs": [f"[{ts_short}] Job {job_id[:8]} queued for '{asset['original_filename']}'."],
        "timeline": None,
        "source_metadata": asset["metadata"],
        "output_metadata": None,
        "output_path": None,
        "error": None,
        "created_at": now_iso,
        "updated_at": now_iso,
        "created_epoch": time.time(),
    }

    await persist_job_record(job_record)

    task = asyncio.create_task(execute_editing_pipeline(job_id, asset))
    background_tasks_registry.add(task)
    task.add_done_callback(background_tasks_registry.discard)

    return serialize_job_response(job_record)


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll real-time job status (`Queued` -> `Analyzing` -> `Generating Voiceover` -> `Rendering FFmpeg` -> `Done`)."""
    job = await fetch_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return serialize_job_response(job)


@app.get("/api/download/{job_id}")
async def download_rendered_video(
    job_id: str,
    inline: bool = Query(default=False, description="Serve inline for <video> playback when True."),
) -> FileResponse:
    """Download or stream the final rendered MP4 video."""
    job = await fetch_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "Done" or not job.get("output_path"):
        raise HTTPException(
            status_code=409,
            detail=f"Rendered output is not ready yet (current status: {job['status']}).",
        )

    out_path = Path(job["output_path"])
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Rendered MP4 file has expired from storage.")

    if inline:
        return FileResponse(path=out_path, media_type="video/mp4")

    return FileResponse(
        path=out_path,
        media_type="video/mp4",
        filename=out_path.name,
    )


@app.get("/api/jobs")
async def list_recent_jobs(limit: int = Query(default=12, ge=1, le=50)) -> dict[str, Any]:
    """Return recent editing jobs from Async SQLite so browser refreshes retain work history."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT job_id FROM render_jobs ORDER BY created_epoch DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()

    items = []
    for row in rows:
        rec = await fetch_job_record(row["job_id"])
        if rec:
            items.append(serialize_job_response(rec))
    return {"jobs": items}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, Any]:
    """Delete a completed or failed job record and reclaim its rendered MP4 disk space."""
    job = await fetch_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.get("output_path"):
        Path(job["output_path"]).unlink(missing_ok=True)
    shutil.rmtree(WORKDIR_ROOT / job_id, ignore_errors=True)
    active_jobs_cache.pop(job_id, None)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM render_jobs WHERE job_id = ?", (job_id,))
        await db.commit()

    return {"deleted": True, "job_id": job_id}


# --- Static Frontend Mounting (Supports Docker Vite Build AND Zero-Build Local Dev) ---
ACTIVE_WEB_DIR = (
    STATIC_BUILD_DIR if (STATIC_BUILD_DIR / "index.html").exists() else FRONTEND_SRC_DIR
)

if (ACTIVE_WEB_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(ACTIVE_WEB_DIR / "assets")), name="assets")


@app.get("/", response_model=None)
async def serve_spa_index():
    index_file = ACTIVE_WEB_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            "<h1>CineFlow AI Backend Running</h1><p>Frontend assets not found.</p>",
            status_code=200,
        )
    return FileResponse(index_file, media_type="text/html")


@app.get("/{asset_name}", response_model=None)
async def serve_root_frontend_asset(asset_name: str):
    """Serve top-level static assets (`style.css`, `app.js`, `favicon.svg`) safely without path traversal."""
    if "/" in asset_name or "\\" in asset_name or asset_name.startswith("."):
        raise HTTPException(status_code=404, detail="Not found")

    candidate = (ACTIVE_WEB_DIR / asset_name).resolve()
    if not str(candidate).startswith(str(ACTIVE_WEB_DIR)) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")

    return FileResponse(candidate)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "10000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
