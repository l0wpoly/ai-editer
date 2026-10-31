"""
FFmpeg Multimedia & Video Rendering Engine (`services/video_engine.py`).

Responsibilities:
- Asynchronous `ffprobe` inspection (duration, visual dimensions accounting for rotation,
  frame rate, codecs, audio stream detection, aspect ratio).
- Poster thumbnail & timestamped keyframe grid extraction.
- Memory-safe segment trimming, speed ramping (`setpts` / `atempo`), aspect-ratio
  normalization (`original`, `9:16`, `16:9`, `1:1`), and visual/audio transitions (`cut`, `fade`, `flash`).
- Lossless stream concatenation and studio-grade dynamic voiceover mixing with
  `sidechaincompress` background audio ducking (plus expression-based volume envelope fallback).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("cineflow.video_engine")

ProgressCallback = Callable[[int, str], Awaitable[None]]


class MediaEngineError(RuntimeError):
    """Raised when FFmpeg or FFprobe fails during media inspection or rendering."""


class VideoMetadata(BaseModel):
    """Normalized technical metadata extracted via `ffprobe`."""

    duration: float = Field(..., description="Total duration in seconds")
    width: int = Field(..., description="Display width in pixels (post-rotation)")
    height: int = Field(..., description="Display height in pixels (post-rotation)")
    fps: float = Field(default=30.0, description="Frames per second")
    has_audio: bool = Field(default=False, description="True if at least one audio stream is present")
    video_codec: str = Field(default="unknown", description="Primary video codec name")
    audio_codec: Optional[str] = Field(default=None, description="Primary audio codec name if present")
    bitrate_kbps: int = Field(default=0, description="Overall container bitrate in kbps")
    size_bytes: int = Field(default=0, description="File size on disk in bytes")
    aspect_ratio_display: str = Field(default="16:9", description="Human-readable aspect ratio")


class RenderSegmentSpec(BaseModel):
    """Complete render specification for a single timeline cut segment."""

    index: int
    start: float = Field(..., ge=0.0, description="Start timestamp in source video (seconds)")
    end: float = Field(..., gt=0.0, description="End timestamp in source video (seconds)")
    action: str = Field(default="cut", description="cut | highlight | speed_ramp | intro | outro")
    transition: str = Field(default="cut", description="cut | fade | flash")
    playback_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Playback speed multiplier")
    scene_label: str = Field(default="Scene", description="Short descriptive title for UI timeline")
    energy_score: int = Field(default=7, ge=1, le=10, description="Visual pacing intensity (1-10)")
    voiceover_text: str = Field(default="", description="Polished spoken narration script")
    orpheus_text: str = Field(default="", description="Script with optional vocal direction tags")
    voiceover_audio_path: Optional[str] = Field(default=None, description="Local path to synthesized VO clip")
    voiceover_duration: float = Field(default=0.0, description="Probed duration of synthesized VO clip")
    tts_engine_used: Optional[str] = Field(default=None, description="TTS backend that rendered this line")
    timeline_start: float = Field(default=0.0, description="Start offset in final edited video (seconds)")
    timeline_end: float = Field(default=0.0, description="End offset in final edited video (seconds)")

    @property
    def source_duration(self) -> float:
        return max(0.1, self.end - self.start)

    @property
    def target_effective_duration(self) -> float:
        return max(0.2, self.source_duration / max(0.5, min(2.0, self.playback_speed)))


ASPECT_TARGETS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}


def compute_aspect_label(width: int, height: int) -> str:
    """Compute a clean display aspect ratio string such as `16:9` or `9:16`."""
    if width <= 0 or height <= 0:
        return "unknown"
    ratio = width / height
    common = [
        (16 / 9, "16:9"),
        (9 / 16, "9:16"),
        (1.0, "1:1"),
        (4 / 3, "4:3"),
        (3 / 4, "3:4"),
        (4 / 5, "4:5"),
        (21 / 9, "21:9"),
    ]
    for target_ratio, label in common:
        if abs(ratio - target_ratio) < 0.04:
            return label
    gcd_val = math.gcd(width, height)
    return f"{width // gcd_val}:{height // gcd_val}"


class VideoEngine:
    """Asynchronous FFmpeg & FFprobe wrapper engineered for container deployments."""

    def __init__(self) -> None:
        self.ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        self.ffprobe_bin = os.getenv("FFPROBE_BIN", "ffprobe")
        self.threads = max(1, int(os.getenv("FFMPEG_THREADS", "2")))
        self.preset = os.getenv("FFMPEG_PRESET", "veryfast")

    def check_ffmpeg_available(self) -> dict[str, Any]:
        """Synchronously inspect FFmpeg & FFprobe installation status for health checks."""
        ffmpeg_path = shutil.which(self.ffmpeg_bin)
        ffprobe_path = shutil.which(self.ffprobe_bin)
        return {
            "available": bool(ffmpeg_path and ffprobe_path),
            "ffmpeg_path": ffmpeg_path,
            "ffprobe_path": ffprobe_path,
            "threads": self.threads,
            "preset": self.preset,
        }

    async def _run_cmd(
        self,
        args: list[str],
        timeout_sec: float = 600.0,
        step_name: str = "ffmpeg",
    ) -> tuple[str, str]:
        """Execute a subprocess asynchronously with timeout protection and stderr diagnostics."""
        logger.debug("Executing %s command: %s", step_name, " ".join(args))
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise MediaEngineError(
                f"Binary '{args[0]}' is not installed or not found on PATH. "
                "Ensure FFmpeg is installed locally or run inside the provided Dockerfile."
            ) from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise MediaEngineError(
                f"{step_name} timed out after {timeout_sec:.0f}s while processing media."
            ) from exc

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            tail = "\n".join(stderr_text.strip().splitlines()[-18:])
            logger.error("%s exited with code %s:\n%s", step_name, proc.returncode, tail)
            raise MediaEngineError(f"{step_name} failed (exit {proc.returncode}): {tail}")

        return stdout_text, stderr_text

    async def probe_video(self, video_path: Path) -> VideoMetadata:
        """Inspect container and stream metadata of an uploaded video file."""
        if not video_path.exists():
            raise MediaEngineError(f"Media file does not exist: {video_path}")

        args = [
            self.ffprobe_bin,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        stdout, _ = await self._run_cmd(args, timeout_sec=45.0, step_name="ffprobe")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise MediaEngineError("Failed to parse ffprobe JSON output.") from exc

        streams: list[dict[str, Any]] = payload.get("streams", [])
        fmt: dict[str, Any] = payload.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video_stream:
            raise MediaEngineError("Uploaded file does not contain a valid decodable video stream.")

        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        raw_width = int(video_stream.get("width") or 0)
        raw_height = int(video_stream.get("height") or 0)
        if raw_width <= 0 or raw_height <= 0:
            raise MediaEngineError("Invalid video dimensions detected (0x0).")

        # Account for smartphone rotation metadata (.MOV / .MP4 display matrix)
        rotation = self._extract_rotation_degrees(video_stream)
        if abs(rotation) in (90, 270):
            width, height = raw_height, raw_width
        else:
            width, height = raw_width, raw_height

        # Parse frame rate reliably (e.g. "30000/1001" -> 29.97)
        fps = self._parse_frame_rate(
            video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1"
        )

        # Resolve duration from format or stream fallback
        duration_raw = fmt.get("duration") or video_stream.get("duration") or 0.0
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            duration = 0.0

        if duration <= 0.15:
            raise MediaEngineError(
                f"Video duration ({duration:.2f}s) is too short or corrupted for editing."
            )

        size_bytes = video_path.stat().st_size
        try:
            bitrate_kbps = int(int(fmt.get("bit_rate", 0)) / 1000)
        except (TypeError, ValueError):
            bitrate_kbps = int((size_bytes * 8) / max(1.0, duration) / 1000)

        return VideoMetadata(
            duration=round(duration, 3),
            width=width,
            height=height,
            fps=round(fps, 2),
            has_audio=audio_stream is not None,
            video_codec=str(video_stream.get("codec_name", "unknown")),
            audio_codec=str(audio_stream.get("codec_name")) if audio_stream else None,
            bitrate_kbps=max(0, bitrate_kbps),
            size_bytes=size_bytes,
            aspect_ratio_display=compute_aspect_label(width, height),
        )

    async def probe_audio_duration(self, audio_path: Path) -> float:
        """Return exact duration in seconds of a WAV or MP3 audio file."""
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return 0.0
        args = [
            self.ffprobe_bin,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(audio_path),
        ]
        stdout, _ = await self._run_cmd(args, timeout_sec=20.0, step_name="ffprobe-audio")
        data = json.loads(stdout)
        dur = data.get("format", {}).get("duration")
        if dur is None:
            streams = data.get("streams", [])
            if streams:
                dur = streams[0].get("duration", 0.0)
        try:
            return max(0.0, float(dur or 0.0))
        except (TypeError, ValueError):
            return 0.0

    async def extract_poster(
        self,
        video_path: Path,
        output_jpg: Path,
        timestamp: float = 0.5,
    ) -> Path:
        """Extract a high-quality JPEG preview thumbnail from the source video."""
        output_jpg.parent.mkdir(parents=True, exist_ok=True)
        seek_ts = max(0.0, timestamp)
        args = [
            self.ffmpeg_bin,
            "-y",
            "-ss",
            f"{seek_ts:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale='min(960,iw)':-2",
            "-q:v",
            "3",
            str(output_jpg),
        ]
        await self._run_cmd(args, timeout_sec=30.0, step_name="extract-poster")
        return output_jpg

    async def extract_keyframes(
        self,
        video_path: Path,
        output_dir: Path,
        duration: float,
        count: int = 16,
    ) -> list[tuple[float, Path]]:
        """
        Extract evenly-spaced timestamped JPEG frames across the video.
        Used for multimodal Gemini fallback on very large videos (>100 MB).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_count = max(4, min(24, count))
        margin = min(0.25, duration * 0.04)
        usable = max(0.2, duration - 2 * margin)
        step = usable / max(1, frame_count - 1)

        results: list[tuple[float, Path]] = []
        for idx in range(frame_count):
            ts = round(min(duration - 0.05, margin + idx * step), 2)
            out_file = output_dir / f"frame_{idx:02d}_{int(ts * 100):06d}.jpg"
            args = [
                self.ffmpeg_bin,
                "-y",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(640,iw)':-2",
                "-q:v",
                "4",
                str(out_file),
            ]
            try:
                await self._run_cmd(args, timeout_sec=20.0, step_name=f"keyframe-{idx}")
                if out_file.exists() and out_file.stat().st_size > 0:
                    results.append((ts, out_file))
            except MediaEngineError as exc:
                logger.warning("Skipping keyframe at %.2fs: %s", ts, exc)

        if not results:
            raise MediaEngineError("Failed to extract any keyframes from source video.")
        return results

    @staticmethod
    def resolve_target_dimensions(
        source_width: int,
        source_height: int,
        aspect_ratio: str,
    ) -> tuple[int, int]:
        """Resolve even-integer output dimensions for the requested aspect preset."""
        if aspect_ratio in ASPECT_TARGETS:
            return ASPECT_TARGETS[aspect_ratio]

        # Preserve original aspect ratio capped cleanly at 1080p short-edge / 1920p long-edge
        max_long_edge = 1920
        long_edge = max(source_width, source_height)
        scale_factor = min(1.0, max_long_edge / max(1, long_edge))
        target_w = max(2, int(round(source_width * scale_factor / 2) * 2))
        target_h = max(2, int(round(source_height * scale_factor / 2) * 2))
        return target_w, target_h

    @classmethod
    def build_video_filter_chain(
        cls,
        source_width: int,
        source_height: int,
        aspect_ratio: str,
        playback_speed: float,
        transition: str,
        effective_duration: float,
    ) -> str:
        """
        Construct the FFmpeg `-vf` filter graph for a single segment:
        - Speed adjustment (`setpts`)
        - Aspect-ratio smart center-crop / even-dimension scaling
        - Constant 30 fps & `yuv420p` web-compatible pixel format
        - Transition effects (`fade`, `flash`, `cut`)
        """
        speed = max(0.5, min(2.0, playback_speed))
        filters: list[str] = ["setpts=PTS-STARTPTS"]

        if abs(speed - 1.0) > 0.01:
            pts_factor = 1.0 / speed
            filters.append(f"setpts={pts_factor:.5f}*PTS")

        target_w, target_h = cls.resolve_target_dimensions(source_width, source_height, aspect_ratio)
        if aspect_ratio in ASPECT_TARGETS:
            filters.append(
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase"
            )
            filters.append(f"crop={target_w}:{target_h}")
        else:
            filters.append(f"scale={target_w}:{target_h}")

        filters.extend(["setsar=1", "fps=30", "format=yuv420p"])

        fade_dur = min(0.35, max(0.12, effective_duration * 0.16))
        if transition == "fade" and effective_duration > fade_dur * 2.1:
            fade_out_st = max(0.05, effective_duration - fade_dur)
            filters.append(f"fade=t=in:st=0:d={fade_dur:.3f}")
            filters.append(f"fade=t=out:st={fade_out_st:.3f}:d={fade_dur:.3f}")
        elif transition == "flash" and effective_duration > 0.45:
            flash_dur = min(0.24, effective_duration * 0.2)
            filters.append(f"fade=t=in:st=0:d={flash_dur:.3f}:color=white")

        return ",".join(filters)

    @staticmethod
    def build_audio_filter_chain(
        playback_speed: float,
        transition: str,
        effective_duration: float,
    ) -> str:
        """
        Construct the FFmpeg `-af` filter graph for a single segment's background audio:
        - Timestamp reset (`asetpts`)
        - Pitch-preserving speed adjustment (`atempo` within `[0.5, 2.0]`)
        - Stereo 44.1 kHz sample format normalization
        - Micro de-click fades (or full audio cross-fade matching video transition)
        """
        speed = max(0.5, min(2.0, playback_speed))
        filters: list[str] = ["asetpts=PTS-STARTPTS"]

        if abs(speed - 1.0) > 0.01:
            filters.append(f"atempo={speed:.4f}")

        filters.append("aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo")

        if transition == "fade" and effective_duration > 0.6:
            afade_d = min(0.28, effective_duration * 0.15)
            out_st = max(0.05, effective_duration - afade_d)
            filters.append(f"afade=t=in:st=0:d={afade_d:.3f}")
            filters.append(f"afade=t=out:st={out_st:.3f}:d={afade_d:.3f}")
        elif effective_duration > 0.1:
            # 20ms micro-fade eliminates PCM boundary clicks at hard cuts
            out_st = max(0.02, effective_duration - 0.025)
            filters.append("afade=t=in:st=0:d=0.020")
            filters.append(f"afade=t=out:st={out_st:.3f}:d=0.020")

        return ",".join(filters)

    async def _render_single_segment(
        self,
        source_video: Path,
        source_meta: VideoMetadata,
        seg: RenderSegmentSpec,
        out_clip: Path,
        aspect_ratio: str,
    ) -> float:
        """Extract, transform, and normalize one cut segment to an intermediate MP4."""
        clip_src_dur = max(0.35, seg.end - seg.start)
        eff_dur = clip_src_dur / max(0.5, min(2.0, seg.playback_speed))

        vf = self.build_video_filter_chain(
            source_width=source_meta.width,
            source_height=source_meta.height,
            aspect_ratio=aspect_ratio,
            playback_speed=seg.playback_speed,
            transition=seg.transition,
            effective_duration=eff_dur,
        )

        if source_meta.has_audio:
            af = self.build_audio_filter_chain(
                playback_speed=seg.playback_speed,
                transition=seg.transition,
                effective_duration=eff_dur,
            )
            args = [
                self.ffmpeg_bin,
                "-y",
                "-ss",
                f"{seg.start:.3f}",
                "-t",
                f"{clip_src_dur:.3f}",
                "-i",
                str(source_video),
                "-vf",
                vf,
                "-af",
                af,
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-crf",
                "21",
                "-threads",
                str(self.threads),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-shortest",
                str(out_clip),
            ]
        else:
            # Source video has no audio track: synthesize silent stereo audio stream
            args = [
                self.ffmpeg_bin,
                "-y",
                "-ss",
                f"{seg.start:.3f}",
                "-t",
                f"{clip_src_dur:.3f}",
                "-i",
                str(source_video),
                "-f",
                "lavfi",
                "-t",
                f"{eff_dur:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-vf",
                vf,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-crf",
                "21",
                "-threads",
                str(self.threads),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-shortest",
                str(out_clip),
            ]

        await self._run_cmd(args, timeout_sec=240.0, step_name=f"segment-{seg.index}")
        probed = await self.probe_video(out_clip)
        return probed.duration

    async def _prepare_voiceover_clip(
        self,
        raw_audio: Path,
        slot_duration: float,
        output_wav: Path,
    ) -> float:
        """
        Normalize a synthesized voiceover track to 44.1 kHz stereo WAV and apply
        gentle tempo compression (up to 1.22x) if speech slightly exceeds segment slot.
        """
        raw_dur = await self.probe_audio_duration(raw_audio)
        if raw_dur <= 0.05:
            return 0.0

        filters: list[str] = ["asetpts=PTS-STARTPTS"]
        max_slot = max(0.35, slot_duration - 0.08)

        if raw_dur > max_slot:
            needed_speed = min(1.22, raw_dur / max_slot)
            if needed_speed > 1.02:
                filters.append(f"atempo={needed_speed:.4f}")

        filters.append("aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo")
        filters.append("afade=t=in:st=0:d=0.015")

        args = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(raw_audio),
            "-af",
            ",".join(filters),
            "-t",
            f"{slot_duration:.3f}",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_wav),
        ]
        await self._run_cmd(args, timeout_sec=45.0, step_name="prepare-vo")
        return await self.probe_audio_duration(output_wav)

    @staticmethod
    def build_sidechain_ducking_graph(
        vo_delays_ms: list[int],
        total_duration: float,
        ducking_level: float,
    ) -> str:
        """
        Build the primary `filter_complex` audio graph using `sidechaincompress`
        so background audio ducks smoothly whenever narration speaks.
        """
        count = len(vo_delays_ms)
        parts: list[str] = []
        vo_labels: list[str] = []

        for idx, delay_ms in enumerate(vo_delays_ms, start=1):
            label = f"vo_{idx}"
            vo_labels.append(f"[{label}]")
            delay_filter = f"adelay={delay_ms}|{delay_ms}" if delay_ms > 0 else "anull"
            parts.append(
                f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"{delay_filter}[{label}]"
            )

        dur_str = f"{max(0.5, total_duration):.3f}"
        if count == 1:
            parts.append(
                f"{vo_labels[0]}apad,atrim=0:{dur_str},asetpts=PTS-STARTPTS,volume=1.10[vo_bus]"
            )
        else:
            joined = "".join(vo_labels)
            parts.append(
                f"{joined}amix=inputs={count}:duration=longest:dropout_transition=0:normalize=0,"
                f"apad,atrim=0:{dur_str},asetpts=PTS-STARTPTS,volume=1.10[vo_bus]"
            )

        parts.append("[vo_bus]asplit=2[vo_sc][vo_mix]")

        # Map ducking_level (0.05..0.60) to compression ratio (12:1 down to 3:1)
        clamped_duck = max(0.05, min(0.65, ducking_level))
        ratio = max(3.0, min(14.0, round(1.3 / clamped_duck, 1)))
        bg_base_vol = 0.92

        parts.append(
            f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"volume={bg_base_vol:.2f}[bg_in]"
        )
        parts.append(
            f"[bg_in][vo_sc]sidechaincompress=threshold=0.018:ratio={ratio:.1f}:"
            f"attack=30:release=380:makeup=1[bg_ducked]"
        )
        parts.append(
            "[bg_ducked][vo_mix]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "alimiter=limit=0.95[aout]"
        )
        return ";".join(parts)

    @staticmethod
    def build_envelope_ducking_graph(
        vo_windows: list[tuple[int, float, float]],
        total_duration: float,
        ducking_level: float,
    ) -> str:
        """
        Fallback `filter_complex` audio graph using frame-evaluated `volume` windows
        in case a minimal FFmpeg distribution lacks `sidechaincompress` or `normalize=0`.
        """
        count = len(vo_windows)
        parts: list[str] = []
        vo_labels: list[str] = []
        between_terms: list[str] = []

        for idx, (delay_ms, win_start, win_end) in enumerate(vo_windows, start=1):
            label = f"vo_{idx}"
            vo_labels.append(f"[{label}]")
            delay_filter = f"adelay={delay_ms}|{delay_ms}" if delay_ms > 0 else "anull"
            parts.append(
                f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"{delay_filter}[{label}]"
            )
            between_terms.append(f"between(t,{max(0.0, win_start - 0.05):.3f},{win_end + 0.12:.3f})")

        dur_str = f"{max(0.5, total_duration):.3f}"
        if count == 1:
            parts.append(
                f"{vo_labels[0]}apad,atrim=0:{dur_str},asetpts=PTS-STARTPTS,volume=1.10[vo_mix]"
            )
        else:
            joined = "".join(vo_labels)
            parts.append(
                f"{joined}amix=inputs={count}:duration=longest:dropout_transition=0,"
                f"apad,atrim=0:{dur_str},asetpts=PTS-STARTPTS,volume={float(count) * 1.05:.2f}[vo_mix]"
            )

        cond = "+".join(between_terms)
        clamped_duck = max(0.05, min(0.65, ducking_level))
        parts.append(
            f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"volume='if(gt({cond},0),{clamped_duck:.2f},0.92)':eval=frame[bg_ducked]"
        )
        parts.append(
            "[bg_ducked][vo_mix]amix=inputs=2:duration=first:dropout_transition=0,"
            "volume=2.0,alimiter=limit=0.95[aout]"
        )
        return ";".join(parts)

    async def render_timeline(
        self,
        source_video: Path,
        source_meta: VideoMetadata,
        segments: list[RenderSegmentSpec],
        work_dir: Path,
        output_path: Path,
        aspect_ratio: str = "original",
        ducking_level: float = 0.18,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> tuple[Path, VideoMetadata, list[RenderSegmentSpec]]:
        """
        Execute the full FFmpeg rendering workflow:
        1. Segment trimming, speed ramping, aspect-ratio scaling/cropping, and transitions
        2. Lossless concatenation of normalized segments
        3. Voiceover temporal alignment, sidechain background audio ducking, and final MP4 muxing
        """
        if not segments:
            raise MediaEngineError("Cannot render an empty edit timeline (0 segments).")

        segments_dir = work_dir / "segments"
        vo_norm_dir = work_dir / "vo_norm"
        segments_dir.mkdir(parents=True, exist_ok=True)
        vo_norm_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        segment_files: list[Path] = []
        cursor_sec = 0.0
        total_segs = len(segments)

        # --- Phase 1: Render Normalized Cut Segments ---
        for pos, seg in enumerate(segments):
            if progress_callback:
                pct = 62 + int((pos / max(1, total_segs)) * 20)
                await progress_callback(
                    pct,
                    f"Cutting & encoding segment {pos + 1}/{total_segs} "
                    f"({seg.start:.1f}s - {seg.end:.1f}s, {seg.transition})",
                )

            seg_file = segments_dir / f"seg_{pos:03d}.mp4"
            actual_dur = await self._render_single_segment(
                source_video=source_video,
                source_meta=source_meta,
                seg=seg,
                out_clip=seg_file,
                aspect_ratio=aspect_ratio,
            )
            seg.timeline_start = round(cursor_sec, 3)
            cursor_sec += actual_dur
            seg.timeline_end = round(cursor_sec, 3)
            segment_files.append(seg_file)

        total_stitched_dur = max(0.3, cursor_sec)

        # --- Phase 2: Concatenate Segments ---
        if progress_callback:
            await progress_callback(84, f"Stitching {total_segs} timeline clips ({total_stitched_dur:.1f}s)...")

        stitched_path = work_dir / "stitched_base.mp4"
        if len(segment_files) == 1:
            shutil.copy2(segment_files[0], stitched_path)
        else:
            concat_txt = work_dir / "concat_list.txt"
            lines = []
            for clip in segment_files:
                escaped = clip.resolve().as_posix().replace("'", "'\\''")
                lines.append(f"file '{escaped}'")
            concat_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

            concat_args = [
                self.ffmpeg_bin,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(stitched_path),
            ]
            await self._run_cmd(concat_args, timeout_sec=180.0, step_name="concat-segments")

        # --- Phase 3: Prepare Voiceover Tracks & Mix with Dynamic Audio Ducking ---
        prepared_vo_inputs: list[Path] = []
        vo_delays_ms: list[int] = []
        vo_windows: list[tuple[int, float, float]] = []

        for pos, seg in enumerate(segments):
            if not seg.voiceover_audio_path:
                continue
            raw_vo = Path(seg.voiceover_audio_path)
            if not raw_vo.exists() or raw_vo.stat().st_size == 0:
                continue

            slot_dur = max(0.35, seg.timeline_end - seg.timeline_start)
            norm_wav = vo_norm_dir / f"vo_norm_{pos:03d}.wav"
            norm_dur = await self._prepare_voiceover_clip(raw_vo, slot_dur, norm_wav)
            if norm_dur > 0.05 and norm_wav.exists():
                seg.voiceover_duration = round(norm_dur, 3)
                # Slight 60ms lead-in padding after cut boundary feels natural to human ear
                lead_in = 0.06 if slot_dur - norm_dur > 0.12 else 0.0
                start_offset = seg.timeline_start + lead_in
                delay_ms = max(0, int(round(start_offset * 1000)))
                prepared_vo_inputs.append(norm_wav)
                vo_delays_ms.append(delay_ms)
                vo_windows.append((delay_ms, start_offset, start_offset + norm_dur))

        if not prepared_vo_inputs:
            # No voiceover tracks present: finalize stitched video directly with faststart
            if progress_callback:
                await progress_callback(95, "Finalizing MP4 container with HTTP faststart...")
            finalize_args = [
                self.ffmpeg_bin,
                "-y",
                "-i",
                str(stitched_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            await self._run_cmd(finalize_args, timeout_sec=120.0, step_name="finalize-no-vo")
        else:
            if progress_callback:
                await progress_callback(
                    90,
                    f"Mixing {len(prepared_vo_inputs)} voiceover tracks with dynamic sidechain audio ducking...",
                )

            input_flags: list[str] = ["-i", str(stitched_path)]
            for wav_file in prepared_vo_inputs:
                input_flags.extend(["-i", str(wav_file)])

            sidechain_graph = self.build_sidechain_ducking_graph(
                vo_delays_ms=vo_delays_ms,
                total_duration=total_stitched_dur,
                ducking_level=ducking_level,
            )
            mix_args = [
                self.ffmpeg_bin,
                "-y",
                *input_flags,
                "-filter_complex",
                sidechain_graph,
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            try:
                await self._run_cmd(mix_args, timeout_sec=240.0, step_name="mix-sidechain")
            except MediaEngineError as exc:
                logger.warning(
                    "Sidechain ducking filter failed (%s); falling back to envelope volume ducking.",
                    exc,
                )
                envelope_graph = self.build_envelope_ducking_graph(
                    vo_windows=vo_windows,
                    total_duration=total_stitched_dur,
                    ducking_level=ducking_level,
                )
                fallback_args = [
                    self.ffmpeg_bin,
                    "-y",
                    *input_flags,
                    "-filter_complex",
                    envelope_graph,
                    "-map",
                    "0:v:0",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
                await self._run_cmd(fallback_args, timeout_sec=240.0, step_name="mix-envelope-fallback")

        if progress_callback:
            await progress_callback(98, "Verifying rendered MP4 stream integrity...")

        output_meta = await self.probe_video(output_path)
        return output_path, output_meta, segments

    @staticmethod
    def _extract_rotation_degrees(video_stream: dict[str, Any]) -> int:
        """Read rotation metadata from stream tags or `side_data_list` display matrix."""
        tags = video_stream.get("tags") or {}
        if "rotate" in tags:
            try:
                return int(float(tags["rotate"])) % 360
            except (TypeError, ValueError):
                pass

        for side_data in video_stream.get("side_data_list") or []:
            if "rotation" in side_data:
                try:
                    return int(float(side_data["rotation"])) % 360
                except (TypeError, ValueError):
                    pass
        return 0

    @staticmethod
    def _parse_frame_rate(rate_str: str) -> float:
        """Safely convert FFprobe rational frame rate strings (`30000/1001`) to float."""
        try:
            frac = Fraction(rate_str)
            if frac.denominator == 0:
                return 30.0
            val = float(frac)
            return val if 1.0 <= val <= 240.0 else 30.0
        except (ValueError, ZeroDivisionError):
            return 30.0
