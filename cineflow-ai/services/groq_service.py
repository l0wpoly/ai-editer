"""
Groq High-Speed Script Optimization & Voiceover Synthesis Service (`services/groq_service.py`).

Responsibilities:
1. Low-latency script pacing & tone optimization via Groq LPU (`llama-3.3-70b-versatile`):
   - Tightens word counts to strictly match each segment's effective duration (~2.35 words/sec).
   - Generates both expressive Orpheus-tagged script (`[cheerful]`, `[excited]`, `[dramatic]`,
     `[whispering]`) and clean SSML-prosody parameters (`rate`, `pitch`) for neural fallback.
2. Synchronous multi-segment audio synthesis:
   - Primary: Groq native Orpheus TTS (`canopylabs/orpheus-v1-english`) with automatic
     sentence-chunk splitting (<=185 chars/request) and lossless PCM WAV concatenation.
   - Automatic resilient fallback (or explicit mode): Microsoft Edge Neural TTS (`edge-tts`)
     with prosody rate/pitch control when Groq TTS hits rate limits or account terms gates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import edge_tts
from groq import AsyncGroq, Groq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from services.video_engine import RenderSegmentSpec, VideoEngine

logger = logging.getLogger("cineflow.groq_service")

ProgressCallback = Callable[[int, str], Awaitable[None]]


class GroqServiceError(RuntimeError):
    """Raised when both script optimization and voiceover synthesis fail unrecoverably."""


class PolishedLineSpec(BaseModel):
    """Pacing-calibrated voiceover line returned by Groq LLM."""

    index: int
    clean_text: str = Field(default="")
    orpheus_text: str = Field(default="")
    prosody_rate: str = Field(default="+0%")
    prosody_pitch: str = Field(default="+0Hz")


# Unified voice persona catalog mapping Groq Orpheus voices <-> Neural Edge TTS voices
VOICE_CATALOG: dict[str, dict[str, str]] = {
    "troy": {
        "groq_voice": "troy",
        "edge_voice": "en-US-GuyNeural",
        "label": "Troy — Energetic Male Narrator",
    },
    "daniel": {
        "groq_voice": "daniel",
        "edge_voice": "en-GB-RyanNeural",
        "label": "Daniel — Cinematic Documentary Male",
    },
    "austin": {
        "groq_voice": "austin",
        "edge_voice": "en-US-ChristopherNeural",
        "label": "Austin — Warm Conversational Male",
    },
    "autumn": {
        "groq_voice": "autumn",
        "edge_voice": "en-US-AriaNeural",
        "label": "Autumn — Expressive Female Narrator",
    },
    "diana": {
        "groq_voice": "diana",
        "edge_voice": "en-US-JennyNeural",
        "label": "Diana — Crisp Studio Female",
    },
    "hannah": {
        "groq_voice": "hannah",
        "edge_voice": "en-AU-NatashaNeural",
        "label": "Hannah — Bright Upbeat Female",
    },
}

TAG_STRIP_RE = re.compile(r"(\[[a-zA-Z_\-\s]{2,20}\]|<[a-zA-Z_\-\s]{2,20}>)")
RATE_VALID_RE = re.compile(r"^[+-]\d{1,2}%$")
PITCH_VALID_RE = re.compile(r"^[+-]\d{1,2}Hz$")


def strip_vocal_tags(text: str) -> str:
    """Remove bracketed Orpheus vocal direction tags to obtain clean spoken text."""
    cleaned = TAG_STRIP_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def split_text_for_orpheus(text: str, max_chars: int = 185) -> list[str]:
    """
    Split longer voiceover lines on natural sentence/clause boundaries so every
    request stays safely under Orpheus's 200-character sweet spot.
    """
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        else:
            if current:
                chunks.append(current)
                current = ""
            # Further split long single sentence on commas/words
            words = sentence.split(" ")
            sub = ""
            for word in words:
                cand = f"{sub} {word}".strip() if sub else word
                if len(cand) <= max_chars:
                    sub = cand
                else:
                    if sub:
                        chunks.append(sub)
                    sub = word[:max_chars]
            if sub:
                current = sub

    if current:
        chunks.append(current)
    return chunks


def concat_wav_files(wav_paths: list[Path], output_wav: Path, pause_ms: int = 75) -> Path:
    """Concatenate multiple identically-formatted PCM WAV chunks with a brief natural breath pause."""
    if len(wav_paths) == 1:
        shutil.copy2(wav_paths[0], output_wav)
        return output_wav

    params = None
    frames_buffer: list[bytes] = []

    for idx, wav_file in enumerate(wav_paths):
        with wave.open(str(wav_file), "rb") as reader:
            cur_params = reader.getparams()
            if params is None:
                params = cur_params
            data = reader.readframes(cur_params.nframes)
            frames_buffer.append(data)

            if idx < len(wav_paths) - 1 and pause_ms > 0:
                silence_frames = int(cur_params.framerate * (pause_ms / 1000.0))
                frame_width = cur_params.nchannels * cur_params.sampwidth
                frames_buffer.append(b"\x00" * (silence_frames * frame_width))

    if params is None:
        raise GroqServiceError("No valid WAV chunks available to concatenate.")

    with wave.open(str(output_wav), "wb") as writer:
        writer.setparams(params)
        for chunk_bytes in frames_buffer:
            writer.writeframes(chunk_bytes)

    return output_wav


def _is_retryable_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    # Do NOT retry model_terms_required or invalid_request_error (400); fall back immediately
    if "model_terms_required" in msg or "terms" in msg or "400" in msg:
        return False
    return any(t in msg for t in ("429", "rate_limit", "503", "502", "timeout", "connection"))


class GroqVoiceoverService:
    """Low-latency Groq LPU script polisher and dual-engine TTS voiceover synthesizer."""

    def __init__(self) -> None:
        self.llm_model = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
        self.fallback_llm_model = "llama-3.1-8b-instant"
        self.tts_model = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
        self.default_tts_engine = os.getenv("DEFAULT_TTS_ENGINE", "auto").lower()

    def is_configured(self) -> bool:
        key = os.getenv("GROQ_API_KEY", "").strip()
        return bool(key and key != "your_groq_api_key_here")

    def _get_api_key(self) -> str:
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key or key == "your_groq_api_key_here":
            raise GroqServiceError(
                "GROQ_API_KEY is missing or unconfigured. Add a valid Groq API key to your environment."
            )
        return key

    async def polish_and_synthesize(
        self,
        segments: list[RenderSegmentSpec],
        user_prompt: str,
        narrative_tone: str,
        voice_persona: str,
        work_dir: Path,
        video_engine: VideoEngine,
        tts_engine_preference: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[RenderSegmentSpec]:
        """
        1. Polish voiceover pacing & inject expressive vocal tags via Groq LPU.
        2. Synthesize synchronous audio files for each narrated segment.
        """
        narrated_indices = [i for i, s in enumerate(segments) if s.voiceover_text.strip()]
        if not narrated_indices:
            logger.info("No segments contain voiceover lines; skipping TTS synthesis.")
            return segments

        persona_key = voice_persona.lower() if voice_persona.lower() in VOICE_CATALOG else "troy"
        voice_cfg = VOICE_CATALOG[persona_key]
        engine_mode = (tts_engine_preference or self.default_tts_engine or "auto").lower()

        # --- Step 1: Groq LPU High-Speed Script Pacing & Polish ---
        if self.is_configured():
            if progress_callback:
                await progress_callback(
                    44,
                    f"Groq LPU ({self.llm_model}) optimizing script pacing & vocal expressions...",
                )
            polished_map = await self._polish_script_with_groq(
                segments=segments,
                user_prompt=user_prompt,
                narrative_tone=narrative_tone,
            )
        else:
            if engine_mode == "groq":
                raise GroqServiceError("GROQ_API_KEY is required when Groq TTS is explicitly selected.")
            logger.warning("GROQ_API_KEY not configured; using Gemini script directly with neural edge-tts.")
            polished_map = {}

        # --- Step 2: Synthesize Audio per Segment ---
        vo_dir = work_dir / "voiceovers"
        vo_dir.mkdir(parents=True, exist_ok=True)

        groq_tts_available = self.is_configured() and engine_mode in ("auto", "groq")
        total_vo = len(narrated_indices)

        for step_num, seg_idx in enumerate(narrated_indices, start=1):
            seg = segments[seg_idx]
            polished = polished_map.get(seg.index)

            if polished and polished.clean_text.strip():
                seg.voiceover_text = polished.clean_text.strip()
                seg.orpheus_text = (polished.orpheus_text or polished.clean_text).strip()
                rate = polished.prosody_rate if RATE_VALID_RE.match(polished.prosody_rate) else "+0%"
                pitch = polished.prosody_pitch if PITCH_VALID_RE.match(polished.prosody_pitch) else "+0Hz"
            else:
                clean = strip_vocal_tags(seg.voiceover_text)
                seg.voiceover_text = clean
                seg.orpheus_text = seg.orpheus_text or clean
                rate = "+6%" if narrative_tone == "energetic" else "+0%"
                pitch = "+0Hz"

            if not seg.voiceover_text:
                continue

            if progress_callback:
                pct = 48 + int((step_num / max(1, total_vo)) * 12)
                await progress_callback(
                    pct,
                    f"Synthesizing voiceover {step_num}/{total_vo} ({voice_cfg['label'].split(' ')[0]})...",
                )

            synthesized_path: Optional[Path] = None
            engine_used: Optional[str] = None

            # Primary attempt: Groq Orpheus Native TTS
            if groq_tts_available:
                try:
                    wav_target = vo_dir / f"vo_seg_{seg.index:03d}.wav"
                    synthesized_path = await self._synthesize_groq_orpheus(
                        orpheus_text=seg.orpheus_text or seg.voiceover_text,
                        groq_voice=voice_cfg["groq_voice"],
                        output_wav=wav_target,
                        temp_dir=vo_dir,
                    )
                    engine_used = f"groq:{self.tts_model}"
                except Exception as groq_tts_exc:
                    err_str = str(groq_tts_exc)
                    logger.warning(
                        "Groq Orpheus TTS unavailable for segment %d (%s); switching to neural edge-tts.",
                        seg.index,
                        err_str,
                    )
                    # If account hasn't accepted Orpheus model terms or model is disabled, avoid repeating 400s
                    if "model_terms_required" in err_str.lower() or "400" in err_str or "404" in err_str:
                        groq_tts_available = False

            # Seamless fallback (or primary when engine_mode == "edge"): Microsoft Edge Neural TTS
            if synthesized_path is None:
                mp3_target = vo_dir / f"vo_seg_{seg.index:03d}.mp3"
                synthesized_path = await self._synthesize_edge_tts(
                    clean_text=seg.voiceover_text,
                    edge_voice=voice_cfg["edge_voice"],
                    rate=rate,
                    pitch=pitch,
                    output_mp3=mp3_target,
                )
                engine_used = f"edge-tts:{voice_cfg['edge_voice']}"

            if synthesized_path and synthesized_path.exists() and synthesized_path.stat().st_size > 0:
                dur = await video_engine.probe_audio_duration(synthesized_path)
                seg.voiceover_audio_path = str(synthesized_path)
                seg.voiceover_duration = round(dur, 3)
                seg.tts_engine_used = engine_used

        return segments

    async def _polish_script_with_groq(
        self,
        segments: list[RenderSegmentSpec],
        user_prompt: str,
        narrative_tone: str,
    ) -> dict[int, PolishedLineSpec]:
        """Use Groq's high-speed Llama inference to calibrate word budgets & vocal directions."""
        api_key = self._get_api_key()
        async_client = AsyncGroq(api_key=api_key)

        input_items = []
        for seg in segments:
            if not seg.voiceover_text.strip():
                continue
            eff_dur = round(seg.target_effective_duration, 2)
            max_words = max(2, int(eff_dur * 2.35))
            input_items.append(
                {
                    "index": seg.index,
                    "scene_label": seg.scene_label,
                    "slot_seconds": eff_dur,
                    "max_words": max_words,
                    "draft_script": seg.voiceover_text,
                }
            )

        if not input_items:
            return {}

        system_msg = (
            "You are a broadcast voiceover director optimizing narration timing for an AI video editor.\n"
            "For each input segment, polish `draft_script` so that:\n"
            "1. Word count NEVER exceeds `max_words` (critical so speech finishes cleanly inside `slot_seconds`).\n"
            "2. Flow across consecutive segments feels cohesive, punchy, and natural when spoken aloud.\n"
            "3. `clean_text` contains ONLY plain spoken words and punctuation (NO brackets or stage tags).\n"
            "4. `orpheus_text` contains the exact same words optionally prefixed with at most ONE valid "
            "Orpheus vocal tag from `[cheerful]`, `[excited]`, `[dramatic]`, `[whispering]` when it enhances tone.\n"
            "5. `prosody_rate` is a slight speed adjustment string formatted like `\"+6%\"`, `\"+0%\"`, or `\"-4%\"`.\n"
            "6. `prosody_pitch` is formatted like `\"+2Hz\"`, `\"+0Hz\"`, or `\"-2Hz\"`.\n"
            "Respond strictly as a JSON object with key `\"lines\"` containing an array of objects: "
            "`{\"index\": int, \"clean_text\": str, \"orpheus_text\": str, \"prosody_rate\": str, \"prosody_pitch\": str}`."
        )

        user_msg = json.dumps(
            {
                "editing_brief": user_prompt,
                "narrative_tone": narrative_tone,
                "segments": input_items,
            },
            ensure_ascii=False,
        )

        for model_name in (self.llm_model, self.fallback_llm_model):
            try:
                completion = await self._call_groq_chat(
                    client=async_client,
                    model_name=model_name,
                    system_msg=system_msg,
                    user_msg=user_msg,
                )
                raw_json = completion.choices[0].message.content or "{}"
                payload = json.loads(raw_json)
                lines_raw: list[dict[str, Any]] = payload.get("lines", [])
                result: dict[int, PolishedLineSpec] = {}
                for item in lines_raw:
                    parsed = PolishedLineSpec.model_validate(item)
                    parsed.clean_text = strip_vocal_tags(parsed.clean_text)
                    result[parsed.index] = parsed
                return result
            except Exception as exc:
                logger.warning("Groq script polish on model '%s' failed (%s).", model_name, exc)

        return {}

    @retry(
        retry=retry_if_exception(_is_retryable_rate_limit),
        wait=wait_exponential(multiplier=1.2, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_groq_chat(
        self,
        client: AsyncGroq,
        model_name: str,
        system_msg: str,
        user_msg: str,
    ) -> Any:
        return await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )

    @retry(
        retry=retry_if_exception(_is_retryable_rate_limit),
        wait=wait_exponential(multiplier=1.2, min=1, max=8),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    async def _synthesize_groq_orpheus(
        self,
        orpheus_text: str,
        groq_voice: str,
        output_wav: Path,
        temp_dir: Path,
    ) -> Path:
        """Synthesize WAV audio via Groq `audio.speech.create` with automatic <=185-char chunking."""
        api_key = self._get_api_key()
        chunks = split_text_for_orpheus(orpheus_text, max_chars=185)
        if not chunks:
            raise GroqServiceError("Empty text passed to Groq Orpheus TTS.")

        def _sync_generate() -> Path:
            sync_client = Groq(api_key=api_key)
            chunk_files: list[Path] = []
            for c_idx, chunk_str in enumerate(chunks):
                chunk_wav = temp_dir / f"{output_wav.stem}_part{c_idx:02d}.wav"
                response = sync_client.audio.speech.create(
                    model=self.tts_model,
                    voice=groq_voice,
                    input=chunk_str,
                    response_format="wav",
                )
                response.write_to_file(str(chunk_wav))
                chunk_files.append(chunk_wav)

            concat_wav_files(chunk_files, output_wav, pause_ms=70)
            for cf in chunk_files:
                if cf != output_wav:
                    cf.unlink(missing_ok=True)
            return output_wav

        return await asyncio.to_thread(_sync_generate)

    @retry(
        retry=retry_if_exception(lambda _: True),
        wait=wait_exponential(multiplier=1.2, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _synthesize_edge_tts(
        self,
        clean_text: str,
        edge_voice: str,
        rate: str,
        pitch: str,
        output_mp3: Path,
    ) -> Path:
        """Synthesize high-definition neural speech via Microsoft Edge TTS (`edge-tts`)."""
        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=edge_voice,
            rate=rate,
            pitch=pitch,
        )
        await communicate.save(str(output_mp3))
        if not output_mp3.exists() or output_mp3.stat().st_size == 0:
            raise GroqServiceError("edge-tts completed without writing audio bytes.")
        return output_mp3
