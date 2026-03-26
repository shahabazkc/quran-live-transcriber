"""
recitation_ws.py
────────────────
Dedicated recitation WebSocket with ordered, queued processing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .audio import pcm_frames_to_float32
from .models import (
    MODELS,
    get_pipeline_runtime_info,
    load_model_fresh,
    transcribe_chunk_detailed,
    unload_model,
)
from .quran_content import load_surah
from .recitation_matcher import RecitationMatcher, RecognizedWord

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recitation-whisper")

# Hybrid protection:
# - Per-session queue cap limits buffered chunks for each reciter
# - Global semaphore limits active transcriptions across all sessions
GLOBAL_ACTIVE_TRANSCRIBE_CAP = 5
_global_transcribe_sem = asyncio.Semaphore(GLOBAL_ACTIVE_TRANSCRIBE_CAP)


async def handle_recitation_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    session_id = str(uuid.uuid4())

    model_id: str | None = None
    gpu_index = 1
    pipe = None
    language = "arabic"
    per_session_cap = 200
    batch_size = 1
    process_interval_ms = 0
    stopped = False
    accepting_audio = False
    mispronounced_count = 0
    min_voice_rms = 0.003

    matcher: RecitationMatcher | None = None
    surah_slug = "surah-mulk"
    queue: asyncio.Queue[dict] = asyncio.Queue()
    chunk_index = 0

    async def _send(obj: dict):
        try:
            await websocket.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    async def _switch_model(next_model_id: str, next_gpu_index: int):
        nonlocal pipe, model_id
        if model_id == next_model_id and pipe is not None:
            return
        if pipe is not None:
            await loop.run_in_executor(_executor, unload_model, pipe)
            pipe = None
        pipe = await loop.run_in_executor(_executor, load_model_fresh, next_model_id, next_gpu_index)
        model_id = next_model_id

    def _pcm_rms(raw_pcm: bytes) -> float:
        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2))) / 32768.0

    async def _emit_queue_status():
        await _send(
            {
                "type": "queue_status",
                "pending_session": queue.qsize(),
                "global_active_cap": GLOBAL_ACTIVE_TRANSCRIBE_CAP,
                "global_available_slots": _global_transcribe_sem._value,  # noqa: SLF001
            }
        )

    async def _worker():
        nonlocal mispronounced_count, stopped
        while not stopped or not queue.empty():
            if process_interval_ms > 0:
                await asyncio.sleep(process_interval_ms / 1000.0)
            if queue.empty():
                await asyncio.sleep(0.01)
                continue

            batch = []
            while len(batch) < max(1, batch_size) and not queue.empty():
                batch.append(await queue.get())

            for item in batch:
                idx = item["idx"]
                try:
                    if pipe is None or matcher is None:
                        await _send({"type": "error", "message": "Session not configured."})
                        continue

                    audio_np = pcm_frames_to_float32(item["pcm"], item["src_rate"])
                    async with _global_transcribe_sem:
                        detailed = await loop.run_in_executor(
                            _executor, transcribe_chunk_detailed, pipe, audio_np, language, False
                        )

                    recognized = [
                        RecognizedWord(text=w.get("text", ""), confidence=w.get("confidence"))
                        for w in detailed.get("words", [])
                    ]
                    match = matcher.consume(recognized)
                    for ev in match["word_events"]:
                        if ev.get("status") == "mispronounced":
                            mispronounced_count += 1

                    await _send(
                        {
                            "type": "progress",
                            "chunk_index": idx,
                            "transcript_text": detailed.get("text", ""),
                            "current_position": match["current_position"],
                            "next_expected": match["next_expected"],
                            "word_events": match["word_events"],
                            "completed_words": match["completed_words"],
                            "is_complete": match["is_complete"],
                        }
                    )
                except Exception as exc:
                    await _send({"type": "error", "message": str(exc)})
                finally:
                    queue.task_done()
                    await _emit_queue_status()

    worker_task = asyncio.create_task(_worker())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send({"type": "error", "message": "Invalid JSON"})
                continue

            mtype = msg.get("type")
            if mtype == "config":
                model_key = msg.get("model", "")
                selected_model_id = MODELS.get(model_key, model_key)
                language = str(msg.get("language", "arabic"))
                surah_slug = str(msg.get("surah_slug", "surah-mulk"))
                per_session_cap = int(msg.get("max_pending", 5))
                batch_size = int(msg.get("max_batch_size", 1))
                process_interval_ms = int(msg.get("process_interval_ms", 0))
                min_voice_rms = float(msg.get("min_voice_rms", 0.003))
                gpu_index = int(msg.get("gpu_index", 1))
                accepting_audio = True
                stopped = False

                surah = load_surah(surah_slug)
                matcher = RecitationMatcher(surah["flattened_words"])
                await _switch_model(selected_model_id, gpu_index)

                await _send(
                    {
                        "type": "ready",
                        "session_id": session_id,
                        "surah_slug": surah_slug,
                        "total_words": surah["total_words"],
                        "runtime": get_pipeline_runtime_info(pipe),
                    }
                )
                await _emit_queue_status()

            elif mtype == "audio_chunk":
                if not accepting_audio:
                    await _send(
                        {
                            "type": "queue_backpressure",
                            "message": "Session is stopped. Send config to start again.",
                        }
                    )
                    continue

                if queue.qsize() >= max(1, per_session_cap):
                    # Keep backlogging as requested; only emit status (no reject).
                    await _send(
                        {
                            "type": "queue_status",
                            "pending_session": queue.qsize(),
                            "over_soft_limit": True,
                            "soft_limit": per_session_cap,
                        }
                    )

                b64 = msg.get("data", "")
                src_rate = int(msg.get("src_rate", 16000))
                try:
                    pcm = base64.b64decode(b64)
                except Exception:
                    await _send({"type": "error", "message": "Invalid base64 audio data"})
                    continue

                if _pcm_rms(pcm) < min_voice_rms:
                    await _send(
                        {
                            "type": "queue_status",
                            "pending_session": queue.qsize(),
                            "dropped_silent_chunk": True,
                        }
                    )
                    continue

                chunk_index += 1
                await queue.put({"idx": chunk_index, "pcm": pcm, "src_rate": src_rate})
                await _emit_queue_status()

            elif mtype == "stop":
                accepting_audio = False
                await queue.join()
                completed_words = matcher.cursor if matcher else 0
                await _send(
                    {
                        "type": "final_summary",
                        "session_id": session_id,
                        "completed_words": completed_words,
                        "mispronounced_count": mispronounced_count,
                    }
                )
                await _emit_queue_status()

            else:
                await _send({"type": "error", "message": f"Unknown message type: {mtype}"})

    except WebSocketDisconnect:
        log.info("Recitation WebSocket disconnected")
    finally:
        stopped = True
        await queue.join()
        worker_task.cancel()
        if pipe is not None:
            await loop.run_in_executor(_executor, unload_model, pipe)
