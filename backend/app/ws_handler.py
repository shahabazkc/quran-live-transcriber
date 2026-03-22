"""
ws_handler.py
─────────────
WebSocket handler for real-time audio transcription.

Protocol
────────
Client → Server:
  { "type": "config", "model": "<model_label or model_id>", "language": "arabic" }
  { "type": "audio_chunk", "data": "<base64 int16 PCM>", "src_rate": 16000 }
  { "type": "stop" }

Server → Client:
  { "type": "ready" }
  { "type": "transcript", "text": "...", "chunk_index": N }
  { "type": "error", "message": "..." }

Design: transcription runs in a background thread pool so the WebSocket
loop never blocks — the mic keeps streaming chunks uninterrupted.
"""

import asyncio
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from .audio import pcm_frames_to_float32
from .models import MODELS, load_model, transcribe_chunk

log = logging.getLogger(__name__)

# Shared thread pool — one worker per connection is fine; Whisper is GIL-bound
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="whisper")


async def handle_ws(websocket: WebSocket):
    await websocket.accept()

    pipe        = None
    chunk_index = 0
    loop        = asyncio.get_event_loop()

    # In-flight transcription tasks — we don't await them so the mic loop
    # continues unblocked while transcription runs concurrently.
    pending: set[asyncio.Future] = set()

    async def _send(obj: dict):
        try:
            await websocket.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    def _run_transcribe(audio_np, idx: int):
        """Blocking work — runs in thread pool."""
        return transcribe_chunk(pipe, audio_np), idx

    async def _dispatch_chunk(raw_pcm: bytes, src_rate: int):
        nonlocal chunk_index
        if pipe is None:
            await _send({"type": "error", "message": "Model not loaded. Send config first."})
            return

        chunk_index += 1
        idx = chunk_index

        audio_np = pcm_frames_to_float32(raw_pcm, src_rate)

        future = loop.run_in_executor(_executor, _run_transcribe, audio_np, idx)
        pending.add(future)

        def _on_done(fut: asyncio.Future):
            pending.discard(fut)
            if fut.cancelled() or fut.exception():
                err = fut.exception()
                asyncio.run_coroutine_threadsafe(
                    _send({"type": "error", "message": str(err)}), loop
                )
                return
            text, i = fut.result()
            asyncio.run_coroutine_threadsafe(
                _send({"type": "transcript", "text": text, "chunk_index": i}), loop
            )

        future.add_done_callback(_on_done)

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
                # Accept either a display label or a raw HF model ID
                model_id  = MODELS.get(model_key, model_key)
                try:
                    pipe = await loop.run_in_executor(_executor, load_model, model_id)
                    await _send({"type": "ready"})
                except Exception as e:
                    await _send({"type": "error", "message": f"Model load failed: {e}"})

            elif mtype == "audio_chunk":
                b64     = msg.get("data", "")
                src_rate = int(msg.get("src_rate", 16000))
                try:
                    pcm_bytes = base64.b64decode(b64)
                except Exception:
                    await _send({"type": "error", "message": "Invalid base64 audio data"})
                    continue
                await _dispatch_chunk(pcm_bytes, src_rate)

            elif mtype == "stop":
                # Wait for all in-flight transcriptions to finish
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                break

            else:
                await _send({"type": "error", "message": f"Unknown message type: {mtype}"})

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    finally:
        # Cancel any remaining futures on disconnect
        for fut in list(pending):
            fut.cancel()
