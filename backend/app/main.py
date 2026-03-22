"""
main.py
───────
FastAPI application entrypoint.

Routes
──────
GET  /api/devices              → list input audio devices
GET  /api/devices/{index}/rate → best sample rate for a device
GET  /api/models               → available model labels + IDs
WS   /ws/transcribe            → WebSocket streaming transcription
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .audio import get_input_devices, get_best_sample_rate
from .models import MODELS
from .ws_handler import handle_ws

app = FastAPI(title="Quran Transcriber API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/devices")
async def list_devices():
    """Return available microphone input devices."""
    devices = get_input_devices()
    return {"devices": [{"label": label, "index": idx} for label, idx in devices.items()]}


@app.get("/api/devices/{index}/rate")
async def device_rate(index: int):
    """Return the best supported sample rate for a device index."""
    rate = get_best_sample_rate(index)
    return {"index": index, "rate": rate}


@app.get("/api/models")
async def list_models():
    """Return available Whisper model options."""
    return {
        "models": [{"label": label, "id": model_id} for label, model_id in MODELS.items()]
    }


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await handle_ws(websocket)
