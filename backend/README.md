# Quran Transcriber — FastAPI Backend

Real-time Arabic speech-to-text via WebSocket streaming using fine-tuned Whisper models.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Server starts at `http://localhost:8000`

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/devices` | List available microphone input devices |
| GET | `/api/devices/{index}/rate` | Best sample rate for a device |
| GET | `/api/models` | Available Whisper models |
| WS | `/ws/transcribe` | Streaming transcription WebSocket |

## WebSocket Protocol

**Client → Server:**
```json
{ "type": "config", "model": "Whisper Medium — Quran fine-tune", "language": "arabic" }
{ "type": "audio_chunk", "data": "<base64 int16 PCM>", "src_rate": 16000 }
{ "type": "stop" }
```

**Server → Client:**
```json
{ "type": "ready" }
{ "type": "transcript", "text": "...", "chunk_index": 1 }
{ "type": "error", "message": "..." }
```

## Models

| Label | HuggingFace ID |
|-------|----------------|
| Whisper Medium — Quran fine-tune | `shahabazkc10/whisper-medium-ar-quran-mix-norm` |
| Whisper Small — Quran fine-tune | `shahabazkc10/whisper-small-ar-quran-mix-norm` |
| Whisper Large-v3 — OpenAI baseline | `openai/whisper-large-v3` |
| Whisper Large-v3-Turbo — Quran fine-tune | `shahabazkc10/whisper-large-v3-turbo-ar-quran-mix-norm` |
