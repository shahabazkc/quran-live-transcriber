"""
audio.py
────────
Audio processing helpers for the FastAPI backend.
Ported from audio_utils.py — server-side only (no VADRecorder).
"""

import io
import struct
import wave
import numpy as np
import pyaudio

# ── Constants ─────────────────────────────────────────────────────────────────
FORMAT            = pyaudio.paInt16
CHANNELS          = 1
WHISPER_RATE      = 16_000
PREFERRED_RATES   = [16000, 44100, 48000, 22050, 8000]
FRAMES_PER_BUFFER = 1024

import re as _re

_HOST_API_PRIORITY = {
    "mme":                      0,
    "windows wasapi":           1,
    "windows directsound":      2,
    "core audio":               0,
    "pulseaudio":               0,
    "pipewire":                 0,
    "alsa":                     1,
    "jack audio connection kit":1,
    "asio":                     0,
}

_STRIP_RE = _re.compile(
    r'(\s*\(hw:\d+,\d+\)'
    r'|\s*\[plughw:\d+,\d+\]'
    r'|\s*\(\d+\s+in,.*?\)'
    r'|\s*@[^)]*'
    r'|\s*-\s*\d+$'
    r'|\s+\d+$'
    r')',
    _re.IGNORECASE,
)

_FRIENDLY = [
    ("macbook pro microphone",   "MacBook Pro Microphone"),
    ("macbook air microphone",   "MacBook Air Microphone"),
    ("built-in microphone",      "Built-in Microphone"),
    ("built-in input",           "Built-in Microphone"),
    ("internal microphone",      "Internal Microphone"),
    ("realtek",                  "Realtek Microphone"),
    ("default",                  "System Default Microphone"),
    ("pulse",                    "PulseAudio Default"),
    ("pipewire",                 "PipeWire Default"),
]

_LOOPBACK_FRAGMENTS = {
    "stereo mix", "what u hear", "wave out mix", "loopback",
    "monitor of", "virtual", "vb-audio", "blackhole",
    "soundflower", "voicemeeter",
}


def _clean_name(raw: str) -> str:
    name = _STRIP_RE.sub("", raw).strip()
    low  = name.lower()
    for fragment, friendly in _FRIENDLY:
        if fragment in low:
            return friendly
    return name or raw.strip()


def _host_priority(api_name: str) -> int:
    return _HOST_API_PRIORITY.get(api_name.lower().strip(), 99)


def _is_loopback(raw_name: str) -> bool:
    low = raw_name.lower()
    return any(f in low for f in _LOOPBACK_FRAGMENTS)


def _probe_device(p: pyaudio.PyAudio, idx: int, rate: int) -> bool:
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=idx,
            frames_per_buffer=512,
            start=False,
        )
        stream.close()
        return True
    except Exception:
        return False


def get_input_devices() -> dict[str, int]:
    """Return {clean_label: device_index} for all available input devices."""
    p = pyaudio.PyAudio()

    default_idx = -1
    try:
        default_idx = p.get_default_input_device_info()["index"]
    except Exception:
        pass

    candidates: list[dict] = []
    for i in range(p.get_device_count()):
        try:
            info     = p.get_device_info_by_index(i)
            api_info = p.get_host_api_info_by_index(info["hostApi"])
        except Exception:
            continue

        if info["maxInputChannels"] < 1:
            continue
        if _is_loopback(info["name"]):
            continue

        api_name   = api_info["name"]
        probe_rate = int(info.get("defaultSampleRate", 16000))
        if probe_rate not in (8000, 16000, 22050, 44100, 48000):
            probe_rate = 16000

        if not _probe_device(p, i, probe_rate):
            continue

        candidates.append({
            "idx":        i,
            "raw_name":   info["name"],
            "clean":      _clean_name(info["name"]),
            "api":        api_name,
            "priority":   _host_priority(api_name),
            "is_default": (i == default_idx),
        })

    p.terminate()

    if not candidates:
        return {}

    groups: dict[str, dict] = {}
    for d in candidates:
        key = d["clean"].lower()
        existing = groups.get(key)
        if existing is None:
            groups[key] = d
        elif d["priority"] < existing["priority"]:
            groups[key] = d
        elif d["priority"] == existing["priority"] and d["is_default"]:
            groups[key] = d

    ordered = sorted(
        groups.values(),
        key=lambda d: (0 if d["is_default"] else 1, d["clean"].lower()),
    )

    result: dict[str, int] = {}

    defaults = [d for d in ordered if d["is_default"]]
    if defaults:
        result["Default Microphone"] = defaults[0]["idx"]

    for d in ordered:
        label = d["clean"]
        if label in result:
            label = f"{label} (2)"
        if label not in result:
            result[label] = d["idx"]

    return result


def get_best_sample_rate(device_index: int) -> int:
    """Return the best supported sample rate for a device."""
    p = pyaudio.PyAudio()
    try:
        for rate in PREFERRED_RATES:
            try:
                supported = p.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=CHANNELS,
                    input_format=FORMAT,
                )
                if supported:
                    return rate
            except Exception:
                continue
        info = p.get_device_info_by_index(device_index)
        return int(info.get("defaultSampleRate", 16000))
    finally:
        p.terminate()


# ── Signal helpers ────────────────────────────────────────────────────────────

def amplify(raw_bytes: bytes, gain: float) -> bytes:
    """Scale int16 PCM by *gain*, hard-clipping to int16 range."""
    if gain == 1.0:
        return raw_bytes
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
    return samples.tobytes()


def _rms(raw_bytes: bytes) -> float:
    """Normalised RMS of a raw int16 PCM buffer (0.0 – 1.0)."""
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2))) / 32768.0


def pcm_frames_to_float32(
    raw_bytes: bytes,
    src_rate: int,
    dst_rate: int = WHISPER_RATE,
) -> np.ndarray:
    """
    Convert raw int16 PCM bytes → float32 [-1, 1], resampling if needed.
    Returns 1-D float32 array ready for the Whisper pipeline.
    """
    import librosa
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if src_rate != dst_rate:
        audio = librosa.resample(audio, orig_sr=src_rate, target_sr=dst_rate)
    return audio


def pcm_to_wav_bytes(raw_bytes: bytes, rate: int) -> bytes:
    """Encode raw int16 PCM bytes → in-memory WAV bytes at *rate*."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(raw_bytes)
    return buf.getvalue()
