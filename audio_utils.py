"""
audio_utils.py
──────────────
Reusable audio recording helpers built on PyAudio + energy-based VAD.

Root-cause fix (2026-03-23)
────────────────────────────
Many mics (especially built-in laptop/USB mics) only support 16000 Hz or
44100 Hz natively.  If you ask PyAudio for 44100 Hz on a 16000 Hz mic, the
OS silently delivers 16 kHz samples into a buffer tagged as 44100 Hz.  The
WAV file then plays back 2.75× too fast and sounds garbled / chipmunk.

We fix this by probing the device for the best supported rate from a
priority list, opening the stream at that detected rate, and writing the
WAV at that same rate.  Resampling to 16 kHz happens only in memory when
feeding Whisper — the stored file is always at the true capture rate.

Public API
──────────
get_input_devices()               → dict[label, device_index]
get_best_sample_rate(device_index)→ int   (actual supported rate)
amplify(raw_bytes, gain)          → bytes
pcm_frames_to_float32(frames, src_rate, dst_rate) → np.ndarray
pcm_to_wav_bytes(frames, rate)    → bytes

VADRecorder(device_index, gain, silence_threshold,
            min_silence_ms, min_chunk_ms, max_chunk_ms)
  .start()
  .stop()               → list[bytes]
  .pop_ready_chunk()    → list[bytes] | None
  .all_frames           → list[bytes]
  .record_rate          → int   (actual rate used)
  .duration_seconds()   → float
  .current_rms          → float
  .is_speech            → bool
  .chunks_emitted       → int
"""

import io
import queue
import threading
import wave
import numpy as np
import pyaudio

# ── Constants ─────────────────────────────────────────────────────────────────
FORMAT            = pyaudio.paInt16
CHANNELS          = 1
WHISPER_RATE      = 16000

# Probe order: prefer 16kHz (Whisper-native, most mics support it),
# then 44100, then 48000, then 22050, then 8000 as last resort.
PREFERRED_RATES   = [16000, 44100, 48000, 22050, 8000]

FRAMES_PER_BUFFER = 1024   # ~23ms @ 44100, ~64ms @ 16000 — fine-grained VAD


# ── Device discovery ──────────────────────────────────────────────────────────

import re as _re
import sys as _sys

# Host API priority per platform: lower = more preferred.
# On Windows prefer MME over WASAPI for input — WASAPI exclusive mode fails on
# many BT headsets; MME goes through the Windows audio engine and works for
# any device that the OS considers "ready".
_HOST_API_PRIORITY = {
    # Windows  (MME is the safest for recording; WASAPI shared-mode also works
    #           but ranks lower because it triggers -9999 on disconnected BT)
    "mme":                      0,
    "windows wasapi":           1,
    "windows directsound":      2,
    # macOS
    "core audio":               0,
    # Linux
    "pulseaudio":               0,
    "pipewire":                 0,
    "alsa":                     1,
    "jack audio connection kit":1,
    # Generic
    "asio":                     0,
}

# Suffixes that WASAPI/ALSA append — strip them for display
_STRIP_RE = _re.compile(
    r'(\s*\(hw:\d+,\d+\)'        # ALSA  hw:0,0
    r'|\s*\[plughw:\d+,\d+\]'    # ALSA  plughw
    r'|\s*\(\d+\s+in,.*?\)'      # PyAudio "(2 in, 0 out)"
    r'|\s*@[^)]*'                 # Windows  "@System32/drivers/..."
    r'|\s*-\s*\d+$'              # trailing  "- 0"
    r'|\s+\d+$'                  # trailing  " 2"
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

# Virtual / loopback devices — never show these
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
    """
    Actually try to open the device for input at *rate*.
    Returns True only if it succeeds — this is the key filter that eliminates
    saved-but-disconnected Bluetooth devices and any other ghosts.
    Windows error -9999 ("Unanticipated host error") means the device exists
    in the registry but is not physically available right now.
    """
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=idx,
            frames_per_buffer=512,
            start=False,   # don't start streaming — just verify it opens
        )
        stream.close()
        return True
    except Exception:
        return False


def get_input_devices() -> dict[str, int]:
    """
    Return {clean_label: device_index} containing ONLY devices that are
    physically available right now — mirrors GMeet / Chrome behaviour.

    Steps
    ─────
    1. Enumerate all PyAudio input devices.
    2. Skip loopbacks / virtual devices.
    3. **Probe each device** — actually try to open it.  Devices that fail
       (disconnected BT, disabled, driver error -9999) are silently dropped.
    4. Group duplicates by cleaned name; keep the best host-API entry per group.
    5. Put "Default Microphone" first; rest alphabetical.
    """
    p = pyaudio.PyAudio()

    default_idx = -1
    try:
        default_idx = p.get_default_input_device_info()["index"]
    except Exception:
        pass

    # Collect candidates
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
        # Normalise uncommon rates for the probe
        if probe_rate not in (8000, 16000, 22050, 44100, 48000):
            probe_rate = 16000

        # ── THE KEY CHECK ──────────────────────────────────────────────────
        if not _probe_device(p, i, probe_rate):
            continue   # not available right now — skip (BT ghost, disabled…)
        # ───────────────────────────────────────────────────────────────────

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

    # Group by cleaned name; keep best host-API entry
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

    # Sort: default first, then alphabetical
    ordered = sorted(
        groups.values(),
        key=lambda d: (0 if d["is_default"] else 1, d["clean"].lower()),
    )

    result: dict[str, int] = {}

    # "Default Microphone" alias at top
    defaults = [d for d in ordered if d["is_default"]]
    if defaults:
        result["Default Microphone"] = defaults[0]["idx"]

    for d in ordered:
        label = d["clean"]
        if label in result:          # same clean name as the default alias
            label = f"{label} (2)"
        if label not in result:      # guard against any remaining collision
            result[label] = d["idx"]

    return result


def get_best_sample_rate(device_index: int) -> int:
    """
    Try opening the device at each rate in PREFERRED_RATES and return the
    first one that succeeds.  Falls back to the device's default rate.
    """
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
        # fallback: use whatever the device reports as default
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
    frames: list[bytes],
    src_rate: int,
    dst_rate: int = WHISPER_RATE,
) -> np.ndarray:
    """
    Concatenate int16 PCM frames → float32 [-1, 1].
    Resamples src_rate → dst_rate only if they differ.
    Returns 1-D float32 array ready for the Whisper pipeline.
    """
    import librosa
    raw   = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if src_rate != dst_rate:
        audio = librosa.resample(audio, orig_sr=src_rate, target_sr=dst_rate)
    return audio


def pcm_to_wav_bytes(frames: list[bytes], rate: int) -> bytes:
    """Encode raw PCM frame list → in-memory WAV bytes at *rate*."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


# ── VAD Recorder ─────────────────────────────────────────────────────────────

class VADRecorder:
    """
    Continuous mic recorder with energy-based Voice Activity Detection.

    On start() the recorder probes the device for its best supported sample
    rate — so the WAV is always written at the correct rate and plays back
    at natural speed.

    The background thread classifies every ~23 ms frame as speech or silence.
    When it observes `min_silence_ms` of continuous silence after at least
    `min_chunk_ms` of speech, it seals the segment into a ready queue.

    Main thread calls pop_ready_chunk() — returns next sealed segment or None.

    Parameters
    ──────────
    silence_threshold  float  RMS below this = silence  (default 0.01)
    min_silence_ms     int    Pause duration to trigger a split  (default 600)
    min_chunk_ms       int    Minimum utterance length to bother transcribing (default 800)
    max_chunk_ms       int    Hard ceiling — split even without a pause (default 30000)
    """

    def __init__(
        self,
        device_index:      int,
        gain:              float = 3.0,
        silence_threshold: float = 0.01,
        min_silence_ms:    int   = 600,
        min_chunk_ms:      int   = 800,
        max_chunk_ms:      int   = 30_000,
    ):
        self.device_index      = device_index
        self.gain              = gain
        self.silence_threshold = silence_threshold
        self.min_silence_ms    = min_silence_ms
        self.min_chunk_ms      = min_chunk_ms
        self.max_chunk_ms      = max_chunk_ms

        # detected at start()
        self._record_rate: int = WHISPER_RATE

        self._lock         = threading.Lock()
        self._all_frames:  list[bytes] = []
        self._ready_queue: queue.Queue[list[bytes]] = queue.Queue()
        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None

        self._current_rms:    float = 0.0
        self._is_speech:      bool  = False
        self._chunks_emitted: int   = 0
        self._open_error:     str | None = None

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def record_rate(self) -> int:
        return self._record_rate

    def start(self):
        # Detect actual supported rate BEFORE opening the stream
        self._record_rate    = get_best_sample_rate(self.device_index)
        self._all_frames     = []
        self._current_rms    = 0.0
        self._is_speech      = False
        self._chunks_emitted = 0
        self._stop_event.clear()
        while not self._ready_queue.empty():
            try:
                self._ready_queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[bytes]:
        """Stop mic thread and return all accumulated frames."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=4)
        return self.all_frames

    def pop_ready_chunk(self) -> list[bytes] | None:
        """Non-blocking. Returns next sealed speech segment, or None."""
        try:
            return self._ready_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def all_frames(self) -> list[bytes]:
        with self._lock:
            return list(self._all_frames)

    def duration_seconds(self) -> float:
        with self._lock:
            return len(self._all_frames) * FRAMES_PER_BUFFER / self._record_rate

    @property
    def current_rms(self) -> float:
        return self._current_rms

    @property
    def is_speech(self) -> bool:
        return self._is_speech

    @property
    def open_error(self) -> str | None:
        """Set if the mic stream failed to open (e.g. device disconnected)."""
        return self._open_error

    @property
    def chunks_emitted(self) -> int:
        return self._chunks_emitted

    # ── internal ──────────────────────────────────────────────────────────────

    def _seal_chunk(self, speech_frames: list[bytes], silence_tail: list[bytes]):
        total_ms = (len(speech_frames) + len(silence_tail)) * FRAMES_PER_BUFFER / self._record_rate * 1000
        if total_ms >= self.min_chunk_ms:
            self._ready_queue.put(speech_frames + silence_tail)
            self._chunks_emitted += 1

    def _capture_loop(self):
        rate = self._record_rate
        ms_per_frame = FRAMES_PER_BUFFER / rate * 1000.0

        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=FRAMES_PER_BUFFER,
            )
        except OSError as e:
            # Device not available (e.g. BT headset disconnected mid-session,
            # Windows error -9999, etc.)  — store the error so the UI can show it
            self._open_error = str(e)
            p.terminate()
            return
        except Exception as e:
            self._open_error = str(e)
            p.terminate()
            return

        self._open_error = None

        in_speech      = False
        speech_frames: list[bytes] = []
        silence_frames: list[bytes] = []
        silence_ms     = 0.0
        speech_ms      = 0.0

        max_pre_frames    = int(500  / ms_per_frame)
        lead_in_frames    = int(200  / ms_per_frame)
        min_silence_frames = self.min_silence_ms / ms_per_frame
        max_speech_ms      = self.max_chunk_ms

        try:
            while not self._stop_event.is_set():
                raw     = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                boosted = amplify(raw, self.gain)

                with self._lock:
                    self._all_frames.append(boosted)

                rms              = _rms(boosted)
                self._current_rms = rms
                is_speech_frame  = rms > self.silence_threshold

                if is_speech_frame:
                    if not in_speech:
                        # SILENCE → SPEECH: prepend lead-in
                        speech_frames  = silence_frames[-lead_in_frames:]
                        silence_frames = []
                        in_speech      = True
                    speech_frames.append(boosted)
                    speech_ms  += ms_per_frame
                    silence_ms  = 0.0
                    self._is_speech = True

                    # hard ceiling
                    if speech_ms >= max_speech_ms:
                        self._seal_chunk(speech_frames, [])
                        speech_frames  = []
                        silence_frames = []
                        speech_ms      = 0.0
                        silence_ms     = 0.0
                        in_speech      = False
                        self._is_speech = False
                else:
                    if in_speech:
                        silence_frames.append(boosted)
                        silence_ms += ms_per_frame
                        if silence_ms >= self.min_silence_ms:
                            self._seal_chunk(speech_frames, silence_frames)
                            speech_frames  = []
                            silence_frames = []
                            speech_ms      = 0.0
                            silence_ms     = 0.0
                            in_speech      = False
                            self._is_speech = False
                    else:
                        # rolling pre-speech buffer
                        silence_frames.append(boosted)
                        if len(silence_frames) > max_pre_frames:
                            silence_frames.pop(0)
                        self._is_speech = False
        finally:
            if speech_frames:
                self._seal_chunk(speech_frames, silence_frames)
            stream.stop_stream()
            stream.close()
            p.terminate()