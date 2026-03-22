"""
audio_utils.py
──────────────
Reusable audio recording helpers built on PyAudio.

Key design
──────────
• Records at the mic's native high-quality rate (44100 Hz by default).
• Gain is applied to every captured frame so the stored PCM is already loud.
• ContinuousRecorder runs the mic in a background thread — the main thread
  never blocks waiting for audio.  It can read a snapshot of the last N
  seconds at any time without interrupting the stream.

Public API
──────────
get_input_devices()        → dict[label, device_index]
amplify(raw_bytes, gain)   → bytes
pcm_to_wav_bytes(frames, rate) → bytes   (in-memory WAV at *rate*)
ContinuousRecorder         — context-manager / manual start-stop class
"""

import io
import wave
import threading
import collections
import numpy as np
import pyaudio

# ── Constants ─────────────────────────────────────────────────────────────────
FORMAT            = pyaudio.paInt16
CHANNELS          = 1
RECORD_RATE       = 44100   # native capture rate — high quality
WHISPER_RATE      = 16000   # what Whisper expects
FRAMES_PER_BUFFER = 4096    # ~93 ms per callback at 44100 Hz


# ── Device discovery ──────────────────────────────────────────────────────────

def get_input_devices() -> dict:
    """Return {label: device_index} for every device that has input channels."""
    p = pyaudio.PyAudio()
    devices = {}
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            label = f"{info['name']}  (idx {i})"
            devices[label] = i
    p.terminate()
    return devices


# ── Signal helpers ────────────────────────────────────────────────────────────

def amplify(raw_bytes: bytes, gain: float) -> bytes:
    """Scale int16 PCM by *gain*, hard-clipping to int16 range."""
    if gain == 1.0:
        return raw_bytes
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
    return samples.tobytes()


def pcm_frames_to_float32(frames: list[bytes], src_rate: int,
                            dst_rate: int = WHISPER_RATE) -> np.ndarray:
    """
    Concatenate int16 PCM frames, convert to float32 [-1, 1],
    and resample from src_rate → dst_rate using librosa.
    Returns a 1-D float32 numpy array ready for the Whisper pipeline.
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


# ── Continuous threaded recorder ──────────────────────────────────────────────

class ContinuousRecorder:
    """
    Keeps the microphone open in a background daemon thread.
    The main thread can call:
      • snapshot_frames(last_n_seconds)  → list[bytes]  — last N seconds of PCM
      • all_frames                        → list[bytes]  — every frame since start
      • stop()                            — stop the mic thread cleanly

    Usage
    -----
        rec = ContinuousRecorder(device_index=0, gain=3.0)
        rec.start()
        ...
        chunk = rec.snapshot_frames(6)   # last 6 s, non-blocking
        ...
        rec.stop()
        wav = pcm_to_wav_bytes(rec.all_frames, RECORD_RATE)
    """

    def __init__(
        self,
        device_index: int,
        gain: float = 3.0,
        rate: int = RECORD_RATE,
        frames_per_buffer: int = FRAMES_PER_BUFFER,
    ):
        self.device_index     = device_index
        self.gain             = gain
        self.rate             = rate
        self.frames_per_buffer = frames_per_buffer

        self._lock        = threading.Lock()
        self._all_frames: list[bytes] = []
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

    # ── public ────────────────────────────────────────────────────────────────

    def start(self):
        """Open mic and begin capturing in background thread."""
        self._all_frames = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def all_frames(self) -> list[bytes]:
        with self._lock:
            return list(self._all_frames)

    def snapshot_frames(self, last_n_seconds: float) -> list[bytes]:
        """
        Return a copy of the frames captured in the last *last_n_seconds*.
        Thread-safe; does NOT pause the recording.
        """
        frames_needed = int(last_n_seconds * self.rate / self.frames_per_buffer)
        with self._lock:
            return list(self._all_frames[-frames_needed:]) if frames_needed else []

    def duration_seconds(self) -> float:
        with self._lock:
            return len(self._all_frames) * self.frames_per_buffer / self.rate

    # ── internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frames_per_buffer,
        )
        try:
            while not self._stop_event.is_set():
                raw = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                boosted = amplify(raw, self.gain)
                with self._lock:
                    self._all_frames.append(boosted)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()