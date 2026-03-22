import streamlit as st
import pyaudio
import wave
import io
import numpy as np
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Audio Recorder", page_icon="🎙️", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #f0ece4;
}
.stApp { background: #0d0d0d; }

h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -1px;
    color: #f0ece4;
}
.subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: -10px;
    margin-bottom: 32px;
}
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 24px;
}
.status-idle      { background:#1e1e1e; color:#666; border:1px solid #333; }
.status-recording { background:#2a0a0a; color:#ff4d4d; border:1px solid #ff4d4d;
                    animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

div.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 10px 28px !important;
    transition: all .15s ease !important;
}
div.stButton > button[kind="primary"] {
    background: #f0ece4 !important; color: #0d0d0d !important; border: none !important;
}
div.stButton > button[kind="primary"]:hover { background: #ff4d4d !important; color: #fff !important; }
div.stButton > button[kind="secondary"] {
    background: transparent !important; color: #f0ece4 !important; border: 1px solid #444 !important;
}
div.stButton > button[kind="secondary"]:hover { border-color: #f0ece4 !important; }

.recordings-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; color: #666; letter-spacing: 3px; text-transform: uppercase;
    border-bottom: 1px solid #222; padding-bottom: 8px; margin-bottom: 20px;
}
.rec-card {
    background: #161616; border: 1px solid #222; border-radius: 6px;
    padding: 16px 20px; margin-bottom: 8px;
}
.rec-meta { font-family:'Space Mono',monospace; font-size:0.68rem; color:#666; margin-bottom:4px; }
.rec-name { font-weight:700; font-size:0.95rem; margin-bottom:2px; }

.sidebar-label {
    font-family:'Space Mono',monospace; font-size:0.68rem; color:#666;
    letter-spacing:1px; text-transform:uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FORMAT   = pyaudio.paInt16
CHANNELS = 1

# ── Helper: list input devices ────────────────────────────────────────────────
@st.cache_data
def get_input_devices():
    p = pyaudio.PyAudio()
    devices = {}
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            label = f"{info['name']}  (idx {i})"
            devices[label] = i
    p.terminate()
    return devices

# ── Helper: amplify raw PCM bytes ─────────────────────────────────────────────
def amplify(raw_bytes: bytes, gain: float) -> bytes:
    """Scale int16 PCM samples by `gain`, clipping to int16 range."""
    if gain == 1.0:
        return raw_bytes
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
    return samples.tobytes()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<p class="sidebar-label">🎚 Input Device</p>', unsafe_allow_html=True)

devices = get_input_devices()
if not devices:
    st.sidebar.error("No input devices found.")
    st.stop()

device_label = st.sidebar.selectbox("Microphone", list(devices.keys()))
device_index = devices[device_label]

st.sidebar.markdown('<p class="sidebar-label" style="margin-top:18px">⚡ Gain / Amplification</p>', unsafe_allow_html=True)
gain = st.sidebar.slider(
    "Input gain  (1× = original, 4× = loud)",
    min_value=1.0, max_value=8.0, value=3.0, step=0.5,
    help="Boost the recorded volume. Use 2–4× if audio sounds too quiet."
)

st.sidebar.markdown('<p class="sidebar-label" style="margin-top:18px">⚙ Audio Parameters</p>', unsafe_allow_html=True)
FRAMES_PER_BUFFER = int(st.sidebar.text_input("Frames per buffer", 3200))
RATE              = int(st.sidebar.text_input("Sample rate (Hz)",   16000))

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("recording", False), ("frames", []), ("recordings", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎙️ Audio Recorder")
st.markdown('<p class="subtitle">Live capture · Gain boost · Playback · Download</p>',
            unsafe_allow_html=True)

# Status pill
if st.session_state.recording:
    st.markdown('<span class="status-pill status-recording">● Recording…</span>',
                unsafe_allow_html=True)
else:
    st.markdown('<span class="status-pill status-idle">○ Idle</span>',
                unsafe_allow_html=True)

# Show active mic + gain
st.markdown(
    f'<p style="font-family:Space Mono,monospace;font-size:.7rem;color:#555;margin-bottom:20px;">'
    f'MIC → {device_label.split("  (")[0]} &nbsp;|&nbsp; GAIN → {gain}×</p>',
    unsafe_allow_html=True,
)

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    start_clicked = st.button("▶ Start", type="primary",  use_container_width=True,
                               disabled=st.session_state.recording)
with col2:
    stop_clicked  = st.button("■ Stop",  type="secondary", use_container_width=True,
                               disabled=not st.session_state.recording)

# ── Start ─────────────────────────────────────────────────────────────────────
if start_clicked:
    st.session_state.recording = True
    st.session_state.frames    = []
    st.rerun()

# ── Capture loop ──────────────────────────────────────────────────────────────
if st.session_state.recording and not stop_clicked:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,      # ← selected mic
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    info_ph = st.empty()
    CHUNK_ITERATIONS = 10   # ~2 s per rerun cycle
    for _ in range(CHUNK_ITERATIONS):
        raw = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        st.session_state.frames.append(amplify(raw, gain))   # ← gain applied

    stream.stop_stream()
    stream.close()
    p.terminate()

    secs = round(len(st.session_state.frames) * FRAMES_PER_BUFFER / RATE, 1)
    info_ph.markdown(
        f'<p style="font-family:Space Mono,monospace;font-size:.72rem;color:#555;">'
        f'{len(st.session_state.frames)} chunks · {secs} s recorded so far…</p>',
        unsafe_allow_html=True,
    )
    st.rerun()

# ── Stop & save ───────────────────────────────────────────────────────────────
if stop_clicked and st.session_state.frames:
    st.session_state.recording = False

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(st.session_state.frames))
    wav_bytes = buf.getvalue()

    duration_s = round(len(st.session_state.frames) * FRAMES_PER_BUFFER / RATE, 1)
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec_name   = f"Recording {len(st.session_state.recordings) + 1}"

    st.session_state.recordings.insert(0, {
        "name":       rec_name,
        "timestamp":  timestamp,
        "wav_bytes":  wav_bytes,
        "duration_s": duration_s,
        "device":     device_label.split("  (")[0],
        "gain":       gain,
    })
    st.session_state.frames = []
    st.rerun()

elif stop_clicked:
    st.session_state.recording = False
    st.warning("No audio captured — try recording for a moment first.")
    st.rerun()

# ── Recordings list ───────────────────────────────────────────────────────────
if st.session_state.recordings:
    st.markdown('<p class="recordings-header">Recorded Audio</p>', unsafe_allow_html=True)

    for i, rec in enumerate(st.session_state.recordings):
        st.markdown(
            f'<div class="rec-card">'
            f'<div class="rec-name">{rec["name"]}</div>'
            f'<div class="rec-meta">'
            f'{rec["timestamp"]} &nbsp;·&nbsp; {rec["duration_s"]} s'
            f' &nbsp;·&nbsp; {rec["device"]}'
            f' &nbsp;·&nbsp; gain {rec["gain"]}×'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.audio(rec["wav_bytes"], format="audio/wav")
        st.download_button(
            label="⬇ Download WAV",
            data=rec["wav_bytes"],
            file_name=f"{rec['name'].replace(' ', '_')}.wav",
            mime="audio/wav",
            key=f"dl_{i}",
        )
        st.markdown("---")