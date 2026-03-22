"""
quran_transcriber.py
─────────────────────
Live Quran audio → Arabic transcription.

Architecture
────────────
• ContinuousRecorder (from audio_utils) keeps the mic open in a daemon thread
  at 44100 Hz — audio NEVER stops or restarts between chunks.
• Every `chunk_secs` seconds a Streamlit rerun fires.  The main thread calls
  recorder.snapshot_frames(chunk_secs) to grab the last N seconds of PCM,
  resamples to 16 kHz, and feeds it to the Whisper pipeline — all while the
  mic thread keeps recording without interruption.
• On Stop the full accumulated PCM is encoded to a WAV at 44100 Hz for
  download / playback.

Usage
─────
    streamlit run quran_transcriber.py
    (audio_utils.py must be in the same directory)
"""

import time
import numpy as np
import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from datetime import datetime

from audio_utils import (
    ContinuousRecorder,
    get_input_devices,
    pcm_to_wav_bytes,
    pcm_frames_to_float32,
    RECORD_RATE,
    WHISPER_RATE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Quran Transcriber", page_icon="📖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#0a0a0f; color:#e8e4db; }
.stApp { background:#0a0a0f; }

.app-title { font-family:'Syne',sans-serif; font-weight:800; font-size:2.2rem;
             letter-spacing:-1px; color:#e8e4db; margin-bottom:2px; }
.app-sub   { font-family:'Space Mono',monospace; font-size:0.7rem; color:#555;
             letter-spacing:2px; text-transform:uppercase; margin-bottom:28px; }

.pill { display:inline-block; padding:4px 14px; border-radius:999px;
        font-family:'Space Mono',monospace; font-size:0.68rem;
        letter-spacing:1px; text-transform:uppercase; margin-bottom:18px; }
.pill-idle      { background:#1a1a22; color:#555;    border:1px solid #2a2a35; }
.pill-recording { background:#1a0a0a; color:#ff5555; border:1px solid #ff5555;
                  animation:blink 1.2s ease-in-out infinite; }
.pill-done      { background:#0a1a0a; color:#55cc77; border:1px solid #55cc77; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.4} }

.sec-label { font-family:'Space Mono',monospace; font-size:0.65rem; color:#444;
             letter-spacing:3px; text-transform:uppercase;
             border-bottom:1px solid #1e1e28; padding-bottom:6px; margin-bottom:14px; }

.arabic-live { font-family:'Amiri',serif; font-size:1.65rem; line-height:2.1;
               direction:rtl; text-align:right; color:#f0d080;
               background:#121218; border:1px solid #2a2a35; border-radius:8px;
               padding:18px 22px; min-height:80px; margin-bottom:8px; }
.arabic-full { font-family:'Amiri',serif; font-size:1.4rem; line-height:2.1;
               direction:rtl; text-align:right; color:#c8c0a8;
               background:#0e0e16; border:1px solid #1e1e28; border-radius:8px;
               padding:18px 22px; min-height:120px; }
.chunk-badge { font-family:'Space Mono',monospace; font-size:0.6rem; color:#333;
               text-align:left; direction:ltr; margin-top:6px; }

.info-box { background:#131318; border:1px solid #2a2a35; border-radius:6px;
            padding:12px 16px; margin-bottom:14px;
            font-family:'Space Mono',monospace; font-size:0.68rem; color:#666; line-height:1.7; }

.rec-card { background:#131318; border:1px solid #2a2a35; border-radius:6px;
            padding:14px 18px; margin-bottom:10px; }
.rec-name { font-weight:700; font-size:0.9rem; margin-bottom:3px; }
.rec-meta { font-family:'Space Mono',monospace; font-size:0.65rem; color:#555; }

div.stButton > button {
    font-family:'Space Mono',monospace !important; font-size:0.74rem !important;
    letter-spacing:1.5px !important; text-transform:uppercase !important;
    border-radius:4px !important; padding:10px 22px !important;
    transition:all .15s ease !important;
}
div.stButton > button[kind="primary"]  { background:#e8e4db !important; color:#0a0a0f !important; border:none !important; }
div.stButton > button[kind="primary"]:hover  { background:#f0d080 !important; }
div.stButton > button[kind="secondary"]{ background:transparent !important; color:#e8e4db !important; border:1px solid #333 !important; }
div.stButton > button[kind="secondary"]:hover { border-color:#e8e4db !important; }
div.stButton > button:disabled { opacity:.35 !important; cursor:not-allowed !important; }

div[data-baseweb="select"] > div { background:#131318 !important; border-color:#2a2a35 !important; }
div[data-baseweb="select"] span  { color:#e8e4db !important; }
.stSlider label, .stSelectbox label {
    font-family:'Space Mono',monospace !important; font-size:0.7rem !important; color:#666 !important; }

section[data-testid="stSidebar"] { background:#0e0e16 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════
MODELS = {
    "Whisper Medium — Quran fine-tune":         "shahabazkc10/whisper-medium-ar-quran-mix-norm",
    "Whisper Small — Quran fine-tune":          "shahabazkc10/whisper-small-ar-quran-mix-norm",
    "Whisper Large-v3 — OpenAI baseline":       "openai/whisper-large-v3",
    "Whisper Large-v3-Turbo — Quran fine-tune": "shahabazkc10/whisper-large-v3-turbo-ar-quran-mix-norm",
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model — this may take a minute…")
def load_model(model_name: str):
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    processor   = AutoProcessor.from_pretrained(model_name)
    model       = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        device=device,
    )
    return pipe


def transcribe_audio(pipe, audio_np: np.ndarray) -> str:
    result = pipe(
        audio_np,
        generate_kwargs={"language": "arabic", "task": "transcribe"},
    )
    return result["text"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
defaults = {
    "recording":        False,
    "recorder":         None,   # ContinuousRecorder instance
    "chunks":           [],     # list of transcribed strings
    "latest_chunk_txt": "",
    "final_wav":        None,
    "final_duration":   0.0,
    "session_ts":       None,
    "last_chunk_time":  0.0,    # time.monotonic() of last transcription trigger
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
sb = st.sidebar
sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.65rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">Model</p>', unsafe_allow_html=True)
model_label = sb.selectbox("ASR Model", list(MODELS.keys()), label_visibility="collapsed")
model_name  = MODELS[model_label]
sb.markdown(f'<div class="info-box">📦 {model_name}</div>', unsafe_allow_html=True)

sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.65rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:16px;margin-bottom:8px">Microphone</p>', unsafe_allow_html=True)
devices = get_input_devices()
if not devices:
    sb.error("No input devices found.")
    st.stop()
device_label = sb.selectbox("Mic", list(devices.keys()), label_visibility="collapsed")
device_index = devices[device_label]

sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.65rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:16px;margin-bottom:4px">Transcription interval (s)</p>', unsafe_allow_html=True)
chunk_secs = sb.slider("Chunk", min_value=3, max_value=10, value=6, step=1,
                        label_visibility="collapsed",
                        help="How many seconds of audio each transcription chunk covers.")
sb.markdown(f'<div class="info-box">Mic records <b style="color:#f0d080">continuously</b>.<br>Transcription runs every <b style="color:#f0d080">{chunk_secs}s</b> in parallel.</div>', unsafe_allow_html=True)

sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.65rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:16px;margin-bottom:4px">Input gain</p>', unsafe_allow_html=True)
gain = sb.slider("Gain", min_value=1.0, max_value=8.0, value=3.0, step=0.5,
                 label_visibility="collapsed",
                 help="Amplify mic input. 2–4× works well for most mics.")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
pipe = load_model(model_name)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="app-title">📖 Quran Live Transcriber</p>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Continuous mic · Chunked ASR · Whisper fine-tuned</p>', unsafe_allow_html=True)

if st.session_state.recording:
    rec = st.session_state.recorder
    dur = rec.duration_seconds() if rec else 0
    st.markdown(f'<span class="pill pill-recording">● Recording — {dur:.0f}s captured</span>', unsafe_allow_html=True)
elif st.session_state.final_wav:
    st.markdown('<span class="pill pill-done">✓ Session complete</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="pill pill-idle">○ Ready</span>', unsafe_allow_html=True)

st.markdown(
    f'<p style="font-family:Space Mono,monospace;font-size:.65rem;color:#444;margin-bottom:18px;">'
    f'MODEL → {model_label} &nbsp;|&nbsp; MIC → {device_label.split("  (")[0]}'
    f' &nbsp;|&nbsp; INTERVAL → {chunk_secs}s &nbsp;|&nbsp; GAIN → {gain}×</p>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2, _ = st.columns([1, 1, 4])
with c1:
    start_clicked = st.button("▶ Start", type="primary",  use_container_width=True,
                               disabled=st.session_state.recording)
with c2:
    stop_clicked  = st.button("■ Stop",  type="secondary", use_container_width=True,
                               disabled=not st.session_state.recording)

# ── Start ─────────────────────────────────────────────────────────────────────
if start_clicked:
    # Clean up old recorder if any
    if st.session_state.recorder:
        try:
            st.session_state.recorder.stop()
        except Exception:
            pass

    rec = ContinuousRecorder(device_index=device_index, gain=gain, rate=RECORD_RATE)
    rec.start()

    st.session_state.recording        = True
    st.session_state.recorder         = rec
    st.session_state.chunks           = []
    st.session_state.latest_chunk_txt = ""
    st.session_state.final_wav        = None
    st.session_state.final_duration   = 0.0
    st.session_state.session_ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.last_chunk_time  = time.monotonic()
    st.rerun()

# ── Stop ──────────────────────────────────────────────────────────────────────
if stop_clicked:
    st.session_state.recording = False
    rec = st.session_state.recorder
    if rec:
        all_frames = rec.all_frames       # grab before stopping
        duration   = rec.duration_seconds()
        rec.stop()
        if all_frames:
            wav = pcm_to_wav_bytes(all_frames, RECORD_RATE)
            st.session_state.final_wav      = wav
            st.session_state.final_duration = round(duration, 1)
        else:
            st.warning("Nothing was recorded.")
    st.rerun()

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<p class="sec-label">Latest chunk</p>', unsafe_allow_html=True)
    live_ph = st.empty()

with right_col:
    st.markdown('<p class="sec-label">Full transcript (all chunks)</p>', unsafe_allow_html=True)
    full_ph = st.empty()


def render_transcripts():
    latest = st.session_state.latest_chunk_txt or "—"
    live_ph.markdown(f'<div class="arabic-live">{latest}</div>', unsafe_allow_html=True)
    if st.session_state.chunks:
        joined = " ".join(st.session_state.chunks)
        count  = len(st.session_state.chunks)
        full_ph.markdown(
            f'<div class="arabic-full">{joined}</div>'
            f'<p class="chunk-badge">{count} chunk(s) transcribed</p>',
            unsafe_allow_html=True,
        )
    else:
        full_ph.markdown(
            '<div class="arabic-full" style="color:#333;">Transcript appears here…</div>',
            unsafe_allow_html=True,
        )


render_transcripts()

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE TRANSCRIPTION LOOP
# The mic thread is ALWAYS running — we just check the wall clock and when
# chunk_secs have elapsed we grab a snapshot and transcribe it.
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.recording:
    rec  = st.session_state.recorder
    now  = time.monotonic()
    elapsed = now - st.session_state.last_chunk_time

    if elapsed >= chunk_secs:
        # ── Transcribe the last chunk_secs of audio ───────────────────────────
        frames = rec.snapshot_frames(chunk_secs)      # non-blocking read
        if frames:
            with st.spinner("Transcribing…"):
                try:
                    audio_np = pcm_frames_to_float32(frames, RECORD_RATE, WHISPER_RATE)
                    text     = transcribe_audio(pipe, audio_np)
                except Exception as e:
                    text = f"[error: {e}]"
            st.session_state.latest_chunk_txt = text
            st.session_state.chunks.append(text)
            st.session_state.last_chunk_time  = time.monotonic()
            st.rerun()
        else:
            # Not enough audio yet — wait a little and rerun
            time.sleep(0.5)
            st.rerun()
    else:
        # Wait out the rest of this chunk interval, then rerun
        remaining = chunk_secs - elapsed
        time.sleep(min(remaining, 1.0))   # sleep max 1 s so UI stays responsive
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RECORDED AUDIO + DOWNLOADS  (shown after Stop)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.final_wav:
    st.markdown("---")
    st.markdown('<p class="sec-label">Recorded audio</p>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="rec-card">'
        f'<div class="rec-name">Session recording</div>'
        f'<div class="rec-meta">'
        f'{st.session_state.session_ts}'
        f' &nbsp;·&nbsp; {st.session_state.final_duration}s'
        f' &nbsp;·&nbsp; {device_label.split("  (")[0]}'
        f' &nbsp;·&nbsp; {RECORD_RATE} Hz &nbsp;·&nbsp; gain {gain}×'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.audio(st.session_state.final_wav, format="audio/wav")

    safe_ts = st.session_state.session_ts.replace(":", "-").replace(" ", "_")
    d1, d2, _ = st.columns([1, 1, 2])
    with d1:
        st.download_button(
            "⬇ Download WAV",
            data=st.session_state.final_wav,
            file_name=f"quran_{safe_ts}.wav",
            mime="audio/wav",
        )
    if st.session_state.chunks:
        full_text = "\n".join(
            f"[Chunk {i+1}] {t}" for i, t in enumerate(st.session_state.chunks)
        )
        with d2:
            st.download_button(
                "⬇ Download Transcript",
                data=full_text.encode("utf-8"),
                file_name=f"transcript_{safe_ts}.txt",
                mime="text/plain",
            )