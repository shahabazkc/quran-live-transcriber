"""
quran_transcriber.py
─────────────────────
Live Quran audio → Arabic transcription driven by Voice Activity Detection.

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
    VADRecorder,
    get_input_devices,
    get_best_sample_rate,
    pcm_to_wav_bytes,
    pcm_frames_to_float32,
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
             letter-spacing:2px; text-transform:uppercase; margin-bottom:24px; }

.pill { display:inline-block; padding:4px 14px; border-radius:999px;
        font-family:'Space Mono',monospace; font-size:0.68rem;
        letter-spacing:1px; text-transform:uppercase; margin-bottom:14px; }
.pill-idle      { background:#1a1a22; color:#555;    border:1px solid #2a2a35; }
.pill-speech    { background:#0a1a2a; color:#55aaff; border:1px solid #55aaff;
                  animation:blink 0.8s ease-in-out infinite; }
.pill-silence   { background:#1a1a0a; color:#aaaa55; border:1px solid #888833; }
.pill-done      { background:#0a1a0a; color:#55cc77; border:1px solid #55cc77; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.4} }

.level-wrap  { display:flex; align-items:center; gap:10px; margin-bottom:16px; }
.level-label { font-family:'Space Mono',monospace; font-size:0.6rem; color:#444;
               text-transform:uppercase; letter-spacing:1px; min-width:80px; }
.level-bar-bg   { flex:1; height:6px; background:#1e1e28; border-radius:3px; }
.level-bar-fill { height:6px; border-radius:3px; transition:width .1s ease; }

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
               padding:18px 22px; min-height:140px; }
.chunk-badge { font-family:'Space Mono',monospace; font-size:0.6rem; color:#333;
               text-align:left; direction:ltr; margin-top:6px; }

.info-box { background:#131318; border:1px solid #2a2a35; border-radius:6px;
            padding:12px 16px; margin-bottom:12px;
            font-family:'Space Mono',monospace; font-size:0.67rem; color:#666; line-height:1.8; }

.rate-badge { display:inline-block; padding:3px 10px; border-radius:4px;
              background:#0a1a0a; border:1px solid #1a3a1a;
              font-family:'Space Mono',monospace; font-size:0.63rem; color:#55cc77; }

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

.rec-card { background:#131318; border:1px solid #2a2a35; border-radius:6px;
            padding:14px 18px; margin-bottom:10px; }
.rec-name { font-weight:700; font-size:0.9rem; margin-bottom:3px; }
.rec-meta { font-family:'Space Mono',monospace; font-size:0.65rem; color:#555; }

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
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_name: str):
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    processor   = AutoProcessor.from_pretrained(model_name)
    model       = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    ).to(device)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        device=device,
    )


def transcribe_frames(pipe, frames: list[bytes], src_rate: int) -> str:
    audio_np = pcm_frames_to_float32(frames, src_rate, WHISPER_RATE)
    result   = pipe(audio_np, generate_kwargs={"language": "arabic", "task": "transcribe"})
    return result["text"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
defaults = {
    "recording":        False,
    "recorder":         None,
    "chunks":           [],
    "latest_chunk_txt": "",
    "final_wav":        None,
    "final_duration":   0.0,
    "session_ts":       None,
    "detected_rate":    None,   # shown in UI after start
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
sb = st.sidebar

sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.63rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">Model</p>', unsafe_allow_html=True)
model_label = sb.selectbox("Model", list(MODELS.keys()), label_visibility="collapsed")
model_name  = MODELS[model_label]
sb.markdown(f'<div class="info-box">📦 {model_name}</div>', unsafe_allow_html=True)

sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.63rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:14px;margin-bottom:8px">Microphone</p>', unsafe_allow_html=True)
devices = get_input_devices()
if not devices:
    sb.error("No input devices found.")
    st.stop()
device_label = sb.selectbox("Mic", list(devices.keys()), label_visibility="collapsed")
device_index = devices[device_label]

# Show detected sample rate for selected mic
detected_rate = get_best_sample_rate(device_index)
sb.markdown(
    f'<div class="info-box">'
    f'<span class="rate-badge">✓ {detected_rate} Hz</span> &nbsp; detected for this mic<br>'
    f'<span style="color:#333;font-size:0.60rem;">WAV saved at this rate — correct playback guaranteed.</span>'
    f'</div>',
    unsafe_allow_html=True,
)

sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.63rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:14px;margin-bottom:4px">Input gain</p>', unsafe_allow_html=True)
gain = sb.slider("Gain", 1.0, 8.0, 3.0, 0.5, label_visibility="collapsed")

sb.markdown('<p style="font-family:Space Mono,monospace;font-size:.63rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:18px;margin-bottom:6px">🎚 VAD Settings</p>', unsafe_allow_html=True)
sb.markdown('<div class="info-box">Chunks split at natural <b style="color:#f0d080">pauses</b> — no fixed timer.</div>', unsafe_allow_html=True)

silence_threshold = sb.slider(
    "Silence threshold  (RMS)",
    min_value=0.001, max_value=0.10, value=0.010, step=0.001, format="%.3f",
    help="RMS below this = silence. Raise if noise keeps triggering; lower if soft voice is missed.",
)
min_silence_ms = sb.slider(
    "Min pause duration  (ms)",
    min_value=200, max_value=2000, value=600, step=50,
    help="Gap must last this long to count as a chunk boundary.",
)
min_chunk_ms = sb.slider(
    "Min chunk length  (ms)",
    min_value=300, max_value=5000, value=800, step=100,
    help="Chunks shorter than this are discarded.",
)
max_chunk_ms = sb.slider(
    "Max chunk length  (ms)",
    min_value=5000, max_value=60000, value=30000, step=1000, format="%d ms",
    help="Hard ceiling — split even without a pause.",
)

sb.markdown(
    f'<div class="info-box">'
    f'Pause ≥ <b style="color:#f0d080">{min_silence_ms} ms</b> &nbsp;·&nbsp; '
    f'RMS threshold <b style="color:#f0d080">{silence_threshold:.3f}</b><br>'
    f'Min/Max chunk <b style="color:#f0d080">{min_chunk_ms} ms / {max_chunk_ms//1000} s</b>'
    f'</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
pipe = load_model(model_name)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="app-title">📖 Quran Live Transcriber</p>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Pause-triggered VAD · Continuous mic · Auto sample-rate detection</p>', unsafe_allow_html=True)

# Status pill + level meter
rec: VADRecorder | None = st.session_state.recorder

if st.session_state.recording and rec:
    rms       = rec.current_rms
    is_speech = rec.is_speech
    dur       = rec.duration_seconds()
    emitted   = rec.chunks_emitted
    rate_used = rec.record_rate

    if is_speech:
        st.markdown('<span class="pill pill-speech">🎙 Speaking…</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-silence">— Listening for voice…</span>', unsafe_allow_html=True)

    pct       = min(rms / 0.10, 1.0) * 100
    bar_color = "#55aaff" if is_speech else "#555"
    st.markdown(
        f'<div class="level-wrap">'
        f'<span class="level-label">Level</span>'
        f'<div class="level-bar-bg">'
        f'<div class="level-bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div>'
        f'</div>'
        f'<span style="font-family:Space Mono,monospace;font-size:.6rem;color:#444;min-width:120px;">'
        f'{rms:.4f} RMS &nbsp;·&nbsp; {dur:.0f}s &nbsp;·&nbsp; {emitted} chunk(s) &nbsp;·&nbsp; '
        f'<span class="rate-badge">{rate_used} Hz</span></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.final_wav:
    st.markdown('<span class="pill pill-done">✓ Session complete</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="pill pill-idle">○ Ready</span>', unsafe_allow_html=True)

st.markdown(
    f'<p style="font-family:Space Mono,monospace;font-size:.63rem;color:#3a3a4a;margin-bottom:18px;">'
    f'MODEL → {model_label} &nbsp;|&nbsp; MIC → {device_label.split("  (")[0]}'
    f' &nbsp;|&nbsp; GAIN → {gain}× &nbsp;|&nbsp; RATE → <span style="color:#55cc77">{detected_rate} Hz</span></p>',
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
    if st.session_state.recorder:
        try:
            st.session_state.recorder.stop()
        except Exception:
            pass

    new_rec = VADRecorder(
        device_index      = device_index,
        gain              = gain,
        silence_threshold = silence_threshold,
        min_silence_ms    = min_silence_ms,
        min_chunk_ms      = min_chunk_ms,
        max_chunk_ms      = max_chunk_ms,
    )
    new_rec.start()

    st.session_state.recording        = True
    st.session_state.recorder         = new_rec
    st.session_state.chunks           = []
    st.session_state.latest_chunk_txt = ""
    st.session_state.final_wav        = None
    st.session_state.final_duration   = 0.0
    st.session_state.session_ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.detected_rate    = new_rec.record_rate
    st.rerun()

# ── Stop ──────────────────────────────────────────────────────────────────────
if stop_clicked:
    st.session_state.recording = False
    if st.session_state.recorder:
        rec_rate   = st.session_state.recorder.record_rate
        all_frames = st.session_state.recorder.stop()
        duration   = len(all_frames) * 1024 / rec_rate
        if all_frames:
            # WAV written at the ACTUAL capture rate — correct playback speed
            st.session_state.final_wav      = pcm_to_wav_bytes(all_frames, rec_rate)
            st.session_state.final_duration = round(duration, 1)
            st.session_state.detected_rate  = rec_rate
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
            f'<p class="chunk-badge">{count} chunk(s)</p>',
            unsafe_allow_html=True,
        )
    else:
        full_ph.markdown(
            '<div class="arabic-full" style="color:#2a2a3a;">Transcript appears after first pause…</div>',
            unsafe_allow_html=True,
        )


render_transcripts()

# ═══════════════════════════════════════════════════════════════════════════════
# VAD POLL LOOP
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.recording and st.session_state.recorder:
    rec   = st.session_state.recorder

    # Check if the mic failed to open (e.g. BT headset disconnected)
    if rec.open_error:
        st.session_state.recording = False
        st.error(
            f"🎙️ Microphone error: **{rec.open_error}**\n\n"
            f"The selected device is not available. "
            f"Please check that your microphone / headset is connected and try again."
        )
        st.rerun()

    chunk = rec.pop_ready_chunk()

    if chunk:
        with st.spinner("Transcribing…"):
            try:
                # Pass the actual record rate so resample is correct
                text = transcribe_frames(pipe, chunk, rec.record_rate)
            except Exception as e:
                text = f"[error: {e}]"
        st.session_state.latest_chunk_txt = text
        st.session_state.chunks.append(text)
        st.rerun()
    else:
        time.sleep(0.3)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RECORDED AUDIO + DOWNLOADS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.final_wav:
    st.markdown("---")
    st.markdown('<p class="sec-label">Recorded audio</p>', unsafe_allow_html=True)
    rate_used = st.session_state.detected_rate or "?"
    st.markdown(
        f'<div class="rec-card">'
        f'<div class="rec-name">Session recording</div>'
        f'<div class="rec-meta">'
        f'{st.session_state.session_ts}'
        f' &nbsp;·&nbsp; {st.session_state.final_duration}s'
        f' &nbsp;·&nbsp; {device_label.split("  (")[0]}'
        f' &nbsp;·&nbsp; <span class="rate-badge">✓ {rate_used} Hz</span>'
        f' &nbsp;·&nbsp; gain {gain}×'
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
        full_text = "\n".join(f"[Chunk {i+1}] {t}" for i, t in enumerate(st.session_state.chunks))
        with d2:
            st.download_button(
                "⬇ Download Transcript",
                data=full_text.encode("utf-8"),
                file_name=f"transcript_{safe_ts}.txt",
                mime="text/plain",
            )