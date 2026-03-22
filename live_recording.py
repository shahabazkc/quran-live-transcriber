import streamlit as st
import torch
import numpy as np
import av
import threading
import librosa
import time

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "shahabazkc10/whisper-large-ar-quran-mix-norm"

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    return processor, model, device


processor, model, device = load_model()

# =========================
# TRANSCRIBE
# =========================
def transcribe(audio_np):
    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device).to(model.dtype) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            max_length=64,     # ⚡ faster
            num_beams=1,        # ⚡ fastest
            language="ar",
            task="transcribe"
        )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    print("TEXT:", text)
    return text


# =========================
# AUDIO PROCESSOR
# =========================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []
        self.lock = threading.Lock()

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()

        # stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        audio = audio.astype(np.float32)

        # normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        with self.lock:
            self.buffer.extend(audio.tolist())

        return frame


# =========================
# SESSION STATE
# =========================
if "full_text" not in st.session_state:
    st.session_state.full_text = ""


# =========================
# UI
# =========================
st.title("Qur'an Hifz Assistant (LIVE OPTIMIZED)")

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# =========================
# MAIN PROCESSING
# =========================
if ctx.audio_processor:
    with ctx.audio_processor.lock:
        buffer_len = len(ctx.audio_processor.buffer)

    print("BUFFER LEN:", buffer_len)

    # 🔥 CONFIG
    INPUT_SR = 48000
    TARGET_SR = 16000
    CHUNK_SEC = 6
    OVERLAP_SEC = 2

    CHUNK_SIZE = INPUT_SR * CHUNK_SEC
    OVERLAP = INPUT_SR * OVERLAP_SEC

    if buffer_len > CHUNK_SIZE:
        with ctx.audio_processor.lock:
            chunk = ctx.audio_processor.buffer[:CHUNK_SIZE]

            # 🔥 sliding window
            ctx.audio_processor.buffer = ctx.audio_processor.buffer[CHUNK_SIZE - OVERLAP:]

        chunk = np.array(chunk)

        print("PROCESSING CHUNK:", len(chunk))

        # 🔥 resample
        chunk = librosa.resample(chunk, orig_sr=INPUT_SR, target_sr=TARGET_SR)

        # 🔥 normalize again
        if np.max(np.abs(chunk)) > 0:
            chunk = chunk / np.max(np.abs(chunk))

        # 🔥 silence filter
        if np.max(np.abs(chunk)) < 0.01:
            print("Skipping silence")
        else:
            text = transcribe(chunk)

            if text.strip():
                st.session_state.full_text += " " + text

        st.rerun()

# =========================
# DISPLAY
# =========================
st.subheader("Live Transcription:")
st.write(st.session_state.full_text)

if st.button("Clear"):
    st.session_state.full_text = ""


# =========================
# KEEP LOOP RUNNING
# =========================
time.sleep(0.5)
st.rerun()