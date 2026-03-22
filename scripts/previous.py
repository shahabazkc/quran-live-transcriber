import streamlit as st
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# =========================
# LOAD MODEL
# =========================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"My device is {device}")

    MODEL_NAME = "shahabazkc10/whisper-large-v3-turbo-ar-quran-mix-norm"
    print(f"Loading model: {MODEL_NAME}...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        # local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model.eval()
    return processor, model, device


processor, model, device = load_model()

# =========================
# TRANSCRIBE FUNCTION
# =========================
def transcribe_chunk(chunk):
    inputs = processor(
        chunk,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device).to(model.dtype) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            max_length=128,
            num_beams=2
        )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return text


# =========================
# UI
# =========================
st.title("Qur'an Hifz Assistant (Chunk Mode)")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    audio, sr = librosa.load(uploaded_file, sr=16000)

    chunk_duration = 5  # seconds
    chunk_size = chunk_duration * sr

    full_text = ""
    progress = st.progress(0)
    output_box = st.empty()

    st.info("Transcribing in chunks...")

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]

        if len(chunk) < 1000:
            continue

        text = transcribe_chunk(chunk)

        full_text += text + " "

        # show live updates
        output_box.write(full_text)

        progress.progress(min((i + chunk_size) / len(audio), 1.0))

    st.success("Done!")

    st.subheader("Final Transcription:")
    st.write(full_text)