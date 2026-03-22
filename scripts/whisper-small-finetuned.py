import streamlit as st
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from download_utils import show_download_button

# =========================
# MODEL CONFIGURATION
# =========================
MODEL_NAME = "shahabazkc10/whisper-small-ar-quran-mix-norm"
MODEL_DISPLAY_NAME = "Whisper Small (Finetuned-Quran)"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource(show_spinner="Loading Model...")
def load_turbo_model():
    # This is the specific Quran-tuned Small model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

    # We use the same pipeline settings as the Large-v3 for a fair test
    turbo_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        device=device
    )
    
    return turbo_pipe

turbo_pipe = load_turbo_model()

# =========================
# UI
# =========================
st.title(f"🎤 {MODEL_DISPLAY_NAME}")
st.info(f"**Model:** {MODEL_NAME}")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe"):
        with st.status("Processing...") as status:
            st.write("Reading audio...")
            audio_data, _ = librosa.load(uploaded_file, sr=16000)
            
            st.write("Transcribing...")
            result = turbo_pipe(
                audio_data,
                generate_kwargs={"language": "arabic", "task": "transcribe"}
            )
            
            # Store in session state to persist across reruns
            st.session_state.transcription_result = result["text"]
            st.session_state.uploaded_filename = uploaded_file.name
            status.update(label="Done!", state="complete")
    
    # Display transcription from session state (no reset on reload)
    if "transcription_result" in st.session_state:
        st.subheader("Transcription Output:")
        st.write(st.session_state.transcription_result)
        
        # Download button - won't cause page reset
        show_download_button(
            st.session_state.transcription_result,
            MODEL_DISPLAY_NAME,
            st.session_state.uploaded_filename
        )

### Logic for side-by-side comparison
st.divider()
st.markdown("""
**What to look for in the comparison:**
1. **Speed:** Turbo should be roughly 3x to 5x faster.
2. **Accuracy:** Check if Turbo misses any 'Harakaat' or small connecting words (like 'و' or 'ل').
3. **Hallucination:** See if Turbo repeats phrases or stops early compared to your Large-v3 output.
""")