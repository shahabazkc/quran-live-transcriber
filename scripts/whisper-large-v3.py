import streamlit as st
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig, pipeline
from download_utils import show_download_button

# =========================
# MODEL CONFIGURATION
# =========================
MODEL_NAME = "openai/whisper-large-v3"
MODEL_DISPLAY_NAME = "Whisper Large-v3 (OpenAI)"

# =========================
# LOAD MODEL (EXTREME LOW VRAM)
# =========================
@st.cache_resource(show_spinner="Loading Large-v3 (Optimized for 4GB GPU)...")
def load_model():
    
    # Advanced 4-bit config to save every byte of VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # This splits the model across GPU/RAM if needed
        low_cpu_mem_usage=True
    )

    # We set batch_size=1 to ensure it doesn't overwhelm your 4GB VRAM
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=1, 
    )
    
    return asr_pipeline

# Check if CUDA is available for the UI
device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.write(f"**Model:** {MODEL_NAME}")
st.sidebar.write(f"Running on: **{device_label}**")

# =========================
# PROCESSING LOGIC
# =========================
try:
    asr_pipe = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")

st.title(f"🎤 {MODEL_DISPLAY_NAME}")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Start Transcription"):
        # STEP 1: LOAD AUDIO
        with st.status("Processing...", expanded=True) as status:
            st.write("Reading audio file...")
            audio_data, _ = librosa.load(uploaded_file, sr=16000)
            
            # STEP 2: TRANSCRIBE
            st.write("Transcribing (this may take a minute for Large-v3)...")
            try:
                result = asr_pipe(
                    audio_data,
                    generate_kwargs={"language": "arabic", "task": "transcribe"}
                )
                # Store in session state to persist across reruns
                st.session_state.transcription_result = result["text"]
                st.session_state.uploaded_filename = uploaded_file.name
                st.success("Transcription Complete!")
                status.update(label="Process Finished", state="complete")
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            
            status.update(label="Process Finished", state="complete")
    
    # Display transcription from session state (no reset on reload)
    if "transcription_result" in st.session_state:
        st.subheader("Output:")
        st.write(st.session_state.transcription_result)
        
        # Download button - won't cause page reset
        show_download_button(
            st.session_state.transcription_result,
            MODEL_DISPLAY_NAME,
            st.session_state.uploaded_filename
        )