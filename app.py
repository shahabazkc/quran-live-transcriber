import streamlit as st
import torch
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# =========================
# MODEL LOAD
# =========================
device = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "shahabazkc10/whisper-medium-ar-quran-mix-norm"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to(device)

# =========================
# SURAH RAHMAN TEXT (shortened sample for MVP)
# =========================
EXPECTED_TEXT = """
اَلرَّحْمٰنُۙ عَلَّمَ الْقُرْاٰنَۙ خَلَقَ الْاِنْسَانَۙ عَلَّمَهُ الْبَیَانَ اَلشَّمْسُ وَ الْقَمَرُ بِحُسْبَانٍۙ وَّ النَّجْمُ وَ الشَّجَرُ یَسْجُدٰنِ وَ السَّمَآءَ رَفَعَهَا وَ وَضَعَ الْمِیْزَانَۙ اَلَّا تَطْغَوْا فِی الْمِیْزَانِ وَ اَقِیْمُوا الْوَزْنَ بِالْقِسْطِ وَ لَا تُخْسِرُوا الْمِیْزَانَ وَ الْاَرْضَ وَ ضَعَهَا لِلْاَنَامِۙ فِیْهَا فَاكِهَةٌ وَ النَّخْلُ ذَاتُ الْاَكْمَامِۖ وَ الْحَبُّ ذُو الْعَصْفِ وَ الرَّیْحَانُۚ فَبِاَیِّ اٰلَآءِ رَبِّكُمَا تُكَذِّبٰنِ خَلَقَ الْاِنْسَانَ مِنْ صَلْصَالٍ كَالْفَخَّارِۙ وَ خَلَقَ الْجَآنَّ مِنْ مَّارِجٍ مِّنْ نَّارٍۚ فَبِاَیِّ اٰلَآءِ رَبِّكُمَا تُكَذِّبٰنِ رَبُّ الْمَشْرِقَیْنِ وَ رَبُّ الْمَغْرِبَیْنِۚ فَبِاَیِّ اٰلَآءِ رَبِّكُمَا تُكَذِّبٰنِ مَرَجَ الْبَحْرَیْنِ یَلْتَقِیٰنِۙ بَیْنَهُمَا بَرْزَخٌ لَّا یَبْغِیٰنِۚ فَبِاَیِّ اٰلَآءِ رَبِّكُمَا تُكَذِّبٰنِ یَخْرُجُ مِنْهُمَا اللُّؤْلُؤُ وَ الْمَرْجَانُۚ فَبِاَیِّ اٰلَآءِ رَبِّكُمَا تُكَذِّبٰنِ وَ لَهُ الْجَوَارِ الْمُنْشَئَاتُ فِی الْبَحْرِ كَالْاَعْلَامِۚ فَبِاَیِّ اٰلَآءِ رَبِّكُمَا تُكَذِّبٰنِ
"""

# =========================
# AUDIO RECORD FUNCTION
# =========================
def record_audio(duration=5, fs=16000):
    st.info("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten()

# =========================
# TRANSCRIBE FUNCTION
# =========================
def transcribe(audio):
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device).to(model.dtype) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            max_length=256,
            num_beams=3,
            temperature=0.0
        )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return text

# =========================
# SIMPLE MATCH FUNCTION
# =========================
def match_text(pred, expected):
    pred_words = pred.split()
    exp_words = expected.split()

    result = []
    for word in exp_words:
        if word in pred_words:
            result.append(f"🟢 {word}")
        else:
            result.append(f"🔴 {word}")

    return " ".join(result)

# =========================
# UI
# =========================
st.title("Qur'an Hifz Assistant (MVP)")

duration = st.slider("Recording Duration (seconds)", 3, 15, 5)

if st.button("🎤 Record and Transcribe"):
    audio = record_audio(duration)

    st.audio(audio, sample_rate=16000)

    text = transcribe(audio)

    st.subheader("Your Recitation:")
    st.write(text)

    st.subheader("Expected (Surah Rahman):")
    st.write(EXPECTED_TEXT)

    st.subheader("Matching Result:")
    st.write(match_text(text, EXPECTED_TEXT))