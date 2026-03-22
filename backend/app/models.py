"""
models.py
─────────
Whisper model loading and transcription logic.
Models are LRU-cached so each model loads only once.
"""

from functools import lru_cache
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

MODELS: dict[str, str] = {
    "Whisper Medium — Quran fine-tune":         "shahabazkc10/whisper-medium-ar-quran-mix-norm",
    "Whisper Small — Quran fine-tune":          "shahabazkc10/whisper-small-ar-quran-mix-norm",
    "Whisper Large-v3 — OpenAI baseline":       "openai/whisper-large-v3",
    "Whisper Large-v3-Turbo — Quran fine-tune": "shahabazkc10/whisper-large-v3-turbo-ar-quran-mix-norm",
}


@lru_cache(maxsize=4)
def load_model(model_id: str):
    """Load and cache a Whisper pipeline by HuggingFace model ID."""
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    processor   = AutoProcessor.from_pretrained(model_id)
    model       = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
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


def transcribe_chunk(pipe, audio_np: np.ndarray) -> str:
    """Run Whisper ASR on a float32 16kHz mono numpy array."""
    result = pipe(audio_np, generate_kwargs={"language": "arabic", "task": "transcribe"})
    return result["text"].strip()
