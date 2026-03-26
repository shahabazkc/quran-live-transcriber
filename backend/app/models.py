"""
models.py
─────────
Whisper model loading and transcription logic.
Models are LRU-cached so each model loads only once.
"""

import gc
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


def _build_pipeline(model_id: str, gpu_index: int | None = None):
    """Build a Whisper pipeline by HuggingFace model ID."""
    if torch.cuda.is_available():
        if gpu_index is None:
            gpu_index = 0
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_index = max(0, min(gpu_index, gpu_count - 1))
            torch.cuda.set_device(gpu_index)
            torch_device = f"cuda:{gpu_index}"
            pipeline_device = gpu_index
        else:
            torch_device = "cpu"
            pipeline_device = -1
    else:
        torch_device = "cpu"
        pipeline_device = -1

    torch_dtype = torch.float16 if torch_device.startswith("cuda") else torch.float32
    processor   = AutoProcessor.from_pretrained(model_id)
    model       = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True,
    ).to(torch_device)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        ignore_warning=True,
        device=pipeline_device,
    )


@lru_cache(maxsize=4)
def load_model(model_id: str):
    """Load and cache a Whisper pipeline by HuggingFace model ID."""
    return _build_pipeline(model_id)


def load_model_fresh(model_id: str, gpu_index: int | None = None):
    """
    Load a non-cached model pipeline.
    Used in recitation mode so switching models can explicitly release context.
    """
    return _build_pipeline(model_id, gpu_index=gpu_index)


def unload_model(pipe) -> None:
    """Best-effort cleanup for a loaded pipeline."""
    if pipe is None:
        return
    try:
        if hasattr(pipe, "model"):
            del pipe.model
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_chunk(pipe, audio_np: np.ndarray) -> str:
    """Run Whisper ASR on a float32 16kHz mono numpy array."""
    result = pipe(audio_np, generate_kwargs={"language": "arabic", "task": "transcribe"})
    return result["text"].strip()


def transcribe_chunk_detailed(
    pipe, audio_np: np.ndarray, language: str = "arabic", use_word_timestamps: bool = False
) -> dict:
    """
    Return text and lightweight word-level chunks.
    Whisper pipelines may not always expose confidence, so it can be None.
    """
    try:
        if use_word_timestamps:
            result = pipe(
                audio_np,
                return_timestamps="word",
                generate_kwargs={"language": language, "task": "transcribe"},
            )
        else:
            result = pipe(
                audio_np,
                generate_kwargs={"language": language, "task": "transcribe"},
            )
    except Exception:
        # Fallback for very short/noisy chunks where timestamp decoding can fail.
        result = pipe(
            audio_np,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
    text = str(result.get("text", "")).strip()

    words = []
    chunks = result.get("chunks", [])
    if isinstance(chunks, list) and chunks:
        for item in chunks:
            token = str(item.get("text", "")).strip()
            if token:
                words.append({"text": token, "confidence": item.get("confidence")})
    else:
        for token in text.split():
            token = token.strip()
            if token:
                words.append({"text": token, "confidence": None})

    return {"text": text, "words": words}


def get_pipeline_runtime_info(pipe) -> dict:
    """Return runtime device info used by the loaded pipeline."""
    using_cuda = torch.cuda.is_available()
    device_type = "cuda" if using_cuda else "cpu"
    gpu_name = None
    gpu_index = None
    if using_cuda:
        try:
            gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
        except Exception:
            gpu_name = "Unknown CUDA GPU"
            gpu_index = 0
    return {
        "device_type": device_type,
        "gpu_name": gpu_name,
        "gpu_index": gpu_index,
    }
