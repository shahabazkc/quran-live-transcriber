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
import os
import shutil
from pathlib import Path
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

MODELS: dict[str, str] = {
    "Faster: Whisper Medium — Quran fine-tune":       "faster:shahabazkc10/whisper-medium-ar-quran-mix-norm",
    "Faster: Whisper Small — Quran fine-tune":        "faster:shahabazkc10/whisper-small-ar-quran-mix-norm",
    "Faster: Whisper Large-v3-Turbo — Quran":         "faster:shahabazkc10/whisper-large-v3-turbo-ar-quran-mix-norm",
    "Transformers: Whisper Medium — Quran fine-tune": "transformers:shahabazkc10/whisper-medium-ar-quran-mix-norm",
    "Transformers: Whisper Small — Quran fine-tune":  "transformers:shahabazkc10/whisper-small-ar-quran-mix-norm",
    "Transformers: Whisper Large-v3 — OpenAI":        "transformers:openai/whisper-large-v3",
}

class EnginePipeline:
    def __init__(self, engine_type: str, inner_model):
        self.engine_type = engine_type
        self.model = inner_model # the actual faster_whisper model or hf pipeline

def _convert_ct2(model_id: str) -> str:
    # Remove prefix if present during call
    cache_dir = Path("ct2_models") / model_id.replace("/", "_")
    if cache_dir.exists() and (cache_dir / "model.bin").exists():
        return str(cache_dir)
        
    print(f"Converting {model_id} to CTranslate2... this may take a moment.")
    os.makedirs(cache_dir.parent, exist_ok=True)
    import subprocess
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import shutil
    
    local_hf_dir = Path("ct2_models") / (model_id.replace("/", "_") + "_hf")
    
    if not local_hf_dir.exists():
        print(f"Downloading base HF model to {local_hf_dir} for conversion...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model.save_pretrained(local_hf_dir)
        processor.save_pretrained(local_hf_dir)
        
        # Inject missing tokenizer.json
        if not (local_hf_dir / "tokenizer.json").exists():
            base_model = "openai/whisper-medium"
            if "small" in model_id.lower():
                base_model = "openai/whisper-small"
            elif "large" in model_id.lower():
                base_model = "openai/whisper-large-v3"
            print(f"Fetching tokenizer.json from {base_model}...")
            tok_path = hf_hub_download(repo_id=base_model, filename="tokenizer.json")
            shutil.copy(tok_path, local_hf_dir / "tokenizer.json")

    cmd = [
        "ct2-transformers-converter", 
        "--model", str(local_hf_dir), 
        "--output_dir", str(cache_dir),
        "--quantization", "float16" # use float16 to keep it fast
    ]
    subprocess.run(cmd, check=True)
    
    # Cleanup local_hf_dir to save space
    shutil.rmtree(local_hf_dir, ignore_errors=True)
    return str(cache_dir)


def _build_pipeline(model_ref: str, gpu_index: int | None = None):
    """Build a Whisper pipeline or load faster-whisper by ID string prefix."""
    is_faster = model_ref.startswith("faster:")
    model_id = model_ref.split(":", 1)[-1] if ":" in model_ref else model_ref
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

    if is_faster:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed.")
        ct2_path = _convert_ct2(model_id)
        # load faster whisper
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cw = "float16" if dev == "cuda" else "float32"
        idx = gpu_index if torch.cuda.is_available() and gpu_index is not None else 0
        model = WhisperModel(ct2_path, device=dev, device_index=idx, compute_type=cw)
        return EnginePipeline("faster", model)

    torch_dtype = torch.float16 if torch_device.startswith("cuda") else torch.float32
    processor   = AutoProcessor.from_pretrained(model_id)
    model       = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True,
    ).to(torch_device)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        ignore_warning=True,
        device=pipeline_device,
    )
    return EnginePipeline("transformers", pipe)


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


def transcribe_chunk(wrapper: EnginePipeline, audio_np: np.ndarray) -> str:
    """Run Whisper ASR on a float32 16kHz mono numpy array."""
    if wrapper.engine_type == "faster":
        segments, _ = wrapper.model.transcribe(audio_np, language="ar", task="transcribe")
        return "".join([s.text for s in segments]).strip()
        
    result = wrapper.model(audio_np, generate_kwargs={"language": "arabic", "task": "transcribe"})
    return result["text"].strip()


def transcribe_chunk_detailed(
    wrapper: EnginePipeline, audio_np: np.ndarray, language: str = "arabic", use_word_timestamps: bool = False
) -> dict:
    """
    Return text and lightweight word-level chunks.
    Whisper pipelines may not always expose confidence, so it can be None.
    """
    if wrapper.engine_type == "faster":
        try:
            segments, _ = wrapper.model.transcribe(
                audio_np, 
                language=language if language != "arabic" else "ar", 
                task="transcribe", 
                word_timestamps=use_word_timestamps
            )
            text = ""
            words = []
            for s in segments:
                text += s.text
                if use_word_timestamps and s.words:
                    for w in s.words:
                        words.append({"text": w.word.strip(), "confidence": w.probability})
                else:
                    for token in s.text.split():
                        t = token.strip()
                        if t:
                            words.append({"text": t, "confidence": None})
            return {"text": text.strip(), "words": words}
        except Exception:
            text = ""
            words = []
            return {"text": text, "words": words}

    pipe = wrapper.model

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
