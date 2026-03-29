raise Exception("TEST")
print("Script started")

from transformers import WhisperForConditionalGeneration

print("Loading model...")

model = WhisperForConditionalGeneration.from_pretrained(
    r"C:\Users\shaha\.cache\huggingface\hub\models--shahabazkc10--whisper-medium-ar-quran-mix-norm\snapshots\fc0e7c6cf8c4a7666d70007d5610a0ba1276c6f8"
)

print("Model loaded, saving...")

model.save_pretrained(r"C:\Users\shaha\whisper-pt-model")

print("Done!")