"""
quran_content.py
────────────────
Load and normalize ayah/word data for recitation mode.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


CONTENT_DIR = Path(__file__).resolve().parent / "content"
SURAH_INDEX_FILE = CONTENT_DIR / "surahs.json"


def _tokenize_words(text: str) -> list[str]:
    return [word.strip() for word in text.split() if word.strip()]


@lru_cache(maxsize=8)
def load_surah(slug: str) -> dict:
    path = CONTENT_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"Surah file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid surah data format")

    # Backward compatibility:
    # - legacy format: { "1": "..." }
    # - scalable format: { "meta": {...}, "indices": {...}, "ayahs": { "1": "..." } }
    if "ayahs" in raw and isinstance(raw["ayahs"], dict):
        ayah_source = raw["ayahs"]
        meta = raw.get("meta", {})
        indices = raw.get("indices", {})
    else:
        ayah_source = raw
        meta = {}
        indices = {}

    ayah_items: list[tuple[int, str]] = []
    for key, value in ayah_source.items():
        ayah_items.append((int(key), str(value).strip()))
    ayah_items.sort(key=lambda x: x[0])

    ayahs = []
    flattened = []
    global_index = 0
    for ayah_index, text in ayah_items:
        words = _tokenize_words(text)
        payload_words = []
        for word_index, word in enumerate(words):
            payload_words.append(
                {
                    "word_index": word_index,
                    "global_word_index": global_index,
                    "text": word,
                }
            )
            flattened.append(
                {
                    "ayah_index": ayah_index,
                    "word_index": word_index,
                    "global_word_index": global_index,
                    "text": word,
                }
            )
            global_index += 1

        ayahs.append(
            {
                "ayah_index": ayah_index,
                "text_ar": text,
                "words": payload_words,
            }
        )

    return {
        "surah": {
            "id": int(meta.get("id", 67)),
            "slug": str(meta.get("slug", slug)),
            "name_ar": str(meta.get("name_ar", "الملك")),
            "name_en": str(meta.get("name_en", "Al-Mulk")),
            "ayah_count": int(meta.get("ayah_count", len(ayahs))),
            "order": int(meta.get("order", 67)),
            "revelation_type": str(meta.get("revelation_type", "Meccan")),
        },
        "indices": indices,
        "ayahs": ayahs,
        "flattened_words": flattened,
        "total_words": len(flattened),
    }


def list_surahs() -> list[dict]:
    if not SURAH_INDEX_FILE.exists():
        # Safe fallback if global index is missing: discover surah files dynamically.
        rows: list[dict] = []
        for path in CONTENT_DIR.glob("*.json"):
            if path.name == SURAH_INDEX_FILE.name:
                continue
            slug = path.stem
            try:
                data = load_surah(slug)
                surah = data["surah"]
                rows.append(
                    {
                        "id": surah["id"],
                        "slug": surah["slug"],
                        "name_ar": surah["name_ar"],
                        "name_en": surah["name_en"],
                        "ayah_count": surah["ayah_count"],
                        "order": surah["order"],
                        "juz": data.get("indices", {}).get("juz", []),
                        "hizb": data.get("indices", {}).get("hizb", []),
                        "enabled": True,
                    }
                )
            except Exception:
                # Ignore non-surah JSON files in content folder.
                continue
        rows.sort(key=lambda row: int(row.get("order", row.get("id", 9999))))
        return rows

    payload = json.loads(SURAH_INDEX_FILE.read_text(encoding="utf-8"))
    rows = payload.get("surahs", [])
    rows = [row for row in rows if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("order", row.get("id", 9999))))
    return rows
