"""
recitation_matcher.py
─────────────────────
Word-wise matching for guided Quran recitation UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
ARABIC_PUNCT_RE = re.compile(r"[^\u0621-\u063A\u0641-\u064A0-9\s]")


def normalize_arabic_token(token: str) -> str:
    token = token.strip()
    token = ARABIC_DIACRITICS_RE.sub("", token)
    token = ARABIC_PUNCT_RE.sub("", token)
    return token


@dataclass
class RecognizedWord:
    text: str
    confidence: float | None = None


class RecitationMatcher:
    def __init__(
        self,
        flattened_words: list[dict],
        match_threshold: float = 0.72,
        lookahead_window: int = 8,
        backtrack_window: int = 10,
    ):
        self.words = flattened_words
        self.match_threshold = match_threshold
        self.lookahead_window = max(1, lookahead_window)
        self.backtrack_window = max(0, backtrack_window)
        self.cursor = 0

    def _score(self, spoken: str, expected: str) -> float:
        return SequenceMatcher(None, spoken, expected).ratio()

    def _make_position(self, idx: int) -> dict | None:
        if idx < 0 or idx >= len(self.words):
            return None
        w = self.words[idx]
        return {
            "ayah_index": w["ayah_index"],
            "word_index": w["word_index"],
            "global_word_index": w["global_word_index"],
            "text": w["text"],
        }

    def consume(self, recognized_words: list[RecognizedWord]) -> dict:
        events: list[dict] = []
        last_confidence: float | None = None

        for rec in recognized_words:
            if self.cursor >= len(self.words):
                break

            spoken_norm = normalize_arabic_token(rec.text)

            if not spoken_norm:
                continue

            confidence = rec.confidence
            last_confidence = confidence if confidence is not None else last_confidence

            expected = self.words[self.cursor]
            expected_norm = normalize_arabic_token(expected["text"])
            base_score = self._score(spoken_norm, expected_norm)

            # Confidence-first: when available and clearly low, mark as mispronounced
            # unless a strong future match is found.
            if confidence is not None and confidence < 0.45:
                pass

            # Flexible matching range:
            # - backward range allows user to revisit previously recited words
            # - forward range allows skip-ahead without getting stuck
            start_idx = max(0, self.cursor - self.backtrack_window)
            end_idx = min(len(self.words), self.cursor + 1 + self.lookahead_window)

            best_idx = self.cursor
            best_score = base_score
            for idx in range(start_idx, end_idx):
                if idx == self.cursor:
                    continue
                cand_norm = normalize_arabic_token(self.words[idx]["text"])
                cand_score = self._score(spoken_norm, cand_norm)
                if cand_score > best_score:
                    best_score = cand_score
                    best_idx = idx

            if best_score >= self.match_threshold and best_idx < self.cursor:
                revisited = self.words[best_idx]
                events.append(
                    {
                        "ayah_index": revisited["ayah_index"],
                        "word_index": revisited["word_index"],
                        "global_word_index": revisited["global_word_index"],
                        "status": "correct",
                        "confidence": round(float(confidence), 3) if confidence is not None else None,
                        "match_score": round(float(best_score), 3),
                        "reason": "revisited",
                    }
                )
                self.cursor = best_idx + 1
                continue

            if best_score >= self.match_threshold and best_idx > self.cursor:
                for miss_idx in range(self.cursor, best_idx):
                    missed = self.words[miss_idx]
                    events.append(
                        {
                            "ayah_index": missed["ayah_index"],
                            "word_index": missed["word_index"],
                            "global_word_index": missed["global_word_index"],
                            "status": "mispronounced",
                            "confidence": round(float(confidence), 3) if confidence is not None else None,
                            "match_score": 0.0,
                            "reason": "skipped_ahead",
                        }
                    )
                matched = self.words[best_idx]
                events.append(
                    {
                        "ayah_index": matched["ayah_index"],
                        "word_index": matched["word_index"],
                        "global_word_index": matched["global_word_index"],
                        "status": "correct",
                        "confidence": round(float(confidence), 3) if confidence is not None else None,
                        "match_score": round(float(best_score), 3),
                    }
                )
                self.cursor = best_idx + 1
                continue

            if best_score >= self.match_threshold:
                events.append(
                    {
                        "ayah_index": expected["ayah_index"],
                        "word_index": expected["word_index"],
                        "global_word_index": expected["global_word_index"],
                        "status": "correct",
                        "confidence": round(float(confidence), 3) if confidence is not None else None,
                        "match_score": round(float(best_score), 3),
                    }
                )
                self.cursor += 1
            else:
                events.append(
                    {
                        "ayah_index": expected["ayah_index"],
                        "word_index": expected["word_index"],
                        "global_word_index": expected["global_word_index"],
                        "status": "mispronounced",
                        "confidence": round(float(confidence), 3) if confidence is not None else None,
                        "match_score": round(float(best_score), 3),
                        "reason": "phonetic_mismatch",
                    }
                )

        current = self._make_position(self.cursor)
        next_expected = self._make_position(self.cursor + 1)

        if current is not None:
            events.append(
                {
                    "ayah_index": current["ayah_index"],
                    "word_index": current["word_index"],
                    "global_word_index": current["global_word_index"],
                    "status": "current",
                    "confidence": last_confidence,
                }
            )
        if next_expected is not None:
            events.append(
                {
                    "ayah_index": next_expected["ayah_index"],
                    "word_index": next_expected["word_index"],
                    "global_word_index": next_expected["global_word_index"],
                    "status": "predicted_next",
                    "confidence": None,
                }
            )

        return {
            "word_events": events,
            "current_position": current,
            "next_expected": next_expected,
            "completed_words": self.cursor,
            "is_complete": self.cursor >= len(self.words),
        }
