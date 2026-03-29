"""
recitation_matcher.py
─────────────────────
Word-wise matching for guided Quran recitation UI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
ARABIC_PUNCT_RE = re.compile(r"[^\u0621-\u063A\u0641-\u064A0-9\s]")
SPACE_RE = re.compile(r"\s+")
ALEF_VARIANTS_RE = re.compile(r"[إأآٱ]")
HAMZA_WAW_RE = re.compile(r"[ؤ]")
HAMZA_YEH_RE = re.compile(r"[ئ]")

log = logging.getLogger(__name__)

def normalize_arabic_token(token: str) -> str:
    token = token.strip()
    token = ARABIC_DIACRITICS_RE.sub("", token)
    token = ARABIC_PUNCT_RE.sub("", token)
    token = ALEF_VARIANTS_RE.sub("ا", token)
    token = HAMZA_WAW_RE.sub("و", token)
    token = HAMZA_YEH_RE.sub("ي", token)
    token = token.replace("ى", "ي").replace("ة", "ه")
    return token

@dataclass
class RecognizedWord:
    text: str
    confidence: float | None = None

class RecitationMatcher:
    def __init__(
        self,
        flattened_words: list[dict],
        minimum_match_score_threshold: float = 0.65,
        forward_search_limit: int = 6,      
        backward_search_limit: int = 2,     
        minimum_words_for_matching: int = 5,
        fuzzy_token_tolerance: float = 0.72,
        phrase_detection_tolerance: float = 0.74,
        stop_words: list[str] | None = None,
        special_phrases: list[str] | None = None,
    ):
        self.words = flattened_words
        self.minimum_match_score_threshold = minimum_match_score_threshold
        self.forward_search_limit = max(1, forward_search_limit)
        self.backward_search_limit = max(0, backward_search_limit)
        self.minimum_words_for_matching = max(1, minimum_words_for_matching)
        self.fuzzy_token_tolerance = max(0.4, min(0.98, fuzzy_token_tolerance))
        self.phrase_detection_tolerance = max(0.5, min(0.98, phrase_detection_tolerance))
        raw_stop_words = stop_words if stop_words is not None else ["ال", "الله", "الا"]
        
        # User requested to comment out Bismillah logic for now:
        # raw_phrases = special_phrases if special_phrases is not None else ["بسم الله الرحمن الرحيم", "اعوذ بالله من الشيطان الرجيم"]
        raw_phrases = []

        self.stop_words = {normalize_arabic_token(w) for w in raw_stop_words if normalize_arabic_token(w)}
        self.special_phrases = [self._normalize_text(p) for p in raw_phrases if self._normalize_text(p)]
        self.special_phrase_tokens = [{t for t in p.split() if t} for p in self.special_phrases]
        self.cursor = 0
        self._build_ayah_index()

    def _build_ayah_index(self):
        self.ayah_words: dict[int, list[dict]] = {}
        self.ayah_starts: dict[int, int] = {}
        self.ayah_order: list[int] = []
        for w in self.words:
            ayah_idx = int(w["ayah_index"])
            if ayah_idx not in self.ayah_words:
                self.ayah_words[ayah_idx] = []
                self.ayah_starts[ayah_idx] = int(w["global_word_index"])
                self.ayah_order.append(ayah_idx)
            self.ayah_words[ayah_idx].append(w)
        self.ayah_order.sort()

    def _score(self, spoken: str, expected: str) -> float:
        return SequenceMatcher(None, spoken, expected).ratio()

    def _normalize_text(self, text: str) -> str:
        base = normalize_arabic_token(text)
        return SPACE_RE.sub(" ", base).strip()

    def _tokenize_normalized(self, text: str) -> list[str]:
        norm = self._normalize_text(text)
        return [t for t in norm.split(" ") if t]

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

    def _current_ayah_index(self) -> int | None:
        pos = self._make_position(self.cursor)
        return int(pos["ayah_index"]) if pos is not None else None

    # New implementation of scoring based on user requests:
    def _score_ayah_alignment(self, transcript_tokens: list[str], ayah_idx: int) -> tuple[float, int, int]:
        ayah_dicts = self.ayah_words.get(ayah_idx, [])
        ayah_tokens = [self._normalize_text(w["text"]) for w in ayah_dicts]
        if not transcript_tokens or not ayah_tokens:
            return 0.0, 0, len(transcript_tokens)

        cursor = 0
        weighted_total = 0.0
        weighted_matched = 0.0
        matched_count = 0

        for tok in transcript_tokens:
            tok_weight = float(max(1, len(tok)))
            # If word is stop word, reduce its weight in jumping logic
            if tok in self.stop_words or len(tok) <= 2:
                tok_weight = 0.5 

            weighted_total += tok_weight
            best_score = 0.0
            best_idx = None
            
            # search forward in ayah
            for idx in range(cursor, min(len(ayah_tokens), cursor + 5)):
                sc = self._score(tok, ayah_tokens[idx])
                if sc > best_score:
                    best_score = sc
                    best_idx = idx
                if sc > 0.97:
                    break
                    
            if best_idx is not None and best_score >= self.fuzzy_token_tolerance:
                matched_count += 1
                weighted_matched += best_score * tok_weight
                cursor = best_idx + 1

        score = (weighted_matched / weighted_total) if weighted_total > 0 else 0.0
        unmatched_count = max(0, len(transcript_tokens) - matched_count)
        return score, matched_count, unmatched_count

    def _get_ayah_window(self, current_ayah: int, direction: str) -> list[int]:
        if current_ayah not in self.ayah_order:
            return []
        pivot = self.ayah_order.index(current_ayah)
        if direction == "forward":
            # 5-6 ayahs forward
            return self.ayah_order[pivot + 1 : pivot + 1 + self.forward_search_limit]
        if direction == "backward":
            # 1-2 ayahs backward
            start = max(0, pivot - self.backward_search_limit)
            return self.ayah_order[start:pivot]
        return [current_ayah]

    def _align_tokens_to_cursor(self, tokens: list[str], confidence: float | None) -> list[dict]:
        """Align tokens to the current sequence, allowing 1-2 words of backward movement."""
        events: list[dict] = []
        window_start = max(0, self.cursor - 2)
        window_end = min(len(self.words), self.cursor + max(10, len(tokens) * 2))
        
        search_cursor = self.cursor

        for tok in tokens:
            best_score = 0.0
            best_idx = None
            
            # Limit how far forward we can jump. 
            # If the token is a stop word or very short, we don't allow it to leap forward and cause random skips.
            is_weak_word = tok in self.stop_words or len(tok) <= 2
            forward_lookahead = 1 if is_weak_word else 4
            
            # Check 2 words back up to `forward_lookahead` words forward from search_cursor
            local_start = max(window_start, search_cursor - 2)
            local_end = min(window_end, search_cursor + forward_lookahead)
            
            for idx in range(local_start, local_end):
                expected = self.words[idx]
                expected_norm = self._normalize_text(expected["text"])
                sc = self._score(tok, expected_norm)
                if sc > best_score:
                    best_score = sc
                    best_idx = idx
                if sc > 0.97:
                    break
                    
            if best_idx is not None and best_score >= self.fuzzy_token_tolerance:
                # Mark any skipped words as mispronounced so the UI doesn't leave them blank
                for skipped_idx in range(search_cursor, best_idx):
                    expected_skipped = self.words[skipped_idx]
                    events.append({
                        "ayah_index": expected_skipped["ayah_index"],
                        "word_index": expected_skipped["word_index"],
                        "global_word_index": expected_skipped["global_word_index"],
                        "status": "mispronounced",
                        "confidence": None,
                        "match_score": 0.0,
                        "reason": "skipped",
                    })

                expected = self.words[best_idx]
                events.append({
                    "ayah_index": expected["ayah_index"],
                    "word_index": expected["word_index"],
                    "global_word_index": expected["global_word_index"],
                    "status": "correct",
                    "confidence": round(float(confidence), 3) if confidence is not None else None,
                    "match_score": round(float(best_score), 3),
                })
                search_cursor = best_idx + 1
            else:
                # If we couldn't match, we assume it's a mispronounced word for the CURRENT search_cursor
                if search_cursor < len(self.words):
                    # But if it's just "الا" hallucination, don't penalize the expected word, just ignore the noise.
                    if not is_weak_word:
                        expected = self.words[search_cursor]
                        events.append({
                            "ayah_index": expected["ayah_index"],
                            "word_index": expected["word_index"],
                            "global_word_index": expected["global_word_index"],
                            "status": "mispronounced",
                            "confidence": round(float(confidence), 3) if confidence is not None else None,
                            "match_score": 0.0,
                            "reason": "phonetic_mismatch",
                        })

        if search_cursor > self.cursor:
             self.cursor = search_cursor
             
        return events

    def consume(self, recognized_words: list[RecognizedWord]) -> dict:
        events: list[dict] = []
        last_confidence: float | None = None
        joined = " ".join((r.text or "").strip() for r in recognized_words if (r.text or "").strip())
        raw_tokens = self._tokenize_normalized(joined)
        
        if recognized_words:
            last_confidence = recognized_words[-1].confidence

        if self.cursor >= len(self.words):
            return {
                "word_events": [],
                "current_position": None,
                "next_expected": None,
                "completed_words": self.cursor,
                "is_complete": True,
                "chunk_match_score": 0.0,
                "matched_words_count": 0,
                "unmatched_words_count": 0,
                "valid_word_count": 0,
                "low_confidence": True,
                "low_confidence_reason": "complete",
            }

        current_ayah = self._current_ayah_index()
        if current_ayah is None:
            current_ayah = self.ayah_order[0]
            
        valid_word_count = len(raw_tokens)
        
        # JUMP LOGIC
        # Score candidate forward ayahs to see if we missed a transition.
        jump_ayah = current_ayah
        chunk_score, matched_count, unmatched_count = self._score_ayah_alignment(raw_tokens, current_ayah)
        
        # 1. Forward Matching
        best_forward = (None, 0.0, 0, 0)
        for ayah_idx in self._get_ayah_window(current_ayah, "forward"):
            sc, mcount, ucount = self._score_ayah_alignment(raw_tokens, ayah_idx)
            if sc > best_forward[1]:
                best_forward = (ayah_idx, sc, mcount, ucount)

        if best_forward[1] >= self.minimum_match_score_threshold and best_forward[1] > chunk_score:
            jump_ayah = best_forward[0]
            chunk_score, matched_count, unmatched_count = best_forward[1], best_forward[2], best_forward[3]
        else:
            # 2. Backward Matching (Only if valid_word_count >= threshold and forward/current failed)
            if valid_word_count >= self.minimum_words_for_matching and chunk_score < self.minimum_match_score_threshold:
                best_backward = (None, 0.0, 0, 0)
                for ayah_idx in self._get_ayah_window(current_ayah, "backward"):
                    sc, mcount, ucount = self._score_ayah_alignment(raw_tokens, ayah_idx)
                    if sc > best_backward[1]:
                        best_backward = (ayah_idx, sc, mcount, ucount)
                        
                if best_backward[1] >= self.minimum_match_score_threshold:
                    jump_ayah = best_backward[0]
                    chunk_score, matched_count, unmatched_count = best_backward[1], best_backward[2], best_backward[3]

        if jump_ayah != current_ayah:
            # We are jumping. Reset cursor to the start of the jump ayah.
            self.cursor = self.ayah_starts.get(jump_ayah, self.cursor)
            
        # Irrespective of jumping, align tokens to sequence
        if len(raw_tokens) > 0:
            events.extend(self._align_tokens_to_cursor(raw_tokens, last_confidence))
            low_confidence = False
        else:
            low_confidence = True
            
        current = self._make_position(self.cursor)
        next_expected = self._make_position(min(len(self.words)-1, self.cursor + 1)) if self.cursor < len(self.words) else None

        if current is not None:
            events.append({
                "ayah_index": current["ayah_index"],
                "word_index": current["word_index"],
                "global_word_index": current["global_word_index"],
                "status": "current",
                "confidence": last_confidence,
            })
        if next_expected is not None:
            events.append({
                "ayah_index": next_expected["ayah_index"],
                "word_index": next_expected["word_index"],
                "global_word_index": next_expected["global_word_index"],
                "status": "predicted_next",
                "confidence": None,
            })

        return {
            "word_events": events,
            "current_position": current,
            "next_expected": next_expected,
            "completed_words": self.cursor,
            "is_complete": self.cursor >= len(self.words),
            "chunk_match_score": round(float(chunk_score), 3),
            "matched_words_count": matched_count,
            "unmatched_words_count": unmatched_count,
            "valid_word_count": valid_word_count,
            "low_confidence": low_confidence,
            "low_confidence_reason": "too_few_words" if low_confidence else "",
        }

