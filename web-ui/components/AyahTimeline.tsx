"use client";

import { SurahAyah } from "@/lib/recitationApi";
import { WordEvent } from "@/lib/useRecitationSocket";

type WordState = "idle" | "correct" | "current" | "predicted_next" | "mispronounced";

function keyFor(ayahIndex: number, wordIndex: number) {
  return `${ayahIndex}:${wordIndex}`;
}

function stateClasses(state: WordState): string {
  if (state === "current") return "bg-[#2a3555] border-[#6f96ff] text-[#dce8ff]";
  if (state === "predicted_next") return "bg-[#1e2539] border-[#4e638f] text-[#b6c7f0]";
  if (state === "correct") return "bg-[#1b3323] border-[#2f7a49] text-[#9de3b3]";
  if (state === "mispronounced") return "bg-[#3a1515] border-[#d24a4a] text-[#ffb4b4]";
  return "bg-[#11131a] border-[#252835] text-[#e8e4db]";
}

interface AyahTimelineProps {
  ayahs: SurahAyah[];
  events: WordEvent[];
}

export default function AyahTimeline({ ayahs, events }: AyahTimelineProps) {
  const statusMap = new Map<string, WordState>();
  for (const ev of events) {
    if (ev.status === "current" || ev.status === "predicted_next" || ev.status === "mispronounced" || ev.status === "correct") {
      statusMap.set(keyFor(ev.ayah_index, ev.word_index), ev.status);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {ayahs.map((ayah) => (
        <div key={ayah.ayah_index} className="border border-[#1f2330] rounded-lg p-4 bg-[#0e1018]">
          <div className="font-mono text-[0.62rem] text-[#7e859a] mb-2">Ayah {ayah.ayah_index}</div>
          <div dir="rtl" className="leading-9 flex flex-wrap gap-2 justify-start">
            {ayah.words.map((word) => {
              const state = statusMap.get(keyFor(ayah.ayah_index, word.word_index)) ?? "idle";
              const mispronounced = state === "mispronounced";
              return (
                <span
                  key={`${ayah.ayah_index}-${word.word_index}`}
                  className={`inline-flex items-center gap-1 px-2.5 py-1 rounded border transition-colors ${stateClasses(state)}`}
                >
                  {mispronounced && <span aria-label="mispronounced" title="Mispronounced" className="text-[#ff5e5e]">●</span>}
                  <span className="font-[var(--font-amiri)] text-xl">{word.text}</span>
                </span>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
