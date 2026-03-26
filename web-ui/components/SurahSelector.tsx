"use client";

import { SurahSummary } from "@/lib/recitationApi";

interface SurahSelectorProps {
  surahs: SurahSummary[];
  value: string;
  onChange: (slug: string) => void;
  disabled?: boolean;
}

export default function SurahSelector({ surahs, value, onChange, disabled }: SurahSelectorProps) {
  return (
    <div>
      <label className="block font-mono text-[0.63rem] text-[#444] uppercase tracking-widest mb-1">
        Surah
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-[#131318] border border-[#2a2a35] text-[#e8e4db] text-sm rounded px-3 py-2 font-mono focus:outline-none focus:border-[#555] disabled:opacity-40"
      >
        {surahs.map((s) => (
          <option key={s.slug} value={s.slug}>
            {s.name_en} ({s.name_ar})
          </option>
        ))}
      </select>
    </div>
  );
}
