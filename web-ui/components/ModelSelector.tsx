"use client";

const MODELS = [
  { label: "Whisper Medium — Quran fine-tune",         id: "shahabazkc10/whisper-medium-ar-quran-mix-norm" },
  { label: "Whisper Small — Quran fine-tune",          id: "shahabazkc10/whisper-small-ar-quran-mix-norm" },
  { label: "Whisper Large-v3 — OpenAI baseline",       id: "openai/whisper-large-v3" },
  { label: "Whisper Large-v3-Turbo — Quran fine-tune", id: "shahabazkc10/whisper-large-v3-turbo-ar-quran-mix-norm" },
];

interface ModelSelectorProps {
  value: string;
  onChange: (label: string) => void;
  disabled?: boolean;
}

export default function ModelSelector({ value, onChange, disabled }: ModelSelectorProps) {
  const selected = MODELS.find((m) => m.label === value) ?? MODELS[0];

  return (
    <div>
      <label className="block font-mono text-[0.63rem] text-[#444] uppercase tracking-widest mb-1">
        Model
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-[#131318] border border-[#2a2a35] text-[#e8e4db] text-sm rounded px-3 py-2 font-mono focus:outline-none focus:border-[#555] disabled:opacity-40"
      >
        {MODELS.map((m) => (
          <option key={m.id} value={m.label}>
            {m.label}
          </option>
        ))}
      </select>
      <p className="mt-1 font-mono text-[0.58rem] text-[#333] truncate">{selected.id}</p>
    </div>
  );
}
