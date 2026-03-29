"use client";

const MODELS = [
  { label: "Faster: Whisper Medium — Quran fine-tune",       id: "faster:shahabazkc10/whisper-medium-ar-quran-mix-norm" },
  { label: "Faster: Whisper Small — Quran fine-tune",        id: "faster:shahabazkc10/whisper-small-ar-quran-mix-norm" },
  { label: "Faster: Whisper Large-v3-Turbo — Quran",         id: "faster:shahabazkc10/whisper-large-v3-turbo-ar-quran-mix-norm" },
  { label: "Transformers: Whisper Medium — Quran fine-tune", id: "transformers:shahabazkc10/whisper-medium-ar-quran-mix-norm" },
  { label: "Transformers: Whisper Small — Quran fine-tune",  id: "transformers:shahabazkc10/whisper-small-ar-quran-mix-norm" },
  { label: "Transformers: Whisper Large-v3 — OpenAI",        id: "transformers:openai/whisper-large-v3" },
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
