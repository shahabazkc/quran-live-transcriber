"use client";

export type PillState =
  | "idle"
  | "connecting"
  | "ready"
  | "speaking"
  | "silence"
  | "sending"
  | "waiting"
  | "done"
  | "error";

interface StatusPillProps {
  state: PillState;
  extra?: string;
}

const PILL_CONFIG: Record<PillState, { label: string; className: string }> = {
  idle:       { label: "○ Ready",            className: "bg-[#1a1a22] text-[#555] border border-[#2a2a35]" },
  connecting: { label: "◌ Connecting…",      className: "bg-[#1a1a22] text-[#888] border border-[#2a2a35]" },
  ready:      { label: "✓ Model loaded",     className: "bg-[#0a1a0a] text-[#55cc77] border border-[#55cc77]" },
  speaking:   { label: "🎙 Speaking…",       className: "bg-[#0a1a2a] text-[#55aaff] border border-[#55aaff] animate-[blink_0.8s_ease-in-out_infinite]" },
  silence:    { label: "— Listening…",      className: "bg-[#1a1a0a] text-[#aaaa55] border border-[#888833]" },
  sending:    { label: "⟳ Sending chunk",    className: "bg-[#1a0f00] text-[#ffaa44] border border-[#aa6622]" },
  waiting:    { label: "… Transcribing",     className: "bg-[#120a1a] text-[#aa88ff] border border-[#6644aa]" },
  done:       { label: "✓ Session complete", className: "bg-[#0a1a0a] text-[#55cc77] border border-[#55cc77]" },
  error:      { label: "✗ Error",            className: "bg-[#1a0a0a] text-[#ff6655] border border-[#aa3322]" },
};

export default function StatusPill({ state, extra }: StatusPillProps) {
  const { label, className } = PILL_CONFIG[state];
  return (
    <span
      className={`inline-block px-3 py-1 rounded-full font-mono text-[0.68rem] tracking-widest uppercase ${className}`}
    >
      {label}
      {extra && <span className="ml-2 opacity-70">{extra}</span>}
    </span>
  );
}
