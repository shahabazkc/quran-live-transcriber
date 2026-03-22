"use client";

interface LevelMeterProps {
  rms: number;          // 0.0 – 1.0
  isSpeaking: boolean;
  durationSec?: number;
  chunksEmitted?: number;
  sampleRate?: number;
}

export default function LevelMeter({
  rms,
  isSpeaking,
  durationSec,
  chunksEmitted,
  sampleRate,
}: LevelMeterProps) {
  const pct = Math.min(rms / 0.10, 1.0) * 100;
  const color = isSpeaking ? "#55aaff" : "#444";

  return (
    <div className="flex items-center gap-3 mb-4">
      <span className="font-mono text-[0.6rem] text-[#444] uppercase tracking-widest min-w-[50px]">
        Level
      </span>
      <div className="flex-1 h-[5px] bg-[#1e1e28] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-[width] duration-100"
          style={{ width: `${pct.toFixed(1)}%`, background: color }}
        />
      </div>
      <span className="font-mono text-[0.58rem] text-[#444] min-w-[200px]">
        {rms.toFixed(4)} RMS
        {durationSec !== undefined && <> &nbsp;·&nbsp; {durationSec.toFixed(0)}s</>}
        {chunksEmitted !== undefined && <> &nbsp;·&nbsp; {chunksEmitted} chunk(s)</>}
        {sampleRate && (
          <span className="ml-1 px-1.5 py-0.5 rounded bg-[#0a1a0a] border border-[#1a3a1a] text-[#55cc77]">
            {sampleRate} Hz
          </span>
        )}
      </span>
    </div>
  );
}
