"use client";

interface TranscriptPanelProps {
  latestChunk: string;
  allChunks: string[];
}

export default function TranscriptPanel({ latestChunk, allChunks }: TranscriptPanelProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Latest chunk */}
      <div>
        <p className="font-mono text-[0.65rem] text-[#444] uppercase tracking-[3px] border-b border-[#1e1e28] pb-1.5 mb-3">
          Latest chunk
        </p>
        <div
          dir="rtl"
          className="font-amiri text-[1.55rem] leading-[2.1] text-right text-[#f0d080] bg-[#121218] border border-[#2a2a35] rounded-lg px-5 py-4 min-h-[80px]"
          style={{ fontFamily: "'Amiri', serif" }}
        >
          {latestChunk || <span className="text-[#2a2a3a]">—</span>}
        </div>
      </div>

      {/* Full transcript */}
      <div>
        <p className="font-mono text-[0.65rem] text-[#444] uppercase tracking-[3px] border-b border-[#1e1e28] pb-1.5 mb-3">
          Full transcript ({allChunks.length} chunk{allChunks.length !== 1 ? "s" : ""})
        </p>
        <div
          dir="rtl"
          className="font-amiri text-[1.35rem] leading-[2.1] text-right text-[#c8c0a8] bg-[#0e0e16] border border-[#1e1e28] rounded-lg px-5 py-4 min-h-[140px] max-h-[400px] overflow-y-auto"
          style={{ fontFamily: "'Amiri', serif" }}
        >
          {allChunks.length > 0 ? (
            allChunks.join(" ")
          ) : (
            <span className="text-[#2a2a3a]">Transcript appears after first pause…</span>
          )}
        </div>
      </div>
    </div>
  );
}
