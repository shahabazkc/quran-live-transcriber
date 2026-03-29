"use client";
import { useEffect, useMemo } from "react";

interface AudioPlayerProps {
  wavBlob: Blob | null;
  sessionTimestamp: string;
  chunks: string[];
}

export default function AudioPlayer({ wavBlob, sessionTimestamp, chunks }: AudioPlayerProps) {
  const audioUrl = useMemo(() => (wavBlob ? URL.createObjectURL(wavBlob) : null), [wavBlob]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  if (!wavBlob) return null;

  const safeTs = sessionTimestamp.replace(/:/g, "-").replace(/ /g, "_");
  const downloadWav = () => {
    if (!audioUrl) return;
    const a = document.createElement("a");
    a.href = audioUrl;
    a.download = `quran_${safeTs}.wav`;
    a.click();
  };
  const downloadTranscript = () => {
    if (chunks.length === 0) return;
    const text = chunks.map((t, i) => `[Chunk ${i + 1}] ${t}`).join("\n");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `transcript_${safeTs}.txt`;
    a.click();
  };

  return (
    <div className="mt-6 border-t border-[#1e1e28] pt-6">
      <p className="font-mono text-[0.65rem] text-[#444] uppercase tracking-[3px] border-b border-[#1e1e28] pb-1.5 mb-4">
        Recorded audio
      </p>
      <div className="bg-[#131318] border border-[#2a2a35] rounded-lg px-4 py-3 mb-4">
        <p className="font-semibold text-sm mb-1">Session recording</p>
        <p className="font-mono text-[0.63rem] text-[#555]">{sessionTimestamp}</p>
      </div>
      {audioUrl && (
        <audio controls src={audioUrl} className="w-full mb-4" />
      )}
      <div className="flex gap-3">
        <button
          onClick={downloadWav}
          className="px-4 py-2 bg-transparent border border-[#333] text-[#e8e4db] font-mono text-[0.72rem] uppercase tracking-widest rounded hover:border-[#e8e4db] transition-colors"
        >
          ↓ Download WAV
        </button>
        {chunks.length > 0 && (
          <button
            onClick={downloadTranscript}
            className="px-4 py-2 bg-transparent border border-[#333] text-[#e8e4db] font-mono text-[0.72rem] uppercase tracking-widest rounded hover:border-[#e8e4db] transition-colors"
          >
            ↓ Download Transcript
          </button>
        )}
      </div>
    </div>
  );
}
