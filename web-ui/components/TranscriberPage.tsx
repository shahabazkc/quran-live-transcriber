"use client";
/**
 * TranscriberPage.tsx
 * Main transcriber component. Wires together:
 *   - useVADRecorder (mic → VAD → chunks)
 *   - useWebSocket (chunks → backend → transcripts)
 *
 * Concurrent design: mic keeps running while chunks are transcribed.
 * onChunk() fires → sendChunk() fires-and-forgets → mic loop continues.
 * Backend processes chunks concurrently and streams back transcripts.
 */
import { useState, useCallback, useRef } from "react";

import StatusPill, { PillState }  from "./StatusPill";
import LevelMeter                 from "./LevelMeter";
import MicSelector                from "./MicSelector";
import ModelSelector              from "./ModelSelector";
import VADControls, { VADSettings } from "./VADControls";
import TranscriptPanel            from "./TranscriptPanel";
import AudioPlayer                from "./AudioPlayer";

import { useVADRecorder }         from "@/lib/useVADRecorder";
import { useWebSocket }           from "@/lib/useWebSocket";

const DEFAULT_VAD: VADSettings = {
  gain: 3.0,
  silenceThreshold: 0.01,
  minSilenceMs: 600,
  minChunkMs: 800,
  maxChunkMs: 30000,
};

export default function TranscriberPage() {
  const [micId, setMicId]       = useState("");
  const [model, setModel]       = useState("Whisper Medium — Quran fine-tune");
  const [vad, setVad]           = useState<VADSettings>(DEFAULT_VAD);
  const [pillState, setPill]    = useState<PillState>("idle");
  const [latestChunk, setLatest] = useState("");
  const [allChunks, setAll]     = useState<string[]>([]);
  const [wavBlob, setWavBlob]   = useState<Blob | null>(null);
  const [sessionTs, setSessionTs] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [durationSec, setDurationSec] = useState(0);
  const durationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  // ── WebSocket ────────────────────────────────────────────────────────────
  const { status: wsStatus, connect, sendChunk, sendStop, disconnect } = useWebSocket({
    onTranscript: ({ text }) => {
      setLatest(text);
      setAll((prev) => [...prev, text]);
      // Return to silence pill after receiving transcript — mic is still running
      setPill((p) => (p === "waiting" ? "silence" : p));
    },
    onReady: () => {
      setPill("ready");
    },
    onError: (msg) => {
      setErrorMsg(msg);
      setPill("error");
    },
  });

  // ── VAD Recorder ─────────────────────────────────────────────────────────
  const handleChunk = useCallback(
    (base64PCM: string, srcRate: number) => {
      // Fire-and-forget — mic stays active
      sendChunk(base64PCM, srcRate);
      setPill("waiting");
    },
    [sendChunk]
  );

  const { state: recState, startRecording, stopRecording } = useVADRecorder({
    ...vad,
    onChunk: handleChunk,
  });

  // ── Start ─────────────────────────────────────────────────────────────────
  const handleStart = async () => {
    setErrorMsg("");
    setLatest("");
    setAll([]);
    setWavBlob(null);
    setPill("connecting");

    try {
      await connect(model);
    } catch {
      setPill("error");
      setErrorMsg("Could not connect to backend. Is it running on port 8000?");
      return;
    }

    try {
      await startRecording(micId || undefined);
    } catch {
      setPill("error");
      setErrorMsg("Microphone access denied or device unavailable.");
      disconnect();
      return;
    }

    setPill("silence");
    setSessionTs(new Date().toLocaleString());
    startTimeRef.current = Date.now();

    durationTimerRef.current = setInterval(() => {
      setDurationSec(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
  };

  // ── Stop ──────────────────────────────────────────────────────────────────
  const handleStop = () => {
    if (durationTimerRef.current) {
      clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }

    const result = stopRecording();
    sendStop();
    disconnect();

    if (result) setWavBlob(result.wavBlob);
    setPill("done");
  };

  const isRecording = recState.isRecording;

  // Keep pill in sync with mic speech state (without overriding waiting/error)
  if (isRecording) {
    const expected = recState.isSpeaking ? "speaking" : "silence";
    if (pillState !== "waiting" && pillState !== "error" && pillState !== expected) {
      // use a layout effect pattern inline (avoid hook call in render)
    }
  }

  return (
    <div className="flex min-h-screen bg-[#0a0a0f]">
      {/* ── Sidebar ───────────────────────────────────────────────────────── */}
      <aside className="w-72 shrink-0 bg-[#0e0e16] border-r border-[#1e1e28] p-5 flex flex-col gap-5">
        <div>
          <p className="font-mono text-[0.55rem] text-[#333] uppercase tracking-[3px] mb-0.5">System</p>
          <h1 className="font-syne font-extrabold text-2xl tracking-tight text-[#e8e4db]" style={{ fontFamily: "'Syne', sans-serif" }}>
            Quran<br />Transcriber
          </h1>
          <p className="font-mono text-[0.58rem] text-[#444] mt-1 leading-relaxed">
            VAD · Continuous mic · Arabic ASR
          </p>
        </div>

        <ModelSelector value={model} onChange={setModel} disabled={isRecording} />
        <MicSelector  value={micId}  onChange={setMicId}  disabled={isRecording} />
        <VADControls  settings={vad} onChange={setVad}    disabled={isRecording} />

        {errorMsg && (
          <div className="bg-[#1a0a0a] border border-[#aa3322] rounded p-3 font-mono text-[0.62rem] text-[#ff6655] leading-relaxed">
            {errorMsg}
          </div>
        )}
      </aside>

      {/* ── Main ──────────────────────────────────────────────────────────── */}
      <main className="flex-1 p-8 flex flex-col gap-6 overflow-y-auto">

        {/* Header */}
        <div>
          <StatusPill
            state={
              isRecording
                ? recState.isSpeaking
                  ? "speaking"
                  : pillState === "waiting"
                  ? "waiting"
                  : "silence"
                : pillState
            }
          />
          {isRecording && (
            <LevelMeter
              rms={recState.rmsLevel}
              isSpeaking={recState.isSpeaking}
              durationSec={durationSec}
              chunksEmitted={recState.chunksEmitted}
            />
          )}
          <p className="font-mono text-[0.6rem] text-[#333] mt-2">
            Model → {model}
            {wsStatus === "connected" && (
              <span className="ml-2 text-[#55cc77]">● connected</span>
            )}
          </p>
        </div>

        {/* Controls */}
        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={isRecording}
            className="px-5 py-2.5 bg-[#e8e4db] text-[#0a0a0f] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:bg-[#f0d080] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            ▶ Start
          </button>
          <button
            onClick={handleStop}
            disabled={!isRecording}
            className="px-5 py-2.5 bg-transparent border border-[#333] text-[#e8e4db] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:border-[#e8e4db] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            ■ Stop
          </button>
        </div>

        <hr className="border-[#1e1e28]" />

        {/* Transcript */}
        <TranscriptPanel latestChunk={latestChunk} allChunks={allChunks} />

        {/* Audio player after stop */}
        <AudioPlayer wavBlob={wavBlob} sessionTimestamp={sessionTs} chunks={allChunks} />
      </main>
    </div>
  );
}
