"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import LevelMeter from "./LevelMeter";
import MicSelector from "./MicSelector";
import ModelSelector from "./ModelSelector";
import StatusPill from "./StatusPill";
import SurahSelector from "./SurahSelector";
import VADControls, { VADSettings } from "./VADControls";
import AyahTimeline from "./AyahTimeline";

import { fetchSurahDetail, fetchSurahs, SurahAyah, SurahSummary } from "@/lib/recitationApi";
import { useRecitationSocket, WordEvent } from "@/lib/useRecitationSocket";
import { useVADRecorder } from "@/lib/useVADRecorder";

const WORDWISE_VAD: VADSettings = {
  gain: 5.8,
  silenceThreshold: 0.01,
  minSilenceMs: 240,
  minChunkMs: 280,
  maxChunkMs: 4000,
  preSpeechMs: 650,
  postSpeechMs: 260,
  noiseSuppression: true,
  echoCancellation: true,
};

export default function RecitationPage() {
  const [model, setModel] = useState("Whisper Medium — Quran fine-tune");
  const [micId, setMicId] = useState("");
  const [vad, setVad] = useState<VADSettings>(WORDWISE_VAD);
  const [surahs, setSurahs] = useState<SurahSummary[]>([]);
  const [surahSlug, setSurahSlug] = useState("surah-mulk");
  const [ayahs, setAyahs] = useState<SurahAyah[]>([]);
  const [events, setEvents] = useState<WordEvent[]>([]);
  const [persistentWordEvents, setPersistentWordEvents] = useState<Record<string, WordEvent>>({});
  const [latestTranscript, setLatestTranscript] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [summary, setSummary] = useState<{ completed_words: number; mispronounced_count: number } | null>(null);
  const [durationSec, setDurationSec] = useState(0);
  const [sessionTotalWords, setSessionTotalWords] = useState(0);
  const [runtimeDevice, setRuntimeDevice] = useState<string>("unknown");
  const [wavBlob, setWavBlob] = useState<Blob | null>(null);
  const [sessionTs, setSessionTs] = useState<string>("");
  const [pendingQueue, setPendingQueue] = useState(0);
  const [globalSlots, setGlobalSlots] = useState<{ cap?: number; available?: number }>({});
  const [matchScore, setMatchScore] = useState<number | null>(null);
  const [chunkMatchedWords, setChunkMatchedWords] = useState<number>(0);
  const [chunkUnmatchedWords, setChunkUnmatchedWords] = useState<number>(0);
  const [activeAyahIndex, setActiveAyahIndex] = useState<number | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTsRef = useRef<number>(0);

  useEffect(() => {
    fetchSurahs()
      .then((list) => {
        setSurahs(list);
        if (list.length > 0) setSurahSlug(list[0].slug);
      })
      .catch(() => setErrorMsg("Failed to load surah list"));
  }, []);

  useEffect(() => {
    fetchSurahDetail(surahSlug)
      .then((detail) => {
        setAyahs(detail.ayahs);
        setSessionTotalWords(detail.total_words);
      })
      .catch(() => setErrorMsg("Failed to load surah ayahs"));
  }, [surahSlug]);

  const audioUrl = useMemo(() => (wavBlob ? URL.createObjectURL(wavBlob) : null), [wavBlob]);
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const { status: wsStatus, connect, sendChunk, sendStop, disconnect } = useRecitationSocket({
    onReady: ({ total_words, runtime }) => {
      setSessionTotalWords(total_words);
      if (runtime?.device_type === "cuda") {
        const label = runtime.gpu_name
          ? `GPU (${runtime.gpu_name}${runtime.gpu_index !== undefined && runtime.gpu_index !== null ? ` #${runtime.gpu_index}` : ""})`
          : "GPU (CUDA)";
        setRuntimeDevice(label);
      } else if (runtime?.device_type) {
        setRuntimeDevice(runtime.device_type.toUpperCase());
      } else {
        setRuntimeDevice("unknown");
      }
    },
    onProgress: (payload) => {
      setLatestTranscript(payload.transcript_text);
      setEvents(payload.word_events);
      setActiveAyahIndex(payload.current_position?.ayah_index ?? null);
      setPersistentWordEvents((prev) => {
        const next = { ...prev };
        for (const ev of payload.word_events) {
          if (ev.status === "correct" || ev.status === "mispronounced") {
            const key = `${ev.ayah_index}:${ev.word_index}`;
            next[key] = ev;
          }
        }
        return next;
      });
      setMatchScore(typeof payload.chunk_match_score === "number" ? payload.chunk_match_score : null);
      setChunkMatchedWords(payload.matched_words_count ?? 0);
      setChunkUnmatchedWords(payload.unmatched_words_count ?? 0);
    },
    onSummary: (payload) => {
      setSummary(payload);
    },
    onQueueStatus: (payload) => {
      setPendingQueue(payload.pending_session ?? 0);
      setGlobalSlots({ cap: payload.global_active_cap, available: payload.global_available_slots });
      if (payload.dropped_silent_chunk) {
        setErrorMsg("Silent/blank chunk skipped (not queued).");
      }
    },
    onError: (msg) => {
      setErrorMsg(msg);
    },
  });

  const handleChunk = useCallback(
    (base64PCM: string, srcRate: number) => {
      sendChunk(base64PCM, srcRate);
    },
    [sendChunk]
  );

  const { state: recState, startRecording, stopRecording } = useVADRecorder({
    ...vad,
    emitOnMaxChunk: false,
    forceEmitOnMaxChunk: false,
    onChunk: handleChunk,
  });

  const isRecording = recState.isRecording;

  const handleStart = async () => {
    setErrorMsg("");
    setSummary(null);
    setEvents([]);
    setPersistentWordEvents({});
    setLatestTranscript("");
    setWavBlob(null);
    setMatchScore(null);
    setChunkMatchedWords(0);
    setChunkUnmatchedWords(0);
    setActiveAyahIndex(null);
    setSessionTs(new Date().toLocaleString());

    await connect({
      model,
      surah_slug: surahSlug,
      gpu_index: 1,
      max_pending: 200,
      max_batch_size: 1,
      process_interval_ms: 0,
      min_voice_rms: 0.0005,
      matcher_config: {
        minimum_match_score_threshold: 0.65,
        forward_search_limit: 10,
        backward_search_limit: 4,
        minimum_words_for_matching: 1,
        fuzzy_token_tolerance: 0.72,
        phrase_detection_tolerance: 0.74,
        stop_words: ["ال", "الله", "الا"],
        special_phrases: ["بسم الله الرحمن الرحيم", "اعوذ بالله من الشيطان الرجيم"],
      },
    });
    await startRecording(micId || undefined);

    startTsRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setDurationSec(Math.floor((Date.now() - startTsRef.current) / 1000));
    }, 1000);
  };

  const handleStop = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    const result = stopRecording();
    if (result) {
      setWavBlob(result.wavBlob);
    }
    sendStop();
  };

  const pillState = useMemo(() => {
    if (errorMsg) return "error" as const;
    if (isRecording) return recState.isSpeaking ? ("speaking" as const) : ("silence" as const);
    if (wsStatus === "connecting") return "connecting" as const;
    if (summary) return "done" as const;
    return "idle" as const;
  }, [errorMsg, isRecording, recState.isSpeaking, wsStatus, summary]);

  const displayEvents = useMemo(() => {
    const sticky = Object.values(persistentWordEvents);
    const transient = events.filter((ev) => ev.status === "current" || ev.status === "predicted_next");
    return [...sticky, ...transient];
  }, [persistentWordEvents, events]);

  const totalWrongWords = useMemo(
    () => Object.values(persistentWordEvents).filter((ev) => ev.status === "mispronounced").length,
    [persistentWordEvents]
  );

  return (
    <div className="flex min-h-screen bg-[#0a0a0f]">
      <aside className="w-80 shrink-0 bg-[#0e0e16] border-r border-[#1e1e28] p-5 flex flex-col gap-5">
        <div>
          <p className="font-mono text-[0.55rem] text-[#333] uppercase tracking-[3px] mb-0.5">Recitation</p>
          <h1 className="font-syne font-extrabold text-2xl tracking-tight text-[#e8e4db]">Word-wise Quran Guide</h1>
          <p className="font-mono text-[0.58rem] text-[#444] mt-1 leading-relaxed">Ayah view, current word, next prediction, wrong-word flags</p>
        </div>

        <SurahSelector surahs={surahs} value={surahSlug} onChange={setSurahSlug} disabled={isRecording} />
        <ModelSelector value={model} onChange={setModel} disabled={isRecording} />
        <MicSelector value={micId} onChange={setMicId} disabled={isRecording} />
        <VADControls settings={vad} onChange={setVad} disabled={isRecording} />

        {errorMsg && <div className="bg-[#1a0a0a] border border-[#aa3322] rounded p-3 font-mono text-[0.62rem] text-[#ff6655]">{errorMsg}</div>}
      </aside>

      <main className="flex-1 p-8 flex flex-col gap-6 overflow-y-auto">
        <div>
          <StatusPill state={pillState} />
          {isRecording && (
            <LevelMeter
              rms={recState.rmsLevel}
              isSpeaking={recState.isSpeaking}
              durationSec={durationSec}
              chunksEmitted={recState.chunksEmitted}
            />
          )}
          <p className="font-mono text-[0.62rem] text-[#777b8b] mt-3">
            Surah: {surahSlug} | Total words: {sessionTotalWords} | WS: {wsStatus}
          </p>
          <p className="font-mono text-[0.62rem] text-[#777b8b] mt-1">
            Queue pending: {pendingQueue} | Global slots: {globalSlots.available ?? "-"} / {globalSlots.cap ?? "-"} | Match score:{" "}
            {matchScore ?? "-"}
          </p>
          <p className="font-mono text-[0.62rem] text-[#777b8b] mt-1">
            Wrong words: {totalWrongWords} | Matched vs unmatched: {chunkMatchedWords} / {chunkUnmatchedWords}
          </p>
          <p className="font-mono text-[0.62rem] text-[#777b8b] mt-1">Inference device: {runtimeDevice}</p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={isRecording || !surahSlug}
            className="px-5 py-2.5 bg-[#e8e4db] text-[#0a0a0f] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:bg-[#f0d080] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Start Reciting
          </button>
          <button
            onClick={handleStop}
            disabled={!isRecording}
            className="px-5 py-2.5 bg-transparent border border-[#333] text-[#e8e4db] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:border-[#e8e4db] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Stop
          </button>
          <button
            onClick={disconnect}
            disabled={wsStatus !== "connected" || isRecording}
            className="px-5 py-2.5 bg-transparent border border-[#333] text-[#e8e4db] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:border-[#e8e4db] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            End Session
          </button>
        </div>

        {audioUrl && (
          <div className="bg-[#101522] border border-[#1e2b45] rounded p-3 mt-4">
            <p className="font-mono text-[0.6rem] text-[#8ea1c9] uppercase tracking-widest mb-1">Session audio</p>
            <p className="font-mono text-[0.62rem] text-[#777b8b] mb-2">{sessionTs}</p>
            <audio controls src={audioUrl} className="w-full" />
          </div>
        )}

        <div className="bg-[#101522] border border-[#1e2b45] rounded p-3">
          <p className="font-mono text-[0.6rem] text-[#8ea1c9] uppercase tracking-widest mb-1">Live transcript chunk</p>
          <p className="font-[var(--font-amiri)] text-xl">{latestTranscript || "—"}</p>
        </div>

        <div className="font-mono text-[0.62rem] text-[#8b92a7]">
          Legend: <span className="text-[#dce8ff]">Current</span> · <span className="text-[#b6c7f0]">Next predicted</span> · <span className="text-[#ffb4b4]">Mispronounced (●)</span>
        </div>

        <AyahTimeline ayahs={ayahs} events={displayEvents} activeAyahIndex={activeAyahIndex} />

        {summary && (
          <div className="bg-[#12191f] border border-[#2a3d52] rounded p-3 font-mono text-[0.68rem]">
            Completed words: {summary.completed_words} | Mispronounced: {summary.mispronounced_count}
          </div>
        )}


      </main>
    </div>
  );
}
