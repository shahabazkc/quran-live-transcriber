"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import ModelSelector from "./ModelSelector";
import StatusPill from "./StatusPill";
import { useWebSocket } from "@/lib/useWebSocket";

function arrayBufferToBase64(buffer: ArrayBuffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  const chunkSize = 8192;
  for (let i = 0; i < len; i += chunkSize) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunkSize)));
  }
  return btoa(binary);
}

export default function DropAudioPage() {
  const [model, setModel] = useState("Whisper Medium — Quran fine-tune");
  const [errorMsg, setErrorMsg] = useState("");
  
  const [audioUrl, setAudioUrl] = useState<string>("");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  
  const [isRecording, setIsRecording] = useState(false);
  
  // Local processing state decoupled from websocket connection status
  const [isProcessingLocal, setIsProcessingLocal] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [durationSec, setDurationSec] = useState(0);
  
  const [transcripts, setTranscripts] = useState<Record<number, string>>({});
  
  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setAudioUrl("");
    }
  }, [audioBlob]);

  const { status: wsStatus, connect, sendOfflineAudio, sendFinalizeBatch, disconnect } = useWebSocket({
    onReady: () => {
       // connection established and ready
    },
    onTranscript: (msg) => {
      setTranscripts((prev) => ({
        ...prev,
        [msg.chunkIndex]: msg.text
      }));
    },
    onBatchCompleted: () => {
      setIsProcessingLocal(false);
      setIsCompleted(true);
    },
    onError: (msg) => {
      setErrorMsg(msg);
      setIsProcessingLocal(false);
    },
  });

  // Pre-warm / maintain websocket connection
  useEffect(() => {
     connect(model).catch(err => {
         // silently absorb connection error initially, user can reconnect 
     });
     
     return () => {
         disconnect();
     }
  }, [model, connect, disconnect]);

  const handleProcess = async () => {
    if (!audioBlob) return;
    setIsProcessingLocal(true);
    setIsCompleted(false);
    setErrorMsg("");
    setTranscripts({});

    try {
        if (wsStatus !== "connected") {
            await connect(model);
        }
        
        const arrayBuffer = await audioBlob.arrayBuffer();
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        if (!AudioContextClass) {
            throw new Error("AudioContext not supported in this browser.");
        }
        const actx = new AudioContextClass({ sampleRate: 16000 });
        const audioBuffer = await actx.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);
        const pcm = new Int16Array(channelData.length);
        for (let i = 0; i < channelData.length; i++) {
            let s = Math.max(-1, Math.min(1, channelData[i]));
            pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        const base64 = arrayBufferToBase64(pcm.buffer);
        
        sendOfflineAudio(base64, 16000);
        sendFinalizeBatch(); // tell backend we are done sending fragments for this batch
    } catch (err: any) {
        setErrorMsg(err.message || "Failed to process audio blob");
        setIsProcessingLocal(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        audioChunksRef.current = [];
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setDurationSec(0);
      setIsCompleted(false);
      setTranscripts({});
      const startMs = Date.now();
      recordingTimerRef.current = setInterval(() => {
        setDurationSec(Math.floor((Date.now() - startMs) / 1000));
      }, 1000);
    } catch (err: any) {
      setErrorMsg("Failed to get microphone permissions: " + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAudioBlob(file);
      setIsCompleted(false);
      setTranscripts({});
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      setAudioBlob(file);
      setIsCompleted(false);
      setTranscripts({});
    } else if (file) {
      setErrorMsg("Please drop a valid audio file.");
    }
  };

  const pillState = useMemo(() => {
    if (errorMsg) return "error" as const;
    if (isRecording) return "speaking" as const;
    if (isProcessingLocal) return "connecting" as const;
    return isCompleted ? ("done" as const) : ("idle" as const);
  }, [errorMsg, isRecording, isProcessingLocal, isCompleted]);

  const isBusy = isRecording || isProcessingLocal;

  const combinedText = useMemo(() => {
      const keys = Object.keys(transcripts).map(Number).sort((a, b) => a - b);
      return keys.map(k => transcripts[k]).join(" ");
  }, [transcripts]);

  return (
    <div className="flex min-h-screen bg-[#0a0a0f]">
      <aside className="w-80 shrink-0 bg-[#0e0e16] border-r border-[#1e1e28] p-5 flex flex-col gap-5">
        <div>
          <p className="font-mono text-[0.55rem] text-[#333] uppercase tracking-[3px] mb-0.5">Offline Transcribe</p>
          <h1 className="font-syne font-extrabold text-2xl tracking-tight text-[#e8e4db]">Drop Audio</h1>
          <p className="font-mono text-[0.58rem] text-[#444] mt-1 leading-relaxed">Extract full transcription iteratively without tracking.</p>
        </div>

        <ModelSelector value={model} onChange={setModel} disabled={isBusy} />

        {errorMsg && <div className="bg-[#1a0a0a] border border-[#aa3322] rounded p-3 font-mono text-[0.62rem] text-[#ff6655] break-words">{errorMsg}</div>}
      </aside>

      <main className="flex-1 p-8 flex flex-col gap-6 overflow-y-auto">
        <div>
          <StatusPill state={pillState} />
          {isRecording && (
            <div className="mt-3 flex items-center gap-2 font-mono text-sm text-[#ffb4b4]">
               <div className="w-2 h-2 bg-[#ff5555] rounded-full animate-pulse" />
               Recording: {durationSec}s
            </div>
          )}
          <p className="font-mono text-[0.62rem] text-[#777b8b] mt-3">
            WS Status: {wsStatus}
          </p>
        </div>
        
        <div className="flex flex-col gap-4">
           {/* Upload/Drop Zone */}
           <div 
             onDragOver={handleDragOver} 
             onDrop={handleDrop}
             className="border-2 border-dashed border-[#1e1e28] rounded-xl p-8 flex flex-col items-center justify-center bg-[#0e0e16] hover:bg-[#121620] transition-colors relative"
           >
              <p className="font-mono text-xs text-[#8b92a7] mb-4">Drag and drop audio file here or</p>
              <div className="flex gap-4">
                  <label className="cursor-pointer px-5 py-2.5 bg-[#e8e4db] text-[#0a0a0f] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:bg-[#f0d080] transition-colors">
                     Browse File
                     <input type="file" className="hidden" accept="audio/*" onChange={handleFileUpload} disabled={isBusy} />
                  </label>
                  {!isRecording ? (
                    <button
                        onClick={startRecording}
                        disabled={isBusy}
                        className="px-5 py-2.5 bg-transparent border border-[#aa3322] text-[#ff6655] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:bg-[#220c0c] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                        Start Native Record
                    </button>
                  ) : (
                    <button
                        onClick={stopRecording}
                        className="px-5 py-2.5 bg-[#aa3322] text-[#fff] font-mono text-[0.74rem] uppercase tracking-widest rounded hover:bg-[#ff4433] transition-colors"
                    >
                        Stop Recording
                    </button>
                  )}
              </div>
           </div>
           
           {/* Player and Process actions */}
           {audioUrl && (
             <div className="flex items-center gap-4 bg-[#101522] border border-[#1e2b45] rounded p-4">
                <audio controls src={audioUrl} className="flex-[2] opacity-80" />
                <button
                   onClick={handleProcess}
                   disabled={isBusy}
                   className="flex-[1] px-6 py-2.5 bg-[#dce8ff] text-[#0a0a0f] font-mono font-bold text-[0.74rem] uppercase tracking-widest rounded hover:bg-[#ffffff] transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-center"
                >
                   {isProcessingLocal ? "Processing..." : "Process Audio"}
                </button>
             </div>
           )}
        </div>

        <div className="bg-[#101522] border border-[#1e2b45] rounded-xl p-6 mt-4 flex-1 shadow-lg relative min-h-[200px]">
          
          {isProcessingLocal && (
            <div className="absolute inset-0 z-10 bg-[#101522]/90 flex flex-col items-center justify-center gap-4 rounded-xl">
               <div className="w-10 h-10 border-[3px] border-[#1e2b45] border-t-[#dce8ff] rounded-full animate-spin"></div>
               <p className="font-mono text-[0.7rem] text-[#8ea1c9] uppercase tracking-widest animate-pulse">Running Transcriptions...</p>
            </div>
          )}

          {isCompleted && (
            <div className="mb-6 flex items-center justify-end gap-2 text-[#4ade80]">
               <span className="font-mono text-xs uppercase tracking-wider">Processing Completed</span>
               <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
            </div>
          )}

          {!isProcessingLocal && !isCompleted && !errorMsg && (
             <p className="font-mono text-[0.6rem] text-[#4d5770] uppercase tracking-widest text-center mt-10">
               No transcription available yet. Select/record an audio file to begin.
             </p>
          )}

          {isCompleted && (
             <div className="font-[var(--font-amiri)] text-[1.6rem] leading-[2.5] text-right text-[#e8e4db] whitespace-pre-wrap select-text" dir="rtl">
                {combinedText}
             </div>
          )}
        </div>

      </main>
    </div>
  );
}
