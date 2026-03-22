"use client";
/**
 * useVADRecorder.ts
 * Mic → AudioWorklet → VAD state machine → chunk emission.
 *
 * Key design: the mic NEVER stops or pauses while chunks are being sent.
 * onChunk() is a fire-and-forget callback — the VAD loop continues
 * regardless of how long transcription takes on the backend.
 *
 * VAD state machine (ported from audio_utils.py):
 *   SILENCE → SPEECH (RMS > threshold)
 *   SPEECH  → SILENCE (RMS < threshold for minSilenceMs)
 *   On SPEECH→SILENCE transition: emit chunk if length >= minChunkMs
 *   Hard ceiling at maxChunkMs: force-emit without waiting for silence
 */
import { useRef, useState, useCallback } from "react";
import { rmsFromFloat32, framesToBase64PCM, framesToWavBlob } from "./audioUtils";

export interface VADConfig {
  gain?: number;
  silenceThreshold?: number;  // RMS below this = silence (default 0.01)
  minSilenceMs?: number;      // pause duration to trigger split (default 600)
  minChunkMs?: number;        // discard shorter utterances (default 800)
  maxChunkMs?: number;        // hard ceiling (default 30000)
}

export interface VADRecorderState {
  isRecording: boolean;
  isSpeaking: boolean;
  rmsLevel: number;
  chunksEmitted: number;
}

interface UseVADRecorderOptions extends VADConfig {
  sampleRate?: number;
  onChunk: (base64PCM: string, srcRate: number) => void;
}

export function useVADRecorder({
  gain = 3.0,
  silenceThreshold = 0.01,
  minSilenceMs = 600,
  minChunkMs = 800,
  maxChunkMs = 30000,
  sampleRate = 16000,
  onChunk,
}: UseVADRecorderOptions) {
  const [state, setState] = useState<VADRecorderState>({
    isRecording: false,
    isSpeaking: false,
    rmsLevel: 0,
    chunksEmitted: 0,
  });

  const ctxRef      = useRef<AudioContext | null>(null);
  const streamRef   = useRef<MediaStream | null>(null);
  const workletRef  = useRef<AudioWorkletNode | null>(null);
  const allFramesRef = useRef<Float32Array[]>([]);

  // VAD state (mutable refs — updated in worklet message handler)
  const vadRef = useRef({
    inSpeech:     false,
    speechFrames: [] as Float32Array[],
    silenceFrames: [] as Float32Array[],
    speechMs:     0,
    silenceMs:    0,
    chunksEmitted: 0,
  });

  const FRAMES_PER_BUFFER = 1024;

  const startRecording = useCallback(
    async (deviceId?: string) => {
      try {
        const constraints: MediaStreamConstraints = {
          audio: deviceId
            ? { deviceId: { exact: deviceId }, channelCount: 1, echoCancellation: false, noiseSuppression: false }
            : { channelCount: 1, echoCancellation: false, noiseSuppression: false },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;

        const ctx = new AudioContext({ sampleRate });
        ctxRef.current = ctx;

        await ctx.audioWorklet.addModule("/audio-processor.js");

        const source = ctx.createMediaStreamSource(stream);
        const worklet = new AudioWorkletNode(ctx, "audio-processor");
        workletRef.current = worklet;

        const msPerFrame = (FRAMES_PER_BUFFER / ctx.sampleRate) * 1000;
        const maxPreFrames = Math.ceil(500 / msPerFrame);
        const leadInFrames = Math.ceil(200 / msPerFrame);

        worklet.port.onmessage = (evt: MessageEvent<{ pcm: Float32Array }>) => {
          const raw = evt.data.pcm;

          // Apply gain
          const boosted = new Float32Array(raw.length);
          for (let i = 0; i < raw.length; i++) {
            boosted[i] = Math.max(-1, Math.min(1, raw[i] * gain));
          }

          allFramesRef.current.push(boosted);

          const rms = rmsFromFloat32(boosted);
          const isSpeechFrame = rms > silenceThreshold;
          const vad = vadRef.current;

          if (isSpeechFrame) {
            if (!vad.inSpeech) {
              // SILENCE → SPEECH: prepend lead-in
              vad.speechFrames  = vad.silenceFrames.slice(-leadInFrames);
              vad.silenceFrames = [];
              vad.inSpeech      = true;
            }
            vad.speechFrames.push(boosted);
            vad.speechMs  += msPerFrame;
            vad.silenceMs  = 0;

            // Hard ceiling — emit without waiting for silence
            if (vad.speechMs >= maxChunkMs) {
              const frames = [...vad.speechFrames];
              vad.speechFrames  = [];
              vad.silenceFrames = [];
              vad.speechMs      = 0;
              vad.silenceMs     = 0;
              vad.inSpeech      = false;
              vad.chunksEmitted++;
              const count = vad.chunksEmitted;
              setState((s) => ({ ...s, isSpeaking: false, rmsLevel: rms, chunksEmitted: count }));
              // Fire-and-forget: mic keeps running
              onChunk(framesToBase64PCM(frames), ctx.sampleRate);
              return;
            }

            setState((s) => ({ ...s, isSpeaking: true, rmsLevel: rms }));
          } else {
            if (vad.inSpeech) {
              vad.silenceFrames.push(boosted);
              vad.silenceMs += msPerFrame;

              if (vad.silenceMs >= minSilenceMs) {
                // SPEECH → SILENCE transition
                const totalMs =
                  (vad.speechFrames.length + vad.silenceFrames.length) * msPerFrame;
                if (totalMs >= minChunkMs) {
                  const frames = [...vad.speechFrames, ...vad.silenceFrames];
                  vad.chunksEmitted++;
                  const count = vad.chunksEmitted;
                  setState((s) => ({ ...s, isSpeaking: false, rmsLevel: rms, chunksEmitted: count }));
                  // Fire-and-forget: mic keeps running
                  onChunk(framesToBase64PCM(frames), ctx.sampleRate);
                }
                vad.speechFrames  = [];
                vad.silenceFrames = [];
                vad.speechMs      = 0;
                vad.silenceMs     = 0;
                vad.inSpeech      = false;
              } else {
                setState((s) => ({ ...s, isSpeaking: false, rmsLevel: rms }));
              }
            } else {
              // Rolling pre-speech buffer
              vad.silenceFrames.push(boosted);
              if (vad.silenceFrames.length > maxPreFrames) vad.silenceFrames.shift();
              setState((s) => ({ ...s, isSpeaking: false, rmsLevel: rms }));
            }
          }
        };

        source.connect(worklet);
        worklet.connect(ctx.destination);

        // Reset state
        allFramesRef.current = [];
        vadRef.current = {
          inSpeech: false,
          speechFrames: [],
          silenceFrames: [],
          speechMs: 0,
          silenceMs: 0,
          chunksEmitted: 0,
        };

        setState({ isRecording: true, isSpeaking: false, rmsLevel: 0, chunksEmitted: 0 });
      } catch (err) {
        console.error("Failed to start recording:", err);
        throw err;
      }
    },
    [gain, silenceThreshold, minSilenceMs, minChunkMs, maxChunkMs, sampleRate, onChunk]
  );

  const stopRecording = useCallback((): { wavBlob: Blob; sampleRate: number } | null => {
    const ctx = ctxRef.current;

    // Emit any remaining speech before stopping
    const vad = vadRef.current;
    if (vad.inSpeech && vad.speechFrames.length > 0) {
      const totalMs = (vad.speechFrames.length + vad.silenceFrames.length) *
        ((FRAMES_PER_BUFFER / (ctx?.sampleRate ?? 16000)) * 1000);
      if (totalMs >= minChunkMs) {
        const frames = [...vad.speechFrames, ...vad.silenceFrames];
        onChunk(framesToBase64PCM(frames), ctx?.sampleRate ?? 16000);
      }
    }

    workletRef.current?.disconnect();
    workletRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    const sr = ctx?.sampleRate ?? 16000;
    const allFrames = allFramesRef.current;

    ctx?.close();
    ctxRef.current = null;

    setState({ isRecording: false, isSpeaking: false, rmsLevel: 0, chunksEmitted: 0 });

    if (allFrames.length === 0) return null;
    return { wavBlob: framesToWavBlob(allFrames, sr), sampleRate: sr };
  }, [minChunkMs, onChunk]);

  return { state, startRecording, stopRecording };
}
