"use client";

import { useCallback, useRef, useState } from "react";

export type RecitationSocketStatus = "disconnected" | "connecting" | "connected" | "error";

export interface WordEvent {
  ayah_index: number;
  word_index: number;
  global_word_index: number;
  status: "correct" | "current" | "predicted_next" | "mispronounced";
  confidence: number | null;
  match_score?: number | null;
  reason?: string;
}

export interface ProgressPayload {
  chunk_index: number;
  transcript_text: string;
  current_position: {
    ayah_index: number;
    word_index: number;
    global_word_index: number;
    text: string;
  } | null;
  next_expected: {
    ayah_index: number;
    word_index: number;
    global_word_index: number;
    text: string;
  } | null;
  word_events: WordEvent[];
  completed_words: number;
  is_complete: boolean;
  chunk_match_score?: number | null;
  matched_words_count?: number;
  unmatched_words_count?: number;
  valid_word_count?: number;
  low_confidence?: boolean;
  low_confidence_reason?: string;
}

export interface RecitationMatcherConfig {
  minimum_match_score_threshold?: number;
  forward_search_limit?: number;
  backward_search_limit?: number;
  minimum_words_for_matching?: number;
  fuzzy_token_tolerance?: number;
  phrase_detection_tolerance?: number;
  stop_words?: string[];
  special_phrases?: string[];
}

interface UseRecitationSocketOptions {
  url?: string;
  onReady?: (payload: {
    session_id: string;
    total_words: number;
    runtime?: { device_type: string; gpu_name?: string | null; gpu_index?: number | null };
  }) => void;
  onProgress: (payload: ProgressPayload) => void;
  onSummary?: (payload: { completed_words: number; mispronounced_count: number }) => void;
  onQueueStatus?: (payload: {
    pending_session: number;
    global_active_cap?: number;
    global_available_slots?: number;
    dropped_silent_chunk?: boolean;
  }) => void;
  onError?: (message: string) => void;
}

export function useRecitationSocket({
  url = "ws://localhost:8000/ws/recitation",
  onReady,
  onProgress,
  onSummary,
  onQueueStatus,
  onError,
}: UseRecitationSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<RecitationSocketStatus>("disconnected");

  const connect = useCallback(
    (params: {
      model: string;
      surah_slug: string;
      language?: string;
      gpu_index?: number;
      max_pending?: number;
      max_batch_size?: number;
      process_interval_ms?: number;
      min_voice_rms?: number;
      matcher_config?: RecitationMatcherConfig;
    }): Promise<void> =>
      new Promise((resolve, reject) => {
        if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
          wsRef.current.close();
        }

        setStatus("connecting");
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          setStatus("connected");
          ws.send(
            JSON.stringify({
              type: "config",
              model: params.model,
              surah_slug: params.surah_slug,
              language: params.language ?? "arabic",
              gpu_index: params.gpu_index ?? 1,
              max_pending: params.max_pending ?? 5,
              max_batch_size: params.max_batch_size ?? 1,
              process_interval_ms: params.process_interval_ms ?? 0,
              min_voice_rms: params.min_voice_rms ?? 0.001,
              ...(params.matcher_config ?? {}),
            })
          );
        };

        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "ready") {
              onReady?.({
                session_id: msg.session_id,
                total_words: msg.total_words,
                runtime: msg.runtime,
              });
              resolve();
            } else if (msg.type === "progress") {
              onProgress(msg);
            } else if (msg.type === "final_summary") {
              onSummary?.({
                completed_words: msg.completed_words,
                mispronounced_count: msg.mispronounced_count,
              });
            } else if (msg.type === "queue_status") {
              onQueueStatus?.({
                pending_session: msg.pending_session ?? 0,
                global_active_cap: msg.global_active_cap,
                global_available_slots: msg.global_available_slots,
                dropped_silent_chunk: msg.dropped_silent_chunk,
              });
            } else if (msg.type === "queue_backpressure") {
              onError?.(msg.message ?? "Queue backpressure");
            } else if (msg.type === "error") {
              onError?.(msg.message ?? "Unknown backend error");
            }
          } catch {
            // ignore parse error
          }
        };

        ws.onerror = () => {
          setStatus("error");
          onError?.("WebSocket connection failed");
          reject(new Error("WebSocket connection failed"));
        };

        ws.onclose = () => {
          setStatus("disconnected");
        };
      }),
    [url, onReady, onProgress, onSummary, onQueueStatus, onError]
  );

  const sendChunk = useCallback((base64: string, srcRate: number) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "audio_chunk", data: base64, src_rate: srcRate }));
  }, []);

  const sendOfflineAudio = useCallback((base64: string, srcRate: number) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "offline_audio", data: base64, src_rate: srcRate }));
  }, []);

  const sendStop = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "stop" }));
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus("disconnected");
  }, []);

  return { status, connect, sendChunk, sendOfflineAudio, sendStop, disconnect };
}
