"use client";
/**
 * useWebSocket.ts
 * Manages the WebSocket connection to the FastAPI backend.
 * Designed for concurrent operation — sendChunk() is fire-and-forget;
 * the connection stays open and keeps receiving transcripts independently.
 */
import { useRef, useState, useCallback } from "react";

export type WsStatus = "disconnected" | "connecting" | "connected" | "error";

export interface TranscriptMessage {
  text: string;
  chunkIndex: number;
}

interface UseWebSocketOptions {
  url?: string;
  onTranscript: (msg: TranscriptMessage) => void;
  onReady?: () => void;
  onError?: (message: string) => void;
}

export function useWebSocket({
  url = "ws://localhost:8000/ws/transcribe",
  onTranscript,
  onReady,
  onError,
}: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<WsStatus>("disconnected");

  const connect = useCallback(
    (model: string): Promise<void> => {
      return new Promise((resolve, reject) => {
        if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
          wsRef.current.close();
        }

        setStatus("connecting");
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          setStatus("connected");
          ws.send(JSON.stringify({ type: "config", model, language: "arabic" }));
        };

        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "ready") {
              onReady?.();
              resolve();
            } else if (msg.type === "transcript") {
              onTranscript({ text: msg.text, chunkIndex: msg.chunk_index });
            } else if (msg.type === "error") {
              onError?.(msg.message);
            }
          } catch {
            // ignore parse errors
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
      });
    },
    [url, onTranscript, onReady, onError]
  );

  /** Fire-and-forget: send a base64 PCM chunk. Mic keeps running. */
  const sendChunk = useCallback((base64: string, srcRate: number) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "audio_chunk", data: base64, src_rate: srcRate }));
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

  return { status, connect, sendChunk, sendStop, disconnect };
}
