"use client";
import { useState } from "react";

export interface VADSettings {
  gain: number;
  silenceThreshold: number;
  minSilenceMs: number;
  minChunkMs: number;
  maxChunkMs: number;
  preSpeechMs?: number;
  postSpeechMs?: number;
  noiseSuppression?: boolean;
  echoCancellation?: boolean;
}

interface VADControlsProps {
  settings: VADSettings;
  onChange: (s: VADSettings) => void;
  disabled?: boolean;
}

function Slider({
  label, value, min, max, step, format, onChange, disabled,
}: {
  label: string; value: number; min: number; max: number; step: number;
  format?: (v: number) => string; onChange: (v: number) => void; disabled?: boolean;
}) {
  const display = format ? format(value) : value.toString();
  return (
    <div className="mb-4">
      <div className="flex justify-between mb-1">
        <span className="font-mono text-[0.62rem] text-[#555] uppercase tracking-wider">{label}</span>
        <span className="font-mono text-[0.62rem] text-[#f0d080]">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 disabled:opacity-40"
      />
    </div>
  );
}

export default function VADControls({ settings, onChange, disabled }: VADControlsProps) {
  const [open, setOpen] = useState(false);
  const set = (patch: Partial<VADSettings>) => onChange({ ...settings, ...patch });

  return (
    <div className="border border-[#2a2a35] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-[#131318] font-mono text-[0.65rem] text-[#555] uppercase tracking-widest hover:text-[#888] transition-colors"
      >
        <span>VAD Settings</span>
        <span className="text-[#2a2a35]">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="px-4 py-3 bg-[#0e0e16]">
          <Slider
            label="Input gain"
            value={settings.gain}
            min={1} max={8} step={0.5}
            format={(v) => `${v}×`}
            onChange={(v) => set({ gain: v })}
            disabled={disabled}
          />
          <Slider
            label="Silence threshold (RMS)"
            value={settings.silenceThreshold}
            min={0.001} max={0.1} step={0.001}
            format={(v) => v.toFixed(3)}
            onChange={(v) => set({ silenceThreshold: v })}
            disabled={disabled}
          />
          <Slider
            label="Min pause duration (ms)"
            value={settings.minSilenceMs}
            min={200} max={2000} step={50}
            format={(v) => `${v} ms`}
            onChange={(v) => set({ minSilenceMs: v })}
            disabled={disabled}
          />
          <Slider
            label="Min chunk length (ms)"
            value={settings.minChunkMs}
            min={300} max={5000} step={100}
            format={(v) => `${v} ms`}
            onChange={(v) => set({ minChunkMs: v })}
            disabled={disabled}
          />
          <Slider
            label="Max chunk length"
            value={settings.maxChunkMs}
            min={5000} max={60000} step={1000}
            format={(v) => `${(v / 1000).toFixed(0)} s`}
            onChange={(v) => set({ maxChunkMs: v })}
            disabled={disabled}
          />
          <Slider
            label="Pre-speech buffer (ms)"
            value={settings.preSpeechMs ?? 500}
            min={100}
            max={1500}
            step={50}
            format={(v) => `${v} ms`}
            onChange={(v) => set({ preSpeechMs: v })}
            disabled={disabled}
          />
          <Slider
            label="Post-speech buffer (ms)"
            value={settings.postSpeechMs ?? 200}
            min={50}
            max={1000}
            step={25}
            format={(v) => `${v} ms`}
            onChange={(v) => set({ postSpeechMs: v })}
            disabled={disabled}
          />
          <div className="mt-3 grid grid-cols-1 gap-2">
            <label className="font-mono text-[0.62rem] text-[#777] flex items-center gap-2">
              <input
                type="checkbox"
                checked={settings.noiseSuppression ?? true}
                onChange={(e) => set({ noiseSuppression: e.target.checked })}
                disabled={disabled}
              />
              Noise suppression
            </label>
            <label className="font-mono text-[0.62rem] text-[#777] flex items-center gap-2">
              <input
                type="checkbox"
                checked={settings.echoCancellation ?? true}
                onChange={(e) => set({ echoCancellation: e.target.checked })}
                disabled={disabled}
              />
              Echo cancellation
            </label>
          </div>
          <div className="mt-2 p-2 bg-[#131318] rounded text-[0.6rem] font-mono text-[#444] leading-relaxed">
            Pause ≥ <span className="text-[#f0d080]">{settings.minSilenceMs} ms</span>
            &nbsp;·&nbsp; threshold <span className="text-[#f0d080]">{settings.silenceThreshold.toFixed(3)}</span>
            <br />
            Chunk <span className="text-[#f0d080]">{settings.minChunkMs} ms</span>
            {" / "}
            <span className="text-[#f0d080]">{(settings.maxChunkMs / 1000).toFixed(0)} s</span>
            &nbsp;·&nbsp; gain <span className="text-[#f0d080]">{settings.gain}×</span>
            <br />
            Pre/Post: <span className="text-[#f0d080]">{settings.preSpeechMs ?? 500} ms</span>
            {" / "}
            <span className="text-[#f0d080]">{settings.postSpeechMs ?? 200} ms</span>
          </div>
        </div>
      )}
    </div>
  );
}
