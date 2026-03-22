"use client";
import { useEffect, useState } from "react";

interface Device {
  deviceId: string;
  label: string;
}

interface MicSelectorProps {
  value: string;
  onChange: (deviceId: string) => void;
  disabled?: boolean;
}

export default function MicSelector({ value, onChange, disabled }: MicSelectorProps) {
  const [devices, setDevices] = useState<Device[]>([]);

  useEffect(() => {
    async function load() {
      try {
        // Request permission first so labels are populated
        await navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => s.getTracks().forEach((t) => t.stop()));
        const all = await navigator.mediaDevices.enumerateDevices();
        const inputs = all
          .filter((d) => d.kind === "audioinput")
          .map((d) => ({ deviceId: d.deviceId, label: d.label || `Microphone (${d.deviceId.slice(0, 6)})` }));
        setDevices(inputs);
        if (inputs.length > 0 && !value) onChange(inputs[0].deviceId);
      } catch {
        // permission denied or no devices
      }
    }
    load();
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <label className="block font-mono text-[0.63rem] text-[#444] uppercase tracking-widest mb-1">
        Microphone
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-[#131318] border border-[#2a2a35] text-[#e8e4db] text-sm rounded px-3 py-2 font-mono focus:outline-none focus:border-[#555] disabled:opacity-40"
      >
        {devices.map((d) => (
          <option key={d.deviceId} value={d.deviceId}>
            {d.label}
          </option>
        ))}
      </select>
    </div>
  );
}
