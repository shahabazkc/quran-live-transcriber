/**
 * audioUtils.ts
 * PCM helpers: RMS, base64 encode, WAV encode, resampling.
 */

/** RMS of a Float32 PCM buffer (0.0 – 1.0) */
export function rmsFromFloat32(buffer: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / buffer.length);
}

/** Convert Float32 PCM → Int16 PCM bytes */
export function float32ToInt16(buffer: Float32Array): Int16Array {
  const int16 = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    const s = Math.max(-1, Math.min(1, buffer[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

/** Encode Int16Array → base64 string */
export function int16ToBase64(samples: Int16Array): string {
  const bytes = new Uint8Array(samples.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

/** Float32 frames → base64 Int16 PCM (ready to send to backend) */
export function framesToBase64PCM(frames: Float32Array[]): string {
  const total = frames.reduce((n, f) => n + f.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const f of frames) {
    merged.set(f, off);
    off += f.length;
  }
  return int16ToBase64(float32ToInt16(merged));
}

/** Encode all captured Float32 frames into a WAV Blob for download/playback */
export function framesToWavBlob(frames: Float32Array[], sampleRate: number): Blob {
  const total = frames.reduce((n, f) => n + f.length, 0);
  const int16 = new Int16Array(total);
  let off = 0;
  for (const f of frames) {
    const chunk = float32ToInt16(f);
    int16.set(chunk, off);
    off += chunk.length;
  }

  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = int16.byteLength;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const write = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  write(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  write(36, "data");
  view.setUint32(40, dataSize, true);

  const samples = new Int16Array(buffer, 44);
  samples.set(int16);

  return new Blob([buffer], { type: "audio/wav" });
}
