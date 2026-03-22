/**
 * audio-processor.js
 * AudioWorklet processor — runs in the audio thread.
 * Accumulates 1024-frame chunks of Float32 PCM and posts them to the main thread.
 */
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(1024);
    this._offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // mono channel, 128 frames per quantum

    for (let i = 0; i < samples.length; i++) {
      this._buffer[this._offset++] = samples[i];
      if (this._offset === 1024) {
        // Send a copy to the main thread
        this.port.postMessage({ pcm: this._buffer.slice() });
        this._offset = 0;
      }
    }

    return true; // keep processor alive
  }
}

registerProcessor("audio-processor", AudioProcessor);
