// Mic-capture worklet. Resamples the mic stream down to a true 16 kHz Int16 PCM
// stream (what Gemini Live expects) regardless of the actual hardware rate —
// browsers often ignore a 16 kHz AudioContext request and run at 44.1/48 kHz,
// and sending that mislabeled as 16 kHz garbles recognition.
const TARGET_RATE = 16000;
const CHUNK = 1600; // ~100 ms at 16 kHz

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE; // e.g. 48000/16000 = 3
    this._acc = [];      // pending input samples at hardware rate
    this._out = [];      // resampled 16 kHz samples awaiting a full chunk
    this._pos = 0;       // fractional read cursor into _acc
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) this._acc.push(ch[i]);

    // Linear-interpolate from hardware rate down to 16 kHz.
    while (this._pos + 1 < this._acc.length) {
      const i0 = Math.floor(this._pos);
      const frac = this._pos - i0;
      this._out.push(this._acc[i0] * (1 - frac) + this._acc[i0 + 1] * frac);
      this._pos += this.ratio;
    }
    const consumed = Math.floor(this._pos);
    if (consumed > 0) { this._acc.splice(0, consumed); this._pos -= consumed; }

    while (this._out.length >= CHUNK) {
      const chunk = this._out.splice(0, CHUNK);
      const pcm = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
