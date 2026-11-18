"""The ONNX renderer: Vāgdhenu's website path (render_core.Renderer.render_one) with the model and vocoder in ONNX.

No torch at run time: numpy and onnxruntime, with Vāgdhenu's own text code (`chant_text.py`, `prep_text.py`) and the
torch-free audio front end and tokenizer of his MLX backend (`vagdhenu_mlx.audio`, `vagdhenu_mlx.text`; Vāgdhenu,
Apache-2.0; the MLX backend by aieunlabs). Every step follows his torch renderer and IndicF5's `infer_process` and
`CFM.sample`, so a render matches his torch render of the same weights from the same noise:
  * the reference: his preprocessing (silence trimming, RMS raised to 0.1), its vocos-type log-mel;
  * tokens: the reference text then the line, one per character, -1 as filler;
  * length: his fixed duration (reference seconds + syllables x the bank's seconds per syllable);
  * sampling: Euler steps on the sway time grid (coefficient -1), classifier-free guidance as a batch of two
    (reference mel zeroed and every token the filler for the null pass), the reference frames put back after;
  * the vocoder: BigVGAN on the generated mel; a quiet line (RMS <= 0.04) is redone with the next seed, up to 4 tries;
  * stitching: his gate, his gaps between lines.
Two of his torch renderer's behaviours are kept as they are: BigVGAN's output is not rescaled by the reference's
loudness (only vocos's was), and if a line is split into chunks only the last chunk's mel reaches BigVGAN.
His retry test runs on IndicF5's stock vocos (`vocos_spectrum.onnx`, with the inverse STFT here): the chunks' vocos
waves cross-faded, scaled down by a quiet reference's loudness, divided by 32768 if any sample exceeds 1.5 (his check
for 16-bit output, which a loud peak also trips), then redone if the RMS is at most 0.04. Without that file the test
falls back to BigVGAN's output, which can pick a different try.

Noise: numpy's generator, seeded per try. The "torch" setting draws it with torch's CPU generator as his renderer does;
it exists only for `nandini onnx check verses` (comparing with his renderer from the same noise) and needs torch.
"""
import json
import sys
from pathlib import Path

import numpy as np

from .. import external as X

HOP, SR_, NMEL = 256, 24000, 100
TARGET_RMS = 0.1
MAX_DURATION = 4096


def _src():
    s = str(X.require("vagdhenu") / "src")
    if s not in sys.path:
        sys.path.insert(0, s)


def vocab_file():
    """Vāgdhenu's vocab.txt, checked against VOCAB_SHA256 (as training.data.vocab_file, without its imports)."""
    import hashlib
    from .. import config as C
    v = X.require("vagdhenu") / "src" / "reference_bank" / "vocab.txt"
    got = hashlib.sha256(v.read_bytes()).hexdigest()
    if got != C.VOCAB_SHA256:
        raise SystemExit(f"{v}: sha256 {got}, expected {C.VOCAB_SHA256}")
    return v


def time_grid(steps, sway=-1.0):
    """IndicF5's t: linspace then the sway warp, in float32 as torch computes it."""
    t = np.linspace(0, 1, steps + 1, dtype=np.float32)
    return (t + np.float32(sway) * (np.cos(np.float32(np.pi / 2) * t) - 1 + t)).astype(np.float32)


class Renderer:
    def __init__(self, onnx_dir, bank_path, vocab_file, speed=0.90, nfe=64, cfg=3.0, gap=0.55, gap_halant=0.20,
                 noise="numpy", providers=("CPUExecutionProvider",), threads=0, dit="nandini_dit.onnx",
                 vocoders=("nandini_vocoder.onnx",)):
        import onnxruntime as ort
        _src()
        import prep_text as PT
        import chant_text as CT
        from vagdhenu_mlx import audio as A
        from vagdhenu_mlx.text import load_vocab, tokenize
        self.PT, self.CT, self.A, self.tokenize = PT, CT, A, tokenize
        self.vocab = load_vocab(str(vocab_file))
        self.speed, self.nfe, self.cfg, self.gap, self.gap_halant, self.noise = speed, nfe, cfg, gap, gap_halant, noise
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        d = Path(onnx_dir)
        self.dit = ort.InferenceSession(str(d / dit), so, providers=list(providers))
        # The first vocoder makes the render; any others decode the same mels (`render_all`), for comparing files.
        self.vocs = [ort.InferenceSession(str(d / v), so, providers=list(providers)) for v in vocoders]
        self.voc = self.vocs[0]
        vp = d / "vocos_spectrum.onnx"
        self.vocos_sess = ort.InferenceSession(str(vp), so, providers=list(providers)) if vp.exists() else None
        self._bank = json.loads(Path(bank_path).read_text(encoding="utf-8"))
        self._bdir = Path(bank_path).parent
        self._lut = {}
        for k, v in self._bank.items():
            if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
                continue
            self._lut[k.lower()] = v
            self._lut[v["wav"].replace(".wav", "").lower()] = v
        self._primes = self._bank.get("repeat_primes", {})
        self._refs = {}

    # -------------------------------------------------------------- references
    def _ref(self, wav, ref_text):
        key = (wav, ref_text)
        if key not in self._refs:
            wave, text = self.A.preprocess_ref(str(self._bdir / wav), ref_text, clip_short=True, log=lambda *_: None)
            dur = len(wave) / SR_
            wave, rms = self.A.normalize_rms(wave)
            raw = text  # infer_process sizes chunks on this; the model reads `text`
            if len(text[-1].encode("utf-8")) == 1:
                text = text + " "
            self._refs[key] = dict(wave=wave, text=text, raw_text=raw, rms=rms, mel=self.A.log_mel(wave), dur=dur)
        return self._refs[key]

    def _meter_ref(self, meter):
        key = meter.lower().replace(".wav", "")
        if key not in self._lut:
            key = self.CT.FALLBACK_METER
            if key not in self._lut:
                raise ValueError(f"meter '{meter}' not in bank (and fallback '{self.CT.FALLBACK_METER}' missing)")
        e = self._lut[key]
        return self._ref(e["wav"], e["ref_text"]), float(e.get("sec_per_syll", 0.26))

    # -------------------------------------------------------------- model
    def _rng(self, seed):
        """One noise stream per try, seeded once, as his renderer seeds torch before each try."""
        if self.noise == "torch":
            import torch
            torch.manual_seed(seed)
            return lambda n: torch.randn(n, NMEL).numpy()
        g = np.random.default_rng(seed)
        return lambda n: g.standard_normal((n, NMEL)).astype(np.float32)

    def sample(self, ref, gen_text, duration, draw):
        """CFM.sample for one item: the generated mel after the reference, (frames, 100)."""
        ids = self.tokenize(ref["text"] + gen_text, self.vocab)
        cl = ref["mel"].shape[0]
        lens = max(len(ids), cl)
        n = min(max(lens + 1, duration), MAX_DURATION)
        cond = np.zeros((n, NMEL), np.float32)
        cond[:cl] = ref["mel"][:n]
        mask = np.zeros((n, 1), bool)
        mask[:lens] = True
        step_cond = np.where(mask, cond, 0).astype(np.float32)
        text = np.full((2, n), -1, np.int64)
        text[0, :min(len(ids), n)] = ids[:n]
        cond2 = np.stack([step_cond, np.zeros_like(step_cond)])
        x = draw(n)[None]
        ts = time_grid(self.nfe)
        for k in range(self.nfe):
            t = np.full((2,), ts[k], np.float32)
            f = self.dit.run(None, {"x": np.repeat(x, 2, 0), "cond": cond2, "text": text, "t": t})[0]
            v = f[:1] + (f[:1] - f[1:]) * np.float32(self.cfg) if self.cfg >= 1e-5 else f[:1]
            x = x + (ts[k + 1] - ts[k]) * v
        out = np.where(mask[None], cond[None], x)[0]
        return out[len(ref["wave"]) // HOP:]

    def vocos_wave(self, mel):
        re_, im = self.vocos_sess.run(None, {"mel": mel.T[None].astype(np.float32)})
        return istft(re_[0] + 1j * im[0])

    def vocode(self, mel, voc=None):
        return (voc or self.voc).run(None, {"mel": mel.T[None].astype(np.float32)})[0][0]

    # -------------------------------------------------------------- public
    def render_one(self, text, meter, seed=60, no_sandhi=True, speed=None, sps=None):
        return self.render_all(text, meter, seed, no_sandhi, speed, sps)[0]

    def render_all(self, text, meter, seed=60, no_sandhi=True, speed=None, sps=None):
        """render_one with every vocoder given: [(sr, audio)] in their order, the lines' mels shared."""
        PT, CT = self.PT, self.CT
        padas = text if isinstance(text, list) else CT.split_padas(text)
        if not padas:
            raise ValueError("empty text")
        ref, ref_sps = self._meter_ref(meter)
        if sps is not None:
            ref_sps = float(sps)
        spd = float(speed) if speed is not None else self.speed
        base = (lambda p: PT.model_text(p)) if no_sandhi else (lambda p: PT.model_text_sandhi(p, echo_final=False))
        pieces = [base(p) for p in padas]
        if not no_sandhi:
            pieces = [CT._satva(x) for x in pieces]
        pieces = [CT._vocalic_l(CT._hna_metathesis(CT._danda_fix(CT._visarga_ksha(CT._anusvara_m(x))))) for x in pieces]

        mono = max((CT._rep_depths(CT._aksharas(x))[0] for x in pieces), default=1)
        di = max((CT._rep_depths(CT._aksharas(x))[1] for x in pieces), default=1)
        pick = None
        if di >= 3:
            pick = next((k for k in ["prime_jaya", "prime_chata"]
                         if k in self._primes and self._primes[k].get("di_max", 0) >= di), None) \
                or next((k for k, v in self._primes.items() if isinstance(v, dict) and v.get("di_max", 0) >= di), None)
        if pick is None and mono >= 2 and "prime_mono" in self._primes \
                and self._primes["prime_mono"].get("mono_max", 0) >= mono:
            pick = "prime_mono"
        ref_len = ref["dur"]
        if pick:
            ref = self._ref(self._primes[pick]["wav"], self._primes[pick]["ref_text"])
            ref_len = ref["dur"]

        nsyll = [CT.n_aksharas(x) for x in pieces]
        gaps = [np.zeros(int(self.gap * SR_) + (int(self.gap_halant * SR_) if CT._ends_halant(p) else 0), np.float32)
                for p in pieces]
        segs = [[] for _ in self.vocs]
        for i, p in enumerate(pieces):
            fixd = (ref_len + nsyll[i] * ref_sps) if (ref_sps > 0 and nsyll) else None
            max_chars = int(len(ref["raw_text"].encode("utf-8")) / ref["dur"] * (25 - ref["dur"]))
            chunks = chunk_text(p, max_chars=max_chars)
            y = None
            for att in range(4):
                mel, draw, tests = None, self._rng(seed + att), []
                for c in chunks:  # his renderer's BigVGAN sees the last chunk's mel only
                    dur = (int(fixd * SR_ / HOP) if fixd is not None
                           else len(ref["wave"]) // HOP + int(len(ref["wave"]) // HOP / len(ref["text"].encode("utf-8"))
                                                             * len(c.encode("utf-8")) / spd))
                    mel = self.sample(ref, c, dur, draw)
                    w = self.vocos_wave(mel) if self.vocos_sess is not None else self.vocode(mel)
                    tests.append(w * (ref["rms"] / TARGET_RMS) if ref["rms"] < TARGET_RMS else w)
                y = self.vocode(mel)
                w = cross_fade(tests)
                if self.vocos_sess is not None and np.abs(w).max() > 1.5:
                    w = w / 32768.0
                if float(np.sqrt((w ** 2).mean())) > 0.04:
                    break
            for j, voc in enumerate(self.vocs):
                yj = y if j == 0 else self.vocode(mel, voc)
                mx = np.abs(yj).max()
                segs[j].append(yj / mx * 0.97 if mx > 1 else yj)
        slp = PT.align_slp1(padas[0])
        fric = bool(slp) and slp[0] in ("S", "z", "s", "h")
        halant = CT._ends_halant(pieces[-1])
        outs = []
        for sg in segs:
            if len(sg) == 1:
                outs.append((SR_, CT.gate(sg[0], fric=fric, halant=halant)))
                continue
            b, last = [], len(sg) - 1
            for i, s in enumerate(sg):
                b += [CT.gate(s, fric=(fric and i == 0), halant=(halant and i == last)),
                      gaps[i] if i < len(gaps) else gaps[-1]]
            outs.append((SR_, np.concatenate(b[:-1])))
        return outs


def chunk_text(text, max_chars=135):
    """f5_tts.infer.utils_infer.chunk_text, verbatim (as in Vāgdhenu's MLX backend)."""
    import re
    chunks, cur = [], ""
    for s in re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text):
        if len(cur.encode("utf-8")) + len(s.encode("utf-8")) <= max_chars:
            cur += s + " " if s and len(s[-1].encode("utf-8")) == 1 else s
        else:
            if cur:
                chunks.append(cur.strip())
            cur = s + " " if s and len(s[-1].encode("utf-8")) == 1 else s
    if cur:
        chunks.append(cur.strip())
    return chunks


def run(a):
    """`nandini onnx render --text ... --bank ...`: one verse to a wav, the meter as his website picks it."""
    import time
    import soundfile as sf
    if not a.text:
        raise SystemExit("--text is required")
    bank = Path(a.bank) / "bank.json"
    r = Renderer(a.onnx, bank, vocab_file(), nfe=a.steps, threads=a.threads)
    meter = a.meter
    if not meter:
        b = json.loads(bank.read_text(encoding="utf-8"))
        ms = [k for k, e in b.items() if not k.startswith("_") and isinstance(e, dict) and "wav" in e]
        alias = {k.lower(): k for k in ms} | {b[k]["wav"].removesuffix(".wav").lower(): k for k in ms}
        meter = alias.get((r.CT.detect_meter_key(a.text) or "").lower()) or \
            ("vasantatilakā" if "vasantatilakā" in ms else ms[0])
    t0 = time.time()
    sr, y = r.render_one(a.text, meter, seed=a.seed)
    sf.write(a.out, y, sr)
    print(f"{a.out}: {meter}, {len(y) / sr:.1f} s of audio in {time.time() - t0:.0f} s")


def istft(spec, n_fft=1024, hop=HOP):
    """torch.istft(spec, n_fft, hop, n_fft, hann_window(n_fft), center=True) for one (513, frames) spectrum: vocos's
    `center` head."""
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float64)  # periodic Hann
    frames = np.fft.irfft(spec.astype(np.complex128), n_fft, axis=0) * win[:, None]
    t = spec.shape[1]
    n = n_fft + hop * (t - 1)
    y, env = np.zeros(n), np.zeros(n)
    for k in range(t):
        y[k * hop:k * hop + n_fft] += frames[:, k]
        env[k * hop:k * hop + n_fft] += win ** 2
    s, e = n_fft // 2, n - n_fft // 2
    return (y[s:e] / np.where(env[s:e] > 1e-11, env[s:e], 1.0)).astype(np.float32)


def cross_fade(waves, dur_s=0.15):
    """infer_batch_process's cross-fade of a line's chunks."""
    final = waves[0]
    for nxt in waves[1:]:
        k = min(int(dur_s * SR_), len(final), len(nxt))
        if k <= 0:
            final = np.concatenate([final, nxt])
            continue
        final = np.concatenate([final[:-k], final[-k:] * np.linspace(1, 0, k) + nxt[:k] * np.linspace(0, 1, k),
                                nxt[k:]])
    return final
