"""Su-śrotā Sanskrit ASR (greedy CTC on the Sanskrit token slice), and which syllables a transcript gets wrong."""
import os
import tarfile
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from . import config as C
from .text import to_slp1

# The model handles many languages from one shared vocabulary. Only its Sanskrit tokens are used here.
OFF, V, BLANK = 4096, 256, 5632  # the Sanskrit slice of the aggregate vocabulary (model card)
_model = None


def _load():
    global _model
    if _model is None:
        import nemo.collections.asr as na
        from omegaconf import OmegaConf
        path = hf_hub_download(C.ASR_REPO, C.ASR_FILE, revision=C.ASR_REV)
        # Saved by a NeMo fork. Upstream calls its tokenizer type "agg", and doesn't know its multi-softmax flags.
        with tarfile.open(path) as t:
            cfg = OmegaConf.create(t.extractfile("./model_config.yaml").read().decode())
        cfg.tokenizer.type = "agg"

        def strip(node):
            if OmegaConf.is_dict(node):
                for k in [k for k in node.keys() if k in ("multisoftmax", "multilingual", "language_keys")]:
                    del node[k]
                for k in node.keys():
                    strip(node[k])
        strip(cfg)
        override = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        OmegaConf.save(cfg, override.name)
        _model = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            path, map_location="cpu", override_config_path=override.name, strict=False).eval()
    return _model


def transcribe(path):
    m = _load()
    wav, _ = librosa.load(path, sr=16000, mono=True)
    with torch.no_grad():
        enc, _ = m.forward(input_signal=torch.tensor(wav).unsqueeze(0), input_signal_length=torch.tensor([len(wav)]))
        lp = m.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    # Greedy decoding: take the likeliest token per frame, then collapse repeats and drop blanks.
    ids = lp[:, [BLANK] + list(range(OFF, OFF + V))].argmax(1)
    sub = m.tokenizer.tokenizers_dict["sa"]
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0:
            out.append(sub.ids_to_tokens([i - 1])[0])
        prev = i
    return "".join(out).replace("▁", " ").strip()


def half_cut(y, sr, t):
    """Sample at which to cut a verse in two: the middle of the quietest 20 ms frame (10 ms steps) between 0.4 s before
    and 0.1 s after t, the time the second half should start."""
    f = int(0.02 * sr)
    a, b = max(0, int((t - 0.4) * sr)), min(len(y) - f, int((t + 0.1) * sr))
    rms = [np.sqrt(np.mean(y[s:s + f] ** 2)) for s in range(a, b, f // 2)]
    return a + int(np.argmin(rms)) * (f // 2) + f // 2


def half_point(verse, path):
    """Time (s) at which the verse's second half starts on a recording: the start of syllable n // 2, from the timed
    transcript (the half-verse pause in every meter of this dataset)."""
    _, h, times, dur = transcribe_timed(path)
    return float(syllable_starts(verse, h, times, dur)[verse.n // 2])


def transcribe_halves(path, t):
    """As transcribe, but the clip is cut in two near t (half_cut), each half transcribed alone, and the two transcripts
    joined. On whole clips the ASR sometimes drops a stretch of a long verse; halves are short enough that it doesn't."""
    y, sr = sf.read(path, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    cut = half_cut(y, sr, t)
    out = []
    for z in (y[:cut], y[cut:]):
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, z, sr)
            out.append(transcribe(tmp))
        finally:
            os.remove(tmp)
    return " ".join(out)


def flagged_syllables(verse, hyp):
    """Per syllable: wrong in this transcript, by character alignment with the verse."""
    # Edit-distance alignment of the transcript with the verse, then a walk back through it. A character that was changed or
    # dropped marks its syllable as wrong, and so does an extra character inserted next to it.
    r = verse.chars
    h = "".join(c for c in to_slp1(hyp) if c not in " '")
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    bad = np.zeros(n, dtype=bool)
    inserted_at = np.zeros(n + 1, dtype=bool)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            bad[i - 1] |= r[i - 1] != h[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            bad[i - 1] = True
            i -= 1
        else:
            inserted_at[i] = True
            j -= 1
    flags = np.zeros(verse.n, dtype=bool)
    for k in range(n):
        if bad[k]:
            flags[verse.char_syl[k]] = True
    for k in range(n + 1):
        if inserted_at[k]:
            flags[verse.char_syl[k - 1] if k > 0 else 0] = True
    return flags


def _clean(s):
    return "".join(c for c in to_slp1(s) if c not in " '")


def transcribe_timed(path):
    """As transcribe, also returning the transcript in SLP1 without spaces, each of its characters' time (s), and the
    audio's duration. A character's time is the CTC frame that emitted its token (one frame is about 80 ms)."""
    m = _load()
    wav, _ = librosa.load(path, sr=16000, mono=True)
    with torch.no_grad():
        enc, _ = m.forward(input_signal=torch.tensor(wav).unsqueeze(0), input_signal_length=torch.tensor([len(wav)]))
        lp = m.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    ids = lp[:, [BLANK] + list(range(OFF, OFF + V))].argmax(1)
    stride = len(wav) / 16000 / len(ids)
    sub = m.tokenizer.tokenizers_dict["sa"]
    toks, frames, prev = [], [], -1
    for f, i in enumerate(ids):
        i = int(i)
        if i != prev and i != 0:
            toks.append(sub.ids_to_tokens([i - 1])[0])
            frames.append(f)
        prev = i
    hyp = "".join(toks).replace("▁", " ").strip()
    h = _clean(hyp)
    times = np.full(len(h), np.nan)
    done = 0
    for k in range(len(toks)):
        n = min(len(_clean("".join(toks[:k + 1]).replace("▁", " "))), len(h))
        times[done:n] = frames[k] * stride
        done = max(done, n)
    return hyp, h, times, len(wav) / 16000


def _align(r, h):
    """Edit-distance table between the verse's characters r and the transcript's h."""
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d


def heard(verse, hyp):
    """Per syllable, the transcript's characters aligned to it (SLP1), by the same alignment as flagged_syllables:
    changed characters go to their syllable, inserted ones to the syllable before."""
    r, h = verse.chars, _clean(hyp)
    d = _align(r, h)
    out = [""] * verse.n
    i, j = len(r), len(h)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            s = verse.char_syl[i - 1]
            out[s] = h[j - 1] + out[s]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            s = verse.char_syl[i - 1] if i > 0 else 0
            out[s] = h[j - 1] + out[s]
            j -= 1
    return out


def syllable_starts(verse, h, htimes, dur):
    """Start time of each syllable, from the alignment of the verse with a timed transcript (transcribe_timed).
    Syllables with no aligned character are interpolated. Approximate; for following along while listening."""
    r = verse.chars
    d = _align(r, h)
    rt = np.full(len(r), np.nan)
    i, j = len(r), len(h)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            rt[i - 1] = htimes[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    st = np.full(verse.n, np.nan)
    for k in range(len(r)):
        s = verse.char_syl[k]
        if not np.isnan(rt[k]) and (np.isnan(st[s]) or rt[k] < st[s]):
            st[s] = rt[k]
    ok = ~np.isnan(st)
    if ok.sum() == 0:
        return np.linspace(0, dur, verse.n, endpoint=False)
    x = np.arange(verse.n)
    return np.maximum.accumulate(np.interp(x, x[ok], st[ok]))
