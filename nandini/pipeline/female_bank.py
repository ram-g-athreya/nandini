"""The female reference bank: the same cuts as Vāgdhenu's bank, taken from the kept clips, so his renderer drives the
trained model with her voice (IndicF5 copies the reference's voice, melody and pace).

  nandini female-bank [--run out] [--kept K.csv] [--audio-dir D] [--out training/female_bank] [--dataset-dir D]
                      [--overrides manifests/female_bank_overrides.csv]

For each entry of his bank.json (external/vagdhenu, src/reference_bank; meters, prose and repeat primes): his cut is
located in his recording of the source verse by cross-correlation (where a clip name exists in both styles, the better
match wins), and the same window is cut from the kept clip of that verse (the melody step keeps his timing). bank.json
is copied with his text, so only the voice changes.

Each cut is scored on its own: his cut and hers transcribed alone, compared with the syllables whose start (on his
recording, from the timed transcript) falls inside the window; her errors are syllables wrong on hers and right on his.

--overrides (default: the release's three) changes an entry's attempt (melody/c<j>/ in the run folder) and/or takes the
verse's second half: from the half-verse pause (his start of syllable n // 2, moved to the quietest 20 ms nearby) to the
end of his voice (+0.15 s); its ref_text becomes the second half of his full_verse. An override whose attempt isn't on
disk is skipped, with a warning. Writes OUT/{bank.json, *.wav, cuts.csv}; never writes into an existing OUT.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate, resample_poly

from .. import config as C
from .. import external as X
from ..asr import flagged_syllables, half_cut, syllable_starts, transcribe, transcribe_timed
from ..dataset import key, melody_path, source
from ..text import Verse, deva

OVERRIDES = C.ROOT / "manifests" / "female_bank_overrides.csv"


def load(p):
    y, s = sf.read(p, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    return resample_poly(y, C.SR, s).astype(np.float32) if s != C.SR else y


def locate(ref, full, d=6):
    """Offset (samples) of ref inside full, and the normalised peak correlation (on a 4 kHz downsample)."""
    a, b = ref[::d], full[::d]
    if len(b) < len(a):
        return 0, 0.0
    c = correlate(b, a, mode="valid", method="fft")
    e = np.sqrt(np.convolve(b ** 2, np.ones(len(a)), mode="valid")) * np.linalg.norm(a) + 1e-9
    r = c / e
    k = int(np.argmax(r))
    return k * d, float(r[k])


def voice_end(y, db=-40):
    f = C.SR // 50
    rms = np.array([np.sqrt(np.mean(y[i:i + f] ** 2)) for i in range(0, len(y) - f, f // 2)])
    on = np.flatnonzero(20 * np.log10(rms + 1e-9) > 20 * np.log10(rms.max()) + db)
    return min(len(y), on[-1] * (f // 2) + f + int(0.15 * C.SR))


def hear(y):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, y, C.SR)
        return transcribe(f.name)


def window_verse(verse, starts, a, b):
    """The syllables whose start falls inside [a, b) samples (0.1 s tolerance), as a Verse of their own."""
    idx = [i for i, t in enumerate(starts) if a / C.SR - 0.1 <= t < b / C.SR - 0.1]
    return Verse("".join(deva(verse.slp[s:e]) for s, e in (verse.spans[i] for i in idx)))  # syllable by syllable


def score_cut(verse, starts, his, her, a, b):
    v = window_verse(verse, starts, a, b)
    his_bad = flagged_syllables(v, hear(his[a:b]))
    bad = flagged_syllables(v, hear(her[a:b])) & ~his_bad
    return v.n, int(his_bad.sum()), int(bad.sum())


def entries(bank):
    out = []
    for k, v in bank.items():
        if k == "repeat_primes":
            out += [(p, e, e["source"].split()[0], ("repeat_primes", p)) for p, e in v.items() if isinstance(e, dict)]
        elif isinstance(v, dict) and "wav" in v:
            out.append((k, v, v.get("clip_id") or v["source"].split()[0], (k,)))
    return out


def run(a):
    his_dir = X.require("vagdhenu") / "src" / "reference_bank"
    bank = json.loads((his_dir / "bank.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    man = pd.read_csv(a.manifest)
    man["cid"] = man.clip_id.str.split("/").str[-1].str.removesuffix(".wav")
    from ..dataset import kept_list
    kept = kept_list(a.run, a.kept).set_index("clip_id")
    audio = Path(a.audio_dir or Path(a.run) / "final")
    ov = pd.read_csv(a.overrides).set_index("entry") if a.overrides else pd.DataFrame()
    rows = []
    for name, e, cid, path in entries(bank):
        ref = load(his_dir / e["wav"])
        best = None
        for c in man[man.cid == cid].itertuples():
            y = load(source(c.clip_id, a.dataset_dir))
            off, r = locate(ref, y)
            if best is None or r > best[3]:
                best = (c, y, off, r)
        if best is None:
            print(f"{name}: source {cid} not in the manifest; skipped", flush=True)
            continue
        c, his, off, r = best
        verse = Verse(c.text)
        _, h, ht, dur = transcribe_timed(str(source(c.clip_id, a.dataset_dir)))
        starts = syllable_starts(verse, h, ht, dur)
        a0, b0, j, half, note = off, off + len(ref), int(kept.loc[c.clip_id, "kept_j"]), "same", ""
        her = load(audio / f"{key(c.clip_id)}.wav")
        if name in ov.index:
            o = ov.loc[name]
            p = melody_path(a.run, int(o.attempt), c.clip_id)
            if p.exists():
                her, j, half, note = load(p), int(o.attempt), o.half, o.reason
                if half == "second":
                    a0, b0 = half_cut(his, C.SR, float(starts[verse.n // 2])), voice_end(his)
            else:
                print(f"{name}: override attempt {o.attempt} not on disk ({p}); kept clip used", flush=True)
        n, hw, err = score_cut(verse, starts, his, her, a0, b0)
        sf.write(out / e["wav"], her[a0:b0], C.SR)
        node = bank
        for k in path:
            node = node[k]
        if half == "second":
            node["ref_text"] = node["full_verse"].split("।")[1].replace("॥", "").strip()
            node["dur_s"] = round((b0 - a0) / C.SR, 2)
        if note:
            node["female_note"] = f"attempt {j}, {half} half: {note}"
        rows.append({"entry": name, "index": int(c.index), "clip_id": c.clip_id, "attempt": j, "half": half,
                     "start_s": round(a0 / C.SR, 3), "end_s": round(b0 / C.SR, 3), "match_r": round(r, 3),
                     "syllables": n, "his_wrong_in_cut": hw, "errors_in_cut": err})
        print(rows[-1], flush=True)
    (out / "bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n")
    d = pd.DataFrame(rows)
    d.to_csv(out / "cuts.csv", index=False)
    print(f"{len(d)} entries -> {out}; errors in the cut: " +
          ", ".join(f"{k}: {v}" for k, v in d.errors_in_cut.value_counts().sort_index().items()))
