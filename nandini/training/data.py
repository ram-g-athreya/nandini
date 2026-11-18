"""The training data: each kept clip with Vāgdhenu's `model_text` of its verse, as his training read it.

  nandini training-data [--run out] [--kept KEPT.csv] [--audio-dir DIR] [--out training/dataset] [--text-only]

Runs in the training environment (external/train-env). For each kept verse (--kept, else the run folder's kept_rechecked.csv, else its kept_part*.csv):
the audio is <audio-dir>/<key>.wav (default <run>/final/), the text his `model_text` of the manifest's verse (punctuation
stripped, Kannada script, no sandhi), called from his prep_text.py in external/vagdhenu, not a copy. Stops with nothing
written if a clip is missing, isn't 24 kHz mono, falls outside IndicF5's 0.3–30 s, or has a character outside his
vocab.txt, except TRAIN_TEXT_ALLOW (vocalic ḷ, one word of one clip: his trainer reads it as a space, as on his data).

Writes OUT/training_text.csv (audio paths as given), and unless --text-only, IndicF5's dataset: OUT/raw.arrow (rows
audio_path, text, duration, with absolute paths, so it is made on the machine that trains) and OUT/duration.json, read
back and compared with what was meant to be written.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

from .. import config as C
from .. import external as X
from ..dataset import key


def vocab_file():
    v = X.require("vagdhenu") / "src" / "reference_bank" / "vocab.txt"
    got = hashlib.sha256(v.read_bytes()).hexdigest()
    if got != C.VOCAB_SHA256:
        sys.exit(f"{v}: sha256 {got}, expected {C.VOCAB_SHA256}")
    return v


def model_text_fn():
    sys.path.insert(0, str(X.require("vagdhenu") / "src"))
    from prep_text import model_text  # his code, unchanged
    return model_text


def kept_rows(run, kept):
    from ..dataset import kept_list
    return kept_list(run, kept)


def rows(a):
    model_text = model_text_fn()
    vocab = {line[:-1] for line in vocab_file().open(encoding="utf-8")}
    allow = set(C.TRAIN_TEXT_ALLOW)
    kept = kept_rows(a.run, a.kept)
    man = pd.read_csv(a.manifest).set_index("clip_id")
    audio = Path(a.audio_dir or Path(a.run) / "final")
    out, problems = [], []
    for r in kept.itertuples():
        p = audio / f"{key(r.clip_id)}.wav"
        if not p.exists():
            problems.append(f"missing audio: {p}")
            continue
        info = sf.info(str(p))
        dur = info.frames / info.samplerate
        if info.samplerate != C.SR or info.channels != 1:
            problems.append(f"{p.name}: {info.samplerate} Hz, {info.channels} channels")
        if not C.TRAIN_MIN_S <= dur <= C.TRAIN_MAX_S:
            problems.append(f"{p.name}: {dur:.2f} s is outside {C.TRAIN_MIN_S}–{C.TRAIN_MAX_S} s; IndicF5 would skip it")
        src = man.loc[r.clip_id, "text"]
        text = model_text(src)
        for c in sorted(set(text) & allow - vocab):
            print(f"allowed outside the vocab: U+{ord(c):04X} in {p.name} (read as a space by the trainer)")
        bad = sorted({c for c in text if c not in vocab and c not in allow})
        if bad:
            problems.append(f"{p.name}: not in vocab: {', '.join(f'U+{ord(c):04X}' for c in bad)}")
        out.append({"index": r.index, "clip_id": r.clip_id, "audio": str(p), "duration": dur, "source_text": src,
                    "text": text})
    if problems:
        print(f"{len(problems)} problem(s), nothing written:")
        print("\n".join(problems[:50]))
        sys.exit(1)
    return out


def write_arrow(rows_, out):
    from datasets import Dataset
    from datasets.arrow_writer import ArrowWriter
    data = [{"audio_path": str(Path(r["audio"]).resolve()), "text": r["text"], "duration": r["duration"]} for r in rows_]
    with ArrowWriter(path=str(out / "raw.arrow")) as w:
        for d in data:
            w.write(d)
        w.finalize()  # without it the file stays empty (datasets 2.14.4)
    durations = [d["duration"] for d in data]
    (out / "duration.json").write_text(json.dumps({"duration": durations}, ensure_ascii=False))
    back = Dataset.from_file(str(out / "raw.arrow"))  # as IndicF5's loader reads it
    if back.to_list() != data or json.loads((out / "duration.json").read_text())["duration"] != durations:
        sys.exit("read-back check FAILED: raw.arrow or duration.json differs from the rows written")


def run(a):
    r = rows(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(r)
    df.assign(duration=df.duration.round(4)).to_csv(out / "training_text.csv", index=False)  # paths as given
    if not a.text_only:
        write_arrow(r, out)
    chars = collections.Counter("".join(df.text))
    print(f"{len(df)} rows, {df.duration.sum() / 3600:.2f} h, clips {df.duration.min():.2f}–{df.duration.max():.2f} s, "
          f"{len(chars)} distinct characters, all in vocab or allowed -> {out}"
          f"{' (text only)' if a.text_only else ' (raw.arrow, duration.json)'}")
