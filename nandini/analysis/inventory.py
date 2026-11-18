"""The converter's failure inventory: N_CONVERSIONS conversions per clip, each scored as made ("raw") and after the
melody step ("melody"), per syllable, plus voice consistency. Measures only; `nandini generate` makes the dataset.

  nandini inventory convert --manifest M --out OUT --seed-vc DIR --seed-vc-python PY [--part i/P] [--clips N | --indices a,b]
  nandini inventory score   --manifest M --out OUT [--part i/P] [--clips N]
  nandini inventory report  --out OUT      (after every part has scored; writes OUT/report.md)

A clip's index is its fixed position in the whole dataset, and its conversion j uses seed SEED_STEP * j + index, so a
clip gets the same conversions in any manifest (and the same as nandini generate's attempts). Every step skips work
already on disk, so a stopped run restarts where it left off.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .. import config as C
from ..dataset import conv_path, melody_path, select, source


def convert(args):
    from ..converter import Converter
    m, part = select(args.manifest, args.part, args.indices, args.clips)
    conv = Converter(args.seed_vc, args.seed_vc_python, args.voice)
    log = Path(args.out) / f"convert_times_part{part}.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        for j in range(args.conversions):  # conversion 0 of every clip first, so a stopped run still has whole rounds
            for r in m.itertuples():
                out = conv_path(args.out, j, r.clip_id)
                t = time.time()
                made = conv.make(source(r.clip_id, args.dataset_dir), C.SEED_STEP * j + r.index, out)
                if made:
                    secs = time.time() - t
                    pd.DataFrame([{"clip_id": r.clip_id, "j": j, "seconds": round(secs, 2),
                                   "audio_s": round(sf.info(str(out)).duration, 2)}]).to_csv(
                        log, mode="a", header=not log.exists(), index=False)
                    print(f"c{j} {r.clip_id} {secs:.1f} s", flush=True)
    finally:
        conv.close()


def score(args):
    from ..asr import flagged_syllables, transcribe
    from ..melody import melody_step
    from ..sounds import syllable_classes
    from ..text import Verse
    from ..voice import profile
    m, part = select(args.manifest, args.part, args.indices, args.clips)
    out = Path(args.out)
    syl_f, file_f, emb_f = (out / f"{n}_part{part}.{e}" for n, e in (("syllables", "csv"), ("files", "csv"), ("embeddings", "npz")))
    done = set(pd.read_csv(file_f).clip_id) if file_f.exists() else set()
    emb = dict(np.load(emb_f)) if emb_f.exists() else {}
    if "voice_sample" not in emb:
        emb["voice_sample"], _ = profile(args.voice)
    for r in m.itertuples():
        if r.clip_id in done:
            continue
        paths = [conv_path(out, j, r.clip_id) for j in range(args.conversions)]
        if not all(p.exists() for p in paths):
            print(f"skip {r.clip_id}: conversions missing", flush=True)
            continue
        t0 = time.time()
        verse = Verse(r.text)
        classes = syllable_classes(verse)
        src = source(r.clip_id, args.dataset_dir)
        his_hyp = transcribe(src)
        his_flags = flagged_syllables(verse, his_hyp)
        files, syl = [], {"his": his_flags}
        e, v = profile(src)
        emb[f"{r.clip_id}|his"] = e
        files.append({"clip_id": r.clip_id, "j": -1, "kind": "his", "transcript": his_hyp, **v})
        k = C.K_SEMITONES[r.style]
        for j, p in enumerate(paths):
            mp = melody_path(out, j, r.clip_id)
            if not mp.exists():
                mp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(mp, melody_step(src, p, k), C.SR)
            for kind, path in (("raw", p), ("melody", mp)):
                hyp = transcribe(path)
                syl[(kind, j)] = flagged_syllables(verse, hyp)
                e, v = profile(path)
                emb[f"{r.clip_id}|{kind}|{j}"] = e
                files.append({"clip_id": r.clip_id, "j": j, "kind": kind, "transcript": hyp, **v})
        rows = []
        for s, (text, flags) in enumerate(classes):
            for j in range(args.conversions):
                rows.append({"clip_id": r.clip_id, "style": r.style, "family": r.family, "syllable": s, "text": text,
                             "j": j, "his_flag": bool(his_flags[s]), "raw_flag": bool(syl[("raw", j)][s]),
                             "melody_flag": bool(syl[("melody", j)][s]), **flags})
        pd.DataFrame(rows).to_csv(syl_f, mode="a", header=not syl_f.exists(), index=False)
        pd.DataFrame(files).to_csv(file_f, mode="a", header=not file_f.exists(), index=False)
        np.savez(emb_f, **emb)
        print(f"scored {r.clip_id} in {time.time() - t0:.0f} s", flush=True)


def report(args):
    from .inventory_report import write_report
    write_report(Path(args.out))
