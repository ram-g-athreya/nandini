"""Makes the dataset: for each verse, attempts until one has at most MAX_ERRORS errors, else the best of N_CONVERSIONS.

Attempt j: conversion with seed SEED_STEP * j + the verse's index (as in the inventory, so its conversions are the same
attempts), then the melody step, then Su-śrotā. An error is a syllable Su-śrotā gets right on Vāgdhenu's recording
and wrong on the attempt. Stop at the first attempt with at most MAX_ERRORS; otherwise keep the attempt with the fewest
errors (the earliest on ties).

--scoring halves scores every recording and attempt in two halves instead (asr.transcribe_halves): the cut is at the
start of the second half on Vāgdhenu's recording, moved to the quietest point nearby in each clip. Same errors, same
rule; only what the ASR hears changes.

  nandini generate --seed-vc DIR --seed-vc-python PY [--manifest M] [--out OUT] [--part i/P] [--indices a,b]
                   [--scoring whole|halves]

Writes OUT/attempts_part{i}.csv (every attempt), OUT/kept_part{i}.csv (one row per verse), OUT/final/<key>.wav (the
kept audio) and OUT/settings_part{i}.json (the settings the run used). Conversions and melody-step files are kept under
OUT/conversions/ and OUT/melody/. A stopped run restarts where it left off: finished verses are skipped and files
already on disk are reused.
"""
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

from .. import config as C
from ..dataset import conv_path, key, melody_path, select, source


def run(args):
    from ..asr import flagged_syllables, half_point, transcribe, transcribe_halves
    from ..converter import Converter
    from ..melody import melody_step
    from ..text import Verse
    m, part = select(args.manifest, args.part, args.indices, args.clips)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scoring = getattr(args, "scoring", "whole")
    record = {"settings": C.current(), "attempts": args.conversions, "voice": str(args.voice), "scoring": scoring}
    (out / f"settings_part{part}.json").write_text(json.dumps(record, indent=2) + "\n")
    att_f, kept_f = out / f"attempts_part{part}.csv", out / f"kept_part{part}.csv"
    done = set(pd.read_csv(kept_f).clip_id) if kept_f.exists() else set()
    conv = Converter(args.seed_vc, args.seed_vc_python, args.voice)
    try:
        for r in m.itertuples():
            if r.clip_id in done:
                continue
            verse = Verse(r.text)
            src = source(r.clip_id, args.dataset_dir)
            if scoring == "halves":
                t = half_point(verse, src)
                hear = lambda p: transcribe_halves(p, t)
            else:
                hear = transcribe
            his = flagged_syllables(verse, hear(src))
            counted = sum(not h for h in his)
            best = None
            for j in range(args.conversions):
                seed = C.SEED_STEP * j + r.index
                cp = conv_path(out, j, r.clip_id)
                mp = melody_path(out, j, r.clip_id)
                t0 = time.time()
                conv.make(src, seed, cp)
                t1 = time.time()
                if not mp.exists():
                    mp.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(mp, melody_step(src, cp, C.K_SEMITONES[r.style]), C.SR)
                flags = flagged_syllables(verse, hear(str(mp)))
                errors = sum(f and not h for f, h in zip(flags, his))
                row = {"clip_id": r.clip_id, "index": r.index, "style": r.style, "family": r.family, "j": j, "seed": seed,
                       "syllables": len(his), "counted": counted, "errors": errors,
                       "convert_s": round(t1 - t0, 1), "rest_s": round(time.time() - t1, 1)}
                pd.DataFrame([row]).to_csv(att_f, mode="a", header=not att_f.exists(), index=False)
                print(f"{r.clip_id} attempt {j}: {errors} errors ({row['convert_s']} s + {row['rest_s']} s)", flush=True)
                if best is None or errors < best["errors"]:
                    best = row
                if errors <= C.MAX_ERRORS:
                    break
            final = out / "final" / f"{key(best['clip_id'])}.wav"
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(melody_path(out, best["j"], r.clip_id), final)
            kept = {k: best[k] for k in ("clip_id", "index", "style", "family", "syllables", "counted", "errors")}
            kept |= {"kept_j": best["j"], "attempts": j + 1, "met": best["errors"] <= C.MAX_ERRORS}
            pd.DataFrame([kept]).to_csv(kept_f, mode="a", header=not kept_f.exists(), index=False)
            print(f"kept {r.clip_id}: attempt {best['j']} of {j + 1}, {best['errors']} errors", flush=True)
    finally:
        conv.close()

