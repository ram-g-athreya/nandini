"""The marking pass: for each kept verse of a run, what Su-śrotā hears on Vāgdhenu's recording and on Nandinī's kept
clip, which syllables it gets wrong on each, and roughly when each syllable is sung in each (for the listening page).

  nandini mark [--run out] [--manifest M] [--dataset-dir D] [--part i/P]

Reads RUN/kept_part*.csv and RUN/final/; writes RUN/marks/marks_part{i}.jsonl, one record per verse. Flags come from
the same greedy transcript and alignment as `nandini generate`, so each verse's error count must match the run's
(checked). Resumes: verses already marked are skipped.

Record keys: id (clip), his_hyp / her_hyp (transcripts of Vāgdhenu's recording / Nandinī's clip), his_f / her_f
(per syllable, "1" if wrong), his_t / her_t (syllable start times, s), his_dur / her_dur (s), syl (syllable spans in
the verse's SLP1).
"""
import json
import time
from pathlib import Path

import pandas as pd

from ..dataset import key, source


def read_kept(run):
    files = sorted(Path(run).glob("kept_part*.csv"))
    if not files:
        raise SystemExit(f"no kept_part*.csv in {run}: run `nandini generate` first")
    return pd.concat(pd.read_csv(f) for f in files).sort_values("index").reset_index(drop=True)


def read_marks(run):
    """Every verse's latest complete record."""
    marks = {}
    for f in sorted((Path(run) / "marks").glob("marks_part*.jsonl")):
        for line in f.open():
            m = json.loads(line)
            if "her_hyp" in m:
                marks[m["id"]] = m
    return marks


def run(args):
    import torch

    from ..asr import flagged_syllables, syllable_starts, transcribe_timed
    from ..text import Verse
    i, p = (int(x) for x in args.part.split("/"))
    torch.set_num_threads(max(1, 8 // p))
    kept = read_kept(args.run).iloc[i::p]
    texts = pd.read_csv(args.manifest).set_index("clip_id").text
    out = Path(args.run) / "marks" / f"marks_part{i}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(read_marks(args.run))
    with out.open("a") as f:
        for r in kept.itertuples():
            if r.clip_id in done:
                continue
            t0 = time.time()
            verse = Verse(texts[r.clip_id])
            rec, flags = {"id": r.clip_id}, {}
            for who, path in (("his", source(r.clip_id, args.dataset_dir)),
                              ("her", Path(args.run) / "final" / f"{key(r.clip_id)}.wav")):
                hyp, h, ht, dur = transcribe_timed(path)
                flags[who] = flagged_syllables(verse, hyp)
                rec[f"{who}_hyp"] = hyp
                rec[f"{who}_t"] = [round(float(t), 2) for t in syllable_starts(verse, h, ht, dur)]
                rec[f"{who}_dur"] = round(dur, 2)
            wrong = sum(bool(b and not a) for a, b in zip(flags["his"], flags["her"]))
            assert wrong == r.errors, (r.clip_id, wrong, r.errors)
            rec["his_f"] = "".join("1" if x else "0" for x in flags["his"])
            rec["her_f"] = "".join("1" if x else "0" for x in flags["her"])
            rec["syl"] = [[int(s), int(e)] for s, e in verse.spans]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{r.index} {r.clip_id} {r.errors} errors {time.time() - t0:.1f} s", flush=True)
