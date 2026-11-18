"""Re-check in halves: the verses whose attempts all missed the bar, scored again with each half verse heard alone.

  nandini recheck-halves [--run out] [--manifest M] [--dataset-dir D]

On some long verses Su-śrotā loses its place in a whole clip and misreads a stretch of it, on every attempt alike, so a
good clip can miss the bar. For each verse that missed after `nandini generate` (met is False in its kept_part*.csv),
Vāgdhenu's recording and every attempt on disk (OUT/melody/c<j>/) are cut at the half-verse pause and each half is
transcribed alone (asr.transcribe_halves; the cut is found as for `generate --scoring halves`). Errors are counted as
before, against his recording scored the same way. The selection rule is applied again to these scores: if an attempt
now has at most MAX_ERRORS, the first such attempt (in seed order) is kept instead; otherwise the verse keeps what
`generate` kept. Verses that met the bar are not touched.

Writes:
  OUT/recheck_halves.csv  every re-checked recording and attempt (who = his | attempt) and its errors in halves
  OUT/kept_rechecked.csv  the final kept list, one row per verse: scoring = whole | halves, and the whole-clip choice
                          beside it (whole_kept_j, whole_errors, whole_counted)
  OUT/final/<key>.wav     replaced by the newly kept attempt where it changed (the attempts stay in OUT/melody/)

`training-data` and `female-bank` read OUT/kept_rechecked.csv when it is there. A stopped run restarts where it left off.
"""
import shutil
from pathlib import Path

import pandas as pd

from .. import config as C
from ..dataset import key, melody_path, source

COLS = ["clip_id", "index", "style", "family", "syllables", "counted", "errors", "kept_j", "attempts", "met",
        "scoring", "whole_kept_j", "whole_errors", "whole_counted"]


def choose(errors):
    """The selection rule on {attempt: errors}: the first attempt with at most MAX_ERRORS, else the fewest (earliest on
    ties)."""
    for j in sorted(errors):
        if errors[j] <= C.MAX_ERRORS:
            return j
    return min(errors, key=lambda j: (errors[j], j))


def final_list(kept, rechecked):
    """kept: generate's kept rows; rechecked: recheck_halves.csv rows. Returns the final kept list (COLS)."""
    out = kept.sort_values("index").copy()
    out["scoring"] = "whole"
    out["whole_kept_j"], out["whole_errors"], out["whole_counted"] = out.kept_j, out.errors, out.counted
    att = rechecked[rechecked.who == "attempt"]
    for i, g in att.groupby("index"):
        errs = dict(zip(g.j.astype(int), g.errors.astype(int)))
        j = choose(errs)
        if errs[j] > C.MAX_ERRORS:
            continue  # still missed: keep generate's choice and scores
        row = out["index"] == i
        out.loc[row, ["kept_j", "errors", "counted", "met", "scoring"]] = [
            j, errs[j], int(g.counted.iloc[0]), True, "halves"]
    return out[COLS]


def run(a):
    from ..asr import flagged_syllables, half_point, transcribe_halves
    from ..text import Verse
    out = Path(a.run)
    parts = sorted(out.glob("kept_part*.csv"))
    if not parts:
        raise SystemExit(f"no kept_part*.csv in {out}: run `nandini generate` first")
    kept = pd.concat(pd.read_csv(p) for p in parts)
    texts = pd.read_csv(a.manifest).set_index("clip_id").text
    log = out / "recheck_halves.csv"
    done = set(pd.read_csv(log)["index"]) if log.exists() else set()
    todo = kept[~kept.met & ~kept["index"].isin(done)].sort_values("index")
    print(f"{int((~kept.met).sum())} verses missed the bar; {len(todo)} to re-check", flush=True)
    for r in todo.itertuples():
        verse = Verse(texts[r.clip_id])
        src = source(r.clip_id, a.dataset_dir)
        t = half_point(verse, src)
        his = flagged_syllables(verse, transcribe_halves(src, t))
        counted = int(sum(not h for h in his))
        rows = [{"index": r.index, "clip_id": r.clip_id, "who": "his", "j": "", "counted": counted,
                 "errors": int(sum(his))}]
        for j in range(C.N_CONVERSIONS):
            p = melody_path(out, j, r.clip_id)
            if not p.exists():
                continue
            flags = flagged_syllables(verse, transcribe_halves(str(p), t))
            rows.append({"index": r.index, "clip_id": r.clip_id, "who": "attempt", "j": j, "counted": counted,
                         "errors": int(sum(f and not h for f, h in zip(flags, his)))})
        pd.DataFrame(rows).to_csv(log, mode="a", header=not log.exists(), index=False)
        best = min(x["errors"] for x in rows[1:]) if len(rows) > 1 else None
        print(f"{r.clip_id}: {len(rows) - 1} attempts re-checked, fewest errors {best} (whole: {r.errors})", flush=True)
    rechecked = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["index", "who", "j", "counted", "errors"])
    final = final_list(kept, rechecked)
    final.to_csv(out / "kept_rechecked.csv", index=False)
    swapped = final[final.kept_j != final.whole_kept_j]
    for r in swapped.itertuples():
        shutil.copyfile(melody_path(out, r.kept_j, r.clip_id), out / "final" / f"{key(r.clip_id)}.wav")
    n = int((final.scoring == "halves").sum())
    print(f"kept_rechecked.csv: {int(final.met.sum())} of {len(final)} verses meet the bar; {n} by the re-check "
          f"({len(swapped)} with a different attempt, so final/ changed for them)", flush=True)
