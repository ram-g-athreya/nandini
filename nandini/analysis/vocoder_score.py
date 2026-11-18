"""Scores for the vocoder fine-tune. Runs in this code's environment (Su-śrotā).

  nandini vocoder score --align [--out training/vocoder/align]
  nandini vocoder score --decoded vocos=DIR his=DIR fine-tune=DIR --verses VERSES_CSV --out DIR

--align: the check before training (from `nandini vocoder align`). Passes if his BigVGAN's decodes of the
         training-style mels keep the words (Su-śrotā's mean character errors over the clips within VOCODER_CER_MARGIN
         points of the real clips'; one character is about a point on a verse, so the margin applies to the
         mean) and show his echo on every clip (above the echo rule's per-clip limit). That shows the mels
         match the audio and the pairs are what his vocoder expects.
--decoded: the pass rule, on the same saved mels decoded by each vocoder (`nandini vocoder decode`). Per render: echo
         and shiver (the held note found on the vocos decode, measured at the same times in each) and character errors
         against the verse text. The fine-tune passes if (1) the echo rule holds (mean <= ECHO_MEAN_MAX, every render
         <= ECHO_EACH_MAX); (2) its shiver is below vocos's (95% paired bootstrap interval of the difference entirely
         below 0); (3) its mean character errors are at most vocos's + VOCODER_CER_MARGIN points.
"""
from pathlib import Path

import pandas as pd

from .. import config as C
from .vocoder_measures import boot, echo, echo_ok, held_note, shiver


def _load(p):
    import librosa
    return librosa.load(str(p), sr=24000, mono=True)[0]


def score_align(a):
    from ..asr import transcribe
    from ..pipeline.pitch_shift import cer
    out = Path(a.out)
    d = pd.read_csv(out / "align.csv")
    d["cer_real"] = [cer(t, transcribe(str(out / f"{n}__real.wav"))) for t, n in zip(d.source_text, d.name)]
    d["cer_his"] = [cer(t, transcribe(str(out / f"{n}__his.wav"))) for t, n in zip(d.source_text, d.name)]
    d["echo_shows"] = d.echo_his > C.ECHO_EACH_MAX
    d.to_csv(out / "align_scores.csv", index=False)
    words = bool(d.cer_his.mean() <= d.cer_real.mean() + C.VOCODER_CER_MARGIN)
    ok = words and bool(d.echo_shows.all())
    lines = ["# Before training: his BigVGAN on training-style mels", "",
             "| Clip | Seconds | Lag, samples | CER real % | CER his decode % | Echo real | Echo his decode |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    lines += [f"| `{r.clip_id}` | {r.seconds:.1f} | {r.lag} | {r.cer_real:.2f} | {r.cer_his:.2f} | {r.echo_real:.4f} | "
              f"{r.echo_his:.4f} |" for r in d.itertuples()]
    lines += ["", f"**{'PASSES' if ok else 'FAILS'}:** words kept, mean {d.cer_his.mean():.2f}% against "
              f"{d.cer_real.mean():.2f}% (margin {C.VOCODER_CER_MARGIN} points: {'yes' if words else 'no'}); his echo "
              f"shows on every clip ({'yes' if d.echo_shows.all() else 'no'})."]
    (out / "align_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok


def score_decoded(a):
    from ..asr import transcribe
    from ..pipeline.pitch_shift import cer
    dirs = dict(x.split("=", 1) for x in a.decoded)
    assert {"vocos", "fine-tune"} <= set(dirs), "--decoded needs vocos=DIR and fine-tune=DIR (his=DIR optional)"
    v = pd.read_csv(a.verses)
    text = pd.read_csv(C.ROOT / "manifests" / "all_clips.csv").set_index("clip_id").text
    names = sorted(p.name for p in Path(dirs["vocos"]).glob("*.wav"))
    rows = []
    for name in names:
        n = int(name.rsplit("_", 1)[1].removesuffix(".wav"))
        t = text[v.set_index("n").clip_id[n]]
        y0 = _load(Path(dirs["vocos"]) / name)
        _, t0, t1 = held_note(y0)
        for k, d in dirs.items():
            y = y0 if k == "vocos" else _load(Path(d) / name)
            rows.append({"render": name, "n": n, "vocoder": k, "held_s": round(t1 - t0, 2), "echo": echo(y),
                         "shiver": shiver(y, t0, t1), "cer": cer(t, transcribe(str(Path(d) / name)))})
        print(f"{name} scored", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "scores.csv", index=False)
    w = {k: d[d.vocoder == k].sort_values("render").reset_index(drop=True) for k in dirs}
    ft, vc = w["fine-tune"], w["vocos"]
    r1 = echo_ok(ft.echo, C.ECHO_MEAN_MAX, C.ECHO_EACH_MAX)
    sh = boot(vc.shiver, ft.shiver, C.VOCODER_BOOT, C.VOCODER_BOOT_SEED)
    r2 = sh[2] < 0
    r3 = ft.cer.mean() <= vc.cer.mean() + C.VOCODER_CER_MARGIN
    g = d.groupby("vocoder", sort=False)[["echo", "shiver", "cer"]].agg(["mean", "max"])
    lines = [f"# The vocoder fine-tune on {len(names)} renders", "",
             "| Vocoder | Echo, mean | Echo, highest | Shiver, mean | CER %, mean |", "|:--|--:|--:|--:|--:|"]
    lines += [f"| {k} | {g.loc[k, ('echo', 'mean')]:.4f} | {g.loc[k, ('echo', 'max')]:.4f} | "
              f"{g.loc[k, ('shiver', 'mean')]:.3f} | {g.loc[k, ('cer', 'mean')]:.2f} |" for k in dirs]
    lines += ["", "*Per render in `scores.csv`.*", "",
              f"1. Echo rule (mean ≤ {C.ECHO_MEAN_MAX}, every render ≤ {C.ECHO_EACH_MAX}): **{'met' if r1 else 'not met'}**.",
              f"2. Shiver below vocos's: change {sh[0]:+.3f} [{sh[1]:+.3f}, {sh[2]:+.3f}] (95% paired bootstrap): "
              f"**{'met' if r2 else 'not met'}**.",
              f"3. Character errors at most vocos's + {C.VOCODER_CER_MARGIN}: {ft.cer.mean():.2f} against "
              f"{vc.cer.mean():.2f}: **{'met' if r3 else 'not met'}**.", "",
              f"**The fine-tune {'PASSES' if r1 and r2 and r3 else 'FAILS'}.**"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run(a):
    score_align(a) if a.align else score_decoded(a)
