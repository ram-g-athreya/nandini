"""Scores the checks after training (renders from `nandini speech-check render` / `trend-check render`). Runs in this
code's environment.

  nandini speech-check score [--out training/checks/speech]
  nandini trend-check  score [--out training/checks/trend] [--last 14800] [--before 12000] [--data training/dataset]
                             [--dataset-dir D]

speech check: Su-śrotā's character error rate against the verse text, per render; passes if every render is at most
SPEAKS_LIMIT (30%) at both cfg settings. It shows the model speaks, not how well (these are training verses).

trend check: per verse and checkpoint, the error rate against the text (primary) and voice consistency (secondary, no pass/fail):
the cosine between the render's speaker embedding (ECAPA, as `nandini inventory`) and the mean embedding of the
training clips. Vāgdhenu's recordings and the training clips of the same verses are scored too, as reference points.
Rule on the error rate, paired bootstrap over the 10 verses (10,000 resamples, seed 20261109), --last against --before:
still improving if the 95% interval of the change lies entirely below 0 and the mean drop is at least 0.5 points;
worse if it lies entirely above 0; otherwise levelled off. Writes OUT/scores.csv and OUT/report.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..dataset import source

SPEAKS_LIMIT = 30.0
BOOT, BOOT_SEED, MIN_DROP = 10000, 20261109, 0.5


def score_speaks(a):
    from ..asr import transcribe
    from ..pipeline.pitch_shift import cer
    out = Path(a.out)
    d = pd.read_csv(out / "renders.csv")
    # renders sit in OUT/render/ (older CSVs hold the rendering machine's absolute paths: use the file name only)
    d["cer"] = [cer(t, transcribe(str(out / "render" / Path(r).name))) for t, r in zip(d.source_text, d.render)]
    d.to_csv(out / "scores.csv", index=False)
    ok = bool((d.cer <= SPEAKS_LIMIT).all() and d.not_silent.all())
    lines = ["# Does the model speak?", "", f"Step {d.step.iloc[0]}. Bar: every render at most {SPEAKS_LIMIT:.0f}% "
             "character errors against the text, at both settings.", "", "| Verse | Setting | Seconds | CER % |",
             "|:--|:--|--:|--:|"]
    lines += [f"| `{r.clip_id}` | {r.setting} | {r.seconds:.1f} | {r.cer:.2f} |" for r in d.itertuples()]
    lines += ["", f"**{'PASSES' if ok else 'FAILS'}.** Training verses: this shows the model speaks, not that it "
              "generalises."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok


def boot(a, b, rng):
    diff = np.asarray(b, float) - np.asarray(a, float)
    m = rng.choice(diff, (BOOT, len(diff))).mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def her_mean(data, cache):
    """Mean ECAPA embedding over the training clips (cached)."""
    import librosa
    from ..voice import _embed
    if cache.exists():
        return np.load(cache)
    clips = pd.read_csv(Path(data) / "training_text.csv").audio
    e = np.stack([_embed(librosa.load(p, sr=16000, mono=True)[0]) for p in clips])
    m = e.mean(0)
    m /= np.linalg.norm(m)
    np.save(cache, m)
    return m


def score_web(a):
    import librosa
    from ..asr import transcribe
    from ..pipeline.pitch_shift import cer
    from ..voice import _embed
    out = Path(a.out)
    v = pd.read_csv(out / "verses.csv")
    text = pd.read_csv(C.ROOT / "manifests" / "all_clips.csv").set_index("clip_id").text
    train = pd.read_csv(Path(a.data) / "training_text.csv").set_index("clip_id").audio
    target = her_mean(a.data, out / "training_clips_mean_embedding.npy")
    steps = sorted({int(p.stem.split("_")[0]) for p in out.glob("*_*.wav") if p.stem.split("_")[0].isdigit()})

    def row(label, n, wav, t):
        voice = float(_embed(librosa.load(str(wav), sr=16000, mono=True)[0]) @ target)
        return {"step": label, "n": n, "cer": cer(t, transcribe(str(wav))), "voice": voice}

    rows = []
    for r in v.itertuples():
        t = text[r.clip_id]
        rows.append(row("vagdhenu", r.n, source(r.clip_id, a.dataset_dir), t))
        rows.append(row("training_clip", r.n, train[r.clip_id], t))
        rows += [row(s, r.n, out / f"{s}_{r.n}.wav", t) for s in steps]
        print(f"verse {r.n} scored", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "scores.csv", index=False)
    rng = np.random.default_rng(BOOT_SEED)
    s = d[~d.step.isin(["vagdhenu", "training_clip"])].astype({"step": int})
    first, last = (s[s.step == x].sort_values("n") for x in (a.before, a.last))
    dc, dv = boot(first.cer, last.cer, rng), boot(first.voice, last.voice, rng)
    if dc[2] < 0 and -dc[0] >= MIN_DROP:
        verdict = "still improving"
    elif dc[1] > 0:
        verdict = "worse"
    else:
        verdict = "levelled off"
    g = d.groupby("step", sort=False)[["cer", "voice"]].mean()
    names = {"vagdhenu": "Vāgdhenu's recording", "training_clip": "training clip"}
    lines = [f"# Checks after training: {len(v)} verses", "",
             f"His website path, seed {60}, the given reference bank, vocos. Per render in `scores.csv`.", "",
             f"**Step {a.before:,} → {a.last:,}: {verdict}** (character errors, paired bootstrap over the {len(v)} verses).", "",
             "| Checkpoint | CER %, mean | Voice cosine to the training clips, mean |", "|:--|--:|--:|"]
    lines += [f"| {names.get(str(k), f'step {int(k):,}' if str(k).isdigit() else k)} | {r.cer:.2f} | {r.voice:.3f} |"
              for k, r in g.iterrows()]
    lines += ["", "*Complete table.*", "",
              f"Change {a.before:,} → {a.last:,}: CER {dc[0]:+.2f} points [{dc[1]:+.2f}, {dc[2]:+.2f}]; voice "
              f"{dv[0]:+.3f} [{dv[1]:+.3f}, {dv[2]:+.3f}] (95% paired bootstrap).", "",
              f"Limits: training verses; one seed per render; {len(v)} verses show a large trend, not a small one; "
              "Su-śrotā's blind spots carry over."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run(a):
    {"speech": score_speaks, "trend": score_web}[a.what](a)
