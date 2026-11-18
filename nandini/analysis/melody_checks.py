"""Checks behind the melody step's design, on the inventory's conversions (`nandini inventory convert` and `score`).

  nandini melody-checks timing        does a conversion's timing drift from Vāgdhenu's inside the verse?
  nandini melody-checks voicing       where the melody step imposes Vāgdhenu's voicing on the conversion
  nandini melody-checks loudness      how much the melody step's loudness matching changes each conversion
  nandini melody-checks loudness-cap  the melody step without its boost cap against with it (why the boost is capped)
  options: [--inventory inventory] [--manifest M] [--dataset-dir D] [--listen]  -> INV/melody_checks/

timing: WORLD's voicing on Vāgdhenu's recording and on each conversion, compared frame by frame with the conversion
shifted by every 5 ms from -40 to +40 ms; the shift with the least disagreement (frames where exactly one is voiced,
among frames where either is) is the offset. The rule set beforehand: alignment is fine if the offset is within 20 ms in
at least 90% of conversions. On the released clips it isn't met (57%), but the measure can't resolve offsets on chant:
Vāgdhenu's recordings are almost entirely voiced, with few voicing boundaries to align on, so the disagreement barely
changes across ±40 ms (the report shows this). The melody step adds no alignment because the converter keeps the
source's timing by design: Seed-VC stretches the source's content features to the source's exact length before
generating (length_adjust 1.0).
voicing: frames where Vāgdhenu is voiced and the conversion isn't (the melody step puts a buzz under an unvoiced
sound), and the reverse (it turns a voiced sound into noise), with the longest stretch of each.
loudness: the melody step's energy gain per frame, relative to the clip's median gain, on frames that come out audible.
loudness-cap: makes every melody-step clip again without the cap, scores it with Su-śrotā, and compares with the
inventory's clips (made with the cap): how many come out well below the −20 dBFS target (a sign of spikes the level
setting pulled everything down for), and how many more syllables go wrong. The cap is kept if at most 5% of capped clips
are more than 3 dB low and the upper end of the 95% interval for the change in syllables wrong is at most +0.5 points.
--listen also cuts short listening clips around the worst stretches (Vāgdhenu, the conversion, the melody step).
"""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..dataset import conv_path, key, melody_path, source

MAX_MS, BAR_MS, BAR_SHARE = 40, 20, 0.90
AUDIBLE_DB = 30.0
DB = 10 / np.log(10)  # natural-log power ratio -> dB
WINDOW_S = 0.4
CAP_BOOT, CAP_SEED = 2000, 20261111


def _load(path):
    import librosa
    return librosa.load(path, sr=C.SR, mono=True)[0]


def _voicing(y):
    import pyworld as pw
    f0, _ = pw.harvest(y.astype(np.float64), C.SR, f0_floor=C.F0_FLOOR_HZ, f0_ceil=C.F0_CEILING_HZ,
                       frame_period=C.STEP_MS)
    return f0 > 0


def _rows(a):
    m = pd.read_csv(a.manifest)
    return [dict(r._asdict(), src=str(source(r.clip_id, a.dataset_dir)), inv=a.inventory) for r in m.itertuples()]


def _parallel(fn, rows):
    with ProcessPoolExecutor(max_workers=6) as ex:
        return list(ex.map(fn, rows))


def _out(a):
    out = Path(a.inventory) / "melody_checks"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _by_style(df):
    return list(df.groupby("style")) + [("all", df)]


# ---------- timing ----------

SHIFTS = np.arange(-MAX_MS, MAX_MS + 1, int(C.STEP_MS)) // int(C.STEP_MS)  # in frames


def _disagreement(his, her, s):
    """Her voicing shifted by s frames (s > 0: her audio is late, so her frame t + s is compared with his frame t)."""
    t = np.arange(len(his))
    u = t + s
    ok = (u >= 0) & (u < len(her))
    a, b = his[t[ok]], her[u[ok]]
    either = a | b
    return float((a ^ b)[either].mean()) if either.any() else np.nan


def _offsets(his, her):
    d = np.array([_disagreement(his, her, s) for s in SHIFTS])
    best = min(range(len(SHIFTS)), key=lambda i: (d[i], abs(SHIFTS[i])))
    return int(SHIFTS[best] * C.STEP_MS), d[best], d[list(SHIFTS).index(0)]


def _timing_one(row):
    his = _voicing(_load(row["src"]))
    out = []
    for p in sorted((Path(row["inv"]) / "conversions").glob(f"c*/{key(row['clip_id'])}.wav")):
        off, d_best, d0 = _offsets(his, _voicing(_load(p)))
        out.append({"clip_id": row["clip_id"], "style": row["style"], "family": row["family"],
                    "j": int(p.parent.name[1:]), "offset_ms": off, "disagree_at_0": round(d0, 4),
                    "disagree_at_best": round(d_best, 4)})
    return out


def timing(a):
    out = _out(a)
    df = pd.DataFrame([x for part in _parallel(_timing_one, _rows(a)) for x in part]).sort_values(["style", "clip_id", "j"])
    df.to_csv(out / "timing.csv", index=False)
    within = df.offset_ms.abs() <= BAR_MS
    summ = pd.DataFrame([{"Style": st, "Conversions": len(g),
                          f"Offset within {BAR_MS} ms, %": round(100 * (g.offset_ms.abs() <= BAR_MS).mean(), 1),
                          "Offset 0 ms, %": round(100 * (g.offset_ms == 0).mean(), 1),
                          "Median |offset|, ms": float(g.offset_ms.abs().median()),
                          "Largest |offset|, ms": int(g.offset_ms.abs().max()),
                          "Voicing disagreement at 0 ms, median %": round(100 * g.disagree_at_0.median(), 1),
                          "Voicing disagreement at 0 ms, 90th pct %": round(100 * g.disagree_at_0.quantile(0.9), 1)}
                         for st, g in _by_style(df)])
    dist = df.offset_ms.value_counts().sort_index()
    ok = within.mean() >= BAR_SHARE
    gain = 100 * (df.disagree_at_0 - df.disagree_at_best)
    spread = df.groupby("clip_id").offset_ms.agg(lambda x: x.max() - x.min())
    edge = int((df.offset_ms.abs() == MAX_MS).sum())
    L = ["# Timing inside the clip (voicing agreement)", "",
         f"{len(df)} conversions of {df.clip_id.nunique()} verses. Offset: the shift of the conversion's voicing (every "
         f"{int(C.STEP_MS)} ms, −{MAX_MS} to +{MAX_MS} ms) that best matches Vāgdhenu's; positive means the conversion "
         "runs late. Disagreement: share of frames where exactly one of the two is voiced, among frames where at least "
         "one is.", "", summ.to_markdown(index=False), "",
         pd.DataFrame({"Offset, ms": dist.index, "Conversions": dist.values}).to_markdown(index=False), "",
         f"**Rule:** offset within {BAR_MS} ms in at least {int(BAR_SHARE * 100)}% of conversions → "
         f"{'**met**' if ok else '**not met**'} ({100 * within.mean():.1f}%).", "",
         "**Can the measure resolve offsets?** The best shift lowers disagreement by a median of "
         f"{gain.median():.2f} points (from {100 * df.disagree_at_0.median():.1f}% at 0 ms); the {C.N_CONVERSIONS} "
         f"conversions of one verse give offsets spread over a median of {spread.median():.0f} ms; {edge} conversions sit "
         f"at the edge of the range (±{MAX_MS} ms). A small gain, a wide spread and many offsets at the edge mean the "
         "voicing curve is nearly flat, so the offsets say little about drift. The melody step adds no alignment: the "
         "converter builds each moment of its output from the same moment of the source.", ""]
    (out / "timing.md").write_text("\n".join(L))
    print("\n".join(L))


# ---------- listening clips ----------

def _excerpt(path, start_s, dur_s, pad=0.6):
    y = _load(path)
    s, e = max(0, int((start_s - pad) * C.SR)), int((start_s + dur_s + pad) * C.SR)
    y = y[s:e]
    return y / max(1e-9, np.abs(y).max()) * 0.5


def _listen(a, picks, name_of, folder):
    """One file per pick: Vāgdhenu's recording, the conversion, then the melody step's output (and any extra paths),
    each the same stretch, with a short pause between."""
    import soundfile as sf
    d = _out(a) / folder
    d.mkdir(exist_ok=True)
    gap = np.zeros(int(0.8 * C.SR))
    for i, (r, start, dur, extra) in enumerate(picks, 1):
        paths = [source(r.clip_id, a.dataset_dir), conv_path(a.inventory, r.j, r.clip_id),
                 melody_path(a.inventory, r.j, r.clip_id), *extra]
        parts = [_excerpt(p, start, dur) for p in paths]
        sound = np.concatenate([x for p in parts for x in (p, gap)][:-1])
        sf.write(d / f"{i}_{name_of(r)}_{key(r.clip_id).split('__', 1)[1]}_c{r.j}.wav", sound, C.SR)
    print(f"{len(picks)} listening clips -> {d}")


def _worst(df, col, n, ascending=False):
    picks, seen = [], set()
    for r in df.sort_values(col, ascending=ascending).itertuples():
        if r.clip_id not in seen:
            seen.add(r.clip_id)
            picks.append(r)
        if len(picks) == n:
            break
    return picks


# ---------- voicing ----------

def _runs(mask):
    """(start, length) of each run of True, in frames."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s, e - s))


def _voicing_one(row):
    his = _voicing(_load(row["src"]))
    out = []
    for j in range(C.N_CONVERSIONS):
        her = _voicing(_load(conv_path(row["inv"], j, row["clip_id"])))
        her = her[:len(his)] if len(her) >= len(his) else np.concatenate([her, np.repeat(her[-1:], len(his) - len(her))])
        either = his | her
        c1, c2 = his & ~her, her & ~his
        r1, r2 = _runs(c1), _runs(c2)
        l1 = max(r1, key=lambda x: x[1]) if r1 else (0, 0)
        l2 = max(r2, key=lambda x: x[1]) if r2 else (0, 0)
        out.append({"clip_id": row["clip_id"], "style": row["style"], "j": j,
                    "vagdhenu_voiced_pct": round(100 * his.mean(), 1), "conversion_voiced_pct": round(100 * her.mean(), 1),
                    "case1_pct": round(100 * c1[either].mean(), 2), "case2_pct": round(100 * c2[either].mean(), 2),
                    "case1_longest_ms": int(l1[1] * C.STEP_MS), "case1_at_s": round(l1[0] * C.STEP_MS / 1000, 2),
                    "case2_longest_ms": int(l2[1] * C.STEP_MS), "case2_at_s": round(l2[0] * C.STEP_MS / 1000, 2)})
    return out


def voicing(a):
    out = _out(a)
    df = pd.DataFrame([x for part in _parallel(_voicing_one, _rows(a)) for x in part])
    df.to_csv(out / "voicing.csv", index=False)
    summ = pd.DataFrame([{"Style": st, "Conversions": len(g),
                          "Vāgdhenu voiced, conversion not: median % of frames": round(g.case1_pct.median(), 2),
                          "longest stretch, median ms": float(g.case1_longest_ms.median()),
                          "longest stretch, max ms": int(g.case1_longest_ms.max()),
                          "Conversion voiced, Vāgdhenu not: median % of frames": round(g.case2_pct.median(), 2),
                          "longest stretch, median ms ": float(g.case2_longest_ms.median()),
                          "longest stretch, max ms ": int(g.case2_longest_ms.max())} for st, g in _by_style(df)])
    L = ["# Where the melody step imposes Vāgdhenu's voicing", "",
         "Per frame (WORLD harvest, as in the melody step): Vāgdhenu's voicing, which the melody step imposes, against "
         "the conversion's own. Percentages are of frames where either is voiced.", "",
         summ.to_markdown(index=False), "", "*Per conversion: `voicing.csv`.*", ""]
    (out / "voicing.md").write_text("\n".join(L))
    print("\n".join(L))
    if a.listen:
        picks = [(r, r.case1_at_s, r.case1_longest_ms / 1000, []) for r in _worst(df, "case1_longest_ms", 3)]
        picks += [(r, r.case2_at_s, r.case2_longest_ms / 1000, []) for r in _worst(df, "case2_longest_ms", 1)]
        _listen(a, picks, lambda r: "voicing", "voicing_listen")


# ---------- loudness ----------

def _loudness_one(row):
    from scipy.ndimage import gaussian_filter1d

    from ..melody import _fit, _world
    _, sp, _ = _world(row["src"])
    his_db = DB * np.log(sp.sum(1) + 1e-16)
    audible = his_db >= his_db.max() - AUDIBLE_DB
    out = []
    for j in range(C.N_CONVERSIONS):
        _, spc, _ = _world(conv_path(row["inv"], j, row["clip_id"]))
        spc = _fit(spc, len(sp))
        her_db = DB * np.log(spc.sum(1) + 1e-16)
        gain = DB * gaussian_filter1d(np.log(sp.sum(1) + 1e-16) - np.log(spc.sum(1) + 1e-16), C.ENERGY_SMOOTHING_FRAMES)
        local = gain - np.median(gain[audible])
        la = np.where(audible, local, np.nan)
        i_up, i_dn = int(np.nanargmax(la)), int(np.nanargmin(la))
        out.append({"clip_id": row["clip_id"], "style": row["style"], "j": j,
                    "loudness_corr": round(float(np.corrcoef(his_db[audible], her_db[audible])[0, 1]), 3),
                    "median_abs_db": round(float(np.median(np.abs(local[audible]))), 2),
                    "p95_abs_db": round(float(np.percentile(np.abs(local[audible]), 95)), 2),
                    "max_boost_db": round(float(la[i_up]), 1), "boost_at_s": round(i_up * C.STEP_MS / 1000, 2),
                    "max_cut_db": round(float(la[i_dn]), 1), "cut_at_s": round(i_dn * C.STEP_MS / 1000, 2)})
    return out


def loudness(a):
    out = _out(a)
    df = pd.DataFrame([x for part in _parallel(_loudness_one, _rows(a)) for x in part])
    df.to_csv(out / "loudness.csv", index=False)
    summ = pd.DataFrame([{"Style": st, "Conversions": len(g),
                          "Loudness correlation, median": round(g.loudness_corr.median(), 2),
                          "Gain applied, median dB": round(g.median_abs_db.median(), 1),
                          "Gain, 95th pct, median dB": round(g.p95_abs_db.median(), 1),
                          "Largest boost, median dB": round(g.max_boost_db.median(), 1),
                          "Largest boost, max dB": round(g.max_boost_db.max(), 1),
                          "Largest cut, median dB": round(g.max_cut_db.median(), 1),
                          "Largest cut, min dB": round(g.max_cut_db.min(), 1)} for st, g in _by_style(df)])
    L = ["# How much the melody step's loudness matching changes each conversion", "",
         "The melody step scales each frame of the conversion by Vāgdhenu's frame energy over its own, smoothed. Here "
         "that gain, relative to the clip's median gain, on frames that come out audible (Vāgdhenu's energy within "
         f"{AUDIBLE_DB:.0f} dB of his loudest). Without a cap; the melody step caps boosts at "
         f"{C.ENERGY_MAX_BOOST_DB:g} dB.", "", summ.to_markdown(index=False), "",
         "*Per conversion: `loudness.csv`.*", ""]
    (out / "loudness.md").write_text("\n".join(L))
    print("\n".join(L))
    if a.listen:
        picks = [(r, r.boost_at_s - WINDOW_S / 2, WINDOW_S, []) for r in _worst(df, "max_boost_db", 3)]
        picks += [(r, r.cut_at_s - WINDOW_S / 2, WINDOW_S, []) for r in _worst(df, "max_cut_db", 1, ascending=True)]
        _listen(a, picks, lambda r: "loudness", "loudness_listen")


# ---------- loudness cap ----------

def _uncapped_path(inv, j, clip_id):
    return Path(inv) / "melody_checks" / "uncapped" / f"c{j}" / f"{key(clip_id)}.wav"


def _make_uncapped(row):
    import soundfile as sf

    from ..melody import melody_step
    for j in range(C.N_CONVERSIONS):
        p = _uncapped_path(row["inv"], j, row["clip_id"])
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, melody_step(row["src"], conv_path(row["inv"], j, row["clip_id"]), C.K_SEMITONES[row["style"]],
                                    max_boost_db=None), C.SR)
    return row["clip_id"]


def _level_drop(files):
    import soundfile as sf
    return np.array([C.TARGET_DBFS - 20 * np.log10(np.sqrt(np.mean(sf.read(f)[0] ** 2))) for f in files])


def loudness_cap(a):
    from ..asr import flagged_syllables, transcribe
    from ..text import Verse
    from .selection import load
    out = _out(a)
    rows = _rows(a)
    for i, c in enumerate(_parallel(_make_uncapped, rows), 1):
        if i % 10 == 0:
            print(f"made {i}/{len(rows)}", flush=True)
    syl = out / "uncapped_syllables.csv"
    done = set(pd.read_csv(syl).clip_id) if syl.exists() else set()
    for r in rows:
        if r["clip_id"] in done:
            continue
        t0, verse, recs = time.time(), Verse(r["text"]), []
        for j in range(C.N_CONVERSIONS):
            flags = flagged_syllables(verse, transcribe(_uncapped_path(a.inventory, j, r["clip_id"])))
            recs += [{"clip_id": r["clip_id"], "j": j, "syllable": s, "uncapped_flag": bool(f)} for s, f in enumerate(flags)]
        pd.DataFrame(recs).to_csv(syl, mode="a", header=not syl.exists(), index=False)
        print(f"scored {r['clip_id']} in {time.time() - t0:.0f} s", flush=True)
    capped = load(a.inventory)[["clip_id", "j", "syllable", "his_flag", "melody_flag"]]
    d = capped.merge(pd.read_csv(syl), on=["clip_id", "j", "syllable"], validate="one_to_one")
    d = d[~d.his_flag]
    per = d.groupby("clip_id").agg(n=("melody_flag", "size"), unc=("uncapped_flag", "sum"), cap=("melody_flag", "sum"))
    diff = 100 * (per.cap.sum() - per.unc.sum()) / per.n.sum()
    rng = np.random.default_rng(CAP_SEED)
    boots = []
    for _ in range(CAP_BOOT):
        b = per.iloc[rng.integers(0, len(per), len(per))]
        boots.append(100 * (b.cap.sum() - b.unc.sum()) / b.n.sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    clips = [(r["clip_id"], j) for r in rows for j in range(C.N_CONVERSIONS)]
    drop_unc = _level_drop([_uncapped_path(a.inventory, j, c) for c, j in clips])
    drop_cap = _level_drop([melody_path(a.inventory, j, c) for c, j in clips])
    r1, r2 = (drop_cap > 3).mean() <= 0.05, hi <= 0.5
    t = [{"Melody step": name, "Clips > 0.5 dB below −20 dBFS, %": round(100 * (x > 0.5).mean(), 1),
          "> 3 dB below, %": round(100 * (x > 3).mean(), 1), "> 6 dB below, %": round(100 * (x > 6).mean(), 1),
          "Most below, dB": round(float(x.max()), 1), "Syllables wrong, %": round(100 * f.sum() / len(f), 2)}
         for name, x, f in (("no cap", drop_unc, d.uncapped_flag),
                            (f"boost capped at +{C.ENERGY_MAX_BOOST_DB:g} dB", drop_cap, d.melody_flag))]
    L = ["# The melody step's boost cap", "",
         f"{len(clips)} melody-step clips of the inventory's conversions, without the cap and with it (the inventory's "
         "own). Syllables wrong: right on Vāgdhenu's recording, wrong on the clip (Su-śrotā).", "",
         pd.DataFrame(t).to_markdown(index=False), "",
         f"Difference in syllables wrong, capped minus no cap: {diff:+.2f} points, 95% interval [{lo:+.2f}, {hi:+.2f}] "
         f"(bootstrap over verses, {CAP_BOOT} resamples).", "",
         f"1. Spikes gone (≤ 5% of capped clips more than 3 dB below −20 dBFS): {'met' if r1 else 'not met'} "
         f"({100 * (drop_cap > 3).mean():.1f}%).",
         f"2. Errors not worse (upper end ≤ +0.5 points): {'met' if r2 else 'not met'} ({hi:+.2f}).", ""]
    (out / "loudness_cap.md").write_text("\n".join(L))
    print("\n".join(L))
    if a.listen:
        e = pd.read_csv(out / "loudness.csv") if (out / "loudness.csv").exists() else None
        if e is None:
            print("run `nandini melody-checks loudness` first for the listening clips")
            return
        picks = [(r, r.boost_at_s - WINDOW_S / 2, WINDOW_S, [_uncapped_path(a.inventory, r.j, r.clip_id)])
                 for r in _worst(e, "max_boost_db", 3)]
        _listen(a, picks, lambda r: "capped_then_uncapped", "loudness_cap_listen")
