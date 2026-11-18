"""The pitch shift k per style: how far the melody step raises Vāgdhenu's pitch to sit where the target voice speaks.

  nandini pitch-shift draw                                   -> OUT/order.csv, OUT/draw.csv (the verses to read)
  nandini pitch-shift readings --indic-speak-python PY [--device auto|cuda|mps|cpu]
                                                             -> OUT/readings/NN.wav, OUT/gen_times.csv
  nandini pitch-shift check                                  -> OUT/check.csv (token cap and Su-śrotā CER per reading)
  nandini pitch-shift vagdhenu [--dataset-dir D]             -> OUT/pitch_vagdhenu.csv, OUT/range_vagdhenu.json
  nandini pitch-shift target                                 -> OUT/pitch_target.csv, OUT/range_target.json
  nandini pitch-shift report                                 -> OUT/report.md
  (OUT: --out, default pitch_shift)

Draw: per style, PITCH_PER_STYLE verses in proportion to its meters (largest remainder), plus PITCH_ADD; within each
style and meter, clips in a seeded random order; a text already drawn is skipped. The released draw is
manifests/pitch_shift_verses.csv. Readings: the target voice (Indic-Speak, speaker TARGET_SPEAKER) reads each verse,
in Indic-Speak's own environment. A reading cut off at the token cap or with a character error rate above
PITCH_CER_LIMIT would be replaced (none was). Pitch: Praat autocorrelation via parselmouth, 10 ms frames; two passes
per speaker, 50-800 Hz on all of that speaker's clips, then 0.75 x Q1 to 1.5 x Q3 of pass 1's pooled voiced frames.
Per clip: the mean of 12 log2(F0 / 100 Hz) over voiced frames; per speaker: the mean over clips. k per style is the
target's mean minus Vāgdhenu's, with a bootstrap interval resampling within meters.
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..dataset import metadata, source

VOWEL = re.compile(r"[aAiIuUfFxXeEoO]")
FAMILY_NAMES = {"anustubh": "anuṣṭubh", "tristubh": "triṣṭubh", "jagati": "jagatī", "rucira": "rucirā",
                "vasantatilaka": "vasantatilakā", "malini": "mālinī", "17-syllable": "17-syllable",
                "sardulavikridita": "śārdūlavikrīḍita", "sragdhara": "sragdharā", "other": "other"}


def clips(dataset_dir=None):
    """Every clip of the dataset with its style, meter family and syllable count."""
    from .manifest import family
    rows = []
    for cfg in C.DATASET_CONFIGS:
        for r in pd.read_csv(metadata(cfg, dataset_dir), dtype=str).to_dict("records"):
            rows.append({**r, "config": cfg, "clip_id": f"{cfg}/{r['file_name']}",
                         "n_syll": len(VOWEL.findall(r["text_slp1"])), "family": family(r["text_slp1"])})
    return pd.DataFrame(rows)


# ---------- draw ----------

def allocation(d):
    """Readings per style x meter: PITCH_PER_STYLE x the meter's share of the style's clips, largest remainder,
    plus PITCH_ADD."""
    n = {}
    for s, g in d.groupby("config"):
        q = g.groupby("family").size() / len(g) * C.PITCH_PER_STYLE
        k = np.floor(q).astype(int)
        k[(q - k).sort_values(ascending=False).index[:C.PITCH_PER_STYLE - k.sum()]] += 1
        n.update({(s, f): int(v) for f, v in k.items()})
    for c in map(tuple, C.PITCH_ADD):
        assert n[c] == 0, c
        n[c] = 1
    return n


def draw(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = clips(a.dataset_dir)
    n = allocation(d)
    rng = np.random.default_rng(C.PITCH_DRAW_SEED)
    order = []
    for (s, f) in sorted(n):  # fixed cell order: style, then family name
        g = d[(d.config == s) & (d.family == f)].sort_values("clip_id")
        g = g.iloc[rng.permutation(len(g))].assign(rank=range(1, len(g) + 1))
        order.append(g)
    order = pd.concat(order)[["config", "family", "rank", "clip_id", "duration", "n_syll", "text_devanagari"]]
    order.to_csv(out / "order.csv", index=False)
    taken, rows = set(), []
    for s in sorted(d.config.unique()):  # style_a first; a text already read is skipped
        for f in sorted({f for (t, f) in n if t == s}):
            got = 0
            for r in order[(order.config == s) & (order.family == f)].itertuples():
                if got == n[(s, f)]:
                    break
                if r.text_devanagari in taken:
                    continue
                taken.add(r.text_devanagari)
                rows.append({**r._asdict(), "reason": "drawn"})
                got += 1
    res = pd.DataFrame(rows).drop(columns="Index")
    res.insert(0, "reading", range(1, len(res) + 1))
    res.to_csv(out / "draw.csv", index=False)
    print(res.groupby(["family", "config"]).size().unstack(fill_value=0).to_string())
    print(f"{len(res)} verses -> {out / 'draw.csv'}")


# ---------- readings ----------

def readings(a):
    out = Path(a.out)
    draw_csv = out / "draw.csv"
    if not draw_csv.exists():
        raise SystemExit(f"no {draw_csv}: run `nandini pitch-shift draw` first, or copy manifests/pitch_shift_verses.csv")
    if not a.indic_speak_python:
        raise SystemExit("missing: --indic-speak-python (the Python of an environment with Indic-Speak's requirements)")
    settings = {"repo": C.TARGET_TTS_REPO, "revision": C.TARGET_TTS_REV, "speaker": C.TARGET_SPEAKER,
                "seed": C.TARGET_TTS_SEED, "max_new_tokens": C.TARGET_TTS_MAX_TOKENS, "device": a.device}
    worker = Path(__file__).with_name("target_readings_worker.py")
    subprocess.run([a.indic_speak_python, str(worker), str(draw_csv), str(out / "readings"), json.dumps(settings)],
                   check=True)


# ---------- check ----------

def _normalise(text):
    """Text compared for the character error rate: no spaces, punctuation, dandas, digits or zero-width characters."""
    return re.sub(r"[\s‌‍।॥|,.;:!?'\"()\[\]{}०-९0-9-]", "", text)


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(text, hyp):
    t, h = _normalise(text), _normalise(hyp)
    return 100 * _lev(t, h) / max(1, len(t))


def check(a):
    from ..asr import transcribe
    out = Path(a.out)
    d = pd.read_csv(out / "draw.csv")
    times = pd.read_csv(out / "gen_times.csv").drop_duplicates("reading", keep="last").set_index("reading")
    rows = []
    for r in d.itertuples():
        hyp = transcribe(out / "readings" / f"{r.reading:02d}.wav")
        c = cer(r.text_devanagari, hyp)
        capped = bool(times.loc[r.reading, "capped"])
        rows.append({"reading": r.reading, "config": r.config, "family": r.family, "clip_id": r.clip_id,
                     "audio_s": times.loc[r.reading, "audio_s"], "gen_s": times.loc[r.reading, "gen_s"],
                     "capped": capped, "cer": round(c, 2), "replace": capped or c > C.PITCH_CER_LIMIT, "hyp": hyp})
        print(r.reading, round(c, 1), "REPLACE" if rows[-1]["replace"] else "", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "check.csv", index=False)
    print(int(res["replace"].sum()), "to replace")


# ---------- pitch ----------

def f0(path, floor, ceiling):
    """Voiced-frame F0 in Hz (10 ms frames)."""
    import parselmouth
    p = parselmouth.Sound(str(path)).to_pitch(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    v = p.selected_array["frequency"]
    return v[v > 0]


def st(hz):
    return 12 * np.log2(np.asarray(hz) / 100.0)


def two_pass(items, name):
    """items: list of (key, path). Returns per-clip rows and the range used."""
    p1 = {}
    for i, (k, path) in enumerate(items):
        p1[k] = f0(path, 50, 800)
        if i % 200 == 0:
            print(name, "pass 1", i, flush=True)
    pooled = np.concatenate(list(p1.values()))
    q1, q3 = np.percentile(pooled, [25, 75])
    rng_ = {"pass1_floor": 50, "pass1_ceiling": 800, "q1": round(float(q1), 2), "q3": round(float(q3), 2),
            "floor": round(float(0.75 * q1), 2), "ceiling": round(float(1.5 * q3), 2),
            "pass1_voiced_frames": int(len(pooled))}
    rows = []
    for i, (k, path) in enumerate(items):
        v = f0(path, rng_["floor"], rng_["ceiling"])
        s = st(v)
        rows.append({"key": k, "voiced_frames": len(v), "mean_st": s.mean() if len(v) else np.nan,
                     "sd_st": s.std() if len(v) else np.nan, "p95_st": np.percentile(s, 95) if len(v) else np.nan,
                     "median_hz": np.median(v) if len(v) else np.nan, "p95_hz": np.percentile(v, 95) if len(v) else np.nan,
                     "pass1_voiced_frames": len(p1[k])})
        if i % 200 == 0:
            print(name, "pass 2", i, flush=True)
    return pd.DataFrame(rows), rng_


def vagdhenu(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = clips(a.dataset_dir)
    rows, r = two_pass([(c, source(c, a.dataset_dir)) for c in d.clip_id], "vagdhenu")
    res = d[["clip_id", "config", "family", "duration", "n_syll"]].merge(rows.rename(columns={"key": "clip_id"}))
    res.to_csv(out / "pitch_vagdhenu.csv", index=False)
    (out / "range_vagdhenu.json").write_text(json.dumps(r, indent=1) + "\n")
    print(r, res.groupby("config").mean_st.agg(["mean", "std", "count"]).round(3).to_string())


def target(a):
    out = Path(a.out)
    chk = pd.read_csv(out / "check.csv")
    if chk["replace"].any():
        raise SystemExit(f"{int(chk['replace'].sum())} readings need replacing first (see check.csv)")
    d = pd.read_csv(out / "draw.csv")
    rows, r = two_pass([(int(x), out / "readings" / f"{int(x):02d}.wav") for x in d.reading], "target")
    res = d[["reading", "config", "family", "clip_id"]].merge(rows.rename(columns={"key": "reading"}))
    res.to_csv(out / "pitch_target.csv", index=False)
    (out / "range_target.json").write_text(json.dumps(r, indent=1) + "\n")
    print(r, res.groupby("config").mean_st.agg(["mean", "std", "count"]).round(3).to_string())


# ---------- report ----------

def boot_mean(df, cells, rng):
    """Bootstrap the mean of mean_st, resampling rows within cells."""
    groups = [g.mean_st.to_numpy() for _, g in df.groupby(cells)]
    n = sum(len(g) for g in groups)
    tot = np.zeros(C.PITCH_BOOTSTRAP)
    for g in groups:
        tot += g[rng.integers(0, len(g), (C.PITCH_BOOTSTRAP, len(g)))].sum(1)
    return tot / n


def k_row(name, h, m, cells, rng):
    k = m.mean_st.mean() - h.mean_st.mean()
    b = boot_mean(m, cells, rng) - boot_mean(h, cells, rng)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"Shift": name, "Target readings": len(m), "Target mean (st)": m.mean_st.mean(), "Vāgdhenu clips": len(h),
            "Vāgdhenu mean (st)": h.mean_st.mean(), "k (st)": k, "95% interval": f"[{lo:+.2f}, {hi:+.2f}]",
            "Target mean (Hz)": 100 * 2 ** (m.mean_st.mean() / 12),
            "Vāgdhenu mean (Hz)": 100 * 2 ** (h.mean_st.mean() / 12)}


def report(a):
    out = Path(a.out)
    md = lambda df, f: df.to_markdown(index=False, floatfmt=f)  # noqa: E731
    h = pd.read_csv(out / "pitch_vagdhenu.csv")
    m = pd.read_csv(out / "pitch_target.csv")
    chk = pd.read_csv(out / "check.csv")
    rh, rm = (json.loads((out / f"range_{x}.json").read_text()) for x in ("vagdhenu", "target"))
    rng = np.random.default_rng(C.PITCH_DRAW_SEED)
    styles = sorted(h.config.unique())
    rows = [k_row(f"k ({s})", h[h.config == s], m[m.config == s], "family", rng) for s in styles]
    rows.append(k_row("pooled k", h, m, ["config", "family"], rng))
    K = pd.DataFrame(rows)
    ks = {s: rows[i]["k (st)"] for i, s in enumerate(styles)}

    def by_meter(df, unit):
        t = df.groupby(["family", "config"]).mean_st.agg(["count", "mean"]).unstack()
        t.columns = [f"{c[1]} {unit if c[0] == 'count' else 'mean (st)'}" for c in t.columns]
        return t.reset_index().assign(family=lambda x: x.family.map(FAMILY_NAMES))

    h["shifted_p95_hz"] = 100 * 2 ** ((h.p95_st + h.config.map(ks)) / 12)
    high = h.groupby(["config", "family"]).shifted_p95_hz.agg(["count", "median", "max"]).reset_index()
    high["family"] = high.family.map(FAMILY_NAMES)
    target_p95 = m.p95_hz.median()
    rep = chk[chk["replace"]]
    L = ["# The pitch shift k per style", "",
         f"{len(m)} target readings ({C.TARGET_SPEAKER}, Indic-Speak) against all {len(h):,} Vāgdhenu clips. "
         "Semitones re 100 Hz.", "",
         "## k", "", md(K, ".2f"), "",
         f"*Complete table. Hz columns are the means converted back. Intervals: 95% bootstrap, {C.PITCH_BOOTSTRAP:,} "
         "resamples, within meters (per style) or style x meter (pooled).*", "",
         "k per style as used: " + ", ".join(f"{s} {ks[s]:+.2f} st" for s in styles) + f" (`K_SEMITONES` in "
         f"`nandini/config.py`: {C.K_SEMITONES}).", "",
         "## Readings check", "",
         f"{len(chk)} readings; cut off at the token cap {int(chk.capped.sum())}; character error rate above "
         f"{C.PITCH_CER_LIMIT:.0f}%: {int((chk.cer > C.PITCH_CER_LIMIT).sum())}; replaced: {len(rep)}. Median "
         f"{chk.cer.median():.1f}%, max {chk.cer.max():.1f}%.", "",
         "## Pitch tracking ranges", "",
         md(pd.DataFrame([{"Speaker": "Vāgdhenu", **rh}, {"Speaker": "target", **rm}]), ".1f"), "",
         "*Complete table. Pass 2 range = 0.75 × Q1 to 1.5 × Q3 of pass 1's pooled voiced frames.*", "",
         "## Vāgdhenu's mean by meter within style", "", md(by_meter(h, "clips"), ".2f"), "", "",
         "## The target's mean by meter within style", "", md(by_meter(m, "readings"), ".2f"), "", "",
         "## How high the shifted clips go", "",
         f"Each clip's 95th-percentile pitch plus its style's k, in Hz, against the target readings' median "
         f"95th percentile ({target_p95:.0f} Hz). Clips above it: {int((h.shifted_p95_hz > target_p95).sum())}.", "",
         md(high.rename(columns={"count": "clips", "median": "median Hz", "max": "max Hz"}), ".0f"), ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
