"""The comparison's measures and report, from `nandini compare render`'s folder. Runs in this code's environment.

  nandini compare score [--out compare] [--her-embedding NPY | --data training/dataset] [--dataset-dir D] [--report-only]

Per render: Su-śrotā's character error rate against the verse text; duration; voice (ECAPA cosine to the mean of her
training clips; the lowest 2 s window's cosine to the render's own mean, for a voice switch; median F0; mean log pitch
in semitones, Praat 60-600 Hz). Pairs: melody correlation as registered in round 1 (pitch curves in semitones around
each curve's median, pYIN 70-500 Hz, aligned by DTW on MFCCs, Pearson r over frames voiced in both) and duration ratio.

Sets are reported separately, never pooled:
  A  every system against his recording of the verse (no pass/fail; N is read against V);
  B  H4, registered in round 1, N against V: mean CER at most V's + H4_CER_MARGIN points, mean melody correlation with
     V's render at least H4_MELODY_MIN, mean duration ratio within 1 +- H4_DURATION_TOL; Vs against V beside it;
  C  Vs against his demo MP3s (does our run of his system match his?);
and the voice measures for every system, with the pitch offset N - V per verse against the dataset's shift k
(per style on set A, pooled on set B). Intervals: 95% bootstrap over verses (paired for differences), COMPARE_BOOT
resamples, seed COMPARE_BOOT_SEED. Writes OUT/scores.csv, OUT/pairs.csv and OUT/report.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import external as X

K_POOLED = float(np.mean(list(C.K_SEMITONES.values())))
_feat = {}


def _features(path):
    """pYIN pitch in semitones around its median, and MFCCs, as round 1's melody measure (16 kHz, 10 ms)."""
    import librosa
    key = str(path)
    if key not in _feat:
        y, _ = librosa.load(key, sr=16000)
        f0, _, _ = librosa.pyin(y, fmin=70, fmax=500, sr=16000, hop_length=160)
        mf = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, hop_length=160)
        k = min(len(f0), mf.shape[1])
        _feat[key] = (12 * np.log2(f0[:k] / np.nanmedian(f0)), mf[:, :k])
    return _feat[key]


def melody(a, b):
    """Round 1's melody correlation (archive: checkpoint_trend.melody), unchanged."""
    import librosa
    (sa, ma), (sb, mb) = _features(a), _features(b)
    _, wp = librosa.sequence.dtw(X=ma, Y=mb, metric="cosine")
    x, y = sa[wp[:, 0]], sb[wp[:, 1]]
    ok = ~np.isnan(x) & ~np.isnan(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def seconds(path):
    import soundfile as sf
    try:
        return sf.info(str(path)).duration
    except RuntimeError:  # MP3 on an older libsndfile
        import librosa
        return librosa.get_duration(path=str(path))


def ci(x, rng):
    """Mean and 95% bootstrap interval over verses."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = rng.choice(x, (C.COMPARE_BOOT, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(c, d=2, pct=False):
    f = (lambda v: f"{v:.{d}%}") if pct else (lambda v: f"{v:.{d}f}")
    return f"{f(c[0])} [{f(c[1])}, {f(c[2])}]"


def _her(a, out):
    if a.her_embedding:
        return np.load(a.her_embedding)
    from ..analysis.training_checks import her_mean
    return her_mean(a.data, out / "her_mean_embedding.npy")


def measure(out, a):
    import librosa
    from ..asr import transcribe
    from ..dataset import source
    from ..pipeline.pitch_shift import cer, f0, st
    from ..voice import _embed, profile
    her = _her(a, out)
    rows = []
    for s in ("A", "B", "C"):
        if not (out / s / "verses.csv").exists():
            continue
        v = pd.read_csv(out / s / "verses.csv")
        refs = {}
        if s == "A":
            refs = {r.n: Path(source(r.clip_id, a.dataset_dir)) for r in v.itertuples()}
        elif s == "C":
            refs = {r.n: X.require("vagdhenu") / r.his_audio for r in v.itertuples()}
        systems = [d.name for d in sorted((out / s).iterdir()) if d.is_dir()]
        kinds = [("his", lambda n: refs[n])] if refs else []
        kinds += [(k, (lambda k: lambda n: out / s / k / f"{n}.wav")(k)) for k in systems]
        for r in v.itertuples():
            for k, where in kinds:
                p = where(r.n)
                if not Path(p).exists():
                    continue
                y = librosa.load(str(p), sr=16000, mono=True)[0]
                hz = f0(p, C.F0_FLOOR_HZ, C.F0_CEILING_HZ)
                _, win = profile(str(p))
                rows.append({"set": s, "n": r.n, "system": k, "seconds": round(seconds(p), 2),
                             "cer": round(cer(r.text, transcribe(str(p))), 2),
                             "her_cosine": round(float(_embed(y) @ her), 4), "window_min": round(win["win_cos_min"], 4),
                             "f0_median_hz": round(float(np.median(hz)), 1) if len(hz) else np.nan,
                             "pitch_st": round(float(st(hz).mean()), 3) if len(hz) else np.nan,
                             "style": getattr(r, "style", "")})
            print(f"{s}{r.n} measured", flush=True)
    return pd.DataFrame(rows)


def pairs(out, d):
    """Melody correlation and duration ratio: A every system against his recording; B N and Vs against V; C Vs against
    his MP3."""
    rows = []
    for s, ref, tests in (("A", "his", ("N", "V", "Vs")), ("B", "V", ("N", "Vs")), ("C", "his", ("Vs",))):
        ds = d[d.set == s]
        if ds.empty:
            continue
        sec = ds.set_index(["n", "system"]).seconds
        for n in sorted(ds.n.unique()):
            for k in tests:
                if (n, k) not in sec.index or (n, ref) not in sec.index:
                    continue
                rows.append({"set": s, "n": n, "system": k, "against": ref,
                             "melody": round(melody(_path(out, s, k, n), _path(out, s, ref, n)), 4),
                             "duration_ratio": round(sec[(n, k)] / sec[(n, ref)], 4)})
    return pd.DataFrame(rows)


_ref_paths = {}


def _path(out, s, k, n):
    return _ref_paths[(s, n)] if k == "his" else out / s / k / f"{n}.wav"


def report(out, d, p):
    rng = np.random.default_rng(C.COMPARE_BOOT_SEED)
    L = ["# Nandinī against Vāgdhenu, through his website's path", "",
         "N: Nandinī (ours: female bank, fine-tuned vocoder). V: his `voice_armA`. Vs: his `voice_steer` (what his "
         "website serves). Sets reported separately. Intervals: 95% bootstrap over verses. Per render in "
         "`scores.csv`, per pair in `pairs.csv`.", ""]

    def g(s, k, col, frame=d):
        return frame[(frame.set == s) & (frame.system == k)].sort_values("n").set_index("n")[col]

    # Set B: H4
    if (d.set == "B").any():
        cn, cv = g("B", "N", "cer"), g("B", "V", "cer")
        dc = ci((cn - cv).dropna(), rng)
        mel, dur = ci(g("B", "N", "melody", p), rng), ci(g("B", "N", "duration_ratio", p), rng)
        ok = [cn.mean() <= cv.mean() + C.H4_CER_MARGIN, mel[0] >= C.H4_MELODY_MIN,
              abs(dur[0] - 1) <= C.H4_DURATION_TOL]
        L += [f"## Set B: {len(cn)} test verses, N against V (H4, registered 2026-11-06)", "",
              "| Test | Bar | N | Met |", "|:--|:--|:--|:--|",
              f"| Character errors, N − V (points) | mean ≤ +{C.H4_CER_MARGIN} | {fmt(dc)} (N {cn.mean():.2f}%, V "
              f"{cv.mean():.2f}%) | {'yes' if ok[0] else 'no'} |",
              f"| Melody correlation with V's render | mean ≥ {C.H4_MELODY_MIN} | {fmt(mel, 3)} | {'yes' if ok[1] else 'no'} |",
              f"| Duration, N ÷ V | mean within 1 ± {C.H4_DURATION_TOL} | {fmt(dur, 3)} | {'yes' if ok[2] else 'no'} |",
              "", "*Complete.*", "", f"**H4 {'MET' if all(ok) else 'NOT MET'}** ({sum(ok)} of 3).", ""]
        if (p[(p.set == "B") & (p.system == "Vs")]).shape[0]:
            L += ["Beside it, his website's model against V on the same verses (no pass/fail): character errors Vs − V "
                  f"{fmt(ci((g('B', 'Vs', 'cer') - cv).dropna(), rng))} points; melody correlation "
                  f"{fmt(ci(g('B', 'Vs', 'melody', p), rng), 3)}; duration ratio "
                  f"{fmt(ci(g('B', 'Vs', 'duration_ratio', p), rng), 3)}.", ""]

    # Set A: against his recording
    if (d.set == "A").any():
        L += [f"## Set A: {d[(d.set == 'A') & (d.system == 'his')].shape[0]} training verses, against his recording", "",
              "| System | Melody correlation with his recording | Duration ÷ his | Character errors % |",
              "|:--|--:|--:|--:|"]
        for k in ("N", "V", "Vs"):
            if g("A", k, "melody", p).empty:
                continue
            L.append(f"| {k} | {fmt(ci(g('A', k, 'melody', p), rng), 3)} | {fmt(ci(g('A', k, 'duration_ratio', p), rng), 3)} "
                     f"| {fmt(ci(g('A', k, 'cer'), rng))} |")
        L.append(f"| his recording | — | — | {fmt(ci(g('A', 'his', 'cer'), rng))} |")
        L += ["", "*Complete.*", ""]
        if not g("A", "N", "melody", p).empty and not g("A", "V", "melody", p).empty:
            dm = ci((g("A", "N", "melody", p) - g("A", "V", "melody", p)).dropna(), rng)
            dd = ci((g("A", "N", "duration_ratio", p) - g("A", "V", "duration_ratio", p)).dropna(), rng)
            L += [f"N − V (paired): melody correlation {fmt(dm, 3)}; duration ratio {fmt(dd, 3)}. Read N against V, "
                  "not against 1.0: V is his own model on the same path.", ""]

    # Set C: our run of his system against his demos
    if (p.set == "C").any():
        c = p[p.set == "C"].merge(pd.read_csv(out / "C" / "verses.csv")[["n", "his_meter", "source"]], on="n")
        L += ["## Set C: his 6 demo verses, Vs against his MP3s", "",
              "| Verse | His meter label | Melody correlation | Duration ÷ his |", "|:--|:--|--:|--:|"]
        L += [f"| {r.source} | {r.his_meter} | {r.melody:.3f} | {r.duration_ratio:.3f} |" for r in c.itertuples()]
        L += ["", "*Complete.*", ""]

    # Voice
    L += ["## Voice (no pass/fail)", "",
          "| Set | System | Cosine to her training clips | Lowest window cosine | Median F0, Hz |",
          "|:--|:--|--:|--:|--:|"]
    for s in ("A", "B", "C"):
        for k in ("N", "V", "Vs", "his"):
            x = d[(d.set == s) & (d.system == k)]
            if len(x):
                L.append(f"| {s} | {k} | {fmt(ci(x.her_cosine, rng), 3)} | {fmt(ci(x.window_min, rng), 3)} | "
                         f"{fmt(ci(x.f0_median_hz, rng), 0)} |")
    L += ["", "*Complete.*", ""]
    for s in ("A", "B"):
        n_, v_ = g(s, "N", "pitch_st"), g(s, "V", "pitch_st")
        off = (n_ - v_).dropna()
        if len(off):
            if s == "A":
                k = d[(d.set == "A") & (d.system == "N")].set_index("n")["style"].map(C.K_SEMITONES).reindex(off.index)
                kk = f"k per style (mean {k.mean():+.2f})"
                dev = ci(off - k, rng)
            else:
                kk, dev = f"pooled k ({K_POOLED:+.2f})", ci(off - K_POOLED, rng)
            L.append(f"- **Pitch offset N − V, set {s}:** {fmt(ci(off, rng))} semitones against {kk}; N − V − k "
                     f"{fmt(dev)}; spread across verses (SD) {off.std():.2f}.")
    L += ["", "Limits: one seed per render; Su-śrotā is the only recogniser; set A is training verses; set B covers "
          "six of the dataset's meter families."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def run(a):
    from ..dataset import source
    out = Path(a.out)
    if (out / "A" / "verses.csv").exists():
        for r in pd.read_csv(out / "A" / "verses.csv").itertuples():
            _ref_paths[("A", r.n)] = Path(source(r.clip_id, a.dataset_dir))
    if (out / "C" / "verses.csv").exists():
        for r in pd.read_csv(out / "C" / "verses.csv").itertuples():
            _ref_paths[("C", r.n)] = X.require("vagdhenu") / r.his_audio
    if a.report_only:
        report(out, pd.read_csv(out / "scores.csv"), pd.read_csv(out / "pairs.csv"))
        return
    d = measure(out, a)
    d.to_csv(out / "scores.csv", index=False)
    p = pairs(out, d)
    p.to_csv(out / "pairs.csv", index=False)
    report(out, d, p)
