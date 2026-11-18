"""The inventory's report: every measure and decision rule, from the scored parts in OUT. Writes OUT/report.md.

Syllables the ASR gets wrong in Vāgdhenu's own recording are left out of every count.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["aspirated", "retroflex", "sibilant", "visarga", "anusvara", "long_vowel", "conjunct"]
NS = (3, 5, 7, 10)
BOOT, SEED = 1000, 20261110
RESCUE_MARGIN = 2.0      # rule 1: percentage points
MELODY_SHARE = 10.0      # rule 2: % of flags after the melody step that the melody step made
ALL_FAIL_SHARE = 5.0     # rule 3: % of a class's occurrences failing in every conversion


def _md(df, fmt=".2f"):
    return df.to_markdown(index=False, floatfmt=fmt)


def _boot(per_clip, stat, rng):
    """95% interval of stat over clips resampled with replacement. per_clip: list of DataFrames."""
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(per_clip), len(per_clip))
        vals.append(stat(pd.concat([per_clip[i] for i in pick])))
    return np.percentile(vals, [2.5, 97.5])


def _by_style(s, s_all, out, n_conv, rescued, rng):
    """Section 7: every headline measure per style, with Vāgdhenu's own recording's ASR error rate as the reference."""
    times = list(out.glob("convert_times_part*.csv"))
    dur = (pd.concat(pd.read_csv(x) for x in times).groupby("clip_id").audio_s.first() if times else None)
    groups = [(st, s[s["style"] == st], s_all[s_all["style"] == st]) for st in sorted(s["style"].unique())]
    groups.append(("all", s, s_all))
    cols = {}
    for name, d, da in groups:
        his0 = da[da.j == 0]
        per = d.groupby(["clip_id", "syllable"]).raw_flag.sum()
        clip_raw = d.groupby("clip_id").raw_flag.mean() * 100
        clip_mel = d.groupby("clip_id").melody_flag.mean() * 100
        made = (~d.raw_flag & d.melody_flag).sum()
        cj = d.groupby(["clip_id", "j"]).raw_flag.agg(["sum", "count"])
        pick = cj.loc[cj.groupby(level=0)["sum"].idxmin()]  # each clip's conversion with the fewest flags (lowest j on ties)
        best = 100 * pick["sum"].sum() / pick["count"].sum()
        after = d.melody_flag.sum()
        c = {"Clips": f"{d.clip_id.nunique()}",
             "Syllables in the texts": f"{len(his0)}",
             "Vāgdhenu's recording: syllables flagged, %": f"{his0.his_flag.mean() * 100:.2f}",
             f"Raw: syllables flagged, % (average of {n_conv} conversions, every syllable: same basis as Vāgdhenu's recording)":
                 f"{da.raw_flag.mean() * 100:.2f}",
             f"After melody step: syllables flagged, % (average of {n_conv} conversions, every syllable)":
                 f"{da.melody_flag.mean() * 100:.2f}",
             "Syllables counted (right in Vāgdhenu's recording)": f"{len(per)}",
             f"Raw: syllables flagged, % (average of {n_conv} conversions)": f"{d.raw_flag.mean() * 100:.2f}",
             f"After melody step: syllables flagged, % (average of {n_conv} conversions)": f"{d.melody_flag.mean() * 100:.2f}",
             f"Raw: best of {n_conv}, whole conversion per clip, % (same judge, optimistic)": f"{best:.2f}",
             f"Raw: best of {n_conv} per syllable, syllables still flagged (same judge, optimistic)": f"{(per == n_conv).sum()} ({(per == n_conv).mean() * 100:.2f}%)",
             "Raw: per-clip median, %": f"{clip_raw.median():.2f}",
             "After melody step: per-clip median, %": f"{clip_mel.median():.2f}",
             "Raw: per-clip range, %": f"{clip_raw.min():.1f}–{clip_raw.max():.1f}",
             "Clips with no raw flag": f"{(clip_raw == 0).sum()}",
             "Melody-made share of flags after the melody step, %": f"{100 * made / after if after else 0:.1f}"}
        if dur is not None:
            syl = his0.groupby("clip_id").size()
            c["Vāgdhenu's delivery, syllables per second (median clip)"] = f"{(syl / dur.reindex(syl.index)).median():.2f}"
        cols[name] = c
    t = pd.DataFrame(cols).rename_axis("Measure").reset_index()
    L = ["## 7. By style", "", t.to_markdown(index=False, disable_numparse=True), "",
         "*'Vāgdhenu's recording' is Su-śrotā's flag rate on its own chant, over every syllable of the text: "
         "the judge's own error on the original. Only the two 'every syllable' rows count on the same basis and can be compared "
         "with it; every other row leaves out the syllables it flags on Vāgdhenu's recording. 'Average of "
         f"{n_conv} conversions': flagged syllables over all syllables, which equals the mean of the {n_conv} conversions' rates "
         f"(each conversion covers every syllable). 'Best of {n_conv} per syllable': each syllable taken from any conversion "
         f"that gets it right, so it stays flagged only if all {n_conv} conversions get it wrong. 'Best of': the same "
         "judge picks and scores, so its lucky misses count as successes; an optimistic ceiling for selection, not a "
         "measured result.*", ""]
    sts = [g[0] for g in groups[:-1]]
    if len(sts) == 2:
        a, b = ({k: v for k, v in s[s["style"] == st].groupby("clip_id")} for st in sts)
        def diff(col, base):
            vals = []
            for _ in range(BOOT):
                pa = pd.concat([a[k] for k in rng.choice(list(a), len(a))])
                pb = pd.concat([b[k] for k in rng.choice(list(b), len(b))])
                vals.append((pb[col].mean() - pa[col].mean()) * 100)
            return base, *np.percentile(vals, [2.5, 97.5])
        rows = []
        for col, label in (("raw_flag", "raw"), ("melody_flag", "after melody step")):
            base = (s[s["style"] == sts[1]][col].mean() - s[s["style"] == sts[0]][col].mean()) * 100
            d0, lo, hi = diff(col, base)
            rows.append({"Scoring": label, f"{sts[1]} − {sts[0]}, points": f"{d0:+.2f}", "95% interval": f"[{lo:+.2f}, {hi:+.2f}]"})
        ha = s_all[(s_all["style"] == sts[0]) & (s_all.j == 0)]
        hb = s_all[(s_all["style"] == sts[1]) & (s_all.j == 0)]
        ga, gb = ({k: v for k, v in h.groupby("clip_id")} for h in (ha, hb))
        vals = []
        for _ in range(BOOT):
            pa = pd.concat([ga[k] for k in rng.choice(list(ga), len(ga))])
            pb = pd.concat([gb[k] for k in rng.choice(list(gb), len(gb))])
            vals.append((pb.his_flag.mean() - pa.his_flag.mean()) * 100)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({"Scoring": "his recording", f"{sts[1]} − {sts[0]}, points": f"{(hb.his_flag.mean() - ha.his_flag.mean()) * 100:+.2f}",
                     "95% interval": f"[{lo:+.2f}, {hi:+.2f}]"})
        aa, ab = ({k: v for k, v in s_all[s_all["style"] == st].groupby("clip_id")} for st in sts)
        vals = []
        for _ in range(BOOT):
            pa = pd.concat([aa[k] for k in rng.choice(list(aa), len(aa))])
            pb = pd.concat([ab[k] for k in rng.choice(list(ab), len(ab))])
            vals.append((pb.raw_flag.mean() - pa.raw_flag.mean()) * 100)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        d0 = (s_all[s_all["style"] == sts[1]].raw_flag.mean() - s_all[s_all["style"] == sts[0]].raw_flag.mean()) * 100
        rows.append({"Scoring": "raw, every syllable (same basis as Vāgdhenu's recording)", f"{sts[1]} − {sts[0]}, points": f"{d0:+.2f}",
                     "95% interval": f"[{lo:+.2f}, {hi:+.2f}]"})
        L += [pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True), "",
              "*Difference in flag rate (average of the conversions) between the styles; 95% bootstrap over clips within each style.*", ""]
    return L


def _best_whole(s, n_conv, rng, draws=200):
    """Section 8: flag rate of each clip's best whole conversion (fewest raw flags) when choosing among n conversions,
    averaged over `draws` random subsets of n of the conversions (one subset per draw, shared by all clips)."""
    cj = s.groupby(["style", "clip_id", "j"]).raw_flag.agg(["sum", "count"]).reset_index()
    groups = [(st, cj[cj["style"] == st]) for st in sorted(cj["style"].unique())] + [("all", cj)]
    subsets = [rng.permutation(n_conv) for _ in range(draws)]
    rows = []
    for n in range(1, n_conv + 1):
        r = {"Conversions to choose from": n}
        for name, d in groups:
            vals = []
            for o in subsets:
                b = d[d.j.isin(o[:n])].sort_values(["sum", "j"]).groupby("clip_id").first()
                vals.append(100 * b["sum"].sum() / b["count"].sum())
            r[f"{name}, % of syllables flagged"] = f"{np.mean(vals):.2f}"
        for name, d in groups:
            vals = []
            for o in subsets:
                b = d[d.j.isin(o[:n])].groupby("clip_id")["sum"].min()
                vals.append(100 * (b == 0).mean())
            r[f"{name}, verses with a flag-free conversion, %"] = f"{np.mean(vals):.1f}"
        rows.append(r)
    return ["## 8. Best whole conversion by number of conversions (raw)", "",
            pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True), "",
            f"*Each clip keeps its conversion with the fewest raw flags among n. 'Verses with a flag-free "
            "conversion': at least one of the n conversions has no syllable flagged (not counting syllables flagged on "
            f"Vāgdhenu's recording). Averaged over {draws} random "
            "sets of n of the conversions. Same judge picks and scores, so optimistic. Decision rule 1 is defined on section 5, per syllable.*", ""]


def _kept_before_after(s, s_all, n_conv, rng, draws=200, n_small=5):
    """Section 9: flags of the conversion each clip keeps, on its converter output and after the melody step, by how it
    was chosen. Fewer-than-all choices are averaged over `draws` random sets of conversions."""
    cj = s.groupby(["style", "clip_id", "j"]).agg(raw=("raw_flag", "sum"), mel=("melody_flag", "sum"),
                                                   n=("raw_flag", "size")).reset_index()
    subsets = [rng.permutation(n_conv)[:n_small] for _ in range(draws)]

    def rates(b):
        return 100 * b.raw.sum() / b.n.sum(), 100 * b.mel.sum() / b.n.sum()

    def kept(d, by, js):
        return d[d.j.isin(js)].sort_values([by, "j"]).groupby("clip_id").first()

    L = ["## 9. The kept conversion before and after the melody step", ""]
    for name in sorted(cj["style"].unique()) + ["all"]:
        d = cj if name == "all" else cj[cj["style"] == name]
        small = np.mean([rates(kept(d, "raw", o)) for o in subsets], axis=0)
        rows = [("Average of 10" if n_conv == 10 else f"Average of {n_conv}", rates(d)),
                (f"Best of {n_small}, chosen on converter output", small),
                (f"Best of {n_conv}, chosen on converter output", rates(kept(d, "raw", range(n_conv)))),
                (f"Best of {n_conv}, chosen after the melody step", rates(kept(d, "mel", range(n_conv))))]
        t = pd.DataFrame([{"Conversion kept per clip": r, "Converter output, % of syllables flagged": f"{a:.2f}",
                           "After the melody step, % of syllables flagged": f"{b:.2f}"} for r, (a, b) in rows])
        L += [f"**{name}**", "", t.to_markdown(index=False, disable_numparse=True), ""]
    L += [f"*Syllables flagged by Su-śrotā in the conversion each clip keeps, leaving out the "
          "syllables it flags on Vāgdhenu's recording (so not comparable with Vāgdhenu's own rate; see section 7). 'Best of n': the "
          f"conversion with the fewest flags among n, on the version named. 'Best of {n_small}': averaged over {draws} "
          f"random sets of {n_small} conversions. Whichever version a conversion is chosen on looks better "
          "than average; the other version, nearly the same audio, shows how much of that holds up.*", ""]
    return L


def write_report(out):
    out = Path(out)
    s = pd.concat(pd.read_csv(f) for f in sorted(out.glob("syllables_part*.csv")))
    f = pd.concat(pd.read_csv(x) for x in sorted(out.glob("files_part*.csv")))
    emb = {}
    for x in sorted(out.glob("embeddings_part*.npz")):
        emb.update(dict(np.load(x)))
    n_conv = s.j.nunique()
    excluded = s[s.his_flag & (s.j == 0)].shape[0]
    s_all = s
    s = s[~s.his_flag].copy()
    clips = s.clip_id.nunique()
    rng = np.random.default_rng(SEED)
    L = ["# The converter's failure inventory", "",
         f"{clips} clips, {n_conv} conversions each. Syllables the ASR gets wrong in Vāgdhenu's own recording are left out "
         f"({excluded} syllables); {s[s.j == 0].shape[0]} syllables counted.", ""]

    # 1. failure rate per conversion
    pc = s.groupby(["clip_id", "j"])[["raw_flag", "melody_flag"]].mean() * 100
    L += ["## 1. Failure rate", "",
          _md(pd.DataFrame([{"Scoring": "raw", f"Syllables flagged, % (average of {n_conv} conversions)": s.raw_flag.mean() * 100,
                             "Per conversion, median %": pc.raw_flag.median(), "Per conversion, max %": pc.raw_flag.max()},
                            {"Scoring": "after melody step", f"Syllables flagged, % (average of {n_conv} conversions)": s.melody_flag.mean() * 100,
                             "Per conversion, median %": pc.melody_flag.median(), "Per conversion, max %": pc.melody_flag.max()}])),
          ""]

    # per syllable across conversions
    g = s.groupby(["clip_id", "syllable"])
    per = g.agg(text=("text", "first"), raw_n=("raw_flag", "sum"), mel_n=("melody_flag", "sum"),
                **{c: (c, "first") for c in CLASSES}).reset_index()

    # 2 + 3. per sound class
    rows = []
    for c in CLASSES + ["all"]:
        sub, ps = (s, per) if c == "all" else (s[s[c]], per[per[c]])
        if not len(ps):
            continue
        rows.append({"Sound class": c, "Syllables": len(ps), "Raw flagged %": sub.raw_flag.mean() * 100,
                     "After melody %": sub.melody_flag.mean() * 100,
                     "Fail in all conversions, raw %": (ps.raw_n == n_conv).mean() * 100,
                     "Fail in all, after melody %": (ps.mel_n == n_conv).mean() * 100})
    cls = pd.DataFrame(rows)
    L += ["## 2. By sound class", "", _md(cls), "",
          "*A syllable counts in every class it contains. 'Fail in all' = flagged in every conversion.*", ""]

    # 3. chance or systematic
    flagged = per[per.raw_n > 0]
    base = s.raw_flag.mean()
    pairs = s.pivot_table(index=["clip_id", "syllable"], columns="j", values="raw_flag").to_numpy().astype(bool)
    num = den = 0
    for a in range(n_conv):
        for b in range(n_conv):
            if a != b:
                num += (pairs[:, a] & pairs[:, b]).sum()
                den += pairs[:, a].sum()
    repeat = num / den if den else float("nan")
    hist = flagged.raw_n.value_counts().sort_index()
    L += ["## 3. Chance or systematic (raw)", "",
          f"Syllables flagged in at least one conversion: {len(flagged)} of {len(per)}. Of those, flagged in all "
          f"{n_conv}: {(flagged.raw_n == n_conv).sum()} ({(flagged.raw_n == n_conv).mean() * 100:.1f}%). Repeat rate "
          f"(flagged in conversion j' given flagged in j): {repeat * 100:.1f}%, against a base rate of {base * 100:.1f}%.", "",
          _md(pd.DataFrame({"Conversions failed": hist.index, "Syllables": hist.values}), ".0f"), "",
          "*Flagged syllables by how many conversions they fail in.*", ""]
    allfail = per[per.raw_n == n_conv]
    if len(allfail):
        show = allfail.assign(classes=allfail[CLASSES].apply(lambda r: ", ".join(c for c in CLASSES if r[c]), axis=1))
        L += ["Syllables that fail in every conversion:", "", _md(show[["clip_id", "syllable", "text", "mel_n", "classes"]], ".0f"), "",
              "*mel_n: conversions in which it is also wrong after the melody step.*", ""]

    # 4. where failures come from: the converter or the melody step
    conv_made = s.raw_flag.sum()
    mel_made = (~s.raw_flag & s.melody_flag).sum()
    mel_fixed = (s.raw_flag & ~s.melody_flag).sum()
    after = s.melody_flag.sum()
    share = 100 * mel_made / after if after else 0.0
    L += ["## 4. Where failures come from", "",
          _md(pd.DataFrame([{"Kind": "converter-made (wrong raw)", "Syllable-conversions": conv_made},
                            {"Kind": "melody-made (right raw, wrong after melody)", "Syllable-conversions": mel_made},
                            {"Kind": "melody-fixed (wrong raw, right after melody)", "Syllable-conversions": mel_fixed},
                            {"Kind": "flagged after melody, all", "Syllable-conversions": after}]), ".0f"), "",
          f"Melody-made share of flags after the melody step: **{share:.1f}%**.", ""]

    # 5. rescue curve
    per_clip = [d for _, d in s.groupby("clip_id")]

    def rescued(d, n):
        t = d.pivot_table(index=["clip_id", "syllable"], columns="j", values="raw_flag").astype(bool)
        t = t[t[0]]
        return 100 * (~t[list(range(n))]).any(axis=1).mean() if len(t) else np.nan

    curve = []
    for n in NS:
        if n > n_conv:
            continue
        lo, hi = _boot(per_clip, lambda d: rescued(d, n), rng)
        curve.append({"Conversions": n, "Wrong in conversion 0 but right in another of the first n, %": rescued(s, n), "95% interval": f"[{lo:.1f}, {hi:.1f}]"})
    cur = pd.DataFrame(curve)
    L += ["## 5. Fixed by another conversion (raw)", "", _md(cur, ".1f"), "",
          "*Of the syllables conversion 0 gets wrong, the share that at least one of conversions 0 … n − 1 gets "
          "right, each syllable on its own (different syllables may come from different conversions). Decision rule 1 uses "
          "it. 95% bootstrap over clips.*", ""]

    # 6. voice consistency
    vc = f.groupby("kind")[["win_cos_min", "win_cos_p10", "win_cos_mean"]].mean().reset_index()
    close = []
    vs = emb.get("voice_sample")
    for kind in ("his", "raw", "melody"):
        keys = [k for k in emb if k.split("|")[1:2] == [kind]]
        if vs is not None and keys:
            close.append({"kind": kind, "cos to voice sample": float(np.mean([emb[k] @ vs for k in keys]))})
    between = []
    for kind in ("raw", "melody"):
        cs = []
        for cid in s.clip_id.unique():
            es = [emb[f"{cid}|{kind}|{j}"] for j in range(n_conv) if f"{cid}|{kind}|{j}" in emb]
            cs += [es[a] @ es[b] for a in range(len(es)) for b in range(a + 1, len(es))]
        if cs:
            between.append({"kind": kind, "between conversions, mean cos": float(np.mean(cs)), "min cos": float(np.min(cs))})
    L += ["## 6. Voice consistency (no pass/fail)", "",
          "Within a clip: cosine of each 2 s window to the clip's mean window (mean over clips).", "", _md(vc, ".3f"), "",
          _md(pd.DataFrame(close), ".3f"), "", _md(pd.DataFrame(between), ".3f"), ""]

    # 7. by style
    L += _by_style(s, s_all, out, n_conv, rescued, rng)

    # 8. best whole conversion by n (its own generator, so earlier sections keep their random draws)
    L += _best_whole(s, n_conv, np.random.default_rng(SEED + 8))

    # 9. the kept conversion before and after the melody step
    L += _kept_before_after(s, s_all, n_conv, np.random.default_rng(SEED + 9))

    # rules
    r1 = None
    if len(cur) and (cur.Conversions == 10).any():
        top = cur.loc[cur.Conversions == 10].iloc[0, 1]
        ok = cur[top - cur.iloc[:, 1] <= RESCUE_MARGIN]
        r1 = int(ok.Conversions.min())
    sysc = cls[(cls["Sound class"] != "all") & (cls["Fail in all conversions, raw %"] >= ALL_FAIL_SHARE)]["Sound class"].tolist()
    L += ["## Decision rules", "",
          f"1. **Conversions for the full run:** {r1 if r1 else 'needs all 10 conversions to apply'} "
          f"(smallest n within {RESCUE_MARGIN:.0f} points of n = 10).",
          f"2. **Check after the melody step:** melody-made share {share:.1f}% → "
          f"**{'required' if share >= MELODY_SHARE else 'optional'}** (threshold {MELODY_SHARE:.0f}%).",
          f"3. **Sound classes failing in every conversion in ≥ {ALL_FAIL_SHARE:.0f}% of occurrences:** "
          f"{', '.join(sysc) if sysc else 'none'}.", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
