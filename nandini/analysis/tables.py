"""The marking pass as tables, for analysis.

  nandini tables [--run out] [--manifest M]

Reads RUN/kept_part*.csv and RUN/marks/; writes to RUN/tables/:
  transcripts.csv        one row per verse: Su-śrotā's transcripts of Vāgdhenu's recording and Nandinī's clip,
                         durations, syllables wrong on each, errors
  syllables.csv.gz       one row per verse and syllable: the syllable (Devanagari, SLP1, IAST), wrong on Vāgdhenu's
                         recording / on Nandinī's clip, error (wrong on Nandinī's, right on Vāgdhenu's), what the ASR
                         heard there on each (SLP1, IAST), start time on each
  substitutions.csv, .md every error grouped by the syllable and what was heard on Nandinī's clip, with counts by
                         style and as a share of that syllable's occurrences
"""
from pathlib import Path

import pandas as pd

from .marks import read_kept, read_marks


def run(args):
    from ..asr import heard
    from ..text import Verse, deva, iast
    kept = read_kept(args.run)
    marks = read_marks(args.run)
    texts = pd.read_csv(args.manifest).set_index("clip_id").text
    missing = [c for c in kept.clip_id if c not in marks]
    if missing:
        print(f"note: {len(missing)} of {len(kept)} verses not marked yet (`nandini mark`); tables cover the rest")
    verses, syls = [], []
    for r in kept.itertuples():
        m = marks.get(r.clip_id)
        if not m:
            continue
        v = Verse(texts[r.clip_id])
        hh, nh = heard(v, m["his_hyp"]), heard(v, m["her_hyp"])
        err = [m["her_f"][k] == "1" and m["his_f"][k] == "0" for k in range(v.n)]
        assert sum(err) == r.errors, r.clip_id
        verses.append({"index": r.index, "clip_id": r.clip_id, "style": r.style, "meter": r.family,
                       "syllables": v.n, "errors": r.errors, "vagdhenu_wrong": m["his_f"].count("1"),
                       "nandini_wrong": m["her_f"].count("1"), "vagdhenu_dur_s": m["his_dur"],
                       "nandini_dur_s": m["her_dur"], "vagdhenu_transcript": m["his_hyp"],
                       "nandini_transcript": m["her_hyp"]})
        for k, (a, b) in enumerate(v.spans):
            slp = v.slp[a:b]
            syls.append({"index": r.index, "clip_id": r.clip_id, "style": r.style, "meter": r.family, "syllable": k,
                         "deva": deva(slp), "slp1": slp, "iast": iast(slp),
                         "vagdhenu_wrong": int(m["his_f"][k]), "nandini_wrong": int(m["her_f"][k]),
                         "error": int(err[k]),
                         "vagdhenu_heard_slp1": hh[k], "vagdhenu_heard_iast": iast(hh[k]),
                         "nandini_heard_slp1": nh[k], "nandini_heard_iast": iast(nh[k]),
                         "vagdhenu_start_s": m["his_t"][k], "nandini_start_s": m["her_t"][k]})
    out = Path(args.run) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(verses).to_csv(out / "transcripts.csv", index=False)
    s = pd.DataFrame(syls)
    s.to_csv(out / "syllables.csv.gz", index=False)
    print(f"{len(verses)} verses, {len(s)} syllables, {int(s.error.sum()) if len(s) else 0} errors -> {out}")
    if len(verses) == len(kept):
        substitutions(s, out)


def substitutions(s, out):
    """Every (syllable, what the ASR heard on Nandinī's clip) pair among the errors, with counts by style and how
    often that syllable occurs where it is counted (right on Vāgdhenu's recording), so counts can be read as rates."""
    from ..text import deva
    s = s.fillna("")
    counted = s[s.vagdhenu_wrong == 0]
    occ = counted.groupby("slp1").size().rename("syllable_counted")
    e = s[s.error == 1].copy()
    styles = sorted(s["style"].unique())
    agg = {"syllable": ("iast", "first"), "heard": ("nandini_heard_iast", "first"), "deva": ("deva", "first"),
           "errors": ("error", "size")}
    for st in styles:
        agg[st] = ("style", lambda x, st=st: int((x == st).sum()))
    t = e.groupby(["slp1", "nandini_heard_slp1"]).agg(**agg).reset_index().join(occ, on="slp1")
    t["heard"] = t.heard.replace("", "(nothing)")
    t["heard_deva"] = [deva(h) if h else "(nothing)" for h in t.nandini_heard_slp1]
    t["pct_of_syllable"] = (100 * t.errors / t.syllable_counted).round(1)
    t = t.sort_values(["errors", "syllable"], ascending=[False, True])
    cols = ["syllable", "deva", "heard", "heard_deva", "errors", *styles, "syllable_counted", "pct_of_syllable", "slp1",
            "nandini_heard_slp1"]
    t[cols].to_csv(out / "substitutions.csv", index=False)
    shown = cols[:-2]
    L = ["# What the errors were heard as", "",
         "Every error in the kept clips (a syllable Su-śrotā reads right on Vāgdhenu's recording and wrong on "
         f"Nandinī's), grouped by the syllable and what the ASR heard there on Nandinī's clip. {len(e)} errors in "
         f"{len(t)} distinct pairs. 'Syllable counted' is how often that syllable occurs where it is counted (right on "
         "Vāgdhenu's recording); '% of syllable' is errors as a share of that. What was heard is matched to syllables "
         "by the same character alignment that counts the errors, so extra sounds land on the syllable before them. "
         "Judged by the ASR that chose the kept attempt, so optimistic.", "",
         t[shown].rename(columns={"syllable": "Syllable", "deva": "Devanagari", "heard": "Heard as",
                                  "heard_deva": "Heard (Devanagari)", "errors": "Errors",
                                  "syllable_counted": "Syllable counted",
                                  "pct_of_syllable": "% of syllable"}).to_markdown(index=False), "",
         f"*Complete table, {len(t)} rows, most frequent first.*", ""]
    (out / "substitutions.md").write_text("\n".join(L))
    print(f"substitutions: {len(t)} pairs")
