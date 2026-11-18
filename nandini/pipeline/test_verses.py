"""Test verses for the evaluation (set B): verses outside the dataset, in its meters, that anyone may reuse.

  nandini test-verses [--out manifests/test_verses.csv]

Source: the Digital Corpus of Sanskrit (DCS; Oliver Hellwig, CC BY 4.0), the works in DCS_WORKS at DCS_COMMIT
(`nandini fetch dcs`). Runs in the training environment (his meter detector needs indic_transliteration).

1. Verses: DCS gives each verse's lines (usually two half-verses) in IAST, with sandhi-joined words split at word ends.
   Each line is printed as the dataset prints its text (a word ending in a consonant joined to the next, avagraha
   joined to the word before, no daṇḍas), the lines joined with a space, then written in Devanagari.
2. Meter: his website's detector (`render_core.detect_meter_key`) on that text, unchanged; its name is mapped to the
   dataset's family (METER_FAMILY). Verses it doesn't recognise are left out, so every verse renders with its own
   meter, not his fallback (this also drops verses with a speaker line such as "ṛṣaya ūcuḥ").
3. Static checks, needing no Sanskrit reading: no line of the verse appears in any dataset verse (letters only,
   spelling normalised); every character is in his vocab.txt; the length is within the dataset's range. For the drawn
   verses, the report also gives the longest run of each half-verse's letters found anywhere in the dataset.
4. Draw: TEST_VERSES_N verses, split between the families the source offers in the dataset's proportions (largest
   remainder), each family drawn at random (TEST_VERSES_SEED) from the verses that pass.

Writes the drawn verses (OUT), every candidate with its checks (OUT's stem + _pool.csv) and a report (_report.md).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import external as X

CONSONANT_END = "kgcjṭḍtdnpbmyrlvśṣshṅñṇ"  # IAST letters a consonant-final word ends in (not ḥ or ṃ)


def dcs_verses(root):
    """One row per DCS verse: work, chapter, verse (DCS's sentence counter), lines, sentence ids, IAST."""
    rows = []
    for work in C.DCS_WORKS:
        for f in sorted((root / "dcs/data/conllu/files" / work).glob("*.conllu")):
            chapter, cur, text, sid, counter = None, None, None, None, None
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.startswith("## chapter:"):
                    chapter = line.split(":", 1)[1].strip()
                elif line.startswith("# text ="):
                    text = line.split("=", 1)[1].strip()
                elif line.startswith("# sent_id ="):
                    sid = line.split("=", 1)[1].strip()
                elif line.startswith("# sent_counter ="):
                    counter = int(line.split("=", 1)[1])
                elif line.startswith("# sent_subcounter ="):
                    key = (work, chapter, counter)
                    if cur is not None and cur["key"] == key:
                        cur["lines"].append(text)
                        cur["ids"].append(sid)
                    else:
                        cur = {"key": key, "lines": [text], "ids": [sid]}
                        rows.append(cur)
    return pd.DataFrame([{"work": r["key"][0], "chapter": r["key"][1], "verse": r["key"][2], "lines": len(r["lines"]),
                          "dcs_sent_ids": " ".join(r["ids"]), "iast": " / ".join(r["lines"]), "_lines": r["lines"]}
                         for r in rows])


def as_dataset_prints(iast):
    """Join, in IAST, what DCS splits at word ends, as the dataset prints it: a word ending in a consonant joins the next
    (tad antar -> tadantar, so Devanagari gets तदन्तर्, not तद्अन्तर्), and an avagraha joins the word before."""
    t = re.sub(r"\s+", " ", iast).strip()
    t = re.sub(f"(?<=[{CONSONANT_END}]) (?=\\S)", "", t)
    return re.sub(r" '", "'", t)


def letters(t):
    """Letters only, spelling normalised: a nasal + halant before a consonant as anusvāra (चन्द्र = चंद्र), candrabindu
    as anusvāra; no spaces, daṇḍas or avagraha."""
    t = re.sub("[ङञणनम]्(?=[क-ह])", "ं", t).replace("ँ", "ं")
    return re.sub(r"[^ऀ-ॿ]|[।॥ऽ]", "", t)


def longest_shared(h, texts):
    """The longest run of h's letters found in any of texts, as a share of h."""
    from difflib import SequenceMatcher
    return max(SequenceMatcher(None, h, d, autojunk=False).find_longest_match(0, len(h), 0, len(d)).size
               for d in texts) / len(h)


def quotas(counts, n):
    share = counts / counts.sum() * n
    q = np.floor(share).astype(int)
    for k in (share - q).sort_values(ascending=False).index[: n - q.sum()]:
        q[k] += 1
    return q


def run(a):
    from indic_transliteration import sanscript
    sys.path[:0] = [str(X.require("vagdhenu") / "src")]
    import render_core as RC  # his code, unchanged
    dcs = X.require("dcs")
    ds = pd.read_csv(C.ROOT / "manifests" / "all_clips.csv")
    vocab = set((X.require("vagdhenu") / "src/reference_bank/vocab.txt").read_text(encoding="utf-8").split("\n"))
    ds_letters = [letters(t) for t in ds.text]
    lo, hi = ds.text.str.len().min(), ds.text.str.len().max()

    v = dcs_verses(dcs)
    to_deva = lambda s: sanscript.transliterate(s, sanscript.IAST, sanscript.DEVANAGARI)  # noqa: E731
    v["text"] = [to_deva(" ".join(as_dataset_prints(ln) for ln in ls)) for ls in v["_lines"]]
    v["meter"] = [RC.detect_meter_key(t) or "" for t in v.text]
    v["family"] = v.meter.map(C.METER_FAMILY).fillna("")
    v["recognized"] = v.meter != ""
    v["new"] = [not any(letters(to_deva(as_dataset_prints(ln))) in d for ln in ls for d in ds_letters)
                for ls in v["_lines"]]
    v["in_vocab"] = [set(t) - {" "} <= vocab for t in v.text]
    v["length_ok"] = v.text.str.len().between(lo, hi)
    v["passes"] = v.recognized & v.new & v.in_vocab & v.length_ok
    v = v.drop(columns="_lines")

    ok = v[v.passes]
    have = ok.family.value_counts()
    fam = ds.family.value_counts()
    fam = fam[fam.index.isin(have.index)]
    q = quotas(fam, C.TEST_VERSES_N)
    short = {k: (q[k], have[k]) for k in q.index if have[k] < q[k]}
    if short:
        sys.exit(f"not enough verses for {short} (family: (needed, available))")
    rng = np.random.default_rng(C.TEST_VERSES_SEED)
    picked = []
    for k in q.index:
        pool = ok[ok.family == k].sort_values(["work", "chapter", "verse"])
        picked.append(pool.iloc[np.sort(rng.choice(len(pool), q[k], replace=False))])
    out = (pd.concat(picked).sort_values(["work", "chapter", "verse"]).reset_index(drop=True))
    out.insert(0, "n", range(1, len(out) + 1))
    out["shared_max"] = [round(max(longest_shared(letters(to_deva(as_dataset_prints(ln))), ds_letters)
                                   for ln in t.split(" / ")), 3) for t in out.iast]
    cols = ["n", "work", "chapter", "verse", "meter", "family", "text", "iast", "dcs_sent_ids", "shared_max"]

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(path, index=False)
    v.to_csv(path.with_name(path.stem + "_pool.csv"), index=False)
    left_out = sorted(set(ds.family) - set(q.index))
    lines = [f"# Test verses (set B): {len(out)} verses from the Digital Corpus of Sanskrit", "",
             f"Source: DCS (Oliver Hellwig, CC BY 4.0) at `{C.DCS_COMMIT[:12]}`, works: {', '.join(C.DCS_WORKS)}. "
             f"Seed {C.TEST_VERSES_SEED}.", "",
             "| Step | Verses |", "|:--|--:|",
             f"| Verses in the works | {len(v)} |", f"| His detector recognises the meter | {v.recognized.sum()} |",
             f"| ...and no line appears in the dataset | {(v.recognized & v.new).sum()} |",
             f"| ...and every character is in his vocab | {(v.recognized & v.new & v.in_vocab).sum()} |",
             f"| ...and the length is within the dataset's ({lo}-{hi} characters): passes | {v.passes.sum()} |", "",
             "| Family | Dataset verses | Share of drawn families | Drawn | Available |", "|:--|--:|--:|--:|--:|"]
    lines += [f"| {k} | {fam[k]} | {fam[k] / fam.sum():.1%} | {q[k]} | {have[k]} |" for k in q.index]
    lines += ["", f"*Dataset families with no recognised verse in the source, so not drawn: "
              f"{', '.join(f'{k} ({(ds.family == k).sum()})' for k in left_out)}.*", "",
              f"Verses dropped as already in the dataset: {(v.recognized & ~v.new).sum()} (of the recognised).",
              f"Drawn verses, longest run of a half-verse's letters found in any dataset verse: median "
              f"{out.shared_max.median():.0%}, highest {out.shared_max.max():.0%} (a shared word or two, not a shared "
              "line).",
              f"By work, drawn: {', '.join(f'{k} {n}' for k, n in out.work.value_counts().items())}."]
    path.with_name(path.stem + "_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
