"""The listening page: every kept verse with Vāgdhenu's recording, Nandinī's kept clip and the Devanagari text, the
errors and attempts, and a place to record a verdict and a note.

  nandini listen build [--run out] [--manifest M] [--dataset-dir D]
  nandini listen serve [--run out] [--port 8765]          then open http://localhost:8765/listen/

Reads RUN/kept_part*.csv, RUN/final/ and, if present, RUN/marks/ (`nandini mark`). With marks, each syllable is marked
(red: wrong on Nandinī's clip, right on Vāgdhenu's recording; grey: wrong on both), hovering shows what the ASR heard,
and the syllable being sung is highlighted while playing. Writes RUN/listen/index.html and RUN/listen/vagdhenu/, links
to Vāgdhenu's recordings wherever they are (a local copy with --dataset-dir, else the Hugging Face cache, downloaded on
first use). Nandinī's clips are played from RUN/final/. Nothing is copied, so the page stays small.

Verdicts are kept in the browser (localStorage) and can be exported to and imported from CSV.
"""
import json
import os
from pathlib import Path

import pandas as pd

from ..analysis.marks import read_kept, read_marks
from ..dataset import key, source

TEMPLATE = Path(__file__).with_name("page.html")


def _marks_fields(verse, text, m):
    """The page's per-syllable fields for one verse, or None when the syllables can't be shown as the text."""
    from ..asr import heard
    from ..text import Verse, deva, iast
    syl, last = [], 0
    for a, b in verse.spans:
        syl.append([deva(verse.slp[a:b]), " " in verse.slp[last:a]])
        last = b
    # Verse() reads ॐ as its spoken form ओम्; show ॐ where the text has it.
    oms = text.count("ॐ")
    for s in syl:
        if oms and s[0] == "ओम्":
            s[0], oms = "ॐ", oms - 1
    # Use the syllables when they read back as the text, or differ only by stray marks the comparison ignores anyway
    # (a doubled vowel sign, a stray virama or a lone vowel sign in the source text; shown without them).
    shown = "".join((" " if sp else "") + t for t, sp in syl)
    if shown != text.strip() and Verse(shown).slp != verse.slp:
        return None
    f = {"sy": syl, "hf": m["his_f"], "nf": m["her_f"], "ht": m["his_t"], "nt": m["her_t"]}
    # What the ASR heard, for marked syllables only: [Devanagari, IAST] per recording, else null.
    hh, nh = heard(verse, m["his_hyp"]), heard(verse, m["her_hyp"])
    f["heard"] = [None if m["his_f"][k] == m["her_f"][k] == "0" else [[deva(x), iast(x)] for x in (hh[k], nh[k])]
                  for k in range(verse.n)]
    return f


def build(args):
    from ..text import Verse
    run = Path(args.run)
    kept = read_kept(run)
    marks = read_marks(run)
    texts = pd.read_csv(args.manifest).set_index("clip_id").text
    out = run / "listen"
    links = out / "vagdhenu"
    links.mkdir(parents=True, exist_ok=True)
    rows, plain = [], 0
    for r in kept.itertuples():
        k = key(r.clip_id)
        hers = run / "final" / f"{k}.wav"
        if not hers.exists():
            raise SystemExit(f"missing {hers}")
        link = links / f"{k}.wav"
        if not link.exists():
            link.symlink_to(Path(source(r.clip_id, args.dataset_dir)).resolve())
        rows.append({"i": int(r.index), "id": r.clip_id, "name": k.split("__", 1)[1], "style": r.style,
                     "meter": r.family, "syl": int(r.syllables), "counted": int(r.counted), "err": int(r.errors),
                     "att": int(r.attempts), "kj": int(r.kept_j), "text": texts[r.clip_id],
                     "his": f"vagdhenu/{k}.wav", "hers": os.path.relpath(hers, out)})
        m = marks.get(r.clip_id)
        if m:
            f = _marks_fields(Verse(texts[r.clip_id]), texts[r.clip_id], m)
            if f:
                rows[-1] |= f
            else:
                plain += 1
    page = (TEMPLATE.read_text().replace("__N__", f"{len(rows):,}")
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")))
    (out / "index.html").write_text(page)
    marked = sum("sy" in r for r in rows)
    print(f"{len(rows)} verses, {marked} with syllable marks ({plain} shown as plain text) -> {out / 'index.html'}")
    if not marks:
        print("no marks yet: run `nandini mark` for syllable marks and highlighting")
