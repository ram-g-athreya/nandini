"""The commands that read a run folder (tables, report, listen build), on a tiny synthetic run: no models, no downloads."""
import json

import numpy as np
import pandas as pd
import soundfile as sf

from nandini import cli
from nandini import config as C
from nandini.asr import heard
from nandini.dataset import key
from nandini.text import Verse, deva, iast, to_slp1

MANIFEST = C.ROOT / "manifests" / "all_clips.csv"


def test_deva_and_iast():
    v = Verse(pd.read_csv(MANIFEST).text.iloc[0])
    assert to_slp1("".join(deva(v.slp[a:b]) for a, b in v.spans)) == v.chars
    assert iast("kfzRa") == "kṛṣṇa"


def test_heard_on_exact_transcript():
    text = pd.read_csv(MANIFEST).text.iloc[0]
    v = Verse(text)
    assert "".join(heard(v, text)) == v.chars


def _run(tmp_path):
    """Two verses: one with 0 errors, one with 1, as `nandini generate` and `nandini mark` would leave them."""
    m = pd.read_csv(MANIFEST).head(2)
    run, ds = tmp_path / "out", tmp_path / "vagdhenu-data"
    kept, att = [], []
    for e, r in enumerate(m.itertuples()):
        v = Verse(r.text)
        tone = 0.1 * np.sin(np.linspace(0, 2000, 24000))
        for p in (ds / r.clip_id, run / "final" / f"{key(r.clip_id)}.wav"):
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, tone, 24000)
        kept.append({"clip_id": r.clip_id, "index": r.index, "style": r.style, "family": r.family, "syllables": v.n,
                     "counted": v.n, "errors": e, "kept_j": 0, "attempts": 1, "met": True})
        att.append({"clip_id": r.clip_id, "index": r.index, "style": r.style, "family": r.family, "j": 0, "seed": r.index,
                    "syllables": v.n, "counted": v.n, "errors": e, "convert_s": 1.0, "rest_s": 1.0})
        her_f = "0" * v.n if e == 0 else "1" + "0" * (v.n - 1)
        her_hyp = r.text if e == 0 else r.text[1:]  # drop the first character: the first syllable is wrong
        rec = {"id": r.clip_id, "his_hyp": r.text, "her_hyp": her_hyp, "his_f": "0" * v.n, "her_f": her_f,
               "his_t": [0.1 * k for k in range(v.n)], "her_t": [0.1 * k for k in range(v.n)], "his_dur": 1.0,
               "her_dur": 1.0, "syl": [list(s) for s in v.spans]}
        (run / "marks").mkdir(parents=True, exist_ok=True)
        with open(run / "marks" / "marks_part0.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pd.DataFrame(kept).to_csv(run / "kept_part0.csv", index=False)
    pd.DataFrame(att).to_csv(run / "attempts_part0.csv", index=False)
    return run, ds


def test_tables_report_and_page(tmp_path):
    run, ds = _run(tmp_path)
    cli.main(["tables", "--run", str(run)])
    t = pd.read_csv(run / "tables" / "transcripts.csv")
    assert list(t.errors) == [0, 1]
    s = pd.read_csv(run / "tables" / "syllables.csv.gz")
    assert s.error.sum() == 1 and s.groupby("index").error.sum().tolist() == [0, 1]
    assert (run / "tables" / "substitutions.md").exists()

    cli.main(["report", "--run", str(run)])
    for f in ("report.md", "errors-by-style.png", "attempts-used.png", "met-by-meter.png"):
        assert (run / "report" / f).exists(), f
    assert len(pd.read_csv(run / "kept.csv")) == 2

    cli.main(["listen", "build", "--run", str(run), "--dataset-dir", str(ds)])
    page = (run / "listen" / "index.html").read_text()
    data = json.loads(page[page.index("const DATA = ") + 13:page.index(";\n", page.index("const DATA = "))])
    assert [d["err"] for d in data] == [0, 1] and all("sy" in d for d in data)
    assert data[0]["hers"].startswith("../final/")
    link = run / "listen" / data[0]["his"]
    assert link.is_symlink() and link.resolve() == (ds / data[0]["id"]).resolve()
