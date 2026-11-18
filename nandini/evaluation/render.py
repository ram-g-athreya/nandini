"""Renders for the comparison, through his website's path. Runs in the training environment.

  nandini compare render --system V|Vs [--sets A B C] [--out compare]
  nandini compare render --system N --ema EMA --bank BANK_DIR --vocoder VOCODER [--sets A B] [--out compare]

Every system goes through his renderer (external/vagdhenu, render_core.py) unchanged, as his website runs it (demo/app.py):
the verse text unchanged, the meter from his detector (else his vasantatilakā fallback), `Renderer.render_one(text,
meter, seed=60)` with his defaults. Only what the system is changes: V and Vs are his released models
(COMPARE_MODELS) with his bank and his BigVGAN; N is --ema with --bank and the --vocoder file, which his renderer
loads as it loads his own BigVGAN.

Sets (COMPARE_SETS says which each system renders):
  A  the 61 pitch-shift verses (manifests/pitch_shift_verses.csv), his text from the dataset's metadata;
  B  the 60 test verses (manifests/test_verses.csv);
  C  his 6 demo verses, the text as on his page (docs/index.html), with his demo MP3s as the reference.

Writes OUT/<set>/verses.csv and OUT/<set>/<system>/<n>.wav (one render per verse; existing files are kept, so a cut-off
run resumes), and OUT/<set>/<system>/meters.csv (the meter each verse rendered with, and whether it was recognised).
"""
import html
import re
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

from .. import config as C
from .. import external as X

SETS = ("A", "B", "C")


def demo_verses():
    """His 6 demo verses: text as on his page (line breaks kept), his meter label, source, and his MP3."""
    page = (X.require("vagdhenu") / "docs" / "index.html").read_text(encoding="utf-8")
    rows = []
    for i, fig in enumerate(re.findall(r'<figure class="clip">(.*?)</figure>', page, re.S), 1):
        badge = re.search(r'<span class="badge">(.*?)</span>', fig).group(1)
        src = re.search(r'<span class="src">(.*?)</span>', fig).group(1)
        verse = re.search(r'<p class="verse">(.*?)</p>', fig, re.S).group(1)
        text = html.unescape(re.sub(r"<br\s*/?>", "\n", verse)).strip()
        audio = re.search(r'src="(audio/[^"]+\.mp3)"', fig).group(1)
        rows.append({"n": i, "his_meter": html.unescape(badge), "source": html.unescape(src), "text": text,
                     "his_audio": f"docs/{audio}"})
    if len(rows) != 6:
        sys.exit(f"expected 6 demo verses on his page, found {len(rows)}")
    return pd.DataFrame(rows)


def verses(name, dataset_dir=None):
    """A set's verses: n, text (what goes into the renderer), and what the set's scoring needs."""
    if name == "A":
        from ..training.checks import his_text
        d = pd.read_csv(C.ROOT / "manifests" / "pitch_shift_verses.csv")
        return pd.DataFrame({"n": d.reading, "clip_id": d.clip_id, "style": d.config, "family": d.family,
                             "text": [his_text(c, dataset_dir) for c in d.clip_id]})
    if name == "B":
        d = pd.read_csv(C.ROOT / "manifests" / "test_verses.csv")
        return d[["n", "family", "text"]].assign(source=d.work + " " + d.chapter + ", v" + d.verse.astype(str))
    return demo_verses()


def _weights(system, a):
    """(model, vocoder, bank.json) for a system."""
    from huggingface_hub import hf_hub_download
    if system == "N":
        missing = [o for o in ("ema", "bank", "vocoder") if not getattr(a, o)]
        if missing:
            sys.exit(f"--system N needs {', '.join('--' + m for m in missing)}")
        return a.ema, a.vocoder, str(Path(a.bank) / "bank.json")
    get = lambda f: hf_hub_download(C.VAGDHENU_WEIGHTS_REPO, f, revision=C.VAGDHENU_WEIGHTS_REV)  # noqa: E731
    return (get(C.COMPARE_MODELS[system]), get(C.HIS_VOCODER_FILE),
            str(X.require("vagdhenu") / "src" / "reference_bank" / "bank.json"))


def run(a):
    import json
    sys.path[:0] = [str(X.require("vagdhenu") / "src"), str(X.require("bigvgan"))]
    import render_core as RC  # his code, unchanged
    from ..training.checks import _device
    from ..training.data import vocab_file
    sets = a.sets or list(C.COMPARE_SETS[a.system])
    wrong = [s for s in sets if s not in C.COMPARE_SETS[a.system]]
    if wrong:
        sys.exit(f"{a.system} doesn't render set {', '.join(wrong)} (COMPARE_SETS: {C.COMPARE_SETS[a.system]})")
    model, voc, bank = _weights(a.system, a)
    b = json.loads(Path(bank).read_text(encoding="utf-8"))
    meters = [k for k, v in b.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v]
    alias = {k.lower(): k for k in meters} | {b[k]["wav"].removesuffix(".wav").lower(): k for k in meters}
    out = Path(a.out)
    todo = []
    for s in sets:
        v = verses(s, a.dataset_dir)
        (out / s / a.system).mkdir(parents=True, exist_ok=True)
        if not (out / s / "verses.csv").exists():
            v.to_csv(out / s / "verses.csv", index=False)
        picked = []
        for t in v.text:  # as demo/app.py picks: detected meter via the bank's aliases, else its _FALLBACK_DISPLAY
            k = alias.get((RC.detect_meter_key(t) or "").lower())
            picked.append((k, True) if k else ("vasantatilakā" if "vasantatilakā" in meters else meters[0], False))
        m = v[["n"]].assign(meter=[p[0] for p in picked], recognized=[p[1] for p in picked])
        m.to_csv(out / s / a.system / "meters.csv", index=False)
        todo += [(s, r.n, r.text, mm) for r, mm in zip(v.itertuples(), m.meter)
                 if not (out / s / a.system / f"{r.n}.wav").exists() and (not a.only or r.n in a.only)]
    if not todo:
        print(f"{a.system}: every render is there")
        return
    dev = _device(a.device)
    r_ = RC.Renderer(str(model), str(voc), bank, device=dev, vocab_file=str(vocab_file()))
    print(f"{a.system} on {dev}: {len(todo)} renders ({', '.join(sets)})", flush=True)
    for i, (s, n, text, meter) in enumerate(todo, 1):
        t0 = time.time()
        sr, y = r_.render_one(text, meter, seed=C.WEB_SEED)
        sf.write(out / s / a.system / f"{n}.wav", y, sr)
        print(f"{a.system} {s}{n} ({meter}): {len(y) / sr:.1f} s audio in {time.time() - t0:.1f} s [{i}/{len(todo)}]",
              flush=True)
