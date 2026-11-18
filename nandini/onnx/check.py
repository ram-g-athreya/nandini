"""Checks of the ONNX version against PyTorch. Run in the training environment (both torch and onnxruntime).

  nandini onnx check graphs --ema EMA --vocoder VOCODER --bank BANK_DIR [--onnx training/onnx]
  nandini onnx check verses --ema EMA --vocoder VOCODER --bank BANK_DIR [--onnx training/onnx] [--n 1 2 3 4 5]

graphs: each graph against its torch module on the same real inputs, on CPU: a bank clip as the reference (its mel,
        preprocessed as the renderer does), a verse's text as tokens, a fixed noise draw, at several times t, as the
        batch of two the renderer uses; BigVGAN, and vocos with the numpy inverse STFT, on that clip's mel. The bar: the largest difference at most
        ONNX_GRAPH_TOL of the torch output's range. Writes OUT/check_graphs.json.
verses: test verses rendered by Vāgdhenu's torch renderer and by the ONNX renderer on the CPU, from the same noise
        (torch's CPU generator, seeded as his renderer seeds it), with the meters the comparison used. Writes
        OUT/check_verses/{torch,onnx}/<n>.wav and times.csv; `nandini onnx check score` then compares length and
        Su-śrotā's errors (this code's environment).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import config as C
from .. import external as X


def inputs(bank_dir, frames_gen=300, seed=0):
    """(x, cond, text) for the batch of two, from the bank's first entry, as the renderer builds them."""
    sys.path.insert(0, str(X.require("vagdhenu") / "src"))
    from vagdhenu_mlx import audio as A
    from vagdhenu_mlx.text import load_vocab, tokenize
    import prep_text as PT
    from ..training.data import vocab_file
    bank = json.loads((Path(bank_dir) / "bank.json").read_text(encoding="utf-8"))
    key = next(k for k, v in bank.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v)
    e = bank[key]
    wave, ref_text = A.preprocess_ref(str(Path(bank_dir) / e["wav"]), e["ref_text"], log=lambda *_: None)
    wave, _ = A.normalize_rms(wave)
    mel = A.log_mel(wave)
    gen = PT.model_text(e["full_verse"].split("।")[1] if "।" in e["full_verse"] else e["full_verse"])
    ids = tokenize(ref_text + gen, load_vocab(str(vocab_file())))
    n = mel.shape[0] + frames_gen
    rng = np.random.default_rng(seed)
    x = np.repeat(rng.standard_normal((1, n, 100)).astype(np.float32), 2, axis=0)
    cond = np.zeros((2, n, 100), np.float32)
    cond[0, :mel.shape[0]] = mel
    text = np.full((2, n), -1, np.int64)
    text[0, :min(len(ids), n)] = ids[:n]
    return key, x, cond, text, mel


def graphs(a):
    import onnxruntime as ort
    import torch
    from . import make
    from ..training.vocoder import load_generator
    onnx_dir = Path(a.onnx)
    key, x, cond, text, mel = inputs(a.bank)
    so = ort.SessionOptions()
    res = {"reference": key, "frames": int(x.shape[1]), "tol": C.ONNX_GRAPH_TOL, "dit": [], "vocoder": None}

    Step, Vocoder, _ = make.wrappers()
    step = Step(make.load_dit(a.ema)).eval()
    sess = ort.InferenceSession(str(onnx_dir / "nandini_dit.onnx"), so, providers=["CPUExecutionProvider"])
    for tv in (0.0, 0.3, 0.9):
        t = np.full((2,), tv, np.float32)
        t0 = time.time()
        with torch.no_grad():
            ref = step(*(torch.from_numpy(v) for v in (x, cond, text, t))).numpy()
        t1 = time.time()
        got = sess.run(None, {"x": x, "cond": cond, "text": text, "t": t})[0]
        t2 = time.time()
        d = float(np.abs(got - ref).max()); rng_ = float(ref.max() - ref.min())
        res["dit"].append({"t": tv, "max_abs_diff": d, "range": rng_, "share": d / rng_, "torch_s": round(t1 - t0, 2),
                           "onnx_s": round(t2 - t1, 2)})
        print(f"dit t={tv}: largest difference {d:.3g} of range {rng_:.3g} ({d / rng_:.2e})", flush=True)

    g = load_generator(a.vocoder, "cpu")
    voc = Vocoder(g).eval()
    vs = ort.InferenceSession(str(onnx_dir / "nandini_vocoder.onnx"), so, providers=["CPUExecutionProvider"])
    m = mel.T[None].astype(np.float32)
    with torch.no_grad():
        ref = voc(torch.from_numpy(m)).numpy()
    got = vs.run(None, {"mel": m})[0]
    d = float(np.abs(got - ref).max()); rng_ = float(ref.max() - ref.min())
    res["vocoder"] = {"frames": int(m.shape[2]), "max_abs_diff": d, "range": rng_, "share": d / rng_}
    print(f"vocoder: largest difference {d:.3g} of range {rng_:.3g} ({d / rng_:.2e})", flush=True)

    from f5_tts.infer.utils_infer import load_vocoder
    from .render import istft
    vc = load_vocoder("vocos", device="cpu")
    with torch.no_grad():
        ref = vc.decode(torch.from_numpy(m)).numpy()[0]
    re_, im = ort.InferenceSession(str(onnx_dir / "vocos_spectrum.onnx"), so,
                                   providers=["CPUExecutionProvider"]).run(None, {"mel": m})
    got = istft(re_[0] + 1j * im[0])
    d = float(np.abs(got - ref).max()); rng_ = float(ref.max() - ref.min())
    res["vocos"] = {"max_abs_diff": d, "range": rng_, "share": d / rng_, "same_length": len(got) == len(ref)}
    print(f"vocos (with the numpy inverse STFT): largest difference {d:.3g} of range {rng_:.3g} ({d / rng_:.2e})")

    res["pass"] = (all(r["share"] <= C.ONNX_GRAPH_TOL for r in res["dit"]) and res["vocoder"]["share"] <= C.ONNX_GRAPH_TOL
                   and res["vocos"]["share"] <= C.ONNX_GRAPH_TOL and res["vocos"]["same_length"])
    (onnx_dir / "check_graphs.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"check 1 (each graph against torch, bar {C.ONNX_GRAPH_TOL:g} of range): "
          f"{'PASS' if res['pass'] else 'FAIL'}", flush=True)


def verses(a):
    import time as _t
    import pandas as pd
    import soundfile as sf
    import torch
    # Same noise needs the CPU throughout. Asked for the CPU, IndicF5 still puts vocos on MPS when it can see it (its
    # device is chosen at import), so MPS is hidden from it for this check.
    torch.backends.mps.is_available = lambda: False
    sys.path[:0] = [str(X.require("vagdhenu") / "src"), str(X.require("bigvgan"))]
    import render_core as RC
    from ..evaluation.render import verses as set_verses
    from ..training.data import vocab_file
    from .render import Renderer
    out = Path(a.onnx) / "check_verses"
    v = set_verses("B").set_index("n")
    meters = {}
    bank = json.loads((Path(a.bank) / "bank.json").read_text(encoding="utf-8"))
    ms = [k for k, e in bank.items() if not k.startswith("_") and isinstance(e, dict) and "wav" in e]
    alias = {k.lower(): k for k in ms} | {bank[k]["wav"].removesuffix(".wav").lower(): k for k in ms}
    for n in a.n:
        k = alias.get((RC.detect_meter_key(v.text[n]) or "").lower())
        meters[n] = k if k else ("vasantatilakā" if "vasantatilakā" in ms else ms[0])
    rows = []
    for name in ("torch", "onnx"):
        (out / name).mkdir(parents=True, exist_ok=True)
        if name == "torch":
            r = RC.Renderer(str(a.ema), str(a.vocoder), str(Path(a.bank) / "bank.json"), device="cpu",
                            vocab_file=str(vocab_file()))
        else:
            r = Renderer(a.onnx, Path(a.bank) / "bank.json", vocab_file(), noise="torch")
        for n in a.n:
            t0 = _t.time()
            sr, y = r.render_one(v.text[n], meters[n], seed=C.WEB_SEED)
            sf.write(out / name / f"{n}.wav", y, sr)
            rows.append({"renderer": name, "n": n, "meter": meters[n], "samples": len(y),
                         "seconds": round(len(y) / sr, 3), "render_s": round(_t.time() - t0, 1)})
            print(rows[-1], flush=True)
        del r
    pd.DataFrame(rows).to_csv(out / "times.csv", index=False)


def score(a):
    """Check 2's verdict: the same length to the sample, and Su-śrotā's character errors within ONNX_CHAR_TOL per verse."""
    import pandas as pd
    import soundfile as sf
    from ..asr import transcribe
    from ..evaluation.render import verses as set_verses
    from ..pipeline.pitch_shift import _lev, _normalise
    out = Path(a.onnx) / "check_verses"
    v = set_verses("B").set_index("n")
    rows = []
    for p in sorted((out / "onnx").glob("*.wav"), key=lambda p: int(p.stem)):
        n = int(p.stem)
        q = out / "torch" / p.name
        if not q.exists():
            continue
        t = _normalise(v.text[n])
        e = {k: _lev(t, _normalise(transcribe(str(f)))) for k, f in (("torch", q), ("onnx", p))}
        yt, yo = sf.read(q)[0], sf.read(p)[0]
        same = len(yt) == len(yo)
        rows.append({"n": n, "samples_torch": len(yt), "samples_onnx": len(yo), "same_length": same,
                     "max_abs_diff": float(np.abs(yt - yo).max()) if same else np.nan,
                     "chars": len(t), "errors_torch": e["torch"], "errors_onnx": e["onnx"],
                     "error_gap": abs(e["torch"] - e["onnx"])})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "scores.csv", index=False)
    ok = bool(len(d)) and bool(d.same_length.all()) and bool((d.error_gap <= C.ONNX_CHAR_TOL).all())
    (out / "verdict.json").write_text(json.dumps({"verses": len(d), "pass": ok, "char_tol": C.ONNX_CHAR_TOL}) + "\n")
    print(f"check 2 ({len(d)} verses: same length, errors within {C.ONNX_CHAR_TOL} character): {'PASS' if ok else 'FAIL'}")


def run(a):
    {"graphs": graphs, "verses": verses, "score": score, "small": small, "score-small": score_small}[a.step](a)


SMALL_DITS = ("nandini_dit.onnx", "nandini_dit_int8.onnx", "nandini_dit_r32_int8.onnx")
SMALL_VOCS = ("nandini_vocoder.onnx", "nandini_vocoder_fp16.onnx", "nandini_vocoder_int8.onnx")
MEL_TOL = 0.09  # nats: about the spread between two seeds of the same pipeline (Vāgdhenu docs/MLX_PORT.md §4)


def small(a):
    """The smaller files' renders: each transformer file with every vocoder file, from the same noise (torch's CPU
    generator), on the check verses. Writes OUT/check_small/<transformer>__<vocoder>/<n>.wav."""
    import time as _t
    import soundfile as sf
    import torch  # noqa: F401  (the noise only)
    from ..evaluation.render import verses as set_verses
    from ..training.data import vocab_file
    from .render import Renderer
    out = Path(a.onnx) / "check_small"
    v = set_verses("B").set_index("n")
    meters = dict(pd_read(Path(a.onnx) / "check_verses" / "times.csv"))
    for dit in SMALL_DITS:
        r = Renderer(a.onnx, Path(a.bank) / "bank.json", vocab_file(), noise="torch", dit=dit, vocoders=SMALL_VOCS)
        for n in a.n:
            t0 = _t.time()
            outs = r.render_all(v.text[n], meters[n], seed=C.WEB_SEED)
            for voc, (sr, y) in zip(SMALL_VOCS, outs):
                d = out / f"{Path(dit).stem}__{Path(voc).stem}"
                d.mkdir(parents=True, exist_ok=True)
                sf.write(d / f"{n}.wav", y, sr)
            print({"dit": dit, "n": n, "render_s": round(_t.time() - t0, 1)}, flush=True)
        del r


def pd_read(p):
    import pandas as pd
    d = pd.read_csv(p)
    return d[d.renderer == "onnx"][["n", "meter"]].itertuples(index=False)


def score_small(a):
    """Each smaller combination against the fp32 files' render (same noise): Su-śrotā's character errors within
    ONNX_CHAR_TOL per verse, and mean |log-mel difference| over the verses at most MEL_TOL."""
    import pandas as pd
    import soundfile as sf
    sys.path.insert(0, str(X.require("vagdhenu") / "src"))
    from vagdhenu_mlx.audio import log_mel
    from ..asr import transcribe
    from ..evaluation.render import verses as set_verses
    from ..pipeline.pitch_shift import _lev, _normalise
    out = Path(a.onnx) / "check_small"
    ref = out / f"{Path(SMALL_DITS[0]).stem}__{Path(SMALL_VOCS[0]).stem}"
    v = set_verses("B").set_index("n")
    err = {}
    rows = []
    for d in sorted(out.iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.wav"), key=lambda p: int(p.stem)):
            n = int(p.stem)
            t = _normalise(v.text[n])
            err[(d.name, n)] = _lev(t, _normalise(transcribe(str(p))))
            if d == ref:
                continue
            y, yr = sf.read(p)[0], sf.read(ref / p.name)[0]
            k = min(len(y), len(yr))
            mel_d = float(np.abs(log_mel(y[:k]) - log_mel(yr[:k])).mean())
            rows.append({"files": d.name, "n": n, "same_length": len(y) == len(yr), "mel_diff": round(mel_d, 4)})
    for r in rows:
        r["errors"], r["errors_fp32"] = err[(r["files"], r["n"])], err[(ref.name, r["n"])]
        r["error_gap"] = abs(r["errors"] - r["errors_fp32"])
    d = pd.DataFrame(rows)
    d.to_csv(out / "scores.csv", index=False)
    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in Path(a.onnx).glob("nandini_*.onnx")}
    summ = []
    for files, g in d.groupby("files"):
        dit, voc = files.split("__")
        ok = bool((g.error_gap <= C.ONNX_CHAR_TOL).all()) and float(g.mel_diff.mean()) <= MEL_TOL
        summ.append({"files": files, "mb": sizes.get(dit + ".onnx", 0) + sizes.get(voc + ".onnx", 0),
                     "mel_diff_mean": round(float(g.mel_diff.mean()), 4), "error_gap_max": int(g.error_gap.max()),
                     "pass": ok})
    s = pd.DataFrame(summ).sort_values("mb")
    s.to_csv(out / "summary.csv", index=False)
    print(s.to_string(index=False))
