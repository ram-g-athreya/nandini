"""Renders for the checks after training (scored by `nandini speech-check score` / `trend-check score`, in this code's
own environment).

  nandini speech-check render --ema EMA [--data training/dataset] [--out training/checks/speech]
  nandini trend-check  render --ema EMA [EMA ...] --bank BANK_DIR [--verses CSV] [--save-mel] [--out training/checks/trend] [--dataset-dir D]

Run in the training environment.

speech check: does the model speak? 3 training verses (8–15 s; 2 style_a, 1 style_b, different works; drawn with seed
20261107), each with a 5–8 s training clip of its style as the voice reference, rendered in one piece with IndicF5's own
inference at Vāgdhenu's locked voice_armA settings (euler, nfe 64, sway −0.7, seed 50) at cfg 1.2, and at cfg 3.0 (his
production column) for comparison; the length is fixed to the clip's. Vocoder: vocos. Not his full renderer.

trend check: the 10 check verses (--verses, default manifests/training_check_verses.csv, drawn with seed
20261109) rendered exactly as his website does (demo/app.py): his verse text from the dataset's metadata unchanged,
meter auto-detected (else his vasantatilakā fallback), `Renderer.render_one(text, meter, seed=60)` with his defaults, his
render_core.py unchanged (external/vagdhenu). Changes, all from outside his code: the model (--ema), the reference bank
(--bank, e.g. the female bank), and the last vocoder call (vocos on the same mel, in place of his BigVGAN, which
adds a frame-rate buzz on a female voice). One render per verse and checkpoint: OUT/<step>_<n>.wav; with --save-mel also
OUT/<step>_<n>_mel.npz (the mel of each pāda before the vocoder, and his joining settings), for `nandini vocoder-decode`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .. import config as C
from .. import external as X
from ..dataset import key, metadata
from .data import model_text_fn, vocab_file
from .warm import CFG

SPEAKS_PICK_SEED, SPEAKS_SEED = 20261107, 50
SPEAKS_SETTINGS = (("cfg1.2", 1.2), ("cfg3.0", 3.0))
NFE, SWAY = 64, -0.7
WEB_SEED = 60
CHECK_VERSES = C.ROOT / "manifests" / "training_check_verses.csv"


def _device(d):
    import torch
    if d != "auto":
        return d
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def speaks_pick(rows):
    m = rows.copy()
    m["style"] = m.clip_id.str.split("/").str[0]
    m["work"] = m.clip_id.map(key).str.split("__").str[1]
    m["dur"] = m.duration
    rng = np.random.default_rng(SPEAKS_PICK_SEED)
    out, used = [], set()
    for style in ("style_a", "style_a", "style_b"):
        c = m[(m["style"] == style) & m.dur.between(8, 15) & ~m.work.isin(used)]
        t = c.iloc[rng.integers(len(c))]
        used.add(t.work)
        r = m[(m["style"] == style) & m.dur.between(5, 8) & (m.clip_id != t.clip_id)]
        out.append((t, r.iloc[rng.integers(len(r))]))
    return out


def speaks(a):
    import torch
    from f5_tts.infer.utils_infer import infer_process, load_checkpoint, load_model, load_vocoder, preprocess_ref_audio_text
    from f5_tts.model import DiT
    dev = _device(a.device)
    rows = pd.read_csv(Path(a.data) / "training_text.csv")
    cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=str(vocab_file()), device=dev)
    cfm = load_checkpoint(cfm, a.ema[0], dev, use_ema=True)
    step = torch.load(a.ema[0], map_location="cpu", weights_only=True).get("step")
    voc = load_vocoder("vocos", device=dev)
    out = Path(a.out)
    (out / "render").mkdir(parents=True, exist_ok=True)
    res = []
    for t, r in speaks_pick(rows):
        ref_audio, ref_text = preprocess_ref_audio_text(r.audio, r.text, clip_short=True)
        ref_len = sf.info(ref_audio).duration
        for name, cfg in SPEAKS_SETTINGS:
            torch.manual_seed(SPEAKS_SEED)
            w, sr, _ = infer_process(ref_audio, ref_text, t.text, cfm, voc, mel_spec_type="vocos", nfe_step=NFE,
                                     cfg_strength=cfg, sway_sampling_coef=SWAY, fix_duration=ref_len + t.dur, device=dev)
            w = np.asarray(w, dtype=np.float32)
            f = out / "render" / f"{key(t.clip_id)}__{name}.wav"
            sf.write(f, w, sr)
            rms = float(np.sqrt((w ** 2).mean())) if len(w) else 0.0
            res.append({"step": step, "clip_id": t.clip_id, "setting": name, "render": f"render/{f.name}",  # relative to OUT, so the folder can move
                        "reference": r.clip_id, "seconds": round(len(w) / sr, 2), "rms": round(rms, 4),
                        "not_silent": bool(np.isfinite(w).all() and rms > 0.01), "source_text": t.source_text})
            print(f"{key(t.clip_id)} {name}: {len(w) / sr:.1f} s, rms {rms:.3f}", flush=True)
            pd.DataFrame(res).to_csv(out / "renders.csv", index=False)  # after each render, so a cut-off run can be scored
    print(f"step {step}: {len(res)} renders, all not silent: {all(r['not_silent'] for r in res)}")


def his_text(clip_id, dataset_dir=None):
    style, name = clip_id.split("/", 1)
    md = pd.read_csv(metadata(style, dataset_dir))
    return md.set_index("file_name").text_devanagari[name].strip()  # the website strips its input


def web(a):
    import json
    import torch
    from huggingface_hub import hf_hub_download
    from vocos import Vocos
    sys.path[:0] = [str(X.require("vagdhenu") / "src"), str(X.require("bigvgan"))]
    import render_core as RC  # his code, unchanged
    dev = _device(a.device)
    bank = Path(a.bank) / "bank.json"
    b = json.loads(bank.read_text())
    meters = [k for k, v in b.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v]
    alias = {k.lower(): k for k in meters} | {b[k]["wav"].removesuffix(".wav").lower(): k for k in meters}
    voc_his = hf_hub_download(C.VAGDHENU_WEIGHTS_REPO, "voc_bigvgan_EMA_2026-06-11.pth",
                              revision=C.VAGDHENU_WEIGHTS_REV)  # loaded by his Renderer, not used for the output
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(dev)

    def vocos_decode(mel):
        m = torch.from_numpy(mel)
        if m.dim() == 3 and m.shape[1] != 100 and m.shape[2] == 100:
            m = m.transpose(1, 2)
        with torch.no_grad():
            return vocos.decode(m.to(dev)).squeeze().cpu().numpy().astype(np.float32)

    v = pd.read_csv(getattr(a, "verses", None) or CHECK_VERSES)[["n", "clip_id"]]
    v["text"] = [his_text(c, a.dataset_dir) for c in v.clip_id]
    picked = []
    for t in v.text:
        k = alias.get((RC.detect_meter_key(t) or "").lower())
        picked.append((k, True) if k else ("vasantatilakā" if "vasantatilakā" in meters else meters[0], False))
    v["meter"], v["recognized"] = [p[0] for p in picked], [p[1] for p in picked]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v.to_csv(out / "verses.csv", index=False)
    for ema in a.ema:
        step = torch.load(ema, map_location="cpu", weights_only=True).get("step") or Path(ema).stem  # no step: name
        todo = [r for r in v.itertuples() if not (out / f"{step}_{r.n}.wav").exists()]
        if not todo:
            continue
        r_ = RC.Renderer(str(ema), voc_his, str(bank), device=dev, vocab_file=str(vocab_file()))
        mels, joins = [], {}

        def decode_and_keep(mel):
            mels.append(np.asarray(mel, dtype=np.float32))
            return vocos_decode(mel)

        stitch = r_._stitch

        def stitch_and_keep(segs, gaps, fric=False, halant=False):
            joins.update(gaps=np.array([len(g) for g in gaps]), fric=fric, halant=halant)
            return stitch(segs, gaps, fric=fric, halant=halant)

        r_._bvgan, r_._stitch = decode_and_keep, stitch_and_keep
        for r in todo:
            mels.clear(); joins.clear()
            sr, y = r_.render_one(r.text, r.meter, seed=WEB_SEED)
            sf.write(out / f"{step}_{r.n}.wav", y, sr)
            if getattr(a, "save_mel", False):  # one mel per pāda, and his joining settings: `nandini vocoder-decode`
                np.savez(out / f"{step}_{r.n}_mel.npz", n_pieces=len(mels), **{f"mel_{i}": m for i, m in enumerate(mels)},
                         **joins)
            print(f"step {step} verse {r.n} ({r.meter}{'' if r.recognized else ', fallback'}): {len(y) / sr:.1f} s",
                  flush=True)
        del r_


def run(a):
    {"speech": speaks, "trend": web}[a.what](a)
