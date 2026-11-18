"""The vocoder fine-tune: his BigVGAN fine-tuned on her clips, made as he made his (NVIDIA's BigVGAN-v2 fine-tuned on
IndicF5's vocos-type mel; the final weights the average of the last 6 checkpoints). Run in the training environment,
with the vocoder extras (third_party/vocoder-requirements.txt).

  nandini vocoder data    [--data training/dataset] [--out training/vocoder/data] [--lag N]
  nandini vocoder align   [--data training/dataset] [--out training/vocoder/align]
  nandini vocoder train   [--data training/vocoder/data] [--out training/vocoder/run]
  nandini vocoder monitor --mels DIR [--run training/vocoder/run]
  nandini vocoder average [--run training/vocoder/run] [--stop STEP]
  nandini vocoder decode  --mels DIR --vocoder vocos|his|FILE --out DIR

data:    every training clip's mel, made with IndicF5's own mel function (what the model outputs and his renderer hands
         to BigVGAN), saved beside the clip padded to a whole number of frames, plus NVIDIA's file lists.
align:   before training, his BigVGAN decodes 3 clips from mels made the training way. Reports the time offset between
         the decoded and the real clip (the pairs follow his vocoder's convention: --lag for `data`) and the echo. The
         errors are scored by `nandini vocoder score --align`.
train:   NVIDIA's train.py at BIGVGAN_COMMIT, unchanged, in fine-tuning mode (precomputed mels), with its v2 config.
         Starts from his generator and NVIDIA's released discriminators, with fresh optimisers and the step count at 0.
         A checkpoint every VOCODER_SAVE_EVERY steps. It runs until stopped: `monitor` stops it.
monitor: every VOCODER_MONITOR_EVERY steps, decodes the monitor mels (--mels, from `trend-check render --save-mel`)
         with the newest checkpoint and measures echo and shiver (nandini/analysis/vocoder_measures.py; the held note
         is found on the vocos decode). Stops training at the first check from VOCODER_MIN_STEP where the echo rule holds
         and shiver is unchanged against VOCODER_FLAT_OVER steps earlier (the 95% paired bootstrap interval spans 0),
         or at --cap (default VOCODER_CAP). Then averages the last VOCODER_AVERAGE checkpoints. Keeps only the newest 2 optimiser files.
average: the final weights: the mean of the last VOCODER_AVERAGE generator checkpoints ending at --stop (default: the
         monitor's stop), saved as RUN/vocoder.pt in the form his renderer loads.
decode:  turns saved mels back into audio with a vocoder, joined as his renderer joins them (each pāda's audio scaled
         down if it clips, then his _stitch). With --vocoder vocos it reproduces the render.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .. import config as C
from .. import external as X

SR, HOP = 24000, 256
_MEL = None


def _bigvgan():
    p = X.require("bigvgan")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def _vagdhenu():
    p = X.require("vagdhenu") / "src"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def _device(d="auto"):
    import torch
    if d != "auto":
        return d
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def f5_mel(y):
    """IndicF5's vocos-type mel of 24 kHz audio: [100, len // 256 + 1], float32."""
    global _MEL
    import torch
    from f5_tts.model.modules import MelSpec
    if _MEL is None:
        _MEL = MelSpec(mel_spec_type="vocos")
    with torch.no_grad():
        return _MEL(torch.from_numpy(np.asarray(y, dtype=np.float32))[None]).squeeze(0).numpy()


def _clip(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: {sr} Hz, expected {SR}"
    return y.mean(1)


def _shift(y, lag, n):
    """Delay by lag samples (advance if negative), then pad or cut to n."""
    y = np.concatenate([np.zeros(lag, np.float32), y]) if lag > 0 else y[-lag:]
    return np.pad(y, (0, max(0, n - len(y))))[:n]


def _his_file():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C.VAGDHENU_WEIGHTS_REPO, C.HIS_VOCODER_FILE, revision=C.VAGDHENU_WEIGHTS_REV)


def _config():
    return _bigvgan() / "configs" / C.NV_BIGVGAN_CONFIG


def load_generator(path, device):
    """A BigVGAN generator for decoding, from his file (state dict) or a train.py checkpoint ({"generator": ...})."""
    import torch
    _bigvgan()
    from bigvgan import BigVGAN
    from env import AttrDict
    g = BigVGAN(AttrDict(json.loads(_config().read_text())), use_cuda_kernel=False)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("generator", sd.get("model", sd))
    g.load_state_dict(sd)
    g.remove_weight_norm()
    return g.to(device).eval()


def decoder(name, device):
    """mel (any of [100, T], [1, 100, T], [1, T, 100]) -> float32 audio."""
    import torch
    if name == "vocos":
        from vocos import Vocos
        voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(device)
        run = voc.decode
    else:
        g = load_generator(_his_file() if name == "his" else name, device)
        run = g

    def dec(mel):
        m = torch.from_numpy(np.asarray(mel, dtype=np.float32))
        if m.dim() == 2:
            m = m[None]
        if m.shape[1] != 100 and m.shape[2] == 100:
            m = m.transpose(1, 2)
        with torch.no_grad():
            return run(m.to(device)).squeeze().cpu().numpy().astype(np.float32)
    return dec


def decode_file(npz, dec):
    """One saved render (`trend-check render --save-mel`) back to audio, joined as his renderer joins it."""
    _vagdhenu()
    import render_core as RC
    z = np.load(npz)
    segs = []
    for i in range(int(z["n_pieces"])):
        y = dec(z[f"mel_{i}"])
        mx = np.abs(y).max()
        segs.append(y / mx * 0.97 if mx > 1 else y)
    gaps = [np.zeros(int(n), np.float32) for n in z["gaps"]]
    return RC.Renderer._stitch(None, segs, gaps, fric=bool(z["fric"]), halant=bool(z["halant"]))


def decode(a):
    dev = _device(a.device)
    dec = decoder(a.vocoder, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.mels).glob("*_mel.npz"))
    for f in files:
        sf.write(out / f.name.replace("_mel.npz", ".wav"), decode_file(f, dec), SR)
    print(f"decoded {len(files)} renders with {a.vocoder} into {out}")


def data(a):
    rows = pd.read_csv(Path(a.data) / "training_text.csv")
    out = Path(a.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    (out / "mels").mkdir(parents=True, exist_ok=True)
    names = []
    for i, r in enumerate(rows.itertuples()):
        name = Path(r.audio).stem
        y = _clip(r.audio)
        mel = f5_mel(y)
        sf.write(out / "wavs" / f"{name}.wav", _shift(y, a.lag, mel.shape[1] * HOP), SR, subtype="FLOAT")
        np.save(out / "mels" / f"{name}.npy", mel)
        names.append(name)
        if (i + 1) % 200 == 0:
            print(f"{i + 1} of {len(rows)} clips", flush=True)
    (out / "train.txt").write_text("\n".join(names) + "\n")
    (out / "val.txt").write_text("\n".join(sorted(names)[:C.VOCODER_VAL_CLIPS]) + "\n")
    json.dump({"clips": len(names), "lag": a.lag, "hours": round(float(rows.duration.sum()) / 3600, 2)
               if "duration" in rows else None}, open(out / "data.json", "w"), indent=1)
    print(f"{len(names)} clips: mels and padded audio in {out}, lag {a.lag} samples")


def _lag(ref, y, search=1024):
    """Samples by which y lags ref, from their loudness envelopes (Hilbert, 2 ms smoothing). A vocoder is free to
    change the phase, so the waveforms themselves don't line up reliably; their envelopes do."""
    from scipy.signal import correlate, hilbert
    n = min(len(ref), len(y))
    k = np.ones(48) / 48
    e = [np.convolve(np.abs(hilbert(x[:n])), k, "same") for x in (ref, y)]
    e = [x - x.mean() for x in e]
    c = correlate(e[1], e[0], mode="full", method="fft")
    lags = np.arange(-n + 1, n)
    w = np.abs(lags) <= search
    return int(lags[w][np.argmax(c[w])])


def align(a):
    from ..analysis.vocoder_measures import echo
    rows = pd.read_csv(Path(a.data) / "training_text.csv")
    rows = rows.sort_values("duration" if "duration" in rows else "duration_s", kind="stable")
    pick = rows.iloc[[len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4]]  # a short, a middle and a long clip
    dec = decoder("his", _device(a.device))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for r in pick.itertuples():
        y = _clip(r.audio)
        z = dec(f5_mel(y))
        name = Path(r.audio).stem
        sf.write(out / f"{name}__real.wav", y, SR)
        sf.write(out / f"{name}__his.wav", z, SR)
        res.append({"clip_id": r.clip_id, "name": name, "seconds": round(len(y) / SR, 2), "decoded_samples": len(z),
                    "real_samples": len(y), "lag": _lag(y, z), "echo_real": round(echo(y), 4),
                    "echo_his": round(echo(z), 4), "source_text": r.source_text})
        print(res[-1]["name"], "lag", res[-1]["lag"], "echo real", res[-1]["echo_real"], "his", res[-1]["echo_his"])
    pd.DataFrame(res).to_csv(out / "align.csv", index=False)
    lags = [x["lag"] for x in res]
    print(f"lags {lags} samples: " + ("all within a quarter frame (64 samples) of 0, so `vocoder data` uses --lag 0"
                                     if max(abs(x) for x in lags) < HOP // 4 else "not all near 0: look before choosing --lag"))


def _init(run, device="cpu"):
    """g_00000000: his generator; do_00000000: NVIDIA's released discriminators, fresh optimisers, step 0."""
    import itertools
    import torch
    from huggingface_hub import hf_hub_download
    _bigvgan()
    from bigvgan import BigVGAN
    from discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator
    from env import AttrDict
    h = AttrDict(json.loads(_config().read_text()))
    g = BigVGAN(h)
    sd = torch.load(_his_file(), map_location="cpu", weights_only=False)
    g.load_state_dict(sd.get("model", sd))
    nv = torch.load(hf_hub_download(C.NV_BIGVGAN_REPO, "bigvgan_discriminator_optimizer.pt", revision=C.NV_BIGVGAN_REV),
                    map_location="cpu", weights_only=False)
    mpd, mrd = MultiPeriodDiscriminator(h), MultiScaleSubbandCQTDiscriminator(h)
    mpd.load_state_dict(nv["mpd"])
    mrd.load_state_dict(nv["mrd"])
    og = torch.optim.AdamW(g.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    od = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()), h.learning_rate,
                           betas=[h.adam_b1, h.adam_b2])
    torch.save({"generator": g.state_dict()}, run / "g_00000000")
    torch.save({"mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "optim_g": og.state_dict(),
                "optim_d": od.state_dict(), "steps": 0, "epoch": -1}, run / "do_00000000")


def train(a):
    run, d = Path(a.out).resolve(), Path(a.data).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if not any(run.glob("g_*")):
        _init(run)
        print("start: his generator, NVIDIA's discriminators, fresh optimisers, step 0", flush=True)
    bv = _bigvgan()
    cmd = [sys.executable, str(bv / "train.py"), "--config", str(_config()),
           "--input_wavs_dir", str(d / "wavs"), "--input_mels_dir", str(d / "mels"),
           "--input_training_file", str(d / "train.txt"), "--input_validation_file", str(d / "val.txt"),
           "--list_input_unseen_wavs_dir", str(d / "wavs"), "--list_input_unseen_validation_file", str(d / "val.txt"),
           "--checkpoint_path", str(run), "--fine_tuning", "True", "--debug", "True",
           "--checkpoint_interval", str(C.VOCODER_SAVE_EVERY), "--summary_interval", "100", "--stdout_interval", "100"]
    json.dump({"command": cmd, "config": C.NV_BIGVGAN_CONFIG, "bigvgan_commit": C.BIGVGAN_COMMIT,
               "start_generator": f"{C.VAGDHENU_WEIGHTS_REPO}@{C.VAGDHENU_WEIGHTS_REV}/{C.HIS_VOCODER_FILE}",
               "start_discriminators": f"{C.NV_BIGVGAN_REPO}@{C.NV_BIGVGAN_REV}/bigvgan_discriminator_optimizer.pt"},
              open(run / "command.json", "w"), indent=1)
    # unbuffered, so its step reports reach the log as they happen (a silent log reads as a hang)
    p = subprocess.Popen(cmd, cwd=bv, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    (run / "train.pid").write_text(str(p.pid))
    print(f"training started (pid {p.pid}); stopped by `nandini vocoder monitor`", flush=True)
    sys.exit(p.wait())


def _steps(run, prefix):
    return sorted(int(f.name[len(prefix):]) for f in run.glob(f"{prefix}*") if f.name[len(prefix):].isdigit())


def _loads(path):
    """True once a checkpoint file is fully written."""
    import torch
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def average(a):
    import torch
    run = Path(a.run)
    stop = a.stop or json.loads((run / "stop.json").read_text())["step"]
    keep = [stop - i * C.VOCODER_SAVE_EVERY for i in range(C.VOCODER_AVERAGE)][::-1]
    sds = [torch.load(run / f"g_{k:08d}", map_location="cpu", weights_only=False)["generator"] for k in keep]
    avg = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
    torch.save(avg, run / "vocoder.pt")
    json.dump({"averaged_steps": keep}, open(run / "vocoder.json", "w"))
    print(f"vocoder.pt: the mean of the generator at steps {keep}")


def monitor(a):
    from ..analysis.vocoder_measures import boot, echo, echo_ok, held_note, shiver
    run, mels = Path(a.run), sorted(Path(a.mels).glob("*_mel.npz"))
    dev = _device(a.device)
    mon = run / "monitor"
    mon.mkdir(parents=True, exist_ok=True)
    csv = mon / "monitor.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    vocos = decoder("vocos", dev)
    held = {}
    for f in mels:  # the held note is found once, on the vocos decode, and measured at the same times in every decode
        y = decode_file(f, vocos)
        held[f.name] = held_note(y)[1:]
        if not any(r["step"] == 0 for r in rows if r["verse"] == f.name):
            rows.append({"step": 0, "verse": f.name, "vocoder": "vocos", "echo": echo(y), "shiver": shiver(y, *held[f.name])})
    pd.DataFrame(rows).to_csv(csv, index=False)

    def stop(step, why):
        json.dump({"step": step, "reason": why}, open(run / "stop.json", "w"))
        pid = int((run / "train.pid").read_text()) if (run / "train.pid").exists() else None
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(20)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f"STOP at step {step}: {why}", flush=True)
        average(type("A", (), {"run": run, "stop": step})())

    while True:
        done = {r["step"] for r in rows}
        for k in [s for s in _steps(run, "g_") if s and s % C.VOCODER_MONITOR_EVERY == 0 and s not in done]:
            f = run / f"g_{k:08d}"
            if not _loads(f):
                break
            dec = decoder(str(f), dev)
            for m in mels:
                y = decode_file(m, dec)
                rows.append({"step": k, "verse": m.name, "vocoder": "fine-tune", "echo": echo(y),
                             "shiver": shiver(y, *held[m.name])})
            del dec
            pd.DataFrame(rows).to_csv(csv, index=False)
            d = pd.DataFrame(rows)
            now = d[d.step == k].sort_values("verse")
            e_ok = echo_ok(now.echo, C.ECHO_MEAN_MAX, C.ECHO_EACH_MAX)
            line = f"step {k}: echo mean {now.echo.mean():.4f} (max {now.echo.max():.4f}, rule {'met' if e_ok else 'not met'})"
            line += f", shiver mean {now.shiver.mean():.3f}"
            flat = None
            if (d.step == k - C.VOCODER_FLAT_OVER).any():
                before = d[d.step == k - C.VOCODER_FLAT_OVER].sort_values("verse")
                b = boot(before.shiver.values, now.shiver.values, C.VOCODER_BOOT, C.VOCODER_BOOT_SEED)
                flat = b[1] <= 0 <= b[2]
                line += f", change over {C.VOCODER_FLAT_OVER} steps {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]"
            print(line, flush=True)
            with open(mon / "monitor.log", "a") as fh:
                fh.write(line + "\n")
            for old in _steps(run, "do_")[:-2]:  # optimiser files are 1.4 GB each; only a resume needs them
                (run / f"do_{old:08d}").unlink(missing_ok=True)
            if k >= C.VOCODER_MIN_STEP and e_ok and flat:
                return stop(k, "levelled off: echo rule met, shiver unchanged over the last "
                               f"{C.VOCODER_FLAT_OVER} steps")
            if k >= getattr(a, "cap", C.VOCODER_CAP):
                return stop(k, "cap reached" + ("" if flat else "; shiver still changing"))
        pid = int((run / "train.pid").read_text()) if (run / "train.pid").exists() else None
        if pid and not Path(f"/proc/{pid}").exists() and sys.platform.startswith("linux"):
            print("training process ended without a stop", flush=True)
            sys.exit(1)
        time.sleep(60)


def run(a):
    {"data": data, "align": align, "train": train, "monitor": monitor, "average": average, "decode": decode}[a.step](a)
