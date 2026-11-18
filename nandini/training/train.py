"""Training: Vāgdhenu's finetune_indicf5.py, unmodified (external/vagdhenu), with his settings, to his length.

  nandini train [--data training/dataset] [--warm training/warm.pt] [--out training/run] [--gpus 2] [--nccl-p2p-off]
                [--stop-at 14800]

Runs in the training environment, on NVIDIA GPUs (2 × 48 GB as released; `accelerate launch`, bf16). Settings from
nandini/config.py: TRAIN_EPOCHS (600, which sets the learning-rate schedule), TRAIN_LR, TRAIN_WARMUP,
TRAIN_FRAMES_PER_GPU, TRAIN_SAVE_PER. His trainer saves model_N.pt every TRAIN_SAVE_PER steps and deletes none; beside
it, this keeps the EMA weights of the steps in TRAIN_KEEP (OUT/ema/ema_N.pt) and deletes the rest. model_last.pt is never
touched. Training is stopped once step TRAIN_STOP_AT is saved (his released model is the EMA at step 14,800), or:
- if his warm-start line isn't `missing(non-melspec): 0 | unexpected: 0`;
- if the step count doesn't move for TRAIN_STALL_S seconds (a hang never ends the process by itself);
- if less than --min-free-gb is left on the disk.
Writes OUT/command.json (every setting and the command line), OUT/train.log, OUT/progress.tsv and OUT/summary.json.
IndicF5 at the pinned commit can't resume from its own checkpoints, so an OUT/checkpoints/ that already holds some is
refused, not deleted.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .. import config as C
from .. import external as X
from .data import vocab_file
from .ema import extract

STEP = re.compile(r"step=(\d+)")
LOSS = re.compile(r"loss=([0-9.]+)")
WARM_OK = "missing(non-melspec): 0 | unexpected: 0"


def tail(path, n=65536):
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - n))
        return f.read().decode("utf-8", "replace").replace("\r", "\n")


def command(a, ck):
    script = X.require("vagdhenu") / "training" / "finetune_indicf5.py"
    launch = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(a.gpus),
              "--mixed_precision", "bf16"] + (["--multi_gpu"] if a.gpus > 1 else [])
    return launch + [str(script), "--vocab", str(vocab_file()), "--warm", str(a.warm), "--data_dir", str(a.data),
                     "--save_dir", str(ck), "--wandb_name", "nandini", "--epochs", str(C.TRAIN_EPOCHS),
                     "--lr", str(C.TRAIN_LR), "--bs", str(C.TRAIN_FRAMES_PER_GPU), "--warmup", str(C.TRAIN_WARMUP),
                     "--save_per", str(C.TRAIN_SAVE_PER)]


def prune(ck, ema, keep, log):
    """EMA of finished kept checkpoints, then delete; delete the others. A file unchanged for 90 s is finished."""
    for f in sorted(ck.glob("model_[0-9]*.pt")):
        if time.time() - f.stat().st_mtime < 90:
            continue
        n = int(f.stem.split("_")[1])
        if n in keep:
            try:
                extract(str(f), str(ema / f"ema_{n}.pt"))
            except Exception as e:  # leave the full file; report
                print(f"EMA extraction FAILED for {f}: {e}", file=log, flush=True)
                continue
        f.unlink()
        print(f"{time.strftime('%FT%T')} {'kept EMA of' if n in keep else 'deleted'} step {n}", file=log, flush=True)


def run(a):
    out = Path(a.out)
    ck, ema = out / "checkpoints", out / "ema"
    if list(ck.glob("*.pt")):
        sys.exit(f"{ck} already holds checkpoints; IndicF5 can't resume from them. Not started; nothing deleted.")
    ck.mkdir(parents=True, exist_ok=True)
    ema.mkdir(exist_ok=True)
    for p in (Path(a.data) / "raw.arrow", Path(a.warm)):
        if not p.exists():
            sys.exit(f"missing {p}: run `nandini training-data` and `nandini warm-start` first")
    keep = set(C.TRAIN_KEEP) | {a.stop_at}
    cmd = command(a, ck)
    env = {**os.environ, "WANDB_MODE": "offline", "WANDB_API_KEY": "",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # his report: expandable_segments
    if a.nccl_p2p_off:
        env["NCCL_P2P_DISABLE"] = "1"  # hosts whose GPU-to-GPU path hangs on the first collective
    (out / "command.json").write_text(json.dumps({
        "command": cmd, "stop_at": a.stop_at, "keep": sorted(keep), "gpus": a.gpus,
        "nccl_p2p_disable": a.nccl_p2p_off, "vagdhenu_commit": C.VAGDHENU_COMMIT,
        "indicf5_commit": C.INDICF5_COMMIT, "settings": C.current()}, indent=2) + "\n")
    log_path, prog = out / "train.log", out / "progress.tsv"
    plog = open(out / "prune.log", "a")
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    print(f"training started (pid {proc.pid}); stops after step {a.stop_at}; log {log_path}", flush=True)

    def stop():
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            for _ in range(wait):
                if proc.poll() is not None:
                    return
                time.sleep(1)

    step, moved, warm_seen, reason, said = 0, time.time(), False, None, 0
    while proc.poll() is None:
        time.sleep(30)
        text = tail(log_path)
        found = STEP.findall(text)
        if found and int(found[-1]) != step:
            step, moved = int(found[-1]), time.time()
        with open(prog, "a") as f:
            f.write(f"{int(time.time())}\t{step}\n")
        if not warm_seen and "[warm-start]" in (full := log_path.read_text(errors="replace")):
            warm_seen = True
            line = next(l for l in full.splitlines() if l.startswith("[warm-start]"))
            print(line, flush=True)
            if WARM_OK not in line:
                reason = f"warm start wrong: {line}"
                stop()
                break
        prune(ck, ema, keep, plog)
        if step >= said + 2000:
            said = step // 2000 * 2000
            el = time.time() - start
            print(f"step {step} of {a.stop_at}: {el / 60:.0f} min, about {el * (a.stop_at - step) / max(step, 1) / 60:.0f}"
                  f" min left", flush=True)
        if time.time() - moved > C.TRAIN_STALL_S:
            reason = f"stalled: step {step} unchanged for {C.TRAIN_STALL_S} s"
            stop()
            break
        if shutil.disk_usage(out).free / 1e9 < a.min_free_gb:
            reason = f"disk nearly full ({shutil.disk_usage(out).free / 1e9:.0f} GB free)"
            stop()
            break
        # the step counter passes stop_at only after that step's checkpoint is written (the trainer saves synchronously)
        if step > a.stop_at:
            reason = "stopped"
            stop()
            break
    else:
        reason = reason or f"trainer ended on its own, exit {proc.returncode}"
    if reason == "stopped":
        for _ in range(60):  # let the last kept checkpoint finish and be extracted
            prune(ck, ema, keep, plog)
            if (ema / f"ema_{a.stop_at}.pt").exists():
                break
            time.sleep(15)
    el = time.time() - start
    losses = [float(x) for x in LOSS.findall(log_path.read_text(errors="replace").replace("\r", "\n"))]
    summary = {"reason": reason, "last_step": step, "minutes": round(el / 60, 1),
               "seconds_per_step": round(el / max(step, 1), 3),
               "loss_first_50": round(sum(losses[:50]) / max(1, len(losses[:50])), 3) if losses else None,
               "loss_last_50": round(sum(losses[-50:]) / max(1, len(losses[-50:])), 3) if losses else None,
               "ema": sorted(p.name for p in ema.glob("ema_*.pt"))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    ok = reason == "stopped" and (ema / f"ema_{a.stop_at}.pt").exists()
    sys.exit(0 if ok else 1)
