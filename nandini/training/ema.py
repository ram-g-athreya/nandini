"""Keeps only the EMA weights of a trainer checkpoint (about 1.35 GB, like Vāgdhenu's released voice files), in the form
IndicF5's inference `load_checkpoint` reads ({"ema_model_state_dict": ...}), plus the step.

  nandini extract-ema model_N.pt ema_N.pt

Runs in the training environment. Writes to a temporary name, reloads it and compares, then renames.
"""
import os


def extract(src, dst):
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=True)
    out = {"ema_model_state_dict": ck["ema_model_state_dict"], "step": ck["step"]}
    tmp = f"{dst}.tmp"
    torch.save(out, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=True)
    assert back["step"] == ck["step"] and back["ema_model_state_dict"].keys() == out["ema_model_state_dict"].keys()
    os.replace(tmp, dst)
    print(f"{src} step {ck['step']} -> {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)", flush=True)


def run(a):
    extract(a.src, a.dst)
