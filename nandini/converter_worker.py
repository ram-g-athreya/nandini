"""Runs in Seed-VC's own environment. Reads one JSON request per line on stdin: {"source", "seed", "out"}. Answers each on stdout.
Usage: <seed-vc python> converter_worker.py <seed-vc dir> <voice sample wav> <settings JSON>
The settings (converter repo, revision, checkpoint, config, diffusion steps, guidance) come from nandini.config, passed
by nandini.converter, because this process runs in Seed-VC's environment and doesn't import this package.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

reply = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)  # Seed-VC prints to stdout, which is the protocol channel here

from huggingface_hub import hf_hub_download  # noqa: E402

seed_vc, voice = Path(sys.argv[1]).resolve(), str(Path(sys.argv[2]).resolve())
S = json.loads(sys.argv[3])
ckpt = hf_hub_download(S["repo"], S["checkpoint"], revision=S["revision"])
cfg = hf_hub_download(S["repo"], S["config"], revision=S["revision"])
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(seed_vc)
sys.path.insert(0, str(seed_vc))
sys.argv = sys.argv[:1]
import inference  # noqa: E402
import torch  # noqa: E402

_kept = {}
_load = inference.load_models


def _load_keeping_first(a):
    # Loading initialises layers from the random generator, so each clip's draw depends on this load happening after its seed.
    # Skipping it would save time per clip, but a seed would no longer give the same conversion.
    m = _load(a)
    return _kept.setdefault("models", m)


inference.load_models = _load_keeping_first
print(json.dumps({"ready": True}), file=reply)
# One conversion per request. The seed is set first, so a clip gets the same result whatever order clips run in.
for line in sys.stdin:
    r = json.loads(line)
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(r["seed"])
        inference.main(argparse.Namespace(source=r["source"], target=voice, output=tmp, diffusion_steps=S["diffusion_steps"], length_adjust=1.0,
                                          inference_cfg_rate=S["inference_cfg_rate"], f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0,
                                          checkpoint=ckpt, config=cfg, fp16=False))
        (made,) = Path(tmp).glob("*.wav")
        Path(r["out"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(made, r["out"])
    print(json.dumps({"done": r["out"]}), file=reply)
