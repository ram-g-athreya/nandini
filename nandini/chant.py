"""`nandini chant`: one verse to a wav with the released model, through Vāgdhenu's renderer.

  nandini chant --text "..." [--backend torch|onnx|mlx] [--compact] [--meter M] [--seed 60] [--model nandini-model] [--out out.wav]

Started from this code's environment (`uv run nandini chant`), it downloads the released model (MODEL_REPO at
MODEL_REV) into --model if it isn't there, converts it for MLX once if needed, then re-runs itself in the backend's
environment (scripts/setup_*.sh):
  torch  external/train-env  Vāgdhenu's render_core.Renderer with Nandinī's voice, vocoder and bank
  onnx   external/onnx-env   nandini/onnx/render.py, the full set (onnx/) or with --compact the smaller one (onnx_compact/)
  mlx    external/mlx-env    Vāgdhenu's MLX backend, on the converted weights (mlx/)
The meter is his detector's, else his fallback (vasantatilakā); seed 60 is his website's.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from . import config as C
from . import external as X

VOICE = "voice_nandini_ema_2026-11-15.pt"
VOCODER = "voc_bigvgan_nandini_2026-11-17.pt"
ENVS = {"torch": X.TRAIN_PYTHON, "onnx": X.EXTERNAL / "onnx-env" / "bin" / "python",
        "mlx": X.EXTERNAL / "mlx-env" / "bin" / "python"}
NEEDS = {"torch": ("f5_tts", "torch", "librosa"), "onnx": ("onnxruntime", "pydub"), "mlx": ("mlx",)}
SETUP = {"torch": "scripts/setup_pytorch.sh", "onnx": "scripts/setup_onnx.sh", "mlx": "scripts/setup_mlx.sh"}
FILES = {"torch": [VOICE, VOCODER], "onnx": ["onnx/*"], "mlx": [VOICE, VOCODER]}
COMPACT = ("nandini_dit_r32_int8.onnx", ("nandini_vocoder_int8.onnx",))


def _here(backend):
    """True if this Python can render with the backend."""
    return all(importlib.util.find_spec(m) for m in NEEDS[backend])


def prepare(a):
    """Download the model files the backend needs, and convert them for MLX once."""
    m = Path(a.model)
    files = FILES[a.backend] if not (a.backend == "onnx" and a.compact) else ["onnx_compact/*"]
    want = files + ["reference_bank/*", "vocab.txt"]
    missing = [f for f in want if not list(m.glob(f))]
    if missing:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit(f"{m} lacks {', '.join(missing)}: run `uv run nandini chant ...` from this folder to download it")
        print(f"downloading {C.MODEL_REPO} ({', '.join(missing)}) into {m}", flush=True)
        snapshot_download(C.MODEL_REPO, revision=C.MODEL_REV, local_dir=str(m), allow_patterns=want)
    if a.backend == "mlx" and not (m / "mlx" / "dit.safetensors").exists():
        print(f"converting the model for MLX into {m / 'mlx'}", flush=True)
        src = str(X.require("vagdhenu") / "src")
        env = {**os.environ, "PYTHONPATH": src}
        subprocess.run([sys.executable, "-m", "vagdhenu_mlx.convert", "--voice", str(m / VOICE), "--vocoder",
                        str(m / VOCODER), "--vocab", str(m / "vocab.txt"), "--out", str(m / "mlx")], env=env, check=True)


def render(a):
    """In the backend's environment: the verse to a.out."""
    import soundfile as sf
    m = Path(a.model)
    if a.backend == "torch":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # one of BigVGAN's operations has no MPS version
        sys.path[:0] = [str(X.require("vagdhenu") / "src"), str(X.require("bigvgan"))]
        import torch
        import render_core as RC
        from chant_text import detect_meter_key
        dev = a.device if a.device != "auto" else \
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        r = RC.Renderer(str(m / VOICE), str(m / VOCODER), str(m / "reference_bank" / "bank.json"), device=dev,
                        vocab_file=str(m / "vocab.txt"))
    elif a.backend == "onnx":
        from .onnx.render import Renderer
        sys.path.insert(0, str(X.require("vagdhenu") / "src"))
        from chant_text import detect_meter_key
        d, (dit, vocs) = (m / "onnx_compact", COMPACT) if a.compact else (m / "onnx", ("nandini_dit.onnx",
                                                                                         ("nandini_vocoder.onnx",)))
        r = Renderer(str(d), str(m / "reference_bank" / "bank.json"), str(m / "vocab.txt"), dit=dit, vocoders=vocs)
    else:
        sys.path.insert(0, str(X.require("vagdhenu") / "src"))
        from vagdhenu_mlx import Renderer, detect_meter_key
        r = Renderer(str(m / "mlx"), bank_path=str(m / "reference_bank" / "bank.json"))
    meter = a.meter or detect_meter_key(a.text) or "vasantatilakā"
    sr, audio = r.render_one(a.text, meter, seed=a.seed)
    sf.write(a.out, audio, sr)
    print(f"{a.out}: {len(audio) / sr:.1f} s, meter {meter}, {a.backend}{' compact' if a.compact else ''}")


def run(a, argv):
    if not a.text:
        sys.exit("--text is required")
    prepare(a)
    if _here(a.backend):
        render(a)
        return
    py = ENVS[a.backend]
    if not py.exists():
        sys.exit(f"the {a.backend} backend needs its environment: run {SETUP[a.backend]}")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(C.ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    sys.exit(subprocess.run([str(py), "-m", "nandini.cli", *argv], env=env).returncode)
