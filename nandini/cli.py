"""The `nandini` command (also `python -m nandini.cli`).

  nandini fetch      [seed-vc] [vagdhenu] [bigvgan] [dcs]     outside code (and DCS texts) into external/, at pinned commits
  nandini manifest   --out OUT.csv [--select CLIPS.csv] [--dataset-dir DIR]
  nandini generate   [--seed-vc DIR --seed-vc-python PY] [--manifest M] [--out OUT] [--part i/P] [--indices a,b]
                     [--scoring whole|halves]
  nandini recheck-halves [--run out] [--dataset-dir D]        the verses that missed, re-checked half a verse at a time
  nandini inventory  convert|score|report [--manifest M] [--out OUT] [...]
  nandini mark       [--run out] [--dataset-dir D] [--part i/P]   what the ASR hears on each kept verse
  nandini tables     [--run out]                                 transcripts, per-syllable and substitution tables
  nandini report     [--run out]                                 the dataset report and charts
  nandini listen     build|serve [--run out]                     the listening page
  nandini pitch-shift draw|readings|check|vagdhenu|target|report [--out pitch_shift]   how k per style was measured
  nandini selection  simulate|threshold [--inventory inventory]   the selection rule on the inventory's clips
  nandini melody-checks timing|voicing|loudness|loudness-cap [--inventory inventory] [--listen]
  nandini training-data [--run out] [--kept K.csv] [--audio-dir D] [--out training/dataset]   text + IndicF5's dataset
  nandini warm-start [--out training/warm.pt] [--check-only]     the warm start from the IndicF5 base, checked
  nandini train      [--data training/dataset] [--warm training/warm.pt] [--out training/run] [--gpus 2]
  nandini extract-ema model_N.pt ema_N.pt
  nandini speech-check render --ema E | score                    after training: 3 verses, does the model speak?
  nandini trend-check  render --ema E [E ...] --bank B | score   after training: 10 verses on his website's path, 12,000 vs 14,800
  nandini vocoder    data|align|train|monitor|average|decode|score   his BigVGAN fine-tuned on her clips, as he made his
  nandini test-verses [--out manifests/test_verses.csv]          set B: verses outside the dataset, from DCS (CC BY 4.0)
  nandini compare    render --system V|Vs|N [...] | score        Nandinī against Vāgdhenu on his website's path
  nandini onnx       make | check graphs|verses|score | render   the ONNX version of Nandinī (no torch to render)
  nandini female-bank [--run out] [--kept K.csv] [--audio-dir D] [--out training/female_bank]   reference cuts, her voice
  nandini config     print the settings (nandini/config.py)

training-data, warm-start, train, extract-ema, the checks' render steps and every vocoder step but score run in the training
environment (external/train-env, README, Training; the vocoder adds third_party/vocoder-requirements.txt); started from another Python they re-run themselves there (--train-python chooses another).

Every setting is in nandini/config.py. The options here only say which clips to run and where files go.
"""
import argparse
import json
import sys

from . import config as C

FULL = C.ROOT / "manifests" / "all_clips.csv"
SAMPLE = C.ROOT / "manifests" / "inventory_sample_clips.csv"


def _run_options(p, manifest, out):
    p.add_argument("--manifest", default=str(manifest), help=f"clips to run (default: {manifest.relative_to(C.ROOT)})")
    p.add_argument("--out", default=str(C.ROOT / out), help=f"output folder (default: {out})")
    p.add_argument("--dataset-dir", help="local copy of the dataset (default: download into the Hugging Face cache)")
    p.add_argument("--seed-vc", help="Seed-VC checkout (default: external/seed-vc, from `nandini fetch seed-vc`)")
    p.add_argument("--seed-vc-python", help="Python of Seed-VC's own environment (default: external/seed-vc/.venv)")
    p.add_argument("--voice", default=str(C.VOICE_SAMPLE), help="voice sample to convert to (default: the configured one)")
    p.add_argument("--part", default="0/1", help="run part i of P (default 0/1)")
    p.add_argument("--indices", help="only these manifest indices, comma-separated")
    p.add_argument("--clips", type=int, default=0, help="only the first N clips of the manifest")
    p.add_argument("--conversions", type=int, default=C.N_CONVERSIONS,
                   help=f"attempts per clip (default {C.N_CONVERSIONS})")


def _need_seed_vc(a):
    """Seed-VC from --seed-vc/--seed-vc-python, else external/seed-vc and its .venv (README, Setup)."""
    from pathlib import Path
    from . import external as X
    if a.seed_vc is None:
        a.seed_vc = str(X.require("seed-vc", ", then make its environment (README, Setup)"))
    if a.seed_vc_python is None:
        a.seed_vc_python = str(X.SEED_VC_PYTHON)
        if not Path(a.seed_vc_python).exists():
            sys.exit(f"no Python at {a.seed_vc_python}: make Seed-VC's environment (README, Setup) or pass --seed-vc-python")


CHECKS = ("speech-check", "trend-check")
TRAIN_NEEDS = {"training-data": ("indic_transliteration", "datasets"), "warm-start": ("f5_tts",),
               "train": ("f5_tts", "accelerate"), "extract-ema": ("torch",), "test-verses": ("indic_transliteration",)}
COMPARE_NEEDS = {"render": ("f5_tts",), "score": ()}
ONNX_NEEDS = {"make": ("f5_tts", "onnx"), "small": ("f5_tts", "onnx"), "graphs": ("f5_tts", "onnxruntime"),
              "verses": ("f5_tts", "onnxruntime"), "score": (), "score-small": (),
              "render": ("onnxruntime", "indic_transliteration", "pydub")}


VOCODER_NEEDS = {"data": ("f5_tts",), "align": ("f5_tts", "vocos", "parselmouth"), "decode": ("vocos",),
                 "train": ("f5_tts", "nnAudio", "pesq", "auraloss"), "monitor": ("vocos", "parselmouth"),
                 "average": ("torch",), "score": ()}


def _needs(a):
    if a.command == "vocoder":
        return VOCODER_NEEDS[a.step]
    if a.command == "compare":
        return COMPARE_NEEDS[a.step]
    if a.command == "onnx":
        return ONNX_NEEDS[a.check if a.step == "check" else a.step] if not (a.step == "check" and a.check == "small") \
            else ("torch", "onnxruntime", "indic_transliteration", "pydub")
    if a.command in CHECKS:
        return () if a.step == "score" else ("f5_tts", "vocos")
    if a.command == "training-data" and a.text_only:
        return TRAIN_NEEDS[a.command][:1]
    return TRAIN_NEEDS.get(a.command, ())


def _in_train_env(a, argv):
    """Run here if this Python has what the command needs; else re-run the same command in the training environment."""
    import importlib.util
    import os
    import subprocess
    from pathlib import Path
    from . import external as X
    needs = _needs(a)
    if all(importlib.util.find_spec(m) for m in needs):
        return
    py = Path(a.train_python or X.TRAIN_PYTHON)
    if not py.exists():
        sys.exit(f"{a.command} needs the training environment ({', '.join(needs)}): make it (README, Training) or pass "
                 f"--train-python")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(C.ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    sys.exit(subprocess.run([str(py), "-m", "nandini.cli", *argv], env=env).returncode)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="nandini", description="A female voice for Vāgdhenu's Sanskrit chant.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="fetch outside code into external/ at the pinned commits")
    p.add_argument("names", nargs="*", metavar="NAME", help="seed-vc, vagdhenu, bigvgan or dcs (default: all)")

    p = sub.add_parser("manifest", help="write a manifest of clips with fixed indices")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--dataset-dir", help="local copy of the dataset")
    p.add_argument("--select", help="keep only the clip_ids in this CSV, in its order")

    p = sub.add_parser("generate", help="make the dataset: attempts per verse until one passes the ASR check")
    _run_options(p, FULL, "out")
    p.add_argument("--scoring", choices=["whole", "halves"], default="whole",
                   help="transcribe each clip whole, or in two halves cut at the half-verse pause (default whole)")

    p = sub.add_parser("recheck-halves", help="re-check the verses that missed, each half verse heard alone; writes kept_rechecked.csv")
    p.add_argument("--run", default=str(C.ROOT / "out"), help="the run folder from `nandini generate` (default: out)")
    p.add_argument("--manifest", default=str(FULL), help="manifest with the verses' texts (default: all clips)")
    p.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")

    p = sub.add_parser("inventory", help="measure how the converter gets syllables wrong")
    p.add_argument("step", choices=["convert", "score", "report"])
    _run_options(p, SAMPLE, "inventory")

    def run_folder(p, manifest=True, dataset=False):
        p.add_argument("--run", default=str(C.ROOT / "out"), help="the run folder from `nandini generate` (default: out)")
        if manifest:
            p.add_argument("--manifest", default=str(FULL), help="manifest with the verses' texts (default: all clips)")
        if dataset:
            p.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")

    p = sub.add_parser("mark", help="what the ASR hears on each kept verse, per syllable, with timings")
    run_folder(p, dataset=True)
    p.add_argument("--part", default="0/1", help="run part i of P (default 0/1)")

    p = sub.add_parser("tables", help="transcripts, per-syllable and substitution tables from the marks")
    run_folder(p)

    p = sub.add_parser("report", help="the dataset report and charts")
    run_folder(p, manifest=False)

    p = sub.add_parser("listen", help="the listening page: build it, then serve it")
    p.add_argument("step", choices=["build", "serve"])
    run_folder(p, dataset=True)
    p.add_argument("--port", type=int, default=8765, help="port for serve (default 8765)")

    p = sub.add_parser("pitch-shift", help="measure the pitch shift k per style (the target voice reads drawn verses)")
    p.add_argument("step", choices=["draw", "readings", "check", "vagdhenu", "target", "report"])
    p.add_argument("--out", default=str(C.ROOT / "pitch_shift"), help="output folder (default: pitch_shift)")
    p.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")
    p.add_argument("--indic-speak-python", help="for readings: Python of an environment with Indic-Speak's requirements")
    p.add_argument("--device", default="auto", help="for readings: auto, cuda, mps or cpu (default auto)")

    p = sub.add_parser("selection", help="the selection rule simulated on the inventory, and ≤ 1 against 0 errors")
    p.add_argument("step", choices=["simulate", "threshold"])
    p.add_argument("--inventory", default=str(C.ROOT / "inventory"), help="the inventory's output folder (default: inventory)")

    p = sub.add_parser("melody-checks", help="checks behind the melody step's design, on the inventory's conversions")
    p.add_argument("step", choices=["timing", "voicing", "loudness", "loudness-cap"])
    p.add_argument("--inventory", default=str(C.ROOT / "inventory"), help="the inventory's output folder (default: inventory)")
    p.add_argument("--manifest", default=str(SAMPLE), help="clips (default: the 61-clip sample)")
    p.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")
    p.add_argument("--listen", action="store_true", help="also cut listening clips around the worst stretches")

    def train_python(p):
        p.add_argument("--train-python", help="Python of the training environment (default: external/train-env)")

    p = sub.add_parser("training-data", help="each kept clip with Vāgdhenu's model_text, and IndicF5's dataset files")
    p.add_argument("--run", default=str(C.ROOT / "out"), help="the run folder from `nandini generate` (default: out)")
    p.add_argument("--kept", help="kept list to use instead of the run's kept_rechecked.csv (or kept_part*.csv)")
    p.add_argument("--audio-dir", help="folder of the kept clips (default: RUN/final)")
    p.add_argument("--manifest", default=str(FULL), help="manifest with the verses' texts (default: all clips)")
    p.add_argument("--out", default=str(C.ROOT / "training/dataset"), help="output folder (default: training/dataset)")
    p.add_argument("--text-only", action="store_true", help="only training_text.csv, not raw.arrow/duration.json")
    train_python(p)

    p = sub.add_parser("warm-start", help="the warm-start file from the IndicF5 base, checked as his script loads it")
    p.add_argument("--out", default=str(C.ROOT / "training/warm.pt"), help="output file (default: training/warm.pt)")
    p.add_argument("--check-only", action="store_true", help="only check an existing file")
    train_python(p)

    p = sub.add_parser("train", help="Vāgdhenu's training script, unmodified, with his settings and length")
    p.add_argument("--data", default=str(C.ROOT / "training/dataset"), help="from `nandini training-data` (default: training/dataset)")
    p.add_argument("--warm", default=str(C.ROOT / "training/warm.pt"), help="from `nandini warm-start` (default: training/warm.pt)")
    p.add_argument("--out", default=str(C.ROOT / "training/run"), help="output folder (default: training/run)")
    p.add_argument("--gpus", type=int, default=C.TRAIN_GPUS, help=f"GPUs (default {C.TRAIN_GPUS}, as released)")
    p.add_argument("--stop-at", type=int, default=C.TRAIN_STOP_AT, help=f"last step (default {C.TRAIN_STOP_AT})")
    p.add_argument("--nccl-p2p-off", action="store_true", help="set NCCL_P2P_DISABLE=1 (hosts where GPU-to-GPU hangs)")
    p.add_argument("--min-free-gb", type=float, default=25, help="stop if less disk than this is left (default 25)")
    train_python(p)

    p = sub.add_parser("extract-ema", help="keep only a trainer checkpoint's EMA weights")
    p.add_argument("src")
    p.add_argument("dst")
    train_python(p)

    p = sub.add_parser("speech-check", help="after training, 3 training verses with IndicF5's own inference: does it speak?")
    cs = p.add_subparsers(dest="step", required=True)
    q = cs.add_parser("render", help="render them (training environment)")
    q.add_argument("--ema", nargs=1, required=True, help="EMA checkpoint")
    q.add_argument("--data", default=str(C.ROOT / "training/dataset"), help="from `nandini training-data` (default: training/dataset)")
    q.add_argument("--out", default=str(C.ROOT / "training/checks/speech"), help="output folder (default: training/checks/speech)")
    q.add_argument("--device", default="auto")
    train_python(q)
    q = cs.add_parser("score", help="score the renders (this code's environment)")
    q.add_argument("--out", default=str(C.ROOT / "training/checks/speech"), help="the renders' folder (default: training/checks/speech)")

    p = sub.add_parser("trend-check", help="after training, 10 verses on his website's path: still improving at the end?")
    cs = p.add_subparsers(dest="step", required=True)
    q = cs.add_parser("render", help="one render per verse and checkpoint (training environment)")
    q.add_argument("--ema", nargs="+", required=True, help="EMA checkpoints")
    q.add_argument("--bank", default=str(C.ROOT / "training/female_bank"), help="reference bank folder (default: training/female_bank)")
    q.add_argument("--verses", default=str(C.ROOT / "manifests/training_check_verses.csv"),
                   help="CSV with columns n and clip_id (default: manifests/training_check_verses.csv, the 10 check verses)")
    q.add_argument("--save-mel", action="store_true", help="also save each render's mels, for `nandini vocoder-decode`")
    q.add_argument("--out", default=str(C.ROOT / "training/checks/trend"), help="output folder (default: training/checks/trend)")
    q.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")
    q.add_argument("--device", default="auto")
    train_python(q)
    q = cs.add_parser("score", help="score the renders and compare the two steps (this code's environment)")
    q.add_argument("--out", default=str(C.ROOT / "training/checks/trend"), help="the renders' folder (default: training/checks/trend)")
    q.add_argument("--data", default=str(C.ROOT / "training/dataset"), help="from `nandini training-data` (default: training/dataset)")
    q.add_argument("--last", type=int, default=C.TRAIN_STOP_AT, help=f"last step (default {C.TRAIN_STOP_AT})")
    q.add_argument("--before", type=int, default=C.TRAIN_CHECK_BEFORE,
                   help=f"the step compared with it (default {C.TRAIN_CHECK_BEFORE})")
    q.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")

    p = sub.add_parser("female-bank", help="Vāgdhenu's reference cuts, taken from the kept clips, each scored")
    run_folder(p, dataset=True)
    p.add_argument("--kept", help="kept list to use instead of the run's kept_rechecked.csv (or kept_part*.csv)")
    p.add_argument("--audio-dir", help="folder of the kept clips (default: RUN/final)")
    p.add_argument("--out", default=str(C.ROOT / "training/female_bank"), help="output folder (default: training/female_bank)")
    p.add_argument("--overrides", default=str(C.ROOT / "manifests" / "female_bank_overrides.csv"),
                   help="entries changed by hand, with reasons (default: the release's; '' for none)")

    p = sub.add_parser("vocoder", help="fine-tune his BigVGAN on her clips, as he made his; decode saved mels; score")
    cs = p.add_subparsers(dest="step", required=True)
    V = "training/vocoder"
    q = cs.add_parser("data", help="IndicF5 mels of the training clips, and NVIDIA's file lists (training environment)")
    q.add_argument("--data", default=str(C.ROOT / "training/dataset"), help="from `nandini training-data` (default: training/dataset)")
    q.add_argument("--out", default=str(C.ROOT / V / "data"), help=f"output folder (default: {V}/data)")
    q.add_argument("--lag", type=int, default=0, help="delay the audio by this many samples against its mel (from `vocoder align`)")
    train_python(q)
    q = cs.add_parser("align", help="before training: his BigVGAN on 3 training-style mels (training environment)")
    q.add_argument("--data", default=str(C.ROOT / "training/dataset"), help="from `nandini training-data` (default: training/dataset)")
    q.add_argument("--out", default=str(C.ROOT / V / "align"), help=f"output folder (default: {V}/align)")
    q.add_argument("--device", default="auto")
    train_python(q)
    q = cs.add_parser("train", help="NVIDIA's train.py, unchanged, from his generator; runs until `monitor` stops it")
    q.add_argument("--data", default=str(C.ROOT / V / "data"), help=f"from `vocoder data` (default: {V}/data)")
    q.add_argument("--out", default=str(C.ROOT / V / "run"), help=f"checkpoints and logs (default: {V}/run)")
    train_python(q)
    q = cs.add_parser("monitor", help="echo and shiver on saved mels every few thousand steps; stops training; averages")
    q.add_argument("--mels", required=True, help="folder of *_mel.npz (`trend-check render --save-mel`)")
    q.add_argument("--run", default=str(C.ROOT / V / "run"), help=f"the training run (default: {V}/run)")
    q.add_argument("--cap", type=int, default=C.VOCODER_CAP, help=f"stop here at the latest (default {C.VOCODER_CAP})")
    q.add_argument("--device", default="auto")
    train_python(q)
    q = cs.add_parser("average", help="the final weights: the mean of the last checkpoints")
    q.add_argument("--run", default=str(C.ROOT / V / "run"), help=f"the training run (default: {V}/run)")
    q.add_argument("--stop", type=int, help="the last step averaged (default: the monitor's stop)")
    train_python(q)
    q = cs.add_parser("decode", help="saved mels back to audio with a vocoder, joined as his renderer joins them")
    q.add_argument("--mels", required=True, help="folder of *_mel.npz")
    q.add_argument("--vocoder", required=True, help="vocos, his, or a generator file (e.g. RUN/vocoder.pt)")
    q.add_argument("--out", required=True, help="output folder")
    q.add_argument("--device", default="auto")
    train_python(q)
    q = cs.add_parser("score", help="the check before training (--align), or the pass rule on decoded renders (this code's environment)")
    q.add_argument("--align", action="store_true", help="score `vocoder align`'s output")
    q.add_argument("--out", default=str(C.ROOT / V / "align"), help=f"with --align, its folder (default: {V}/align); else the report folder")
    q.add_argument("--decoded", nargs="+", help="vocos=DIR fine-tune=DIR [his=DIR], each from `vocoder decode` on the same mels")
    q.add_argument("--verses", help="the renders' verses.csv (from `trend-check render`)")

    p = sub.add_parser("onnx", help="the ONNX version of Nandinī: make it, check it against torch, render with it")
    os_ = p.add_subparsers(dest="step", required=True)
    q = os_.add_parser("make", help="the two ONNX files from the PyTorch weights (training environment)")
    q.add_argument("--ema", required=True, help="the model (EMA checkpoint)")
    q.add_argument("--vocoder", required=True, help="the vocoder file (from `nandini vocoder average`)")
    q.add_argument("--out", default=str(C.ROOT / "training/onnx"), help="output folder (default: training/onnx)")
    train_python(q)
    q = os_.add_parser("small", help="smaller files from the fp32 ones: int8 and rank-32 transformer, fp16 and int8 vocoder")
    q.add_argument("--ema", required=True, help="the model (EMA checkpoint), for the rank-32 AdaLN")
    q.add_argument("--onnx", default=str(C.ROOT / "training/onnx"), help="the ONNX folder (default: training/onnx)")
    train_python(q)
    q = os_.add_parser("check", help="the ONNX version against torch: graphs, verses (training environment), score")
    q.add_argument("check", choices=["graphs", "verses", "score", "small", "score-small"])
    q.add_argument("--ema", help="graphs, verses: the model (EMA checkpoint)")
    q.add_argument("--vocoder", help="graphs, verses: the vocoder file")
    q.add_argument("--bank", help="graphs, verses: the reference bank folder (holding bank.json)")
    q.add_argument("--onnx", default=str(C.ROOT / "training/onnx"), help="the ONNX folder (default: training/onnx)")
    q.add_argument("--n", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="verses: test verses (default: 1-5)")
    train_python(q)
    q = os_.add_parser("render", help="render with the ONNX version (onnxruntime, no torch)")
    q.add_argument("--onnx", default=str(C.ROOT / "training/onnx"), help="the ONNX folder (default: training/onnx)")
    q.add_argument("--bank", required=True, help="the reference bank folder (holding bank.json)")
    q.add_argument("--text", help="a verse (Devanagari; lines split at daṇḍas or newlines)")
    q.add_argument("--meter", help="with --text: the meter (default: detected, else vasantatilakā)")
    q.add_argument("--out", default="out.wav", help="with --text: the output wav (default: out.wav)")
    q.add_argument("--seed", type=int, default=C.WEB_SEED, help=f"default {C.WEB_SEED}, his website's")
    q.add_argument("--steps", type=int, default=64, help="sampling steps (default 64, his)")
    q.add_argument("--threads", type=int, default=0, help="onnxruntime threads (default: all cores)")
    train_python(q)

    p = sub.add_parser("compare", help="Nandinī (N) against Vāgdhenu (V, Vs) through his website's path")
    cs = p.add_subparsers(dest="step", required=True)
    q = cs.add_parser("render", help="one system's renders of its sets (training environment)")
    q.add_argument("--system", required=True, choices=list(C.COMPARE_SETS), help="V, Vs (his models) or N (ours)")
    q.add_argument("--sets", nargs="+", choices=["A", "B", "C"], help="default: every set the system renders")
    q.add_argument("--only", nargs="+", type=int, help="only these verse numbers (a trial run)")
    q.add_argument("--ema", help="N: the model (EMA checkpoint)")
    q.add_argument("--bank", help="N: the reference bank folder (holding bank.json)")
    q.add_argument("--vocoder", help="N: the vocoder file (from `nandini vocoder average`)")
    q.add_argument("--out", default=str(C.ROOT / "compare"), help="output folder (default: compare)")
    q.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")
    q.add_argument("--device", default="auto")
    train_python(q)
    q = cs.add_parser("score", help="the measures and the report (this code's environment)")
    q.add_argument("--out", default=str(C.ROOT / "compare"), help="the renders' folder (default: compare)")
    q.add_argument("--her-embedding", help="mean speaker embedding of her training clips (.npy); default: made from --data")
    q.add_argument("--data", default=str(C.ROOT / "training/dataset"), help="training clips, for --her-embedding (default: training/dataset)")
    q.add_argument("--report-only", action="store_true", help="only rebuild report.md from scores.csv and pairs.csv")
    q.add_argument("--dataset-dir", help="local copy of the dataset (default: the Hugging Face cache)")

    p = sub.add_parser("test-verses", help="set B: verses outside the dataset in its meters, from DCS (training environment)")
    p.add_argument("--out", default=str(C.ROOT / "manifests" / "test_verses.csv"),
                   help="drawn verses (default: manifests/test_verses.csv; _pool.csv and _report.md beside it)")
    train_python(p)

    sub.add_parser("config", help="print the settings")

    a = ap.parse_args(argv)
    if a.command == "compare":
        _in_train_env(a, argv)
        if a.step == "render":
            from .evaluation.render import run
        else:
            from .evaluation.score import run
        run(a)
        return
    if a.command == "onnx":
        _in_train_env(a, argv)
        if a.step == "make":
            from .onnx.make import run
        elif a.step == "small":
            from .onnx.small import run
        elif a.step == "check":
            a.step = a.check
            from .onnx.check import run
        else:
            from .onnx.render import run
        run(a)
        return
    if a.command == "vocoder":
        _in_train_env(a, argv)
        if a.step == "score":
            from .analysis.vocoder_score import run
        else:
            from .training.vocoder import run
        run(a)
        return
    if a.command in TRAIN_NEEDS or a.command in CHECKS:
        _in_train_env(a, argv)
        if a.command in CHECKS:
            a.what = a.command.removesuffix("-check")
            if a.step == "score":
                from .analysis.training_checks import run
            else:
                from .training.checks import run
            run(a)
            return
        if a.command == "test-verses":
            from .pipeline.test_verses import run
            run(a)
            return
        from .training import data, ema, train, warm
        {"training-data": data, "warm-start": warm, "train": train, "extract-ema": ema}[a.command].run(a)
        return
    if a.command == "female-bank":
        from .pipeline.female_bank import run
        run(a)
        return
    if a.command == "fetch":
        from .external import run
        run(a)
    elif a.command == "manifest":
        from .pipeline.manifest import run
        run(a)
    elif a.command == "generate":
        _need_seed_vc(a)
        from .pipeline.generate_dataset import run
        run(a)
    elif a.command == "recheck-halves":
        from .pipeline.recheck_halves import run
        run(a)
    elif a.command == "inventory":
        if a.step == "convert":
            _need_seed_vc(a)
        from .analysis import inventory
        getattr(inventory, a.step)(a)
    elif a.command == "mark":
        from .analysis.marks import run
        run(a)
    elif a.command == "tables":
        from .analysis.tables import run
        run(a)
    elif a.command == "report":
        from .analysis.dataset_report import run
        run(a)
    elif a.command == "listen":
        if a.step == "build":
            from .listening.build import build
            build(a)
        else:
            from .listening.serve import serve
            serve(a)
    elif a.command == "pitch-shift":
        from .pipeline import pitch_shift
        getattr(pitch_shift, a.step)(a)
    elif a.command == "selection":
        from .analysis import selection
        getattr(selection, a.step)(a)
    elif a.command == "melody-checks":
        from .analysis import melody_checks
        getattr(melody_checks, a.step.replace("-", "_"))(a)
    elif a.command == "config":
        print(json.dumps(C.current(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
