"""The ONNX version of Nandinī: its two graphs, made from the PyTorch weights. Runs in the training environment.

  nandini onnx make --ema EMA.pt --vocoder VOCODER.pt [--out training/onnx]

Writes, in fp32 (ONNX opset ONNX_OPSET):
  nandini_dit.onnx      one evaluation of IndicF5's transformer: x (noisy mel), cond (reference mel, zero after it),
                        text (token ids, -1 as filler), t (time) -> flow. Batch and length are dynamic. The guided and
                        the unguided ("null") passes go in as a batch of two: IndicF5's null pass is the same call with
                        the reference mel zeroed and every token the filler, so no flags are needed in the graph.
  nandini_vocoder.onnx  the fine-tuned BigVGAN, mel (batch, 100, frames) -> wave (batch, frames * 256), weight norm
                        folded into the weights.
  vocos_spectrum.onnx   IndicF5's stock vocos up to its spectrum (the inverse STFT is done in numpy). Not Nandinī's
                        voice: his renderer redoes a line when vocos's output is quiet, and this reproduces that test.
  manifest.json         inputs, outputs, sizes and sha256 of both, and what they were made from.

The model is built as Vāgdhenu's renderer builds it (render_core.Renderer: IndicF5's DiT with his settings, the EMA
weights), and the vocoder as `nandini vocoder` loads it.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

from .. import config as C
from .. import external as X

DIT_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)  # his render_core.Renderer


def load_dit(ema, device="cpu"):
    """IndicF5's CFM with Nandinī's EMA weights, exactly as his renderer loads a voice; returns its transformer."""
    import torch
    from f5_tts.infer.utils_infer import load_model
    from f5_tts.model import DiT
    from ..training.data import vocab_file
    cfm = load_model(DiT, DIT_CFG, mel_spec_type="vocos", vocab_file=str(vocab_file()), device=device)
    ck = torch.load(ema, map_location="cpu", weights_only=True)
    sd = {k.replace("ema_model.", ""): v for k, v in ck["ema_model_state_dict"].items() if k not in ("initted", "step")}
    cfm.load_state_dict(sd, strict=False)
    return cfm.eval().transformer


def wrappers():
    import torch

    class Step(torch.nn.Module):
        """One transformer evaluation with no CFG flags: the caller builds the null pass into the batch."""
        def __init__(self, dit):
            super().__init__()
            self.dit = dit

        def forward(self, x, cond, text, t):
            return self.dit(x=x, cond=cond, text=text, time=t, mask=None, drop_audio_cond=False, drop_text=False)

    class VocosSpectrum(torch.nn.Module):
        """IndicF5's stock vocos up to its complex spectrum; the inverse STFT (torch.istft, center=True) is done in
        numpy (render.istft), since it doesn't export."""
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, mel):
            x = self.v.head.out(self.v.backbone(mel)).transpose(1, 2)
            mag, p = x.chunk(2, dim=1)
            mag = torch.clip(torch.exp(mag), max=1e2)
            return mag * torch.cos(p), mag * torch.sin(p)

    class Vocoder(torch.nn.Module):
        def __init__(self, g):
            super().__init__()
            self.g = g

        def forward(self, mel):
            return self.g(mel).squeeze(1)

    return Step, Vocoder, VocosSpectrum


def fixed_filters(g):
    """BigVGAN's anti-aliasing filters expand their kernel to the channel count at run time
    (`filter.expand(C, -1, -1)`), which the ONNX exporter can't size. Each Activation1d knows its channel count (its
    SnakeBeta's alpha), so the expanded kernel is stored once and used as is. Same arithmetic, fixed shapes."""
    import types
    import torch.nn.functional as F
    from alias_free_activation.torch.act import Activation1d

    def up(self, x):
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.kernel, stride=self.stride, groups=self.kernel.shape[0])
        return x[..., self.pad_left: -self.pad_right]

    def low(self, x):
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.kernel, stride=self.stride, groups=self.kernel.shape[0])

    n = 0
    for m in g.modules():
        if isinstance(m, Activation1d):
            c = m.act.alpha.numel()
            m.upsample.register_buffer("kernel", m.upsample.filter.expand(c, -1, -1).contiguous().clone())
            m.upsample.forward = types.MethodType(up, m.upsample)
            lp = m.downsample.lowpass
            lp.register_buffer("kernel", lp.filter.expand(c, -1, -1).contiguous().clone())
            lp.forward = types.MethodType(low, lp)
            n += 1
    return n


def example(frames=400, seed=0):
    """Inputs of the shapes the renderer uses: a batch of two (guided, null), `frames` long."""
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, frames, 100, generator=g)
    cond = torch.randn(2, frames, 100, generator=g)
    cond[1] = 0  # the null pass
    cond[0, frames // 2:] = 0  # the reference occupies the start
    text = torch.randint(0, 2000, (2, frames), generator=g)
    text[1] = -1
    text[0, frames // 3:] = -1
    t = torch.full((2,), 0.3)
    return x, cond, text, t


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(a):
    import onnx
    import torch
    from ..training.vocoder import load_generator
    X.require("vagdhenu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Step, Vocoder, VocosSpectrum = wrappers()
    made = {}

    t0 = time.time()
    step = Step(load_dit(a.ema)).eval()
    dit_path = out / "nandini_dit.onnx"
    with torch.no_grad():
        torch.onnx.export(step, example(), str(dit_path), opset_version=C.ONNX_OPSET, do_constant_folding=True,
                          input_names=["x", "cond", "text", "t"], output_names=["flow"],
                          dynamic_axes={"x": {0: "batch", 1: "frames"}, "cond": {0: "batch", 1: "frames"},
                                        "text": {0: "batch", 1: "frames"}, "t": {0: "batch"},
                                        "flow": {0: "batch", 1: "frames"}})
    onnx.checker.check_model(str(dit_path))
    made["nandini_dit.onnx"] = {"inputs": {"x": "float32 [batch, frames, 100]", "cond": "float32 [batch, frames, 100]",
                                           "text": "int64 [batch, frames], -1 = filler", "t": "float32 [batch]"},
                                "outputs": {"flow": "float32 [batch, frames, 100]"}, "from": str(a.ema)}
    print(f"nandini_dit.onnx in {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    g = load_generator(a.vocoder, "cpu")
    print(f"{fixed_filters(g)} anti-aliasing activations given fixed kernels", flush=True)
    voc = Vocoder(g).eval()
    voc_path = out / "nandini_vocoder.onnx"
    with torch.no_grad():
        torch.onnx.export(voc, (torch.randn(1, 100, 200),), str(voc_path), opset_version=C.ONNX_OPSET,
                          do_constant_folding=True, input_names=["mel"], output_names=["wave"],
                          dynamic_axes={"mel": {0: "batch", 2: "frames"}, "wave": {0: "batch", 1: "samples"}})
    onnx.checker.check_model(str(voc_path))
    made["nandini_vocoder.onnx"] = {"inputs": {"mel": "float32 [batch, 100, frames]"},
                                    "outputs": {"wave": "float32 [batch, frames * 256], 24 kHz"}, "from": str(a.vocoder)}
    print(f"nandini_vocoder.onnx in {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    from f5_tts.infer.utils_infer import load_vocoder
    vs = VocosSpectrum(load_vocoder("vocos", device="cpu")).eval()
    with torch.no_grad():
        torch.onnx.export(vs, (torch.randn(1, 100, 200),), str(out / "vocos_spectrum.onnx"), opset_version=C.ONNX_OPSET,
                          do_constant_folding=True, input_names=["mel"], output_names=["real", "imag"],
                          dynamic_axes={"mel": {0: "batch", 2: "frames"}, "real": {0: "batch", 2: "frames"},
                                        "imag": {0: "batch", 2: "frames"}})
    onnx.checker.check_model(str(out / "vocos_spectrum.onnx"))
    made["vocos_spectrum.onnx"] = {"inputs": {"mel": "float32 [batch, 100, frames]"},
                                   "outputs": {"real": "float32 [batch, 513, frames]", "imag": "float32 [batch, 513, frames]"},
                                   "from": "charactr/vocos-mel-24khz (IndicF5's stock vocoder, MIT), for the retry test only"}
    print(f"vocos_spectrum.onnx in {time.time() - t0:.0f} s", flush=True)

    for name, m in made.items():
        p = out / name
        m |= {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {"opset": C.ONNX_OPSET, "precision": "fp32", "torch": torch.__version__, "onnx": onnx.__version__,
                "indicf5_commit": C.INDICF5_COMMIT, "vagdhenu_commit": C.VAGDHENU_COMMIT, "files": made}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v["bytes"] for k, v in made.items()}), file=sys.stderr)
