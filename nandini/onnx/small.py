"""Smaller ONNX files (PLAN-free summary: README, "The ONNX version"). Training environment for the AdaLN step.

  nandini onnx small --ema EMA [--onnx training/onnx]

From the fp32 files of `nandini onnx make`, writes beside them:
  nandini_dit_int8.onnx        transformer weights stored as int8, one scale per output column (de-quantised on load;
                               arithmetic stays float)
  nandini_dit_r32.onnx         AdaLN factorised to rank ADALN_RANK, fp32
  nandini_dit_r32_int8.onnx    both
  nandini_vocoder_fp16.onnx    BigVGAN weights stored as fp16 (cast to fp32 on load)
  nandini_vocoder_int8.onnx    BigVGAN weights stored as int8, one scale per output channel
  small.json                   sizes, and how closely the rank-32 AdaLN reproduces the full one

The rank-32 AdaLN follows the design of Hemanth's browser version of Vāgdhenu (gnumanth/sanskrit-tts-onnx, MIT): every
block's AdaLN reads the same time embedding, so one shared projection of it to 32 dimensions (fitted on the embeddings
of times across [0, 1]) followed by a 32-wide matrix per block replaces the 22 full 1024 x 6144 matrices.
"""
import json
from pathlib import Path

import numpy as np

ADALN_RANK = 32


# ---------------------------------------------------------------- AdaLN, rank 32 (torch, before export)
def lowrank_adaln(dit, rank=ADALN_RANK, n_t=2001):
    """Replace each AdaLN linear with a shared rank-`rank` projection; returns the largest relative error of the
    modulations over n_t times in [0, 1], and the share of the inputs' variance the projection keeps."""
    import torch
    from torch import nn
    with torch.no_grad():
        t = torch.linspace(0, 1, n_t)
        e = torch.nn.functional.silu(dit.time_embed(t))  # what every AdaLN linear reads
        mu = e.mean(0)
        u, s, vt = torch.linalg.svd(e - mu, full_matrices=False)
        kept = float((s[:rank] ** 2).sum() / (s ** 2).sum())
        P = nn.Parameter(vt[:rank].T.contiguous(), requires_grad=False)  # 1024 x rank, shared
        MU = nn.Parameter(mu.contiguous(), requires_grad=False)

        class LowRank(nn.Module):
            def __init__(self, lin):
                super().__init__()
                self.p, self.mu = P, MU
                self.w = nn.Parameter((lin.weight @ P).contiguous(), requires_grad=False)  # out x rank
                self.b = nn.Parameter((lin.bias + lin.weight @ MU).contiguous(), requires_grad=False)

            def forward(self, x):
                return ((x - self.mu) @ self.p) @ self.w.T + self.b

        worst = 0.0
        mods = [blk.attn_norm for blk in dit.transformer_blocks] + [dit.norm_out]
        for m in mods:
            full = m.linear(e)
            low = LowRank(m.linear)
            err = float((low(e) - full).abs().max() / full.abs().max())
            worst = max(worst, err)
            m.linear = low
    return worst, kept


# ---------------------------------------------------------------- weight storage (ONNX graph edits)
def _weights(model, ops):
    """(node, input index, axis of the output channels) for each float weight the given ops read from an initializer."""
    from onnx import helper
    inits = {i.name: i for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type not in ops or len(n.input) < 2 or n.input[1] not in inits:
            continue
        w = inits[n.input[1]]
        if w.data_type != 1 or len(w.dims) < 2:
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if attrs.get("group", 1) != 1:
            continue  # BigVGAN's fixed anti-aliasing kernels: small, and kept exact
        axis = {"MatMul": 1, "Gemm": 0 if attrs.get("transB", 0) else 1, "Conv": 0, "ConvTranspose": 1}[n.op_type]
        out.append((n, 1, axis))
    return out


def store(src, dst, ops, kind):
    """Rewrite `src` with the weights of `ops` stored as int8 (per output channel, DequantizeLinear) or fp16 (Cast)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    new_nodes, done = [], {}
    for n, k, axis in _weights(m, ops):
        name = n.input[k]
        if name not in done:
            w = numpy_helper.to_array(inits[name]).astype(np.float32)
            m.graph.initializer.remove(inits[name])
            if kind == "int8":
                red = tuple(i for i in range(w.ndim) if i != axis)
                scale = np.maximum(np.abs(w).max(axis=red), 1e-12) / 127.0
                shape = [1] * w.ndim; shape[axis] = -1
                q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
                m.graph.initializer.extend([numpy_helper.from_array(q, name + "_q8"),
                                            numpy_helper.from_array(scale.astype(np.float32), name + "_scale"),
                                            numpy_helper.from_array(np.zeros_like(scale, np.int8), name + "_zp")])
                new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q8", name + "_scale", name + "_zp"],
                                                  [name + "_dq"], axis=axis, name=name + "_dequant"))
            else:
                m.graph.initializer.append(numpy_helper.from_array(w.astype(np.float16), name + "_f16"))
                new_nodes.append(helper.make_node("Cast", [name + "_f16"], [name + "_dq"], to=TensorProto.FLOAT,
                                                  name=name + "_cast"))
            done[name] = name + "_dq"
        n.input[k] = done[name]
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return len(done)


def run(a):
    import onnx
    import torch
    from . import make
    d = Path(a.onnx)
    res = {}
    for kind in ("int8",):
        n = store(d / "nandini_dit.onnx", d / "nandini_dit_int8.onnx", ("MatMul", "Gemm"), kind)
        res["nandini_dit_int8.onnx"] = {"weights": n}
    step = make.wrappers()[0](make.load_dit(a.ema)).eval()
    worst, kept = lowrank_adaln(step.dit)
    res["adaln"] = {"rank": ADALN_RANK, "variance_kept": kept, "largest_relative_error": worst}
    print(f"AdaLN rank {ADALN_RANK}: keeps {kept:.6f} of the time embeddings' variance; largest relative error of the "
          f"modulations {worst:.2e}", flush=True)
    with torch.no_grad():
        torch.onnx.export(step, make.example(), str(d / "nandini_dit_r32.onnx"), opset_version=17,
                          do_constant_folding=True, input_names=["x", "cond", "text", "t"], output_names=["flow"],
                          dynamic_axes={"x": {0: "batch", 1: "frames"}, "cond": {0: "batch", 1: "frames"},
                                        "text": {0: "batch", 1: "frames"}, "t": {0: "batch"},
                                        "flow": {0: "batch", 1: "frames"}})
    onnx.checker.check_model(str(d / "nandini_dit_r32.onnx"))
    res["nandini_dit_r32_int8.onnx"] = {"weights": store(d / "nandini_dit_r32.onnx", d / "nandini_dit_r32_int8.onnx",
                                                         ("MatMul", "Gemm"), "int8")}
    for kind in ("fp16", "int8"):
        f = f"nandini_vocoder_{kind}.onnx"
        res[f] = {"weights": store(d / "nandini_vocoder.onnx", d / f, ("Conv", "ConvTranspose"), kind)}
    for f in sorted(d.glob("nandini_*.onnx")):
        res.setdefault(f.name, {})["mb"] = round(f.stat().st_size / 1e6, 1)
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB", flush=True)
    (d / "small.json").write_text(json.dumps(res, indent=1) + "\n")
