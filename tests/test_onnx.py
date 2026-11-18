"""The ONNX version: the sampler's time grid, chunking, weight storage, and the commands. No models needed."""
import numpy as np
import pytest

from nandini import cli
from nandini import config as C
from nandini.onnx.render import chunk_text, time_grid


def test_time_grid_is_indicf5s_sway():
    t = time_grid(64)
    ref = np.linspace(0, 1, 65)
    ref = ref - (np.cos(np.pi / 2 * ref) - 1 + ref)
    assert t.dtype == np.float32 and len(t) == 65 and t[0] == 0 and abs(t[-1] - 1) < 1e-6
    assert np.allclose(t, ref, atol=1e-6) and (np.diff(t) > 0).all()


def test_chunk_text_keeps_short_lines_whole():
    assert chunk_text("ಸನ್ತತಂ ಚಿನ್ತಯೇತ್", max_chars=200) == ["ಸನ್ತತಂ ಚಿನ್ತಯೇತ್"]
    assert len(chunk_text("one, two, three, four", max_chars=8)) > 1


def test_weight_storage_int8_and_fp16(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    from nandini.onnx.small import store
    w = np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 64])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 32])],
                          [numpy_helper.from_array(w, "w")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", C.ONNX_OPSET)], ir_version=8)  # as torch exports
    src = tmp_path / "m.onnx"
    onnx.save(m, str(src))
    x = np.random.default_rng(1).standard_normal((5, 64)).astype(np.float32)
    ref = x @ w
    for kind, tol in (("int8", 0.02), ("fp16", 1e-3)):
        dst = tmp_path / f"m_{kind}.onnx"
        assert store(src, dst, ("MatMul",), kind) == 1
        y = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        assert np.abs(y - ref).max() / np.abs(ref).max() < tol


def test_onnx_commands_parse(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_in_train_env", lambda a, argv: None)
    import nandini.onnx.render as R
    monkeypatch.setattr(R, "run", lambda a: seen.setdefault("render", a))
    cli.main(["onnx", "render", "--bank", "B", "--text", "राम"])
    a = seen["render"]
    assert a.seed == C.WEB_SEED and a.steps == 64 and a.out == "out.wav" and a.onnx.endswith("training/onnx")
    assert cli._needs(a) == ("onnxruntime", "indic_transliteration", "pydub")
