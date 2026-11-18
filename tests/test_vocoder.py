"""The vocoder fine-tune's pieces that run without models: options, the audio shift, the envelope lag, averaging and
the rules."""
import numpy as np
import pytest

from nandini import cli
from nandini import config as C
from nandini.analysis.vocoder_measures import boot, echo_ok
from nandini.training import vocoder as V


def parse(*argv):
    seen = {}

    def fake_run(a):
        seen["a"] = a
    import nandini.training.vocoder as tv
    import nandini.analysis.vocoder_score as vs
    old = tv.run, vs.run, cli._in_train_env
    tv.run = vs.run = fake_run
    cli._in_train_env = lambda a, argv: None
    try:
        cli.main(list(argv))
    finally:
        tv.run, vs.run, cli._in_train_env = old
    return seen["a"]


def test_vocoder_commands_parse():
    a = parse("vocoder", "data")
    assert a.step == "data" and a.lag == 0 and a.out == str(C.ROOT / "training/vocoder/data")
    assert parse("vocoder", "train").out == str(C.ROOT / "training/vocoder/run")
    a = parse("vocoder", "monitor", "--mels", "m")
    assert a.mels == "m" and a.run == str(C.ROOT / "training/vocoder/run")
    assert parse("vocoder", "decode", "--mels", "m", "--vocoder", "his", "--out", "o").vocoder == "his"
    a = parse("vocoder", "score", "--decoded", "vocos=a", "fine-tune=b", "--verses", "v.csv", "--out", "o")
    assert a.decoded == ["vocos=a", "fine-tune=b"] and not a.align
    assert cli._needs(a) == ()


def test_shift_pads_and_cuts():
    y = np.arange(1, 6, dtype=np.float32)
    assert V._shift(y, 0, 7).tolist() == [1, 2, 3, 4, 5, 0, 0]
    assert V._shift(y, 2, 5).tolist() == [0, 0, 1, 2, 3]
    assert V._shift(y, -2, 4).tolist() == [3, 4, 5, 0]


def test_lag_found_on_the_envelope():
    rng = np.random.default_rng(0)
    env = np.repeat(rng.random(60) > 0.5, 1200).astype(float)  # bursts of noise
    ref = env * rng.standard_normal(len(env))
    y = np.concatenate([np.zeros(37), (env * rng.standard_normal(len(env)))[:-37]])  # other noise, same bursts, late
    assert abs(V._lag(ref, y) - 37) <= 3


def test_average_is_the_mean_of_the_last_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    for k in range(0, 9000, 1000):
        torch.save({"generator": {"w": torch.full((2,), float(k))}}, tmp_path / f"g_{k:08d}")
    V.average(type("A", (), {"run": tmp_path, "stop": 8000})())
    avg = torch.load(tmp_path / "vocoder.pt")
    assert C.VOCODER_AVERAGE == 6
    assert torch.allclose(avg["w"], torch.full((2,), float(np.mean(range(3000, 9000, 1000)))))


def test_rules():
    assert echo_ok([0.01, 0.03], C.ECHO_MEAN_MAX, C.ECHO_EACH_MAX)
    assert not echo_ok([0.01, 0.05], C.ECHO_MEAN_MAX, C.ECHO_EACH_MAX)
    assert not echo_ok([0.03, 0.03], C.ECHO_MEAN_MAX, C.ECHO_EACH_MAX)
    m, lo, hi = boot([1, 2, 3, 4], [0, 1, 2, 3], 1000, 1)
    assert m == -1 and lo == hi == -1
