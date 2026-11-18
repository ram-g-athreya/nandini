"""The comparison's command, sets and settings, without models."""
import pytest

from nandini import cli
from nandini import config as C
from nandini import external as X


def parse(*argv):
    seen = {}
    import nandini.evaluation.render as er
    import nandini.evaluation.score as es
    old = er.run, es.run, cli._in_train_env
    er.run = es.run = lambda a: seen.setdefault("a", a)
    cli._in_train_env = lambda a, argv: None
    try:
        cli.main(list(argv))
    finally:
        er.run, es.run, cli._in_train_env = old
    return seen["a"]


def test_compare_commands_parse():
    a = parse("compare", "render", "--system", "N", "--ema", "e", "--bank", "b", "--vocoder", "v", "--only", "1", "3")
    assert (a.system, a.ema, a.bank, a.vocoder, a.only) == ("N", "e", "b", "v", [1, 3])
    assert a.out == str(C.ROOT / "compare") and a.sets is None
    assert cli._needs(a) == ("f5_tts",)
    a = parse("compare", "score")
    assert cli._needs(a) == ()
    with pytest.raises(SystemExit):
        parse("compare", "render", "--system", "X")


def test_settings():
    assert C.COMPARE_SETS == {"V": ("A", "B"), "Vs": ("A", "B", "C"), "N": ("A", "B")}
    assert (C.H4_CER_MARGIN, C.H4_MELODY_MIN, C.H4_DURATION_TOL) == (2.0, 0.8, 0.10)
    assert C.WEB_SEED == 60
    sets = {s for v in C.COMPARE_SETS.values() for s in v}
    n = {"A": 61, "B": 60, "C": 6}
    assert sum(n[s] for v in C.COMPARE_SETS.values() for s in v) == 369 and sets == set(n)


def test_set_b_is_the_test_verses():
    from nandini.evaluation.render import verses
    v = verses("B")
    assert len(v) == 60 and list(v.n) == list(range(1, 61)) and v.text.str.len().gt(0).all()


@pytest.mark.skipif(X.head("vagdhenu") != C.VAGDHENU_COMMIT, reason="needs `nandini fetch vagdhenu`")
def test_set_c_is_his_six_demos():
    from nandini.evaluation.render import demo_verses
    d = demo_verses()
    assert len(d) == 6 and d.his_audio.str.startswith("docs/audio/").all()
    assert all((X.EXTERNAL / "vagdhenu" / p).exists() for p in d.his_audio)
    assert d.text.str.contains("\n").sum() >= 2  # his line breaks kept (the renderer splits pādas on them)
