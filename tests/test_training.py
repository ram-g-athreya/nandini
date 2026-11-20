"""Outside code, training settings and the training commands' plumbing (no models, no downloads, no GPU)."""
import subprocess

import numpy as np
import pytest

from nandini import cli
from nandini import config as C
from nandini import external


def _repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "f.txt").write_text("x")
    for args in (["add", "f.txt"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(path), *args], check=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_fetch_pins_the_commit(tmp_path, monkeypatch):
    """fetch clones and checks out the pinned commit; a commit that isn't there stops it; require() checks HEAD."""
    src = tmp_path / "src"
    src.mkdir()
    commit = _repo(src)
    monkeypatch.setattr(external, "EXTERNAL", tmp_path / "external")
    monkeypatch.setitem(external.REPOS, "demo", (str(src), commit))
    external.fetch("demo")
    assert external.head("demo") == commit and external.require("demo") == tmp_path / "external" / "demo"
    monkeypatch.setitem(external.REPOS, "demo", (str(src), "0" * 40))
    with pytest.raises(subprocess.CalledProcessError):
        external.fetch("demo")
    with pytest.raises(SystemExit, match="nandini fetch demo"):
        external.require("demo")


def test_outside_code_is_pinned_and_ignored():
    for name, (url, commit) in external.REPOS.items():
        assert url.startswith("https://") and len(commit) == 40, name
    assert "/external/" in (C.ROOT / ".gitignore").read_text().split()


def test_training_length_and_schedule():
    """His length, 14,800 steps; on 1,467 clips (49 steps per epoch on 2 GPUs) his 600 epochs end at 29,400, so the
    learning rate at 14,800 is about half its peak (warm-up 500, then linear to 0)."""
    assert (C.TRAIN_STOP_AT, C.TRAIN_EPOCHS, C.TRAIN_GPUS, C.TRAIN_LR) == (14800, 600, 2, 1e-5)
    total = 49 * C.TRAIN_EPOCHS
    lr = (total - C.TRAIN_STOP_AT) / (total - C.TRAIN_WARMUP)
    assert total == 29400 and abs(lr - 0.505) < 0.01
    assert C.TRAIN_STOP_AT in C.TRAIN_KEEP and C.TRAIN_CHECK_BEFORE in C.TRAIN_KEEP  # the last-stretch check
    assert all(k % C.TRAIN_SAVE_PER == 0 for k in C.TRAIN_KEEP)  # every kept step is one the trainer saves


def test_training_commands_need_the_training_environment(tmp_path, monkeypatch):
    """Started without the training packages and with no training environment, a training command says what to make."""
    import importlib.util
    monkeypatch.setattr(external, "TRAIN_PYTHON", tmp_path / "none" / "python")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda m, *a: None if m in ("f5_tts", "accelerate") else real(m, *a))
    with pytest.raises(SystemExit, match="PyTorch environment"):
        cli.main(["warm-start"])


def test_check_verses_are_round_1s():
    """The 10 check verses: drawn from the full manifest with seed 20261109."""
    import pandas as pd
    v = pd.read_csv(C.ROOT / "manifests" / "training_check_verses.csv")
    m = pd.read_csv(C.ROOT / "manifests" / "all_clips.csv")
    assert v.clip_id.tolist() == m.sample(10, random_state=20261109).clip_id.tolist() and v.n.tolist() == list(range(1, 11))


def test_bank_window_keeps_the_syllables_inside():
    """A cut's verse is the syllables starting inside it, converted one syllable at a time."""
    from nandini.pipeline.female_bank import window_verse
    from nandini.text import Verse
    v = Verse("रामो राजमणिः सदा विजयते")
    starts = np.arange(v.n) * 0.3
    w = window_verse(v, starts, 0, int(0.95 * C.SR))  # syllables starting before 0.85 s: the first 3
    assert w.n == 3 and w.slp == "".join(v.slp[s:e] for s, e in v.spans[:3])


def test_bank_overrides_name_real_entries():
    import json
    import pandas as pd
    o = pd.read_csv(C.ROOT / "manifests" / "female_bank_overrides.csv")
    assert set(o.half) <= {"same", "second"} and o.reason.str.len().min() > 20
    bank = external.VAGDHENU / "src" / "reference_bank" / "bank.json"
    if bank.exists():  # after `nandini fetch vagdhenu`
        assert set(o.entry) <= set(json.loads(bank.read_text()))
