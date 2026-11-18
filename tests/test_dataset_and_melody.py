"""Manifests, run selection, output paths and the melody step (on synthetic audio; no models, no downloads)."""
import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf

from nandini import config as C
from nandini.dataset import conv_path, key, melody_path, select
from nandini.melody import melody_step
from nandini.pipeline.manifest import family

MANIFESTS = C.ROOT / "manifests"


def test_manifests():
    full = pd.read_csv(MANIFESTS / "all_clips.csv")
    assert len(full) == 1467
    assert list(full["index"]) == list(range(1, 1468))
    assert full["style"].value_counts().to_dict() == {"style_a": 764, "style_b": 703}
    sample = pd.read_csv(MANIFESTS / "inventory_sample_clips.csv")
    assert len(sample) == 61
    # The sample keeps each clip's fixed index, so its seeds are the full run's.
    merged = sample.merge(full, on="clip_id", suffixes=("", "_full"))
    assert (merged["index"] == merged["index_full"]).all()


def test_family():
    assert family("a" * 32) == "anustubh"
    assert family("a" * 84) == "sragdhara"
    assert family("a" * 50) == "other"


def test_select_parts_cover_every_clip_once():
    m = MANIFESTS / "all_clips.csv"
    parts = [select(m, f"{i}/4")[0] for i in range(4)]
    ids = pd.concat(parts).clip_id
    assert len(ids) == 1467 and ids.is_unique


def test_select_indices_and_paths():
    rows, part = select(MANIFESTS / "all_clips.csv", "0/1", indices="1,1345")
    assert list(rows["index"]) == [1, 1345] and part == 0
    cid = rows.clip_id.iloc[0]
    assert key(cid) == cid.replace("/audio/", "__").removesuffix(".wav")
    assert conv_path("out", 3, cid).parts[-3:] == ("conversions", "c3", key(cid) + ".wav")
    assert melody_path("out", 3, cid).parts[-3:] == ("melody", "c3", key(cid) + ".wav")


def _tone(path, f0, seconds=1.5, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    y = 0.3 * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
    sf.write(path, y, sr)


def test_melody_step(tmp_path):
    """The output follows Vāgdhenu's pitch raised k semitones, at the target level, without clipping."""
    his, conv = tmp_path / "his.wav", tmp_path / "conv.wav"
    _tone(his, 140.0)
    _tone(conv, 230.0, seconds=1.45)  # a conversion a few frames shorter, as happens
    k = 6.74
    y = melody_step(his, conv, k)
    assert abs(len(y) / C.SR - 1.5) < 0.02
    assert np.abs(y).max() <= 0.99
    rms_db = 20 * np.log10(np.sqrt(np.mean(y ** 2)))
    assert abs(rms_db - C.TARGET_DBFS) < 0.5
    f0, _ = pw.harvest(y.astype(np.float64), C.SR, f0_floor=C.F0_FLOOR_HZ, f0_ceil=C.F0_CEILING_HZ,
                       frame_period=C.STEP_MS)
    want = 140.0 * 2 ** (k / 12)
    assert abs(np.median(f0[f0 > 0]) / want - 1) < 0.02
