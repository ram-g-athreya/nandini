"""Settings: one place (nandini/config.py), printed by `nandini config` and recorded by every run."""
import json

from nandini import cli
from nandini import config as C


def test_release_settings():
    """The settings the released audio was made with."""
    assert C.K_SEMITONES == {"style_a": 6.74, "style_b": 6.58}
    assert (C.N_CONVERSIONS, C.SEED_STEP, C.MAX_ERRORS) == (10, 1000, 1)
    assert C.ENERGY_MAX_BOOST_DB == 6.0
    assert C.DATASET_REV == "adda7747c1ea05c45e6b789d23d6b6bbf550918d"
    assert C.VOICE_SAMPLE.is_file()


def test_config_prints_every_setting(capsys):
    cli.main(["config"])
    printed = json.loads(capsys.readouterr().out)
    names = {n for n in vars(C) if n.isupper() and n != "ROOT"}
    assert set(printed) == names
    assert printed["VOICE_SAMPLE"] == "voice/female_voice_sample.wav"
