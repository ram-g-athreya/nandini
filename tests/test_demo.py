"""The demo app (demo/space/app.py) pins the same outside code and weights as the release (no models loaded)."""
import re

from nandini import chant
from nandini import config as C

APP = (C.ROOT / "demo" / "space" / "app.py").read_text(encoding="utf-8")


def _const(name):
    """A string constant from app.py (one per line: NAME = "...")."""
    return re.search(rf'^{name} = "([^"]+)"', APP, re.M).group(1)


def test_demo_pins_match_the_release():
    assert _const("VAGDHENU_COMMIT") == C.VAGDHENU_COMMIT
    assert _const("BIGVGAN_COMMIT") == C.BIGVGAN_COMMIT
    assert _const("NANDINI_REV") == C.MODEL_REV
    assert _const("VAGDHENU_REV") == C.VAGDHENU_WEIGHTS_REV
    assert chant.VOICE in APP and chant.VOCODER in APP
    assert f'"{C.COMPARE_MODELS["Vs"]}"' in APP  # his website's model, as in the comparison


def test_demo_site_is_self_contained():
    """Every audio file and image the website points at is in the site folder."""
    site = C.ROOT / "demo" / "site"
    page = (site / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="((?:audio/)[^"]+|[^":/#]+\.png)"', page)
    assert refs and all((site / r).exists() for r in refs)
