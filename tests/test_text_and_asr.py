"""Text handling and error marking, on the dataset's own verses (no models needed)."""
import numpy as np
import pandas as pd
import pytest

from nandini import config as C
from nandini.asr import flagged_syllables, half_cut
from nandini.pipeline.manifest import FAMILIES
from nandini.text import Verse, to_slp1

MANIFEST = pd.read_csv(C.ROOT / "manifests" / "all_clips.csv")


def test_transliteration():
    assert to_slp1("रामः") == "rAmaH"
    assert to_slp1("कृष्ण") == "kfzRa"
    assert to_slp1("गच्छति") == "gacCati"


def test_syllables_match_meter():
    """Every verse in a named meter has about 4 × the meter's syllables per quarter (the manifest's own rule, ±1)."""
    per_quarter = {f: p for p, f in FAMILIES.items()}
    for r in MANIFEST[MANIFEST.family != "other"].itertuples():
        assert abs(Verse(r.text).n - 4 * per_quarter[r.family]) <= 1, r.clip_id


def test_syllables_cover_the_verse():
    for text in MANIFEST.text.head(50):
        v = Verse(text)
        assert v.spans[0][0] == 0 and v.spans[-1][1] == len(v.slp)
        assert all(a < b for a, b in v.spans)
        assert len(v.char_syl) == len(v.chars) and v.char_syl.max() == v.n - 1


@pytest.fixture
def verse():
    return Verse(MANIFEST.text.iloc[0])


def test_exact_transcript_has_no_errors(verse):
    assert not flagged_syllables(verse, MANIFEST.text.iloc[0]).any()


def test_changed_sound_marks_its_syllable(verse):
    k = 5
    a, b = verse.spans[k]
    vowel = next(i for i in range(a, b) if verse.slp[i] in "aAiIuUeo")
    swap = {"a": "A", "A": "a", "i": "I", "I": "i", "u": "U", "U": "u", "e": "E", "o": "O"}[verse.slp[vowel]]
    hyp = verse.slp[:vowel] + swap + verse.slp[vowel + 1:]
    flags = flagged_syllables(verse, _deva(hyp))
    assert flags.sum() == 1 and flags[k]


def test_dropped_syllable_is_marked(verse):
    k = 3
    a, b = verse.spans[k]
    flags = flagged_syllables(verse, _deva(verse.slp[:a] + verse.slp[b:]))
    assert flags[k]
    assert flags.sum() <= 2  # the alignment may also blame a neighbour


def _deva(slp):
    """SLP1 back to Devanagari, enough for these tests (flagged_syllables reads Devanagari)."""
    from nandini.text import CONS, MATRAS, VOWELS, VIRAMA, MARKS
    cons = {v: k for k, v in CONS.items()}
    vow = {v: k for k, v in VOWELS.items()}
    mat = {v: k for k, v in MATRAS.items()}
    marks = {"M": "ं", "H": "ः"}
    out, i = [], 0
    while i < len(slp):
        c = slp[i]
        if c in cons:
            out.append(cons[c])
            nxt = slp[i + 1] if i + 1 < len(slp) else ""
            if nxt == "a":
                i += 1
            elif nxt in mat:
                out.append(mat[nxt])
                i += 1
            else:
                out.append(VIRAMA)
        elif c in vow:
            out.append(vow[c])
        elif c in marks:
            out.append(marks[c])
        else:
            out.append(c)
        i += 1
    assert MARKS  # imported for completeness of the table
    return "".join(out)


def test_deva_round_trip(verse):
    assert to_slp1(_deva(verse.slp)) == verse.slp


def test_half_cut_finds_the_pause():
    """The cut lands in the quietest frame near t, and never outside 0.4 s before to 0.1 s after it."""
    sr = 16000
    y = np.random.default_rng(0).normal(0, 0.1, 4 * sr).astype("float32")
    y[int(1.80 * sr):int(1.90 * sr)] = 0  # a 100 ms pause, 0.2 s before t
    cut = half_cut(y, sr, 2.0)
    assert 1.80 * sr <= cut <= 1.90 * sr
    y[int(1.80 * sr):int(1.90 * sr)] = 0.1  # pause filled; another one outside the window is ignored
    y[int(1.0 * sr):int(1.1 * sr)] = 0
    assert 1.6 * sr <= half_cut(y, sr, 2.0) <= 2.1 * sr
