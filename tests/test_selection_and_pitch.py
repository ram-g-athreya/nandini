"""The selection simulation and the pitch shift's pure parts (no models, no audio)."""
import numpy as np
import pandas as pd

from nandini import config as C
from nandini.analysis.selection import simulate_rule
from nandini.pipeline.pitch_shift import allocation, cer, st


def _syllables(errors_by_attempt, n=10):
    """One clip whose attempt j has errors_by_attempt[j] errors among n syllables (none wrong on Vāgdhenu's)."""
    rows = []
    for j, e in enumerate(errors_by_attempt):
        for k in range(n):
            rows.append({"clip_id": "c", "style": "style_a", "family": "anustubh", "j": j, "syllable": k,
                         "his_flag": False, "melody_flag": k < e})
    return pd.DataFrame(rows)


def test_rule_stops_at_first_good_attempt():
    v = simulate_rule(_syllables([3, 2, 1, 0]), max_errors=1, max_attempts=10)
    assert (v.attempts.item(), v.kept_j.item(), v.kept_errors.item(), v.met.item()) == (3, 2, 1, True)


def test_rule_keeps_earliest_best_of_all():
    v = simulate_rule(_syllables([4, 2, 3, 2]), max_errors=1, max_attempts=10)
    assert (v.attempts.item(), v.kept_j.item(), v.kept_errors.item(), v.met.item()) == (4, 1, 2, False)


def test_zero_error_rule_needs_more_attempts():
    s = _syllables([1, 1, 0])
    assert simulate_rule(s, 1, 10).attempts.item() == 1 and simulate_rule(s, 0, 10).attempts.item() == 3


def test_allocation_sums_to_per_style_plus_added():
    d = pd.DataFrame({"config": ["style_a"] * 70 + ["style_b"] * 30,
                      "family": ["anustubh"] * 50 + ["tristubh"] * 20 + ["anustubh"] * 30})
    d = pd.concat([d, pd.DataFrame({"config": ["style_a"], "family": ["sardulavikridita"]})])
    n = allocation(d)
    assert sum(v for (s, _), v in n.items() if s == "style_b") == C.PITCH_PER_STYLE
    assert sum(v for (s, _), v in n.items() if s == "style_a") == C.PITCH_PER_STYLE + 1
    assert n[("style_a", "sardulavikridita")] == 1


def test_pitch_helpers():
    assert np.isclose(st(200.0), 12.0)
    assert cer("रामः।", "रामः") == 0
    assert cer("राम", "राम्") > 0
