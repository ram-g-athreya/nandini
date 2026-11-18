"""The re-check in halves: its rule, the final kept list, and which kept list later commands read."""
import pandas as pd

from nandini import config as C
from nandini.dataset import kept_list
from nandini.pipeline.recheck_halves import COLS, choose, final_list

KEPT = pd.DataFrame([
    # met on attempt 0: never re-checked
    {"clip_id": "a", "index": 1, "style": "style_a", "family": "f", "syllables": 40, "counted": 39, "errors": 0,
     "kept_j": 0, "attempts": 1, "met": True},
    # missed; in halves attempt 3 has 1 error, attempt 5 has 0: the first at the bar (3) is kept
    {"clip_id": "b", "index": 2, "style": "style_b", "family": "f", "syllables": 40, "counted": 38, "errors": 4,
     "kept_j": 7, "attempts": 10, "met": False},
    # missed and still misses in halves: generate's choice and scores stay
    {"clip_id": "c", "index": 3, "style": "style_b", "family": "f", "syllables": 40, "counted": 40, "errors": 3,
     "kept_j": 2, "attempts": 10, "met": False},
])


def rechecked():
    rows = [{"index": 2, "who": "his", "j": "", "counted": 37, "errors": 3}]
    rows += [{"index": 2, "who": "attempt", "j": j, "counted": 37, "errors": e}
             for j, e in enumerate([4, 2, 3, 1, 2, 0, 2, 2, 3, 2])]
    rows += [{"index": 3, "who": "his", "j": "", "counted": 40, "errors": 0}]
    rows += [{"index": 3, "who": "attempt", "j": j, "counted": 40, "errors": e}
             for j, e in enumerate([5, 4, 2, 3, 3, 2, 4, 5, 3, 4])]
    return pd.DataFrame(rows)


def test_choose_is_the_selection_rule():
    assert C.MAX_ERRORS == 1
    assert choose({0: 3, 1: 1, 2: 0}) == 1          # the first at the bar, not the fewest
    assert choose({0: 3, 1: 2, 2: 2}) == 1          # none at the bar: the fewest, earliest on ties


def test_final_list():
    f = final_list(KEPT, rechecked()).set_index("index")
    assert list(f.reset_index()[COLS].columns) == COLS
    assert f.loc[1, "scoring"] == "whole" and f.loc[1, "kept_j"] == 0
    b = f.loc[2]
    assert (b.scoring, b.kept_j, b.errors, b.counted, bool(b.met)) == ("halves", 3, 1, 37, True)
    assert (b.whole_kept_j, b.whole_errors, b.whole_counted) == (7, 4, 38)
    c = f.loc[3]
    assert (c.scoring, c.kept_j, c.errors, bool(c.met)) == ("whole", 2, 3, False)


def test_released_kept_list():
    d = pd.read_csv(C.ROOT / "manifests" / "released_kept.csv")
    assert list(d.columns) == COLS and len(d) == 1467 and d["index"].is_unique
    assert int(d.met.sum()) == 1298 and int((d.scoring == "halves").sum()) == 86
    assert int((d.kept_j != d.whole_kept_j).sum()) == 55
    assert (d[d.scoring == "halves"].errors <= C.MAX_ERRORS).all()


def test_kept_list_prefers_the_rechecked_list(tmp_path):
    KEPT.iloc[:2].to_csv(tmp_path / "kept_part0.csv", index=False)
    KEPT.iloc[2:].to_csv(tmp_path / "kept_part1.csv", index=False)
    assert len(kept_list(tmp_path)) == 3
    final_list(KEPT, rechecked()).iloc[:1].to_csv(tmp_path / "kept_rechecked.csv", index=False)
    assert len(kept_list(tmp_path)) == 1
    assert len(kept_list(tmp_path, tmp_path / "kept_part1.csv")) == 1
