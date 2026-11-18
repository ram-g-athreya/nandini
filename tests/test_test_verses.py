"""Set B's text handling and draw, without DCS or models."""
import pandas as pd

from nandini.pipeline.test_verses import as_dataset_prints, letters, longest_shared, quotas


def test_joins_as_the_dataset_prints():
    assert as_dataset_prints("svayaṃ tad antarhṛdaye 'vabhātam") == "svayaṃ tadantarhṛdaye'vabhātam"
    assert as_dataset_prints("yan na  pūrvam") == "yanna pūrvam"
    assert as_dataset_prints("rāmaḥ vanam āgataḥ") == "rāmaḥ vanamāgataḥ"


def test_letters_normalise_spelling():
    assert letters("चन्द्र") == letters("चंद्र")
    assert letters("सोऽजः । x") == "सोजः"


def test_longest_shared():
    assert longest_shared("अबकड", ["xxबकxx", "अब"]) == 0.5


def test_quotas_sum_and_follow_shares():
    q = quotas(pd.Series({"a": 398, "b": 245, "c": 27}), 60)
    assert q.sum() == 60 and q["a"] > q["b"] > q["c"] >= 2
