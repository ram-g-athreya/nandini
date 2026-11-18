"""Write a manifest of Vāgdhenu clips with fixed indices: every clip's position in the whole dataset (style_a then
style_b, each in its metadata order, numbered from 1). Conversion seeds come from this index, so a clip gets the same
conversions in any manifest that contains it.

  nandini manifest --out OUT.csv [--select CLIPS.csv] [--dataset-dir DIR]

--select keeps only the clip_ids listed in CLIPS.csv (a `clip_id` column), in that file's order, with their fixed indices.
"""
import re

import pandas as pd

from .. import config as C
from ..dataset import metadata

VOWEL = re.compile(r"[aAiIuUfFxXeEoO]")  # SLP1 vowels: one per syllable
# syllables per quarter-verse -> meter family
FAMILIES = {8: "anustubh", 11: "tristubh", 12: "jagati", 13: "rucira", 14: "vasantatilaka",
            15: "malini", 17: "17-syllable", 19: "sardulavikridita", 21: "sragdhara"}


def family(slp1):
    n = len(VOWEL.findall(slp1))
    return next((f for p, f in FAMILIES.items() if abs(n - 4 * p) <= 1), "other")


def full(dataset_dir=None):
    rows = []
    for cfg in C.DATASET_CONFIGS:
        for r in pd.read_csv(metadata(cfg, dataset_dir)).itertuples():
            rows.append({"clip_id": f"{cfg}/{r.file_name}", "style": cfg, "family": family(r.text_slp1), "text": r.text_devanagari})
    m = pd.DataFrame(rows)
    m.insert(0, "index", range(1, len(m) + 1))
    return m


def run(args):
    m = full(args.dataset_dir)
    if args.select:
        want = pd.read_csv(args.select).clip_id
        missing = set(want) - set(m.clip_id)
        assert not missing, f"not in the dataset: {sorted(missing)[:5]}"
        m = m.set_index("clip_id").loc[want].reset_index()[["index", "clip_id", "style", "family", "text"]]
    m.to_csv(args.out, index=False)
    print(len(m), "clips ->", args.out)
