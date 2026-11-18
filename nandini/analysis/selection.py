"""The selection rule, simulated on the inventory's clips, and why it stops at at most 1 error rather than 0.

  nandini selection simulate  [--inventory inventory]   -> INV/selection/simulation.md, simulation.csv
  nandini selection threshold [--inventory inventory]   -> INV/selection/threshold.md

No new audio: the inventory (`nandini inventory convert` and `score`) already has, for every clip, N_CONVERSIONS
attempts (a conversion, then the melody step) scored by Su-śrotā. Error: a syllable it gets right on Vāgdhenu's
recording and wrong on the attempt after the melody step. Rule: attempts in seed order; stop at the first with at most
MAX_ERRORS errors, else keep the best of N_CONVERSIONS (the earliest on ties).
"""
from pathlib import Path

import pandas as pd

from .. import config as C


def load(inventory):
    files = sorted(Path(inventory).glob("syllables_part*.csv"))
    if not files:
        raise SystemExit(f"no syllables_part*.csv in {inventory}: run `nandini inventory score` first")
    return pd.concat(pd.read_csv(f) for f in files)


def simulate_rule(s, max_errors, max_attempts):
    """One row per clip: attempts used, the kept attempt and its errors."""
    s = s[~s.his_flag]
    att = (s.groupby(["style", "family", "clip_id", "j"])
           .agg(errors=("melody_flag", "sum"), syllables=("melody_flag", "size")).reset_index())
    rows = []
    for (style, family, clip), g in att.groupby(["style", "family", "clip_id"]):
        g = g.sort_values("j").head(max_attempts)
        best, used = None, 0
        for r in g.itertuples():
            used += 1
            if best is None or r.errors < best.errors:
                best = r
            if r.errors <= max_errors:
                break
        rows.append({"clip_id": clip, "style": style, "family": family, "syllables": best.syllables,
                     "attempts": used, "kept_j": best.j, "kept_errors": best.errors,
                     "met": best.errors <= max_errors})
    return pd.DataFrame(rows)


def summary(v, by, max_errors, max_attempts):
    groups = [(k, g) for k, g in v.groupby(by)] + [("all", v)]
    return pd.DataFrame([{by: k, "Verses": len(g), "Mean syllables": round(g.syllables.mean(), 1),
                          "Mean attempts": round(g.attempts.mean(), 2),
                          f"Met the bar (≤ {max_errors} error)": int(g.met.sum()),
                          f"Used all {max_attempts}": int((g.attempts == max_attempts).sum()),
                          "Kept, % syllables wrong": round(100 * g.kept_errors.sum() / g.syllables.sum(), 2)}
                         for k, g in groups])


def simulate(a):
    out = Path(a.inventory) / "selection"
    out.mkdir(parents=True, exist_ok=True)
    v = simulate_rule(load(a.inventory), C.MAX_ERRORS, C.N_CONVERSIONS)
    v.to_csv(out / "simulation.csv", index=False)
    dist = v.attempts.value_counts().sort_index()
    L = ["# The selection rule, simulated on the inventory's clips", "",
         f"Rule: attempts in seed order; stop at the first with at most {C.MAX_ERRORS} error, else keep the best of "
         f"{C.N_CONVERSIONS}. Error: a syllable Su-śrotā gets right on Vāgdhenu's recording and wrong on the attempt "
         "after the melody step.", "",
         "## By style", "", summary(v, "style", C.MAX_ERRORS, C.N_CONVERSIONS).to_markdown(index=False), "",
         "## By meter", "", summary(v, "family", C.MAX_ERRORS, C.N_CONVERSIONS).to_markdown(index=False), "",
         "## Attempts used", "",
         pd.DataFrame({"Attempts": dist.index, "Verses": dist.values}).to_markdown(index=False), "",
         "*'Kept, % syllables wrong' is judged by the same ASR that decided when to stop, so it is "
         "optimistic.*", ""]
    (out / "simulation.md").write_text("\n".join(L))
    print("\n".join(L))


def threshold(a):
    out = Path(a.inventory) / "selection"
    out.mkdir(parents=True, exist_ok=True)
    s = load(a.inventory)
    n_att = C.N_CONVERSIONS
    rows, runs = [], {}
    for m in (1, 0):
        v = runs[m] = simulate_rule(s, m, n_att)
        for st, g in list(v.groupby("style")) + [("all", v)]:
            rows.append({"style": st, "Stop at": f"≤ {m}" if m else "0", "Verses": len(g),
                         "Mean attempts": round(g.attempts.mean(), 2),
                         "Total attempts": int(g.attempts.sum()),
                         "Kept with 0 errors": int((g.kept_errors == 0).sum()),
                         "Kept with ≤ 1 error": int((g.kept_errors <= 1).sum()),
                         f"Used all {n_att}": int((g.attempts == n_att).sum()),
                         "Kept, % syllables wrong": round(100 * g.kept_errors.sum() / g.syllables.sum(), 2)})
    rules = pd.DataFrame(rows)
    v1 = runs[1]
    total, hard = int(v1.kept_errors.sum()), int(v1.kept_errors[v1.kept_errors >= 2].sum())
    h = s[s.j == 0].groupby(["style", "clip_id"]).agg(n=("his_flag", "size"), f=("his_flag", "sum")).reset_index()
    floor = pd.DataFrame([{"style": st, "Verses": len(g), "Median syllables": g.n.median(),
                           "Misread on Vāgdhenu's recording, %": round(100 * g.f.sum() / g.n.sum(), 2),
                           "Verses with ≥ 1 misread": int((g.f >= 1).sum()),
                           "1 error, % of the median verse": round(100 / g.n.median(), 1)}
                          for st, g in list(h.groupby("style")) + [("all", h)]])
    a1, a0 = rules[rules["Stop at"] == "≤ 1"], rules[rules["Stop at"] == "0"]
    ratio = a0[a0["style"] == "all"]["Total attempts"].item() / a1[a1["style"] == "all"]["Total attempts"].item()
    L = ["# Stopping at ≤ 1 error or at 0", "",
         f"The inventory's {s.clip_id.nunique()} clips, each with {n_att} attempts (a conversion, then the melody step), "
         "scored by Su-śrotā. Error: a syllable it gets right on Vāgdhenu's recording and wrong on the attempt. Both "
         f"rules keep the best of {n_att} if the target is never reached.", "",
         "## The two rules", "", rules.to_markdown(index=False), "",
         f"*Stopping at 0 takes {ratio:.2f} times the attempts of stopping at ≤ 1. Under ≤ 1, {hard} "
         f"of the {total} kept errors are in verses kept with 2 or more errors (none of their {n_att} attempts reached "
         "≤ 1), which no stricter threshold changes. Error rates are judged by the ASR that made the selection, so "
         "they are optimistic, more so for the stricter rule.*", "",
         "## Su-śrotā on Vāgdhenu's own recordings", "", floor.to_markdown(index=False), "",
         "*Vāgdhenu's recordings are taken as correctly pronounced, so these misreads are the judge's "
         "own error on human chanting.*", ""]
    (out / "threshold.md").write_text("\n".join(L))
    print("\n".join(L))
