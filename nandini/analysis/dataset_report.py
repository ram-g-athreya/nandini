"""The dataset report: attempts used, verses that met the bar, verses with 0 errors and the kept error rate, by style and
by meter, three charts, and the verses that missed.

  nandini report [--run out]

Reads RUN/kept_part*.csv and RUN/attempts_part*.csv; writes RUN/kept.csv and RUN/attempts.csv (all parts, in index
order) and RUN/report/ (report.md, errors-by-style.png, attempts-used.png, met-by-meter.png). Error rates come from the
ASR that chose each kept attempt, so they are optimistic.
"""
from pathlib import Path

import pandas as pd

from .. import config as C
from .marks import read_kept

COLOURS = ("#2a78d6", "#eb6834")  # two categorical colours, validated for contrast on a light surface
INK, MUTED, BASE, SURFACE, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#fcfcfb", "#e1e0d9"
METER_NAMES = {"anustubh": "anuṣṭubh", "tristubh": "triṣṭubh", "jagati": "jagatī", "rucira": "rucirā",
               "malini": "mālinī", "vasantatilaka": "vasantatilakā", "sardulavikridita": "śārdūlavikrīḍita",
               "sragdhara": "sragdharā", "17-syllable": "17-syllable", "other": "other"}
BINS = ["0", "1", "2", "3", "4", "5+"]


def summary(v, by):
    groups = [(k, g) for k, g in v.groupby(by)] + [("all", v)]
    return pd.DataFrame([{by: k, "Verses": len(g), "Mean syllables": round(g.syllables.mean(), 1),
                          "Mean attempts": round(g.attempts.mean(), 2),
                          "Kept with 0 errors": int((g.errors == 0).sum()),
                          f"Met the bar (≤ {C.MAX_ERRORS} error)": int(g.met.sum()),
                          "Met, %": round(100 * g.met.mean(), 1),
                          f"Used all {C.N_CONVERSIONS}": int((g.attempts == C.N_CONVERSIONS).sum()),
                          "Kept, % counted syllables wrong": round(100 * g.errors.sum() / g.counted.sum(), 2)}
                         for k, g in groups])


def error_distribution(kept):
    b = kept.errors.clip(upper=5).map(lambda e: BINS[e])
    counts = pd.crosstab(b, kept["style"]).reindex(BINS, fill_value=0)
    pct = (100 * counts / counts.sum()).round(1)
    t = pd.DataFrame({"Errors kept": BINS})
    for st in counts.columns:
        t[f"{st}, verses (%)"] = [f"{c} ({p:.1f}%)" for c, p in zip(counts[st], pct[st])]
    return counts, t


def _axes(plt, figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return fig, ax


def _grouped(plt, table, title, xlabel, path, bar_at=None, bar_text=None):
    """Grouped bars by style, each labelled 'count (pct%)'."""
    fig, ax = _axes(plt, (8.4, 4.2))
    xs = range(len(table))
    w = 0.38
    for k, st in enumerate(table.columns):
        c = table[st]
        p = 100 * c / c.sum()
        pos = [x + (k - 0.5) * w for x in xs]
        ax.bar(pos, c, width=w, color=COLOURS[k % 2], label=f"{st} ({c.sum()} verses)", zorder=2)
        for x, n, q in zip(pos, c, p):
            ax.text(x, n, f"{n}\n({q:.1f}%)", ha="center", va="bottom", fontsize=6.5, color=MUTED, linespacing=1.1)
    ax.set_xticks(list(xs), table.index)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel("Verses", color=MUTED, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_ylim(0, table.values.max() * 1.18)
    if bar_at is not None:
        ax.axvline(bar_at, color=BASE, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(bar_at + 0.05, ax.get_ylim()[1] * 0.97, bar_text, color=MUTED, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _met_by_meter(plt, kept, path):
    m = (kept.groupby("family").agg(verses=("met", "size"), met=("met", "sum"), syl=("syllables", "mean"))
         .sort_values("syl"))
    m["pct"] = 100 * m.met / m.verses
    fig, ax = _axes(plt, (8.4, 4.4))
    ys = range(len(m))
    ax.barh(list(ys), m.pct, color=COLOURS[0], height=0.62, zorder=2)
    for y, (_, r) in zip(ys, m.iterrows()):
        ax.text(r.pct + 1, y, f"{r.pct:.0f}%  ({int(r.met)} of {int(r.verses)})", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(list(ys), [f"{METER_NAMES.get(f, f)}  ·  {r.syl:.0f} syl." for f, r in m.iterrows()])
    ax.tick_params(axis="y", colors=INK)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel(f"Verses that met the bar (≤ {C.MAX_ERRORS} error), %", color=MUTED, fontsize=9)
    all_pct = 100 * kept.met.mean()
    ax.axvline(all_pct, color=BASE, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(all_pct + 0.6, len(m) - 0.45, f"all verses {all_pct:.1f}%", color=MUTED, fontsize=8, va="bottom")
    ax.set_title("Meeting the bar falls with verse length (meters by mean syllables)", loc="left", color=INK,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(args.run)
    kept = read_kept(run_dir)
    att = pd.concat(pd.read_csv(f) for f in sorted(run_dir.glob("attempts_part*.csv"))).sort_values(["index", "j"])
    assert kept.clip_id.is_unique
    assert (att.groupby("clip_id").size().reindex(kept.clip_id).values == kept.attempts.values).all()
    kept.to_csv(run_dir / "kept.csv", index=False)
    att.to_csv(run_dir / "attempts.csv", index=False)
    out = run_dir / "report"
    out.mkdir(exist_ok=True)

    counts, dist_t = error_distribution(kept)
    _grouped(plt, counts, f"Errors kept per verse, by style ({len(kept):,} verses)",
             "Errors in the kept attempt (syllables right on Vāgdhenu's recording, wrong on Nandinī's)",
             out / "errors-by-style.png", C.MAX_ERRORS + 0.5, f"bar: ≤ {C.MAX_ERRORS} error")
    used = pd.crosstab(kept.attempts, kept["style"]).reindex(range(1, C.N_CONVERSIONS + 1), fill_value=0)
    used.index = [str(i) for i in used.index]
    _grouped(plt, used, f"Attempts used per verse, by style ({len(kept):,} verses)",
             f"Attempts used ({C.N_CONVERSIONS} = never met the bar, or met it on the last; best kept)",
             out / "attempts-used.png")
    _met_by_meter(plt, kept, out / "met-by-meter.png")

    miss = kept[~kept.met]
    worst = miss.sort_values("errors", ascending=False).head(15)
    times = att.groupby("style").agg(convert_s=("convert_s", "mean"), rest_s=("rest_s", "mean")).round(1)
    hours = (att.convert_s.sum() + att.rest_s.sum()) / 3600
    L = ["# The dataset", "",
         f"{len(kept):,} verses. Rule: attempts in seed order; stop at the first with at most {C.MAX_ERRORS} error, "
         f"else keep the best of {C.N_CONVERSIONS}. Error: a syllable Su-śrotā gets right on Vāgdhenu's recording and "
         "wrong on the attempt. 'Counted' syllables are those it reads right on Vāgdhenu's recording. Error rates come "
         "from the ASR that chose each kept attempt, so they are optimistic.", "",
         f"Attempts: {len(att):,} ({len(att) / len(kept):.2f} per verse), {hours:.1f} h of attempt time summed over "
         "parts.", "",
         "## By style", "", summary(kept, "style").to_markdown(index=False), "",
         "## Errors kept per verse, by style", "", "![Errors kept per verse, by style](errors-by-style.png)", "",
         dist_t.to_markdown(index=False), "",
         f"*The chart's numbers. 5+ pools 5 to the maximum. 0 to {C.MAX_ERRORS} met the bar; more "
         f"are verses kept with their best of {C.N_CONVERSIONS} attempts.*", "",
         "## Attempts used", "", "![Attempts used per verse, by style](attempts-used.png)", "",
         used.rename_axis("Attempts").reset_index().to_markdown(index=False), "",
         "## By meter", "", "![Verses meeting the bar, by meter](met-by-meter.png)", "",
         summary(kept, "family").to_markdown(index=False), "",
         "## Time per attempt", "", times.reset_index().to_markdown(index=False), "",
         f"## Verses that missed the bar ({len(miss)})", "",
         f"Kept with their best of {C.N_CONVERSIONS} attempts. The {len(worst)} with the most errors:", "",
         worst[["index", "clip_id", "style", "family", "syllables", "counted", "errors", "kept_j"]]
         .to_markdown(index=False), "",
         f"*{len(worst)} of {len(miss)} rows, by errors (most first). All of them: `kept.csv`, met = False.*", ""]
    (out / "report.md").write_text("\n".join(L))
    print(f"{len(kept)} verses, {len(att)} attempts -> {out / 'report.md'}")
