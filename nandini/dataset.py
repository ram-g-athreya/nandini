"""The source recordings and manifests: where a clip's audio is, which clips a run covers, and where its files go."""
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from . import config as C


def key(clip_id):
    """A clip's file stem in outputs: 'style_a/audio/x.wav' -> 'style_a__x'."""
    return clip_id.replace("/audio/", "__").removesuffix(".wav")


def source(clip_id, dataset_dir=None):
    """Vāgdhenu's recording: from a local copy of the dataset if given and present, else the Hugging Face cache at the
    pinned revision."""
    if dataset_dir:
        p = Path(dataset_dir) / clip_id
        if p.is_file():
            return p
    return Path(hf_hub_download(C.DATASET, clip_id, repo_type="dataset", revision=C.DATASET_REV))


def metadata(style, dataset_dir=None):
    """A style's metadata.csv, locally if present, else from the Hugging Face cache."""
    local = Path(dataset_dir) / style / "metadata.csv" if dataset_dir else None
    if local and local.is_file():
        return local
    return Path(hf_hub_download(C.DATASET, f"{style}/metadata.csv", repo_type="dataset", revision=C.DATASET_REV))


def select(manifest, part="0/1", indices=None, clips=0):
    """The manifest's rows this run covers: optionally only some indices or the first N, then every P-th from i."""
    m = pd.read_csv(manifest)
    if indices:
        m = m[m["index"].isin(int(x) for x in str(indices).split(","))]
    if clips:
        m = m.head(clips)
    i, p = (int(x) for x in part.split("/"))
    return m.iloc[i::p], i


def conv_path(out, j, clip_id):
    return Path(out) / "conversions" / f"c{j}" / f"{key(clip_id)}.wav"


def melody_path(out, j, clip_id):
    return Path(out) / "melody" / f"c{j}" / f"{key(clip_id)}.wav"


def kept_list(run, kept=None):
    """The kept list of a run: --kept if given, else OUT/kept_rechecked.csv (after `nandini recheck-halves`), else generate's
    kept_part*.csv."""
    import pandas as pd
    if kept:
        return pd.read_csv(kept)
    if (Path(run) / "kept_rechecked.csv").exists():
        return pd.read_csv(Path(run) / "kept_rechecked.csv")
    parts = sorted(Path(run).glob("kept_part*.csv"))
    if not parts:
        raise SystemExit(f"no kept_rechecked.csv or kept_part*.csv in {run}; pass --kept")
    return pd.concat([pd.read_csv(p) for p in parts]).sort_values("index")
