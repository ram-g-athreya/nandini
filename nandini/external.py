"""Outside code, fetched into external/ (gitignored) at pinned commits, never copied into this repo.

  nandini fetch [seed-vc] [vagdhenu] [bigvgan] [dcs]      (default: all four)

Each is cloned from its own repository and checked out at the commit in nandini/config.py; the checkout's HEAD is then
checked against that commit. dcs (the Digital Corpus of Sanskrit, texts not code) is checked out sparsely: only the
works in DCS_WORKS. A folder already there is moved to the pinned commit, never deleted. Environments are made
separately (README, Setup): external/seed-vc/.venv for the converter, external/train-env for training.
"""
import subprocess
import sys

from . import config as C

EXTERNAL = C.ROOT / "external"
REPOS = {
    "seed-vc": (C.SEED_VC_REPO, C.SEED_VC_COMMIT),
    "vagdhenu": (C.VAGDHENU_REPO, C.VAGDHENU_COMMIT),
    "bigvgan": (C.BIGVGAN_REPO, C.BIGVGAN_COMMIT),
    "dcs": (C.DCS_REPO, C.DCS_COMMIT),
}
SPARSE = {"dcs": ["dcs/data/readme.md"] + [f"dcs/data/conllu/files/{w}/" for w in C.DCS_WORKS]}
SEED_VC = EXTERNAL / "seed-vc"
SEED_VC_PYTHON = SEED_VC / ".venv" / "bin" / "python"
VAGDHENU = EXTERNAL / "vagdhenu"
BIGVGAN = EXTERNAL / "bigvgan"
TRAIN_ENV = EXTERNAL / "train-env"
TRAIN_PYTHON = TRAIN_ENV / "bin" / "python"


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def head(name):
    """The checked-out commit of external/<name>, or None if it isn't there."""
    d = EXTERNAL / name
    if not (d / ".git").exists():
        return None
    return _git("rev-parse", "HEAD", cwd=d)


def fetch(name):
    url, commit = REPOS[name]
    d = EXTERNAL / name
    if not (d / ".git").exists():
        EXTERNAL.mkdir(exist_ok=True)
        print(f"cloning {url} into {d}", flush=True)
        if name in SPARSE:
            _git("clone", "-q", "--filter=blob:none", "--no-checkout", url, str(d))
            _git("sparse-checkout", "set", "--no-cone", *SPARSE[name], cwd=d)
            _git("checkout", "-q", commit, cwd=d)  # fills the empty index with only the sparse paths
        else:
            _git("clone", "-q", url, str(d))
    if head(name) != commit:
        try:
            _git("checkout", "-q", commit, cwd=d)
        except subprocess.CalledProcessError:
            _git("fetch", "-q", "origin", cwd=d)
            _git("checkout", "-q", commit, cwd=d)
    got = head(name)
    if got != commit:
        sys.exit(f"{name}: HEAD is {got}, expected {commit}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=d)
    note = f" (local changes: {len(dirty.splitlines())} files, e.g. a platform patch)" if dirty else ""
    print(f"{name}: {commit[:12]} ok{note}")


def require(name, hint=""):
    """The path of external/<name> at its pinned commit; stop with what to run if it's missing or elsewhere."""
    url, commit = REPOS[name]
    got = head(name)
    if got != commit:
        what = "missing" if got is None else f"at {got[:12]}, not {commit[:12]}"
        sys.exit(f"external/{name} is {what}: run `nandini fetch {name}`{hint}")
    return EXTERNAL / name


def run(args):
    unknown = [n for n in args.names if n not in REPOS]
    if unknown:
        sys.exit(f"unknown: {', '.join(unknown)} (choose from {', '.join(REPOS)})")
    for name in args.names or list(REPOS):
        fetch(name)
