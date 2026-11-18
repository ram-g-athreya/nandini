"""The `nandini` command: settings printing, manifests and argument checks (no models, no downloads)."""
import pandas as pd
import pytest

from nandini import cli
from nandini import config as C


def test_generate_needs_seed_vc(tmp_path, monkeypatch):
    """Without --seed-vc, generate looks in external/seed-vc and says what to run if it isn't there."""
    from nandini import external
    monkeypatch.setattr(external, "EXTERNAL", tmp_path)
    with pytest.raises(SystemExit, match="nandini fetch seed-vc"):
        cli.main(["generate"])


def test_defaults(monkeypatch):
    """generate runs every verse into out/; the inventory runs the 61-clip sample into inventory/."""
    seen = {}
    from nandini.analysis import inventory
    monkeypatch.setattr(inventory, "report", lambda a: seen.update(inv=a))
    cli.main(["inventory", "report"])
    assert seen["inv"].manifest.endswith("manifests/inventory_sample_clips.csv") and seen["inv"].out == str(C.ROOT / "inventory")
    from nandini.pipeline import generate_dataset
    monkeypatch.setattr(generate_dataset, "run", lambda a: seen.update(gen=a))
    cli.main(["generate", "--seed-vc", "x", "--seed-vc-python", "y"])
    g = seen["gen"]
    assert g.manifest.endswith("manifests/all_clips.csv") and g.conversions == C.N_CONVERSIONS
    assert g.out == str(C.ROOT / "out")  # inside this folder, wherever it is run from
    assert g.scoring == "whole"
    cli.main(["generate", "--seed-vc", "x", "--seed-vc-python", "y", "--scoring", "halves"])
    assert seen["gen"].scoring == "halves"


def test_manifest_select(tmp_path, monkeypatch):
    """`nandini manifest --select` keeps the chosen clips with their fixed indices. Uses the shipped full manifest in
    place of the dataset's metadata, so it runs offline."""
    from nandini.pipeline import manifest
    full = pd.read_csv(C.ROOT / "manifests" / "all_clips.csv")
    monkeypatch.setattr(manifest, "full", lambda dataset_dir=None: full)
    out = tmp_path / "m.csv"
    cli.main(["manifest", "--out", str(out), "--select", str(C.ROOT / "manifests" / "inventory_sample_clips.csv")])
    pd.testing.assert_frame_equal(pd.read_csv(out), pd.read_csv(C.ROOT / "manifests" / "inventory_sample_clips.csv"))


def test_check_commands_parse(monkeypatch):
    """speech-check / trend-check: render and score steps, default folders by check."""
    import nandini.cli as cli
    seen = {}
    monkeypatch.setattr(cli, "_in_train_env", lambda a, argv: None)
    import nandini.analysis.training_checks as tc
    monkeypatch.setattr(tc, "run", lambda a: seen.update(what=a.what, step=a.step, out=a.out))
    cli.main(["speech-check", "score"])
    assert seen == {"what": "speech", "step": "score", "out": str(cli.C.ROOT / "training/checks/speech")}
    cli.main(["trend-check", "score", "--before", "12000"])
    assert seen == {"what": "trend", "step": "score", "out": str(cli.C.ROOT / "training/checks/trend")}

