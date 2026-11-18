"""The warm-start file for Vāgdhenu's finetune_indicf5.py, from the IndicF5 base checkpoint.

  nandini warm-start [--out training/warm.pt] [--check-only]

Runs in the training environment. Keeps the base's EMA weights (`ema_model._orig_mod.` prefix stripped; the bundled
vocoder dropped) and renames GRN `weight/bias` to `gamma/beta` if present: IndicF5's code names them gamma/beta,
zero-initialised, and his script loads with strict=False, so a wrong name would leave them at zero without an error.
Then loads it exactly as his script does and checks it: no missing or unexpected keys (other than the mel filter), the
text encoder's GRN tensors non-zero, every loaded tensor equal to the file's. Writes OUT and OUT.sha256.
"""
import hashlib
import sys
from pathlib import Path

from .. import config as C
from .data import vocab_file

PREFIX = "ema_model._orig_mod."
CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)  # his script's model


def make(out):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    raw = load_file(hf_hub_download(C.INDICF5_BASE_REPO, "model.safetensors", revision=C.INDICF5_BASE_REV))
    sd, renamed = {}, 0
    for k, v in raw.items():
        if not k.startswith(PREFIX):
            continue  # the vocoder bundled in the file; training uses none
        k = k[len(PREFIX):]
        if ".grn.weight" in k or ".grn.bias" in k:
            k = k.replace(".grn.weight", ".grn.gamma").replace(".grn.bias", ".grn.beta")
            renamed += 1
        sd[k] = v
    print(f"base {C.INDICF5_BASE_REPO}@{C.INDICF5_BASE_REV[:7]}: {len(raw)} tensors, kept {len(sd)}, "
          f"GRN keys renamed {renamed}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": sd}, out)
    digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    Path(f"{out}.sha256").write_text(f"{digest}  {Path(out).name}\n")
    print(f"saved {out} (sha256 {digest[:12]}…)")


def check(warm):
    """His script's warm-start lines, unchanged, plus the GRN and coverage checks."""
    import torch
    from f5_tts.infer.utils_infer import load_model
    from f5_tts.model import DiT
    sha = Path(f"{warm}.sha256")
    if sha.exists():
        want = sha.read_text().split()[0]
        got = hashlib.sha256(Path(warm).read_bytes()).hexdigest()
        if got != want:
            print(f"sha256 {got} differs from {sha}")
            return False
    cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=str(vocab_file()), device="cpu")
    ck = torch.load(warm, map_location="cpu", weights_only=True)
    sd = ck.get("model_state_dict", ck)
    miss, unexp = cfm.load_state_dict(sd, strict=False)
    n_miss = len([m for m in miss if "mel_spec" not in m])
    print("[warm-start] missing(non-melspec):", n_miss, "| unexpected:", len(unexp), flush=True)
    ok = n_miss == 0 and not unexp
    model = cfm.state_dict()
    text_grn = [k for k in model if ".grn." in k and "text_embed" in k]
    zero = [k for k in text_grn if float(model[k].abs().sum()) == 0]
    print(f"text-encoder GRN tensors {len(text_grn)}, all-zero {len(zero)}")
    ok &= bool(text_grn) and not zero
    differ = [k for k, v in sd.items() if k in model and not torch.equal(model[k], v)]
    not_from_file = [k for k in model if k not in sd and "mel_spec" not in k]
    print(f"tensors differing from the file {len(differ)}, model tensors not in the file {len(not_from_file)}")
    ok &= not differ and not not_from_file
    print("CHECK", "PASSED" if ok else "FAILED")
    return ok


def run(a):
    if not a.check_only:
        make(a.out)
    sys.exit(0 if check(a.out) else 1)
