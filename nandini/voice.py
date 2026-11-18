"""Speaker embeddings over sliding windows, to see whether a clip keeps one voice throughout."""
import librosa
import numpy as np
import torch
from huggingface_hub import snapshot_download

from . import config as C

_enc = None


def _encoder():
    global _enc
    if _enc is None:
        from speechbrain.inference.speaker import EncoderClassifier
        local = snapshot_download(C.SPEAKER_REPO, revision=C.SPEAKER_REV)
        _enc = EncoderClassifier.from_hparams(local, savedir=local,
                                              run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    return _enc


def _embed(wav):
    with torch.no_grad():
        e = _encoder().encode_batch(torch.tensor(wav, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
    return e / np.linalg.norm(e)


def profile(path):
    """Whole-clip embedding, and window statistics: cosine of each 2 s window (1 s hop) to the clip's own mean window."""
    wav = librosa.load(path, sr=16000, mono=True)[0]
    whole = _embed(wav)
    w, h = int(C.WINDOW_S * 16000), int(C.HOP_S * 16000)
    starts = range(0, max(1, len(wav) - w + 1), h)
    win = np.stack([_embed(wav[s:s + w]) for s in starts])
    mean = win.mean(0)
    mean /= np.linalg.norm(mean)
    cos = win @ mean
    return whole, {"windows": len(win), "win_cos_min": float(cos.min()), "win_cos_mean": float(cos.mean()),
                   "win_cos_p10": float(np.percentile(cos, 10))}
