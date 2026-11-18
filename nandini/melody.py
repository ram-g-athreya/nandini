"""The melody step, with WORLD: Vāgdhenu's pitch raised k semitones and its frame energy, on one conversion's
spectral envelope and aperiodicity. k is set per style."""
import librosa
import numpy as np
import pyworld as pw
from scipy.ndimage import gaussian_filter1d

from . import config as C

DB = 10 / np.log(10)  # natural-log power ratio -> dB


def _world(path):
    y = librosa.load(path, sr=C.SR, mono=True)[0].astype(np.float64)
    f0, t = pw.harvest(y, C.SR, f0_floor=C.F0_FLOOR_HZ, f0_ceil=C.F0_CEILING_HZ, frame_period=C.STEP_MS)
    f0 = pw.stonemask(y, f0, t, C.SR)
    return f0, pw.cheaptrick(y, f0, t, C.SR), pw.d4c(y, f0, t, C.SR)


def analyse(path, k):
    """Vāgdhenu's pitch raised k semitones (unvoiced frames stay 0), and its envelope, used for its frame energy."""
    f0, sp, _ = _world(path)
    return f0 * 2 ** (k / 12), sp


# Conversions can differ from the original by a few frames, so they're trimmed or padded (repeating the last frame) to match.
def _fit(a, n):
    return a[:n] if len(a) >= n else np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)])


def melody_step(source_path, conversion_path, k, max_boost_db=...):
    """Audio at SR: Vāgdhenu's melody (+k st) and energy, the conversion's envelope and aperiodicity, at TARGET_DBFS.
    max_boost_db caps how far a frame's energy gain may rise above the clip's median gain (None: no cap; default: the
    configured ENERGY_MAX_BOOST_DB)."""
    if max_boost_db is ...:
        max_boost_db = C.ENERGY_MAX_BOOST_DB
    f0, sp = analyse(source_path, k)
    _, spc, apc = _world(conversion_path)
    spc, apc = _fit(spc, len(sp)), _fit(apc, len(sp))
    gain = gaussian_filter1d(np.log(sp.sum(1) + 1e-16) - np.log(spc.sum(1) + 1e-16), C.ENERGY_SMOOTHING_FRAMES)
    if max_boost_db is not None:
        his_db = DB * np.log(sp.sum(1) + 1e-16)
        typical = np.median(gain[his_db >= his_db.max() - C.ENERGY_AUDIBLE_DB])
        gain = np.minimum(gain, typical + max_boost_db / DB)
    x = pw.synthesize(f0, spc * np.exp(gain)[:, None], np.clip(apc, 1e-3, 1.0), C.SR, C.STEP_MS)
    x = x * 10 ** ((C.TARGET_DBFS - 20 * np.log10(np.sqrt(np.mean(x ** 2)))) / 20)
    return x / max(1.0, np.abs(x).max() / 0.99)
