"""Vocoder measures (echo, shiver, new errors) and the rules that use them. Light imports only (numpy, scipy, librosa,
parselmouth), so they run in the training environment (the monitor) and in this code's environment (the score).

echo:   the frame-rate peak in the cepstrum, at the quefrency of one hop (256 samples, 93.75 Hz at 24 kHz) and its double.
        His BigVGAN adds it on her voice; about 0 for real audio.
shiver: on the held note, the share of the amplitude envelope's modulation power in 5-30 Hz (envelope at 200 Hz, mean
        removed). Vocos adds it on long vowels.
held note: the longest run of voiced 10 ms frames whose pitch moves under 0.3 semitone per frame (Praat pitch, 60-600 Hz).
"""
import numpy as np

SR = 24000


def held_note(y, sr=SR):
    """(seconds, start s, end s) of the longest held note; (0, 0, 0) if none."""
    import parselmouth
    p = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr).to_pitch(
        time_step=0.01, pitch_floor=60, pitch_ceiling=600)
    f, t = p.selected_array["frequency"], p.xs()
    st = np.where(f > 0, 12 * np.log2(np.maximum(f, 1e-9) / 100), np.nan)
    best, cur = (0, 0, 0), None
    for i in range(len(st)):
        ok = not np.isnan(st[i]) and cur is not None and not np.isnan(st[i - 1]) and abs(st[i] - st[i - 1]) < 0.3
        if not np.isnan(st[i]) and cur is None:
            cur = i
        elif not ok and cur is not None:
            if i - cur > best[0]:
                best = (i - cur, cur, i - 1)
            cur = i if not np.isnan(st[i]) else None
    if cur is not None and len(st) - cur > best[0]:
        best = (len(st) - cur, cur, len(st) - 1)
    if best[0] == 0:
        return 0.0, 0.0, 0.0
    return best[0] * 0.01, float(t[best[1]]), float(t[best[2]])


def echo(y):
    import librosa
    S = np.abs(librosa.stft(np.asarray(y, dtype=np.float32), n_fft=8192, hop_length=2048)) + 1e-9
    c = np.fft.irfft(np.log(S), axis=0).mean(1)
    return float(c[255:258].max() + c[511:514].max())


def shiver(y, t0, t1, sr=SR):
    from scipy.signal import hilbert, resample_poly
    seg = np.asarray(y, dtype=np.float64)[int(t0 * sr): int(t1 * sr)]
    if len(seg) < sr // 10:
        return float("nan")
    env = resample_poly(np.abs(hilbert(seg)), 1, sr // 200)
    env = env / (env.mean() + 1e-9) - 1
    P = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    f = np.fft.rfftfreq(len(env), 1 / 200)
    return float(P[(f >= 5) & (f <= 30)].sum() / (P[f > 0.5].sum() + 1e-12))


def boot(a, b, n, seed):
    """Paired bootstrap of mean(b - a): (mean, 2.5%, 97.5%)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[~np.isnan(d)]
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def echo_ok(values, mean_max, each_max):
    v = np.asarray(values, float)
    return bool(v.mean() <= mean_max and v.max() <= each_max)
