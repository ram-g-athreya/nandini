"""Runs in Indic-Speak's own environment: the target voice reads each drawn verse, one wav per reading.

Usage: <indic-speak python> target_readings_worker.py <draw.csv> <out dir> <settings JSON>
Settings (repo, revision, speaker, seed, max_new_tokens, device) come from nandini.config, passed by
nandini.pipeline.pitch_shift, because this process doesn't import this package. Resumes: readings already on disk are
skipped (the seed is the same for every verse). Appends one row per reading to <out dir>/../gen_times.csv.
"""
import csv
import json
import sys
import time
import warnings
from pathlib import Path

from huggingface_hub import snapshot_download

draw_csv, out, S = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
rows = list(csv.DictReader(open(draw_csv, encoding="utf-8")))
local = snapshot_download(S["repo"], revision=S["revision"])
sys.path.insert(0, local)
from inference import TTS  # noqa: E402  (the model repo's own code)

device = S["device"]
if device == "auto":
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time()
tts = TTS(local, device=device)
print(f"model load {time.time() - t0:.1f} s on {device}", flush=True)
out.mkdir(parents=True, exist_ok=True)
log = out.parent / "gen_times.csv"
for r in rows:
    i = int(r["reading"])
    wav_path = out / f"{i:02d}.wav"
    if wav_path.exists():
        continue
    text = r["text_devanagari"]
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wav = tts(text, speaker=S["speaker"], seed=S["seed"], max_new_tokens=S["max_new_tokens"])
    secs = time.time() - t
    capped = any("max_new_tokens" in str(x.message) for x in w)
    tts.save(str(wav_path), wav)
    dur = len(wav) / tts.sample_rate
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["reading", "clip_id", "n_chars", "audio_s", "gen_s", "capped", "device"])
        wr.writerow([i, r["clip_id"], len(text), round(dur, 2), round(secs, 1), capped, device])
    print(i, f"{len(text)} chars, {dur:.1f} s audio in {secs:.1f} s", "CAPPED" if capped else "ok", flush=True)
