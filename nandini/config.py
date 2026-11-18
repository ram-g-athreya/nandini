"""Every setting of the method, and the exact models it was tested with: the settings the released audio was made with.

Code reads them as `config.NAME` when it runs. `nandini config` prints them, and every `nandini generate` run writes
them to its output folder (`settings_part*.json`).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the folder holding nandini/, manifests/, voice/
VOICE_SAMPLE = ROOT / "voice" / "female_voice_sample.wav"  # the voice to convert to; --voice overrides it

# The source recordings: pinned so that everyone starts from the same audio.
DATASET = "prathoshap/vagdhenu-data"
DATASET_REV = "adda7747c1ea05c45e6b789d23d6b6bbf550918d"
DATASET_CONFIGS = ("style_a", "style_b")

# The voice converter (a Seed-VC checkpoint) and the settings it runs with.
CONVERTER_REPO = "DreamSyncCo/IndicVoiceChanger"
CONVERTER_REV = "6fb95c7c236422dc158434b8ecdb0f1b3dad8ce1"
CONVERTER_CKPT = "Indic-seed-uvit-whisper-small-wavenet.pth"
CONVERTER_CONFIG = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
SEED_VC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
DIFFUSION_STEPS = 30
INFERENCE_CFG_RATE = 0.7

# Outside code, fetched by `nandini fetch` into external/ (gitignored) at these commits, never copied into this repo.
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"  # runs the converter; commit SEED_VC_COMMIT above
VAGDHENU_REPO = "https://github.com/prathoshap/vagdhenu.git"  # Vāgdhenu's code: training script, text, renderer
VAGDHENU_COMMIT = "c18927a8a77775753e32248177a5a6bea1019a12"
BIGVGAN_REPO = "https://github.com/NVIDIA/BigVGAN.git"  # imported by Vāgdhenu's renderer when it loads (unused: vocos)
BIGVGAN_COMMIT = "7d2b454564a6c7d014227f635b7423881f14bdac"

# The speech recogniser that judges pronunciation.
ASR_REPO = "prathoshap/sushrota-sanskrit-asr"
ASR_REV = "5452d080bbffa62ff879e3ddd53b42dfa7a1db64"
ASR_FILE = "sushrota_sanskrit_asr_v13b.nemo"

# The speaker-embedding model that measures voice consistency.
SPEAKER_REPO = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_REV = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"

# Attempts per verse, and their seeds: attempt j of a verse uses SEED_STEP * j + the verse's index (its fixed position in
# the whole dataset). A verse keeps its first attempt with at most MAX_ERRORS errors, else its best of N_CONVERSIONS.
# At most 1, not 0: the ASR misreads about 4.7% of syllables on Vāgdhenu's own recordings, so one error is within
# its noise, and a 0-error rule needed about 1.7 times the attempts.
N_CONVERSIONS = 10
SEED_STEP = 1000
MAX_ERRORS = 1

# How K_SEMITONES was measured (`nandini pitch-shift`): the target voice reads PITCH_PER_STYLE verses per style (drawn
# in proportion to each style's meters with seed PITCH_DRAW_SEED, plus one śārdūlavikrīḍita verse that rounding leaves
# out of style_a); k is the gap between the mean log pitch of those readings and of all of Vāgdhenu's clips in the
# style. Pitch: Praat autocorrelation, 10 ms frames, two passes per speaker (50-800 Hz, then 0.75 x Q1 to 1.5 x Q3).
TARGET_TTS_REPO = "bodhan-ai/indic-speak"
TARGET_TTS_REV = "76e0814f189321efe550011850cd288dae366aad"
TARGET_SPEAKER = "Bharati"
TARGET_TTS_SEED = 24
TARGET_TTS_MAX_TOKENS = 4920
PITCH_DRAW_SEED = 20261110
PITCH_PER_STYLE = 30
PITCH_ADD = (("style_a", "sardulavikridita"),)
PITCH_CER_LIMIT = 30.0  # a reading with a higher character error rate (or cut off at the token cap) is replaced
PITCH_BOOTSTRAP = 10000

# The melody step: Vāgdhenu's pitch raised by K_SEMITONES[style], from the mean log pitch of the target voice's reading
# against Vāgdhenu's chant, per style (measured, not chosen).
K_SEMITONES = {"style_a": 6.74, "style_b": 6.58}

# Audio and analysis. A "frame" is one STEP_MS slice of audio; WORLD describes each frame separately.
SR = 24000
STEP_MS = 5.0
TARGET_DBFS = -20.0
F0_FLOOR_HZ, F0_CEILING_HZ = 60, 600  # the WORLD pitch tracker's range
ENERGY_SMOOTHING_FRAMES = 4.0  # smoothing of Vāgdhenu's energy, applied to a conversion
# Cap on the energy gain's boost, in dB above the clip's median gain over frames within ENERGY_AUDIBLE_DB of Vāgdhenu's
# loudest (None: no cap). Without it, near-silent stretches of a conversion are boosted 20-74 dB into spikes.
ENERGY_MAX_BOOST_DB = 6.0  # tested against +12 dB: 6 dB removes the spikes (0.2% of clips pulled > 3 dB low, against 26.6%) without more syllables wrong
ENERGY_AUDIBLE_DB = 30.0

# Voice consistency: speaker embeddings over sliding windows.
WINDOW_S, HOP_S = 2.0, 1.0

# Training: Vāgdhenu's finetune_indicf5.py, unmodified, with his settings (its defaults) and his length. His released
# model (voice_armA) is the EMA checkpoint at step TRAIN_STOP_AT. IndicF5's code is installed in the training environment
# at INDICF5_COMMIT (third_party/train-requirements.txt); the warm start is the IndicF5 base at INDICF5_BASE_REV.
INDICF5_COMMIT = "13f7c4d627cc10111aea8fe9c0039462cacacdc7"
INDICF5_BASE_REPO = "ai4bharat/IndicF5"
INDICF5_BASE_REV = "fb652a96e508091045890dae5bb99fe511facbcf"
VAGDHENU_WEIGHTS_REPO = "prathoshap/vagdhenu"  # his released models; his BigVGAN is loaded by the renderer, then unused
VAGDHENU_WEIGHTS_REV = "0b8b7474c1899471b3d2b6a3ee919ab6c9373ebc"
VOCAB_SHA256 = "d3a5ff6aac12ea4fb50628a66a97b20c1e83b9e5ca356e5afae17b511cda96df"  # his reference_bank/vocab.txt
TRAIN_EPOCHS = 600  # his script's default: sets the learning-rate schedule (warm-up 500, then linear to 0)
TRAIN_LR = 1e-5
TRAIN_WARMUP = 500
TRAIN_FRAMES_PER_GPU = 19200
TRAIN_GPUS = 2
TRAIN_SAVE_PER = 400  # his trainer saves model_N.pt every this many steps; only TRAIN_KEEP are kept, as EMA
TRAIN_STOP_AT = 14800
TRAIN_KEEP = (2000, 4000, 6000, 8000, 10000, 12000, 14000, 14800)
TRAIN_CHECK_BEFORE = 12000  # the last-stretch check: TRAIN_STOP_AT against this (the nearest saved step to 3,000 before)
TRAIN_STALL_S = 600  # stop if the step count doesn't move for this long (a hang never ends the process by itself)
TRAIN_TEXT_ALLOW = ("\u0ce2",)  # vocalic l: outside his vocab, read by his trainer as a space, kept as in his data
TRAIN_MIN_S, TRAIN_MAX_S = 0.3, 30.0  # IndicF5's loader skips clips outside this range


def current():
    """Every setting, by name (paths relative to ROOT when inside it)."""
    out = {}
    for name, v in globals().items():
        if not name.isupper() or name == "ROOT":
            continue
        if isinstance(v, Path):
            v = str(v.relative_to(ROOT) if v.is_relative_to(ROOT) else v)
        elif isinstance(v, tuple):
            v = list(v)
        out[name] = v
    return out

# Vocoder fine-tune (`nandini vocoder`): his BigVGAN fine-tuned on her clips, as he made his (NVIDIA's BigVGAN-v2 on
# IndicF5's vocos-type mel, final weights the average of the last 6 checkpoints). NVIDIA's train.py at BIGVGAN_COMMIT,
# unchanged, with its v2 config; discriminators from NVIDIA's release at NV_BIGVGAN_REV (his aren't released).
HIS_VOCODER_FILE = "voc_bigvgan_EMA_2026-06-11.pth"  # in VAGDHENU_WEIGHTS_REPO at VAGDHENU_WEIGHTS_REV
NV_BIGVGAN_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"
NV_BIGVGAN_REV = "c329ede9e9bbc100ddf5c91e2330a61921262370"
NV_BIGVGAN_CONFIG = "bigvgan_v2_24khz_100band_256x.json"  # in BigVGAN's configs/
VOCODER_SAVE_EVERY = 1000  # a checkpoint every this many steps (a step: 4 random 2.73 s chunks, each from one clip)
VOCODER_AVERAGE = 6  # the final weights: the average of this many checkpoints, ending at the stop
VOCODER_MONITOR_EVERY = 2500  # the monitor decodes its mels with the checkpoint at every multiple of this
VOCODER_MIN_STEP = 10000  # no stop before this
VOCODER_CAP = 30000  # a cost ceiling, not a target
VOCODER_FLAT_OVER = 5000  # "levelled off": shiver unchanged against the checkpoint this many steps earlier
VOCODER_VAL_CLIPS = 3  # NVIDIA's train.py needs a validation list; validation itself is skipped (--debug)
ECHO_MEAN_MAX, ECHO_EACH_MAX = 0.02, 0.04  # round 1's echo rule (93.75 Hz frame-rate peak in the cepstrum)
VOCODER_CER_MARGIN = 0.25  # new errors at most vocos's + this many points (round 1's margin)
VOCODER_BOOT, VOCODER_BOOT_SEED = 10000, 20261109

# Test verses for the evaluation (set B): verses outside the dataset, in its meters, from the Digital Corpus of Sanskrit
# (Hellwig, CC BY 4.0), fetched into external/dcs at DCS_COMMIT (only DCS_WORKS are checked out).
DCS_REPO = "https://github.com/OliverHellwig/sanskrit.git"
DCS_COMMIT = "8aeed5a1343d0e48b64eb32af8c00e8c6eb29359"
DCS_WORKS = ("Bhāgavatapurāṇa", "Mukundamālā")  # devotional Vaiṣṇava verse, the closest open texts to the dataset's
TEST_VERSES_N = 60
TEST_VERSES_SEED = 20261117
# His detector's meter names (render_core.detect_meter_key) to the dataset's meter families.
METER_FAMILY = {"anushtubh": "anustubh", "indravajra": "tristubh", "upendravajra": "tristubh", "upajati": "tristubh",
                "vamshastha": "jagati", "indravamsha": "jagati", "vasantatilaka": "vasantatilaka", "malini": "malini",
                "shikharini": "17-syllable", "mandakranta": "17-syllable", "harini": "17-syllable",
                "prithvi": "17-syllable", "shardulavikridita": "sardulavikridita", "sragdhara": "sragdhara"}

# The comparison with Vāgdhenu: every system through his website's path (demo/app.py: auto meter, his fallback,
# Renderer.render_one(text, meter, seed=60), his defaults). V and Vs are his released models with his bank and his
# BigVGAN; N is ours (--ema) with the female bank (--bank) and the fine-tuned vocoder (--vocoder), all through his
# renderer unchanged. Set A: the 61 pitch-shift verses (his recordings exist); B: manifests/test_verses.csv; C: his 6
# demo verses (docs/index.html at VAGDHENU_COMMIT), Vs only, against his demo MP3s.
COMPARE_MODELS = {"V": "voice_armA_ema_2026-06-11.pt", "Vs": "voice_steer_ema_2026-06-17.pt"}  # in VAGDHENU_WEIGHTS_REPO
COMPARE_SETS = {"V": ("A", "B"), "Vs": ("A", "B", "C"), "N": ("A", "B")}
WEB_SEED = 60  # his website's default seed
# H4, registered in round 1 (2026-11-06), N against V on set B: mean CER within this many points of V's; mean melody
# correlation with V's render at least this; mean duration ratio within 1 +- this.
H4_CER_MARGIN, H4_MELODY_MIN, H4_DURATION_TOL = 2.0, 0.8, 0.10
COMPARE_BOOT, COMPARE_BOOT_SEED = 10000, 20261109
