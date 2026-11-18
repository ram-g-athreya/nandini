# Nandinī: a female voice for Vāgdhenu

Code for making a female Sanskrit chant voice for [Vāgdhenu](https://prathosh.in/vagdhenu/), with no recordings of a
female reciter. Vāgdhenu's released chant (1,467 clips, about 5.3 hours) is all in one male voice, and so is its model.
Nandinī is made in two parts:

1. **The dataset:** each of Vāgdhenu's clips is converted into a female voice, and his melody is put back, so every
   clip keeps his chant (text, timing and tune) in the new voice.
2. **The voice:** IndicF5 is fine-tuned on those clips with Vāgdhenu's own recipe, and the two other parts his renderer
   needs, the reference bank and the vocoder, are remade for the new voice. The result runs in Vāgdhenu's renderer
   unchanged.

## How it works

### Making the dataset

![How each clip is made: up to 10 attempts of voice conversion, melody step and recogniser check per verse](docs/dataset_pipeline.png)

Each verse goes through up to 10 attempts (`nandini generate`):

- **Voice conversion:** a voice converter (IndicVoiceChanger) turns Vāgdhenu's recording into the voice of a 25 s
  sample. Each attempt uses its own fixed seed, so every run makes the same conversions.
- **Melody step:** the converter keeps the new voice but drifts from the chant's pitch, so the clip is resynthesised
  with WORLD on Vāgdhenu's pitch, raised by k semitones (one k per chant style: +6.74 and +6.58), with his voicing and
  loudness. His timing is kept as it is.
- **Recogniser check:** [Su-śrotā](https://huggingface.co/prathoshap/sushrota-sanskrit-asr) transcribes both the clip
  and Vāgdhenu's recording. An **error** is a syllable it hears right in his recording and wrong in the clip, so its
  own mistakes on his chant aren't counted against the clip.
- **Keep or retry:** the first attempt with at most 1 error is kept; if none of the 10 gets there, the one with the
  fewest errors.
- **Re-check in halves:** on long verses the recogniser can lose its place and misread a whole stretch of a good clip.
  So the verses that missed were re-checked with each half verse transcribed on its own, and where an attempt now had
  at most 1 error, the first such attempt replaced the kept one ([how the released audio was made](#making-the-dataset)).

All 1,467 verses are kept, so the dataset has the same verses, styles and meters as Vāgdhenu's: 1,298 (88.5%) with at
most 1 syllable wrong.

### Training the voice

Vāgdhenu's renderer makes chant from three parts: a model, a reference bank and a vocoder. Each is remade for the new
voice, and nothing else changes ([Training](#training)):

- **The model:** IndicF5 fine-tuned on the 1,467 clips with Vāgdhenu's training script and settings, unchanged, to his
  length (14,800 steps; the EMA weights are used, as his are).
- **The reference bank:** for each meter, his renderer copies melody and pace from a short reference clip. The same cuts
  are taken from the same verses of the new dataset, so only the voice differs from his bank.
- **The vocoder:** turns the model's mel spectrogram into sound. IndicF5's own vocoder (vocos) adds a shiver on long held
  notes, and Vāgdhenu's fine-tuned BigVGAN adds an echo on a female voice, so BigVGAN-v2 is fine-tuned on the new clips
  the way he fine-tuned it on his.

Checks after training confirm the model speaks and that training had levelled off. It is then compared with
Vāgdhenu's own models on 60 verses neither has seen ([Comparing with Vāgdhenu](#comparing-with-vāgdhenu)).

## What's in this folder

Everything runs through one command, `nandini`, with every setting in one file, `nandini/config.py`.

| Path | What it is |
|:--|:--|
| `nandini/cli.py` | the `nandini` command and its subcommands |
| `nandini/config.py` | every setting the released audio was made with |
| `nandini/dataset.py` | the source recordings: where a clip's audio is, which clips a run covers, output paths |
| `nandini/pipeline/manifest.py` | `nandini manifest`: writes a manifest of clips with fixed indices |
| `nandini/pipeline/generate_dataset.py` | `nandini generate`: makes the dataset (the loop above) |
| `nandini/analysis/inventory.py` | `nandini inventory`: measures how the converter gets syllables wrong, on a sample of clips |
| `nandini/analysis/inventory_report.py` | the inventory's report |
| `nandini/analysis/marks.py` | `nandini mark`: what the ASR hears on each kept verse, per syllable, with timings |
| `nandini/analysis/tables.py` | `nandini tables`: transcripts, per-syllable and substitution tables |
| `nandini/analysis/dataset_report.py` | `nandini report`: the dataset report and its charts |
| `nandini/analysis/selection.py` | `nandini selection`: the selection rule simulated on the inventory, and ≤ 1 against 0 errors |
| `nandini/analysis/melody_checks.py` | `nandini melody-checks`: timing, voicing, loudness and the boost cap |
| `nandini/pipeline/pitch_shift.py`, `target_readings_worker.py` | `nandini pitch-shift`: how k per style was measured |
| `nandini/listening/` | `nandini listen build` and `serve`: the listening page (`page.html`) and a local server that can seek |
| `nandini/converter.py`, `converter_worker.py` | voice conversion, run in Seed-VC's own environment |
| `nandini/melody.py` | the melody step (WORLD) |
| `nandini/asr.py` | Su-śrotā transcription and per-syllable error marking |
| `nandini/text.py` | Sanskrit text handling: transliteration and syllables |
| `nandini/sounds.py` | sound classes, for grouping errors |
| `nandini/voice.py` | voice consistency, from speaker embeddings |
| `nandini/external.py` | `nandini fetch`: outside code into `external/` at pinned commits |
| `nandini/training/data.py` | `nandini training-data`: each kept clip with Vāgdhenu's `model_text`, and IndicF5's dataset files |
| `nandini/training/warm.py` | `nandini warm-start`: the warm start from the IndicF5 base, checked as his script loads it |
| `nandini/training/train.py` | `nandini train`: his training script, unmodified, with his settings and length |
| `nandini/training/ema.py` | `nandini extract-ema`: a checkpoint's EMA weights only |
| `nandini/training/checks.py`, `nandini/analysis/training_checks.py` | `nandini speech-check` / `trend-check`: renders after training, and their scores |
| `nandini/training/vocoder.py`, `nandini/analysis/vocoder_score.py`, `nandini/analysis/vocoder_measures.py` | `nandini vocoder`: the vocoder fine-tune, decoding saved mels, and its scores |
| `nandini/pipeline/female_bank.py` | `nandini female-bank`: his reference cuts, taken from the kept clips |
| `tests/` | quick checks that need no models or downloads (`pytest`) |
| `manifests/all_clips.csv` | all 1,467 clips: index, clip, style, meter, text |
| `manifests/pitch_shift_verses.csv` | the 61 verses the target voice read to measure k |
| `manifests/inventory_sample_clips.csv` | a 61-clip sample, stratified by meter within each style, for the inventory |
| `manifests/training_check_verses.csv` | the 10 verses the checks after training render (seed 20261109) |
| `manifests/female_bank_overrides.csv` | the bank entries changed by hand, with reasons |
| `manifests/test_verses.csv` | 60 new verses for evaluation, outside the dataset, from the Digital Corpus of Sanskrit (CC BY 4.0; [Test verses](#test-verses)) |
| `manifests/test_verses_pool.csv`, `test_verses_report.md` | every candidate verse with its checks, and how the 60 were drawn |
| `voice/female_voice_sample.wav` | the 25 s sample that sets the female voice |
| `docs/dataset_pipeline.png`, `.tex` | the dataset diagram above, and its TikZ source |
| `third_party/` | pinned requirements for the Seed-VC, Indic-Speak and training environments, and a patch for Seed-VC on Apple silicon (GPL-3.0, like Seed-VC) |
| `external/` | not in the repo (gitignored): outside code fetched by `nandini fetch`, and the Seed-VC and training environments |
| `uv.lock` | the exact package versions of this code's environment for the release run |
| `LICENSE`, `NOTICE` | the code's licence (Apache-2.0), and the parts with their own terms |

## Setup

Four environments, because Seed-VC (which runs IndicVoiceChanger), Indic-Speak and Vāgdhenu's training code each need
package versions of their own. [uv](https://docs.astral.sh/uv/) is assumed; the exact versions used for the release are
pinned. Outside code is never copied into this repo: `nandini fetch` clones it into `external/` (gitignored) at the
commits in `nandini/config.py` and checks each checkout's commit.

**This code** (Python 3.10), from this folder:

```bash
uv sync                      # installs the versions in uv.lock and the `nandini` command; run commands as `uv run nandini ...`
```

`uv.lock` resolves `pyproject.toml` as of the release run. Commands run from this folder, because the manifests and
the voice sample are found relative to it. `python -m nandini.cli` works as well as `nandini`.

**Seed-VC, running IndicVoiceChanger** (Python 3.10; needed by `generate` and `inventory convert`):

```bash
uv run nandini fetch seed-vc                         # external/seed-vc at SEED_VC_COMMIT
uv venv --python 3.10 external/seed-vc/.venv
uv pip install --python external/seed-vc/.venv/bin/python -r third_party/seed-vc-requirements.txt
git -C external/seed-vc apply "$PWD/third_party/seed-vc-mps-float32.patch"    # Apple silicon only
```

`generate` and `inventory convert` then find it there; `--seed-vc DIR --seed-vc-python PY` point elsewhere. On Apple
silicon also set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Training** (Python 3.10, Linux with NVIDIA GPUs for `train`; the other training commands also run on a Mac). Vāgdhenu's
requirements at his commit, with the two fixes found when they were installed: torch held at 2.4.1 (one of his packages
declares a newer torch, so a plain install would upgrade it), and `datasets`/`pyarrow` pinned (his list leaves them
open, and newer versions break his code). `train-constraints.txt` holds every package's tested version.

```bash
uv run nandini fetch vagdhenu bigvgan                # his code at VAGDHENU_COMMIT; BigVGAN, which his renderer imports
uv venv --python 3.10 external/train-env
uv pip install --python external/train-env/bin/python torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121          # on a Mac: leave out --index-url
uv pip install --python external/train-env/bin/python --override third_party/train-overrides.txt \
    -c third_party/train-constraints.txt -r third_party/train-requirements.txt
```

The training commands find it there and re-run themselves in it; `--train-python PY` points elsewhere. The IndicF5
base model is gated: accept its terms on [its model page](https://huggingface.co/ai4bharat/IndicF5) and log in.

**Indic-Speak** (Python 3.12; needed only by `pitch-shift readings`, to re-measure k). The model is gated: accept its
licence on [its model page](https://huggingface.co/bodhan-ai/indic-speak) and log in with `huggingface-cli login`. Its
code downloads with its weights.

```bash
uv venv --python 3.12 ../indic-speak-env
uv pip install --python ../indic-speak-env/bin/python -r third_party/indic-speak-requirements.txt
```

Then pass `--indic-speak-python ../indic-speak-env/bin/python`.

**Models and data** download from Hugging Face on first use, at the revisions pinned in `nandini/config.py`:
- the Vāgdhenu dataset;
- the converter, [IndicVoiceChanger](https://huggingface.co/DreamSyncCo/IndicVoiceChanger), a Seed-VC checkpoint;
- Su-śrotā;
- the speaker model, [ECAPA from SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

`--dataset-dir` reads the clips from a local copy instead.

## Same results, not identical audio

The same code, seeds and package versions give the same choices and nearly the same audio on another machine, but not
identical files. Conversions differ slightly between GPUs, and between a GPU and Apple silicon. Even the melody step,
which runs on the CPU, differs between Linux and macOS: re-making 610 clips on a Mac from the same conversions changed
the level of 21 of them by more than 0.1 dB, with the same package versions. So a rerun on other hardware can keep a
different attempt for some verses, or count an error more or fewer. Results on the same machine and environment are
identical. The released audio was made on an NVIDIA A40 under Linux.

## Getting the recordings

The source recordings are the [Vāgdhenu dataset](https://huggingface.co/datasets/prathoshap/vagdhenu-data). It is
public, needs no login, and is licensed CC-BY-4.0.

**Automatically:** `nandini manifest`, `nandini generate` and `nandini inventory` download what they need from Hugging
Face. They fetch the metadata and each clip the first time it is used, at the revision pinned in the config
(`[dataset] revision`), and keep them in the Hugging Face cache. Nothing needs installing first.

**All at once** (about 0.9 GB): useful on a cloud machine, or when several parts run in parallel:

```bash
huggingface-cli download prathoshap/vagdhenu-data --repo-type dataset \
  --revision adda7747c1ea05c45e6b789d23d6b6bbf550918d --local-dir vagdhenu-data
```

Then pass `--dataset-dir vagdhenu-data` to the commands. The folder should look like this:

```
vagdhenu-data/
  style_a/metadata.csv
  style_a/audio/<clip>.wav     764 clips
  style_b/metadata.csv
  style_b/audio/<clip>.wav     703 clips
```

Use the pinned revision. A later revision could change clips or texts, and the manifests' fixed indices and the seeds
derived from them would then no longer match the released audio.

**Attribution:** audio made from these recordings is derived from Vāgdhenu, so credit Vāgdhenu and its authors when
sharing it, as CC-BY-4.0 requires.

## Making the dataset

```bash
nandini generate
```

By default it runs every verse (`manifests/all_clips.csv`) into `out/`, with the settings in `nandini/config.py`.
`nandini config` prints them.

Attempt j converts the recording with seed 1000 · j + the verse's index, where the index is the verse's fixed position
in the whole dataset. It then applies the melody step, which:
- uses Vāgdhenu's pitch raised k semitones, with its voicing;
- follows Vāgdhenu's frame energy, but caps any frame's boost at 6 dB above the clip's typical gain, which stops near-silent
  stretches of a conversion from being blown up into spikes;
- writes 24 kHz audio at −20 dBFS.

Outputs:
- `out/final/<style>__<clip>.wav`: the kept clip for each verse.
- `out/settings_part*.json`: the settings the run used.
- `out/attempts_part*.csv`: every attempt, with its seed and errors.
- `out/kept_part*.csv`: one row per verse, with the kept attempt, its errors, the attempts used, and whether it met the
  bar.

Every conversion and melody-step file is kept under `out/conversions/` and `out/melody/`.

Options:
- `--part i/P` splits the run into P parallel parts.
- `--indices a,b` runs a subset; `--manifest` and `--out` choose other clips or another folder.
- A stopped run restarts where it left off.
- `--scoring halves` scores in two halves instead of whole clips (see below).

**Scoring in halves (`--scoring halves`):** on some long verses Su-śrotā drops a stretch of a whole clip (verse 1343
reads with 16 or more errors whole and 1 in halves), which wrongly marks good attempts as bad. With this option every
recording and attempt is cut at the half-verse pause, each half is transcribed alone, and the two transcripts are
joined and scored against the whole verse as before. The pause is found from Vāgdhenu's recording (the start of
syllable n // 2) and moved to the quietest 20 ms nearby in each clip. Errors, the bar and the rule are unchanged. It
costs two transcriptions per clip, plus one timed transcription of Vāgdhenu's recording per verse.

**How the released audio was made:**
- **Hardware:** one A40, with 4 parts in parallel.
- **Run size:** 4,766 attempts for 1,467 verses (3.25 per verse) in about 19 hours.
- **Result:** 1,212 verses (82.6%) met the bar, 492 of them with no errors; the other 255 are kept with their best of
  10.
- **Error rate:** 2.19% of counted syllables are wrong. This count comes from the same ASR that chose each attempt, so
  it is optimistic.
- **Re-check in halves:** the 255 verses that missed were scored again in halves, on the same 10 attempts. 86 then had
  an attempt with at most 1 error, and took the first such attempt (55 a different attempt, 31 the same one); the
  other 169 kept their best of 10. Result: 1,298 verses (88.5%) at the bar, and 1.97% of counted syllables wrong (a
  mix of whole and halves scoring). Training used this kept list.

**Why at most 1 error, not 0:** Su-śrotā misreads about 4.7% of syllables on Vāgdhenu's own recordings, so a single
error in a verse (about 2% of a typical 46-syllable verse) is within the judge's own noise. On the 61-clip sample, a
0-error rule needed about 1.7 times the attempts and lowered the error rate only from 2.1% to 1.7%.

## Checking the dataset

Everything after `nandini generate` works from its run folder (`out/`), plus Vāgdhenu's recordings, which come from
`--dataset-dir` or the Hugging Face cache as before:

| Data | Where it comes from |
|:--|:--|
| Vāgdhenu's recordings | `--dataset-dir`, else the Hugging Face cache (downloaded on first use) |
| Nandinī's clips | `out/final/` |
| Errors per verse | `out/kept_part*.csv` |
| Errors per syllable, what the ASR heard, timings | `out/marks/`, written by `nandini mark` |

```bash
nandini mark            # re-runs Su-śrotā on both recordings of every kept verse -> out/marks/ (--part i/P to split)
nandini tables          # -> out/tables/: transcripts.csv, syllables.csv.gz, substitutions.csv and .md
nandini report          # -> out/kept.csv, out/attempts.csv, out/report/: report.md and three charts
nandini listen build    # -> out/listen/index.html
nandini listen serve    # then open http://localhost:8765/listen/
```

- **The listening page** shows every verse with Vāgdhenu's recording, Nandinī's clip and the Devanagari text.
  - **Marks:** red marks errors, grey dotted marks syllables wrong on both recordings, and hovering shows what the ASR
    heard.
  - **While playing,** the syllable being sung is highlighted.
  - **Verdicts:** you can record a verdict and a note per verse. They're kept in the browser, with CSV export and
    import.
  - **Audio:** the page links to the audio wherever it is, so nothing is copied. Nandinī's clips play from
    `out/final/`, and Vāgdhenu's through links in `out/listen/vagdhenu/`.
  - **Serving:** the page needs the local server rather than opening the file directly, so the players can seek.
- **Without marks,** the page still works, but shows the text without syllable marks.
- **`nandini mark`** needs the ASR model but no GPU. It checks that each verse's error count matches the run's.
- **From the released dataset:** the dataset uses the same layout, so `--run` can point at a downloaded copy. That
  needs neither the converter nor the ASR.

## Measuring the converter's failures

`nandini inventory` measures how often the converter gets syllables wrong and why. It was used to choose the settings above.
For each clip in a manifest (the 61-clip sample; `manifests/all_clips.csv` runs the same way):

1. Convert the recording 10 times, with the same seeds as `nandini generate`.
2. Make a second version of each conversion with the melody step.
3. Transcribe Vāgdhenu's recording, every conversion and every melody-step version with Su-śrotā, and mark each
   syllable right or wrong against the text. Syllables Su-śrotā gets wrong on Vāgdhenu's own recording are left out
   of the counts.
4. Measure voice consistency with speaker embeddings over 2 s windows.

```bash
nandini inventory convert
nandini inventory score
nandini inventory report
```

The report gives:
- failure rates, and failures per sound class;
- how many conversions each failing syllable fails in;
- where a failure comes from (the converter or the melody step);
- how many failing syllables extra conversions rescue (3, 5, 7, 10);
- voice consistency;
- every headline measure per style, beside the ASR's error rate on Vāgdhenu's own recording.

By default it runs the 61-clip sample into `inventory/`. `--part`, `--indices`, `--clips` and `--dataset-dir` work as
for `nandini generate`.

## How the settings were chosen

These reproduce the measurements behind `nandini/config.py`. None of them is needed to make the dataset.

**The pitch shift k per style** (`K_SEMITONES`). The target voice reads 61 verses, and k is the gap between the mean log
pitch of those readings and that of Vāgdhenu's clips, per style:

```bash
nandini pitch-shift draw                 # the verses to read (as in manifests/pitch_shift_verses.csv)
nandini pitch-shift readings --indic-speak-python path/to/indic-speak-env/bin/python
nandini pitch-shift check                # token cap and Su-śrotā character error rate per reading
nandini pitch-shift vagdhenu             # Vāgdhenu's pitch, all clips
nandini pitch-shift target               # the readings' pitch
nandini pitch-shift report               # k per style with bootstrap intervals -> pitch_shift/report.md
```

- **The readings need Indic-Speak.** It is gated: accept its licence on Hugging Face first. It also runs in its own
  environment, like Seed-VC; the model's code downloads with its weights.
- **The released readings were made on Apple silicon** (`--device mps`). The other steps need only this package.

**The selection rule and the melody step**, on the inventory's clips (`nandini inventory convert` and `score`):

```bash
nandini selection simulate               # the rule on the inventory's attempts -> inventory/selection/
nandini selection threshold              # stopping at ≤ 1 error against 0, and Su-śrotā's misreads on Vāgdhenu
nandini melody-checks timing             # voicing-based timing check (inconclusive on chant; see its report)
nandini melody-checks voicing            # where the melody step imposes Vāgdhenu's voicing
nandini melody-checks loudness           # how much the loudness matching changes each conversion
nandini melody-checks loudness-cap       # without the 6 dB boost cap against with it (re-scores with Su-śrotā)
```

`--listen` cuts short listening clips around the worst stretches (Vāgdhenu, the conversion, the melody step).

## Training

Vāgdhenu's own recipe, unchanged: his `finetune_indicf5.py` (fetched into `external/vagdhenu`, not copied), his
settings (learning rate 1e-5, warm-up 500, 19,200 frames per GPU, 600 epochs, bf16, 2 GPUs) and his length: his
released model (`voice_armA`) is the EMA checkpoint at step 14,800, and so is this one.

```bash
nandini training-data                    # out/ -> training/dataset/: model_text, checks, raw.arrow + duration.json
nandini warm-start                       # training/warm.pt from the IndicF5 base, checked (0 missing, 0 unexpected)
nandini train                            # training/run/: stops after step 14,800; EMA kept every 2,000 and at 14,800
nandini female-bank                      # training/female_bank/: his reference cuts in her voice, each scored
```

- **Text:** his `model_text` of each verse (punctuation stripped, Kannada script, no sandhi), as his training read it.
  Every character must be in his `vocab.txt` (IndicF5 silently reads unknown ones as a space), except vocalic ḷ in one
  word of one clip, which his own trainer read as a space too.
- **`train`:** his script through `accelerate launch`. Beside it, the EMA weights of the kept steps are extracted and the
  other checkpoints deleted (each is 5.4 GB). It stops on a wrong warm-start line, a step count that hasn't moved for
  10 minutes, or a nearly full disk. `--nccl-p2p-off` is for hosts where the first GPU-to-GPU transfer hangs.
  IndicF5 can't resume from its own checkpoints, so a run folder that already holds some is refused, not deleted.
- **The female bank:** his renderer copies voice, melody and pace from a meter-matched reference clip in its bank. The
  same cuts are taken from the kept clips, so only the voice changes; `manifests/female_bank_overrides.csv` lists the
  three entries changed by hand and why.
- **Run folder:** `training-data` and `female-bank` read `out/` (its `kept_part*.csv` and `final/`); `--kept` and
  `--audio-dir` read another kept list.
- **Where files go:** the default folders (`out/`, `training/`, `inventory/`, `pitch_shift/`) are inside this folder,
  wherever a command is run from, so the commands find each other's outputs without flags. All are gitignored.

**Checks after training** (render in the training environment, score in this one):

```bash
nandini speech-check render --ema training/run/ema/ema_14800.pt    # 3 training verses: does the model speak?
nandini speech-check score                                          # every render at most 30% character errors
nandini trend-check render --ema training/run/ema/ema_12000.pt training/run/ema/ema_14800.pt
nandini trend-check score                                           # 12,000 against 14,800: still improving or levelled off
```

The trend check renders the 10 check verses (`manifests/training_check_verses.csv`) exactly as his website does, with the
female bank and vocos (his BigVGAN adds a frame-rate buzz on a female voice). The score is Su-śrotā's error rate against
the text, with a paired bootstrap over the verses, plus voice consistency against the training clips, reported without
pass/fail. These are training verses: the checks show that the model speaks and that training levelled off, not that it
generalises.

**As released:** RunPod, 2 × RTX A6000 (48 GB each): 0.88 s per step, 216 min for 14,800 steps.

**The vocoder.** The model makes a mel spectrogram; a vocoder turns it into sound. Vocos (IndicF5's own) adds a shiver on
long held notes, which is why Vāgdhenu fine-tuned NVIDIA's BigVGAN-v2 on IndicF5's mel of his voice. His BigVGAN adds a
frame-rate echo on a female voice, so it is fine-tuned again, on these clips, the way he made his: IndicF5's mel of
every training clip, NVIDIA's `train.py` unchanged in its fine-tuning mode with its v2 config, and the final weights the
average of the last 6 checkpoints. It starts from his generator and NVIDIA's released discriminators.

```bash
uv pip install --python external/train-env/bin/python -c third_party/train-constraints.txt -r third_party/vocoder-requirements.txt
nandini vocoder data                     # IndicF5 mels of the training clips, and NVIDIA's file lists
nandini vocoder align                    # before training: his BigVGAN on 3 training-style mels (timing and echo)
nandini vocoder score --align            #   ... and whether it keeps the words
nandini trend-check render --ema training/run/ema/ema_14800.pt --verses V.csv --save-mel --out M   # mels to monitor
nandini vocoder train &                  # NVIDIA's train.py; runs until the monitor stops it
nandini vocoder monitor --mels M         # every 2,500 steps: echo and shiver; stops when levelled off; averages
nandini vocoder decode --mels T --vocoder training/vocoder/run/vocoder.pt --out D   # any saved mels, any vocoder
nandini vocoder score --decoded vocos=D1 fine-tune=D2 --verses T/verses.csv --out R  # the pass rule
```

- **A step** is one update on 4 random 2.73 s chunks, each from one clip, so about 367 steps go through every clip once.
- **The monitor** stops at the first check from step 10,000 where the echo rule holds and shiver hasn't changed over the
  last 5,000 steps (paired bootstrap), or at 30,000 steps.
- **The pass rule:** on renders the monitor never saw, no echo (mean ≤ 0.02, each ≤ 0.04), shiver below vocos's, and
  character errors at most vocos's + 0.25 points.
- **Measures:** echo is the frame-rate peak in the cepstrum; shiver is the share of a held note's loudness modulation at
  5–30 Hz (`nandini/analysis/vocoder_measures.py`).
- `decode` joins each verse's pieces as his renderer does: with vocos it reproduces the render exactly.

## Comparing with Vāgdhenu

Nandinī (N) against Vāgdhenu's own models: V (`voice_armA`, trained like N) and Vs (`voice_steer`, what his website
serves). Every system runs through his website's path, his renderer unchanged: the verse text as given, his meter
detection and fallback, `render_one(text, meter, seed=60)` with his defaults. Only the model, reference bank and vocoder
file differ (V and Vs: his; N: ours).

```bash
nandini compare render --system N --ema EMA --bank BANK_DIR --vocoder VOCODER   # training environment
nandini compare render --system V
nandini compare render --system Vs
nandini compare score                                                          # this environment (Su-śrotā)
```

| Set | Verses | Systems | Compared against |
|:--|:--|:--|:--|
| A | the 61 pitch-shift verses (training verses) | N, V, Vs | Vāgdhenu's recording of each |
| B | the 60 [test verses](#test-verses) | N, V, Vs | V's render (the main test, below) |
| C | his 6 demo verses (`docs/index.html`) | Vs | his demo MP3s: do we run his system as he does? |


- **Main test, set B (N against V), fixed before rendering:** mean character errors at most V's + 2.0 points; mean melody correlation
  with V's render at least 0.8; mean duration ratio within 1 ± 0.10.
- **Melody correlation:** pitch curves (pYIN, 70–500 Hz) in semitones around each curve's own median, aligned by DTW
  on MFCCs, Pearson r over frames voiced in both.
- **Voice, every system (no pass/fail):** speaker cosine to her training clips, the lowest 2 s window's cosine (a voice
  switch), median F0, and the pitch offset N − V per verse against the dataset's shift k.
- Sets are reported separately in `compare/report.md`; per render in `scores.csv`, per pair in `pairs.csv`.

## Test verses

`manifests/test_verses.csv` holds 60 verses for evaluating a chant model on text it never saw. They are in the dataset's
meters but outside it, and anyone may reuse them as data.

```bash
nandini fetch dcs     # the Digital Corpus of Sanskrit, only the two works used, at a pinned commit
nandini test-verses    # rebuilds manifests/test_verses*.csv (training environment)
```

- **Source:** the [Digital Corpus of Sanskrit](https://github.com/OliverHellwig/sanskrit) (Oliver Hellwig, CC BY 4.0):
  the Bhāgavata Purāṇa and the Mukundamālā, devotional verse close to the dataset's.
- **Text:** printed as the dataset prints its text (Devanagari, words joined at consonant ends, avagraha joined, no
  daṇḍas); the DCS reading in IAST and its sentence ids are kept beside it.
- **Kept:** Vāgdhenu's meter detector recognises the meter; no half-verse appears in any dataset verse (letters,
  spelling normalised); every character is in Vāgdhenu's vocabulary; the length is within the dataset's range.
- **Drawn:** 60, split by the dataset's meter-family shares over the families the source offers, seed 20261117.
  Mālinī, sragdharā, rucirā and the dataset's "other" have no recognised verse in these works and aren't covered.

## Settings

Every setting is in `nandini/config.py`, with a comment on each:
- the pinned dataset and model revisions; diffusion steps (30) and guidance (0.7);
- at most 10 attempts per verse, their seeds (seed step 1000), and at most 1 error to stop;
- k per style (+6.74 semitones for style_a, +6.58 for style_b; each is the gap between Vāgdhenu's mean log pitch and
  the female voice's own reading of the same verses);
- the WORLD frame step and pitch range, and the energy boost cap (6 dB);
- the voice sample and the voice-consistency window.

`nandini config` prints them, and every `nandini generate` run writes them to `out/settings_part*.json`.
**Changing any setting means the results no longer match the released audio.**

## Tests

```bash
uv sync --group test
uv run pytest
```

About 6 seconds, offline, with no models loaded. They check that:
- the settings are the release's, and `nandini config` prints every one;
- the manifests have their fixed indices, and the sample keeps the full run's;
- every verse's syllable count fits its meter;
- error marking flags a changed or dropped syllable, and nothing on an exact transcript;
- the melody step lands on Vāgdhenu's pitch + k at −20 dBFS without clipping;
- the command's defaults and argument checks work;
- `nandini tables`, `report` and `listen build` work on a small synthetic run folder;
- the selection rule stops and keeps as it should, and the pitch shift's draw allocation and helpers are right;
- `nandini fetch` checks out the pinned commit and refuses another, and `external/` is gitignored;
- the training length and schedule (14,800 steps; 600 epochs end at 29,400 on these clips), and every kept step is
  one the trainer saves;
- the training commands say what to make when the training environment is missing;
- the check verses are the seed-20261109 draw, and a bank cut keeps the syllables that start inside it.

A full check needs the models and Seed-VC: `nandini generate --indices 389 --out check ...` should keep attempt 0 with 1 error, as in the released run. On the same machine and environment the audio is identical sample for sample; across GPUs, conversions can differ slightly.

## The voice sample

The female voice comes from one 25 s clip, `voice/female_voice_sample.wav`, which sets the voice the converter aims
for (`--voice` replaces it). It was generated with [Indic-Speak](https://huggingface.co/bodhan-ai/indic-speak)
(revision `76e0814`, speaker Bharati), under the Bodhan AI Open Model License (see [Licences](#licences)).

## Licences

The code is licensed under the [Apache License 2.0](LICENSE): use it, change it, build on it and sell what you make,
as long as you keep the licence and the [NOTICE](NOTICE) file and say what you changed.

Some parts carry their own terms (details in [NOTICE](NOTICE)):
- **`voice/female_voice_sample.wav`** is output of [Indic-Speak](https://huggingface.co/bodhan-ai/indic-speak) and is
  under the Bodhan AI Open Model License: keep its credit, and its acceptable-use restrictions apply to whatever you
  make with it.
- **`third_party/seed-vc-mps-float32.patch`** modifies Seed-VC and is GPL-3.0.
- **Audio made with this code** is derived from the Vāgdhenu dataset (CC-BY-4.0): credit Vāgdhenu and its authors.
- **`manifests/test_verses*`** are texts from the Digital Corpus of Sanskrit (Oliver Hellwig), CC BY 4.0: credit the
  DCS.
- **The models the code downloads** keep their own licences: IndicVoiceChanger and Seed-VC (GPL-3.0; Seed-VC runs as
  a separate program and is not included), SpeechBrain's ECAPA model (Apache-2.0), and Su-śrotā (no licence stated
  on its model card).
