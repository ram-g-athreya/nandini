---
title: Nandinī — Sanskrit Chant in a Female Voice
emoji: 🪷
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Sanskrit verse to chant, in Nandinī's female voice or beside Vāgdhenu's
---

# Nandinī — Sanskrit chant in a female voice (demo)

Paste a Sanskrit verse in any Indian script and hear it chanted by [Nandinī](https://huggingface.co/ramgathreya/nandini),
by [Vāgdhenu](https://huggingface.co/prathoshap/vagdhenu), or by both side by side. Both run through Vāgdhenu's own
renderer with his website's settings; only the voice, the vocoder and the reference bank differ.

## Deploying

This folder is the whole Space. Nothing else from the Nandinī repository is needed:

```bash
huggingface-cli upload <user>/nandini-demo . --repo-type space
```

- **Hardware:** ZeroGPU (the app takes a GPU only while a verse renders). Any GPU Space, or a machine with a GPU, also
  works with `python app.py`. On a CPU a verse takes several minutes.
- **First start** clones Vāgdhenu's code and NVIDIA's BigVGAN at pinned commits into `_external/`, then downloads both
  models from Hugging Face at pinned revisions.
- **While `ramgathreya/nandini` is private,** add an `HF_TOKEN` secret that can read it.
- **Limits** are Vāgdhenu's own (`limits.py` in his code): one verse per request and 10 chants per visitor per day.

## Running locally

```bash
pip install -r requirements.txt gradio==5.49.1
python app.py                   # then open http://127.0.0.1:7860
```

On Apple silicon it runs on the GPU (MPS); `VAGDHENU_SRC=/path/to/vagdhenu/src` reuses an existing checkout.

## Licences

The app is Apache-2.0. Vāgdhenu's code and weights are Apache-2.0 (Prof. Prathosh A P, IISc Bengaluru); IndicF5 and
BigVGAN-v2 are MIT. Nandinī's voice comes from a sample made with [Indic-Speak](https://huggingface.co/bodhan-ai/indic-speak)
under the Bodhan AI Open Model License, whose acceptable-use restrictions apply to everything made with it: no voice
clones of real people without consent, no robocalls or voice phishing.
