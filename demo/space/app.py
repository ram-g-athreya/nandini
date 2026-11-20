"""Nandinī — Sanskrit chant in a female voice, beside Vāgdhenu's (a Gradio app for a Hugging Face Space).

Chants a verse with Nandinī (ramgathreya/nandini), or with Nandinī and Vāgdhenu (prathoshap/vagdhenu, the model his
website serves) side by side, to compare. Both go through Vāgdhenu's own renderer, unchanged: his text handling (any Indic script,
Kannada routing, meter detection) and his website's settings. Only the voice, the vocoder and the reference bank differ.

Vāgdhenu's code and NVIDIA's BigVGAN are cloned at pinned commits on first start, never copied here; the weights
download from Hugging Face at pinned revisions. On a ZeroGPU Space the GPU is taken only while a verse renders.

Local run (a CUDA GPU, Apple silicon or the CPU):
    pip install -r requirements.txt gradio==5.49.1 && python app.py
Set VAGDHENU_SRC to an existing checkout's src/ to skip the clone.
"""
import json
import os
import subprocess
import sys
import time

import gradio as gr

try:
    import spaces
    GPU = spaces.GPU
except ImportError:  # outside Hugging Face: no ZeroGPU, run as is
    def GPU(duration=60):
        return lambda f: f

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple silicon: one BigVGAN operation has no MPS kernel

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("NANDINI_DEMO_CACHE", os.path.join(HERE, "_external"))

# Pinned outside code and weights (the same as the Nandinī repository's nandini/config.py)
VAGDHENU_REPO = "https://github.com/prathoshap/vagdhenu.git"
VAGDHENU_COMMIT = "8bd7733dd1aba6352235fe83eb3dfac322a343eb"
BIGVGAN_REPO = "https://github.com/NVIDIA/BigVGAN.git"
BIGVGAN_COMMIT = "7d2b454564a6c7d014227f635b7423881f14bdac"
NANDINI_HF = "ramgathreya/nandini"
NANDINI_REV = "10a125355a3128751d9c0819db889c21c580556c"
VAGDHENU_HF = "prathoshap/vagdhenu"
VAGDHENU_REV = "0b8b7474c1899471b3d2b6a3ee919ab6c9373ebc"
NANDINI_FILES = {"voice": "voice_nandini_ema_2026-11-15.pt", "vocoder": "voc_bigvgan_nandini_2026-11-17.pt"}
VAGDHENU_FILES = {"voice": "voice_steer_ema_2026-06-17.pt", "vocoder": "voc_bigvgan_EMA_2026-06-11.pth"}
SEED = 60  # his website's


def _clone(url, commit, name):
    """A checkout of url at commit under CACHE (made once)."""
    d = os.path.join(CACHE, name)
    if not os.path.isdir(os.path.join(d, ".git")):
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(["git", "clone", "-q", url, d], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()
    if head != commit:
        subprocess.run(["git", "checkout", "-q", commit], cwd=d, check=True)
    return d


SRC = os.environ.get("VAGDHENU_SRC") or os.path.join(_clone(VAGDHENU_REPO, VAGDHENU_COMMIT, "vagdhenu"), "src")
sys.path.insert(0, SRC)
try:
    import bigvgan  # noqa: F401
except ImportError:
    sys.path.insert(0, _clone(BIGVGAN_REPO, BIGVGAN_COMMIT, "BigVGAN"))

import limits  # noqa: E402  (his: one shloka per request, a daily quota per visitor)
from chant_text import detect_meter_key  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

HIS_BANK = os.path.join(SRC, "reference_bank", "bank.json")
VOCAB = os.path.join(SRC, "reference_bank", "vocab.txt")  # IndicF5's tokenizer vocab, the same in both releases

# Meters: the two banks share their keys (the female bank is cut from the same verses as his)
_bank = json.load(open(HIS_BANK, encoding="utf-8"))
METERS = [k for k, v in _bank.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v]
_FIRST = [m for m in ("anuṣṭubh", "upajāti", "śārdūlavikrīḍita", "vasantatilakā", "mālinī") if m in METERS]
METERS = _FIRST + [m for m in METERS if m not in _FIRST]
FALLBACK = "vasantatilakā" if "vasantatilakā" in METERS else METERS[0]
ALIAS = {}
for _k, _v in _bank.items():
    if not _k.startswith("_") and isinstance(_v, dict) and "wav" in _v:
        ALIAS[_k.lower()] = _k
        ALIAS[_v["wav"].removesuffix(".wav").lower()] = _k
AUTO = "__auto__"
METER_CHOICES = [("Auto-detect (recommended)", AUTO)] + [(m, m) for m in METERS]

NANDINI, BOTH = "Nandinī", "Nandinī and Vāgdhenu, side by side"
VAGDHENU = "Vāgdhenu"  # not offered on its own: his own Space chants in his voice

# Sample verses, each complete so the meter can be detected: (label, verse, meter). His detector misses
# drutavilambita and mālinī, so those two give their meter.
_NARASIMHA = ("हठलुठ दल घिष्टोत्कण्ठदष्टोष्ठ विद्युत्\nसटशठ कठिनोरः पीठभित्सुष्ठुनिष्ठाम् ।\n"
              "पठतिनुतव कण्ठाधिष्ठ घोरान्त्रमाला\nदह दह नरसिंहासह्यवीर्याहितं मे ॥")
SAMPLES = [
    ("Kṛṣṇa — Vasudevasutaṃ · anuṣṭubh",
     "वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\nदेवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥", AUTO),
    ("Viṣṇu — Śuklāmbaradharaṃ · anuṣṭubh",
     "शुक्लाम्बरधरं विष्णुं शशिवर्णं चतुर्भुजम् ।\nप्रसन्नवदनं ध्यायेत् सर्वविघ्नोपशान्तये ॥", AUTO),
    ("Sarasvatī — Yā kundendu · śārdūlavikrīḍita",
     "या कुन्देन्दुतुषारहारधवला या शुभ्रवस्त्रावृता\nया वीणावरदण्डमण्डितकरा या श्वेतपद्मासना ।\n"
     "या ब्रह्माच्युतशङ्करप्रभृतिभिर्देवैः सदा वन्दिता\nसा मां पातु सरस्वती भगवती निःशेषजाड्यापहा ॥", AUTO),
    ("Bhāgavatam 1.1.4 — Nigamakalpataroḥ · drutavilambita",
     "निगमकल्पतरोर्गलितं फलं शुकमुखादमृतद्रवसंयुतम् ।\nपिबत भागवतं रसमालयं मुहुरहो रसिका भुवि भावुकाः ॥", "drutavilambita"),
    ("Narasiṃha — a retroflex tongue-twister · mālinī", _NARASIMHA, "mālinī"),
    ("Karāgre vasate · anuṣṭubh",
     "कराग्रे वसते लक्ष्मीः करमध्ये सरस्वती ।\nकरमूले तु गोविन्दः प्रभाते करदर्शनम् ॥", AUTO),
    ("Gaṇeśa — Vakratuṇḍa · anuṣṭubh",
     "वक्रतुण्ड महाकाय सूर्यकोटिसमप्रभ ।\nनिर्विघ्नं कुरु मे देव सर्वकार्येषु सर्वदा ॥", AUTO),
]

_RENDERERS = {}


def _assets(voice):
    """(voice, vocoder, bank.json) for a voice, downloaded once."""
    if voice == NANDINI:
        d = snapshot_download(NANDINI_HF, revision=NANDINI_REV,
                              allow_patterns=[*NANDINI_FILES.values(), "reference_bank/*"])
        return (os.path.join(d, NANDINI_FILES["voice"]), os.path.join(d, NANDINI_FILES["vocoder"]),
                os.path.join(d, "reference_bank", "bank.json"))
    return (hf_hub_download(VAGDHENU_HF, VAGDHENU_FILES["voice"], revision=VAGDHENU_REV),
            hf_hub_download(VAGDHENU_HF, VAGDHENU_FILES["vocoder"], revision=VAGDHENU_REV), HIS_BANK)


_ASSETS = {v: _assets(v) for v in (NANDINI, VAGDHENU)}  # downloaded once at startup, off the GPU


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _renderer(voice):
    if voice not in _RENDERERS:
        from render_core import Renderer
        v, voc, bank = _ASSETS[voice]
        _RENDERERS[voice] = Renderer(v, voc, bank, device=_device(), vocab_file=VOCAB)
    return _RENDERERS[voice]


def _meter(text, choice):
    """(meter, status line)."""
    if choice and choice != AUTO:
        return choice, f"Meter: **{choice}**"
    k = ALIAS.get((detect_meter_key(text) or "").lower())
    if k:
        return k, f"Detected meter: **{k}**"
    return FALLBACK, (f"Couldn't detect the meter, so it is chanted as **{FALLBACK}**. Paste the complete verse, "
                      f"or choose the meter under Advanced.")


# ZeroGPU charges each call for the time it asks for, against a small daily allowance per visitor, so ask for
# little: a verse takes about 20 s on the GPU, the first one included (measured on this Space).
@GPU(duration=45)
def _render_one(voice, text, meter, seed):
    t = time.time()
    out = _renderer(voice).render_one(text, meter, seed=int(seed))
    print(f"rendered {voice}: {len(out[1]) / out[0]:.1f} s of audio in {time.time() - t:.1f} s", flush=True)
    return out


def chant(text, voice, meter_choice, seed, request: gr.Request):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please paste a verse first 🙏")
    msg = limits.validate_one_shloka(text)
    if msg:
        raise gr.Error(msg)
    if not limits.check_and_count(limits.client_ip(request)):
        raise gr.Error(f"You've reached today's limit of {limits.DAILY_LIMIT} chants. Please come back tomorrow.")
    meter, status = _meter(text, meter_choice)
    try:
        if voice == BOTH:
            # one GPU call per voice: ZeroGPU limits how long a visitor's single call may ask for
            n, v = _render_one(NANDINI, text, meter, seed), _render_one(VAGDHENU, text, meter, seed)
        else:
            n, v = _render_one(NANDINI, text, meter, seed), None
    except Exception as e:
        if "quota" in str(e).lower():
            raise gr.Error("Today's free GPU time for your connection is used up. Please try again tomorrow, or sign in "
                           "to Hugging Face on this page for more.")
        raise gr.Error(f"Sorry, rendering failed: {e}")
    return n, v, status


def _show(voice):
    return gr.update(visible=voice == BOTH)


INTRO = """\
# 🪷 Nandinī — Sanskrit chant in a female voice
Turn a **Sanskrit verse into traditional chant**. Paste a verse and press **Chant it**.

- Works with **any Indian script**: Devanagari, Kannada, Telugu, Malayalam and more. It's detected automatically.
- The **meter is detected for you**. (You can set it yourself under *Advanced*.)
- Choose **side by side** to hear [Vāgdhenu](https://prathosh.in/vagdhenu/)'s male voice chant it too.
- Put each line or half-verse on its own line, or separate them with `।` and `॥`.
"""

FOOTER = """\
---
The first chant takes a minute or so while the voice loads; after that it's quicker.
For classical (non-Vedic) verse only; please remove accent marks first.

Nandinī is built on **Vāgdhenu** by Prof. Prathosh A P, Indian Institute of Science, Bengaluru
([website](https://prathosh.in/vagdhenu/) · [code](https://github.com/prathoshap/vagdhenu)), and chants through
Vāgdhenu's own system. Nandinī: [website](https://ram-g-athreya.github.io/nandini/) ·
[code](https://github.com/ram-g-athreya/nandini) · [weights](https://huggingface.co/ramgathreya/nandini) ·
[dataset](https://huggingface.co/datasets/ramgathreya/nandini-data).
Nandinī's voice comes from a sample made with [Indic-Speak](https://huggingface.co/bodhan-ai/indic-speak) (Bodhan AI
Open Model License): no voice clones of real people without their consent, and no robocalls or voice phishing.
"""

# The website's colours: plum for Nandinī (and the page), warm gold as the second colour
PLUM = gr.themes.Color(c50="#fdf2f6", c100="#fbe6ee", c200="#f3cddc", c300="#e6a6c0", c400="#cf7099",
                       c500="#b04477", c600="#9b2d5c", c700="#82244d", c800="#6e1d41", c900="#5a1836", c950="#3a0d22",
                       name="plum")
# The website's fonts: Spectral, with Noto Serif Devanagari for the verses (Spectral has no Devanagari)
THEME = gr.themes.Soft(primary_hue=PLUM, secondary_hue="amber", neutral_hue="stone", text_size=gr.themes.sizes.text_lg,
                       font=[gr.themes.GoogleFont("Spectral"), gr.themes.GoogleFont("Noto Serif Devanagari"),
                             "Georgia", "serif"]).set(
    button_primary_background_fill="#9b2d5c", button_primary_background_fill_hover="#6e1d41",
    button_primary_text_color="#ffffff", body_background_fill="#fdf9fa",
    link_text_color="#9b2d5c", link_text_color_hover="#6e1d41", link_text_color_visited="#9b2d5c")
CSS = """
.gradio-container {max-width: 1100px !important}
#intro h1 {font-weight: 700; color: #6e1d41; letter-spacing: -0.01em}
#verse textarea {font-family: 'Noto Serif Devanagari', 'Spectral', serif; font-size: 1.15rem; line-height: 1.9}
.gallery {font-family: 'Spectral', 'Noto Serif Devanagari', serif}
#chant-btn {font-size: 1.15rem; font-weight: 600}
#advanced .label-wrap .icon {font-size: 0.7rem; color: #8c7b84}  /* the open/close arrow: smaller, as in Vāgdhenu's app */
#samples > .label {display: none}  /* the examples' empty label still shows a list icon */
"""

with gr.Blocks(title="Nandinī — Sanskrit chant", theme=THEME, css=CSS) as demo:
    gr.Markdown(INTRO, elem_id="intro")
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(label="Your verse — one shloka at a time", value=SAMPLES[0][1], lines=4, elem_id="verse",
                             placeholder="Paste a single Sanskrit verse in any Indian script…")
            gr.Markdown(f"<sub>One shloka per chant · up to {limits.DAILY_LIMIT} chants per day.</sub>")
            voice = gr.Radio([NANDINI, BOTH], value=NANDINI, label="Voice")
            with gr.Accordion("⚙️ Advanced (optional)", open=False, elem_id="advanced"):
                meter = gr.Dropdown(METER_CHOICES, value=AUTO, label="Meter (chandas)",
                                    info="Leave on Auto-detect unless you know the meter.")
                seed = gr.Slider(0, 1000, value=SEED, step=1, label="Seed",
                                 info="Change it for a different take of the same verse.")
            btn = gr.Button("🎧 Chant it", variant="primary", size="lg", elem_id="chant-btn")
        with gr.Column(scale=2):
            out_n = gr.Audio(label="Nandinī", type="numpy", autoplay=False)
            out_v = gr.Audio(label="Vāgdhenu", type="numpy", autoplay=False, visible=False)
            status = gr.Markdown("")
    voice.change(_show, voice, out_v)
    btn.click(chant, [txt, voice, meter, seed], [out_n, out_v, status])
    gr.Markdown("### 📜 Sample shlokas — click one to load it, then press **Chant it**\n"
                "Or pick any verse from the **[Bhagavad Gītā](https://sanskritdocuments.org/doc_giitaa/bhagvadnew.html)** "
                "and paste it above.")
    gr.Examples([[t, m] for _, t, m in SAMPLES], [txt, meter], example_labels=[n for n, _, _ in SAMPLES], label="",
                elem_id="samples")
    gr.Markdown(FOOTER)

if __name__ == "__main__":
    demo.launch()
